"""
generate_data.py
=================
Gen dữ liệu fine-tune NER (5 loại khái niệm y tế tiếng Việt) cho vinallama-2.7b-chat,
dùng LLM lớn qua OpenRouter làm "giáo viên" để sinh cặp (system_prompt, input_text, output_text).

Sửa các vấn đề đã bàn (bao gồm bug phát hiện từ output thật):
  1. Ép đủ 5 type: THUỐC, CHẨN_ĐOÁN, TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM
  2. mtsamples chỉ dùng để lấy tên thuốc (RxNorm-style) + mẫu vitals nén -> không dịch nguyên câu
  3. Giữ heading section trong input_text/system_prompt để không mất tín hiệu assertion
  4. Sample mtsamples trải đều nhiều medical_specialty + ép tỉ lệ assertion (isNegated/isFamily/isHistorical/none)
  5. Validate span tự động (text phải là substring chính xác của input_text) trước khi ghi ra file
  6. Few-shot gold thật (từ BTC) neo cứng logic assertion: thuốc lịch sử -> isHistorical,
     nhưng triệu chứng là CHỈ ĐỊNH điều trị đi kèm thì KHÔNG kế thừa assertion của thuốc
  7. Rule cứng cho vitals nén ("VS98.3 12987...") -> PHẢI là TÊN_XÉT_NGHIỆM/KẾT_QUẢ_XÉT_NGHIỆM,
     TUYỆT ĐỐI KHÔNG được gán TRIỆU_CHỨNG (bug thật đã xảy ra ở lần chạy trước)
  8. Rule cứng span tối giản: không được nhét chủ ngữ (bố/mẹ/gia đình) hay mốc thời gian
     vào trong span entity -- những cái đó chỉ thể hiện qua assertions, không phải qua text span
  9. Post-hoc validator (validate_no_leaked_context, validate_vitals_typing) tự động phát
     hiện & loại bỏ vi phạm rule 7/8 thay vì chỉ trông chờ LLM tuân thủ đúng 100%
  10. fix_negated_history_context: câu "không có tiền sử X" hay bị LLM gán NHẦM cả
      isHistorical lẫn sai type (case thật: "dị ứng thuốc" bị gán TRIỆU_CHỨNG +
      isHistorical thay vì CHẨN_ĐOÁN + isNegated) -- marker "tiền sử" nằm trong 1 cụm PHỦ
      ĐỊNH thì phải ưu tiên isNegated, không phải isHistorical. fix_missing_historical_marker
      cũng được sửa để không tự thêm isHistorical nhầm trong đúng case này.
  11. validate_treatment_purpose: loại TRIỆU_CHỨNG dạng cụm MỤC ĐÍCH điều trị (cụm
      động từ như "giảm đau", "hạ sốt") -- khác với triệu chứng cụ thể dạng danh từ
      ("đau nhức", "táo bón") mà gold example thật sự dùng.
  12. validate_procedure_not_diagnosis: loại CHẨN_ĐOÁN thực chất là thủ thuật/phẫu
      thuật (vd "phẫu thuật thay khớp gối") -- ngoài phạm vi 5 type, đề không có THỦ_THUẬT.
  13. Rule prompt (i): CHẨN_ĐOÁN là bệnh nền đang được 1 THUỐC isHistorical điều
      trị/kiểm soát trong section Tiền sử bệnh thì CŨNG gán isHistorical cho chính nó
      (case thật: "Metformin ... điều trị đái tháo đường type 2" -> đái tháo đường type 2
      PHẢI isHistorical). Tách biệt với rule (a) chỉ áp dụng cho TRIỆU_CHỨNG là chỉ định
      điều trị đi kèm (không kế thừa).
  14. [MỚI] RxNorm (RXNCONSO.RRF): nguồn tên thuốc thật đầy đủ liều+dạng bào chế, gộp
      vào drug_pool cùng mtsamples để đa dạng drug_hint hơn.
"""

import os
import re
import json
import time
import random
import string
import csv
import statistics
from pathlib import Path
from collections import Counter
from functools import lru_cache
from dotenv import load_dotenv
import requests
import pandas as pd
try:
    # Import khi dùng như package từ CLI trong ``scripts/data_gen``.
    from .gen_reject import (
        COMMON_LAB_PAIR_RE,
        INCOMPLETE_ENTITY_END_RE,
        NAMED_MEASUREMENT_RE,
        process_record,
    )
except ImportError:
    # Giữ tương thích khi chạy trực tiếp file core cũ.
    from gen_reject import (
        COMMON_LAB_PAIR_RE,
        INCOMPLETE_ENTITY_END_RE,
        NAMED_MEASUREMENT_RE,
        process_record,
    )

load_dotenv()

# ----------------------------------------------------------------------------
# CONFIG 
# ----------------------------------------------------------------------------
API_KEY = os.environ.get("OPENROUTER_API_KEY", "Chưa có key")
MODEL = os.environ.get("GEN_MODEL", "Chọn model đi")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

import xml.etree.ElementTree as ET

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SEED_DIR = BASE_DIR / "data" / "raw" / "Samples"
MTSAMPLES_PATH = BASE_DIR / "data" / "raw" / "mtsamples" / "mtsamples_clean.csv"
ICD10_PATH = BASE_DIR / "data" / "raw" / "icd10" / "icd102019en.xml"
RXNORM_PATH = BASE_DIR / "data" / "raw" / "rxnorm" / "RXNCONSO.RRF"
VIHEALTHQA_PATH = BASE_DIR / "data" / "raw" / "ViHealthQA" / "vihealthqa.csv"
OUTPUT_PATH = BASE_DIR / "data" / "synthetic" / "train.jsonl"
REJECT_PATH = BASE_DIR / "data" / "synthetic" / "reject.jsonl"
N_SAMPLES = 5
MAX_RETRY_PER_SAMPLE = 3
V4_MAX_RETRY_PER_SAMPLE = 2
SLEEP_BETWEEN_CALLS = 1
DEFAULT_PROFILE = "mixed_v5"  # bấm Run: augmentation chống overfit, ưu tiên NER boundary
V4_MAX_COMPLETION_TOKENS = 3200  # JSON của record 1.3k-4.5k ký tự cần dư chỗ cho entities

V4_COMMON_DRUGS_BY_SPECIALTY = {
    "Cardiovascular / Pulmonary": ["amlodipine 5 mg", "metoprolol 25 mg", "furosemide 40 mg", "aspirin 81 mg"],
    "Gastroenterology": ["omeprazole 20 mg", "lactulose 15 ml", "mesalazine 500 mg", "ondansetron 4 mg"],
    "Neurology": ["levetiracetam 500 mg", "carbamazepine 200 mg", "gabapentin 300 mg", "sumatriptan 50 mg"],
    "Orthopedic": ["paracetamol 500 mg", "ibuprofen 400 mg", "alendronate 70 mg", "colchicine 0.5 mg"],
    "Ophthalmology": ["timolol 0.5% nhỏ mắt", "latanoprost 0.005%", "moxifloxacin nhỏ mắt", "prednisolone acetate 1%"],
    "Obstetrics / Gynecology": ["sắt sulfat 325 mg", "acid folic 5 mg", "nifedipine 10 mg", "oxytocin 10 IU"],
    "Urology": ["tamsulosin 0.4 mg", "ciprofloxacin 500 mg", "oxybutynin 5 mg", "finasteride 5 mg"],
    "Nephrology": ["furosemide 40 mg", "sevelamer 800 mg", "calcitriol 0.25 mcg", "epoetin alfa"],
    "ENT - Otolaryngology": ["amoxicillin/clavulanate 875/125 mg", "cetirizine 10 mg", "fluticasone xịt mũi", "ciprofloxacin nhỏ tai"],
    "Hematology - Oncology": ["rituximab 375 mg/m2", "cyclophosphamide 500 mg", "hydroxyurea 500 mg", "sắt sulfat 325 mg"],
    "Dermatology": ["hydrocortisone 1% bôi tại chỗ", "doxycycline 100 mg", "cetirizine 10 mg", "terbinafine 250 mg"],
    "Endocrinology": ["metformin 500 mg", "insulin glargine", "levothyroxine 50 mcg", "propylthiouracil 50 mg"],
    "General Medicine": ["paracetamol 500 mg", "amlodipine 5 mg", "metformin 500 mg", "omeprazole 20 mg"],
}


def _percentile(values, percentile):
    """Percentile tuyến tính nhỏ gọn, tránh phụ thuộc numpy khi chỉ đọc metadata corpus."""
    if not values:
        return 0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


@lru_cache(maxsize=1)
def load_vihealthqa_style_profile(path=VIHEALTHQA_PATH):
    """Chỉ lấy thống kê hình thức ViHealthQA; tuyệt đối không trả nội dung câu gốc.

    V4 dùng các percentile độ dài để chọn kích thước văn bản hợp lý. Không một question,
    answer hay URL nào được đưa vào prompt, nên corpus chỉ đóng vai trò tham khảo phân bố.
    """
    path = Path(path)
    fallback = {
        "available": False,
        "rows": 0,
        "question_p50": 60,
        "answer_p50": 380,
        "combined_p75": 700,
        "combined_p90": 1050,
        "combined_p95": 1350,
    }
    if not path.exists():
        return fallback

    question_lengths = []
    answer_lengths = []
    combined_lengths = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                question = (row.get("question") or "").strip()
                answer = (row.get("answer") or "").strip()
                if not question and not answer:
                    continue
                question_lengths.append(len(question))
                answer_lengths.append(len(answer))
                combined_lengths.append(len(question) + len(answer))
    except (OSError, UnicodeError, csv.Error):
        return fallback

    if not combined_lengths:
        return fallback
    return {
        "available": True,
        "rows": len(combined_lengths),
        "question_p50": (
            round(statistics.median(question_lengths))
            if question_lengths else fallback["question_p50"]
        ),
        "answer_p50": (
            round(statistics.median(answer_lengths))
            if answer_lengths else fallback["answer_p50"]
        ),
        "combined_p75": _percentile(combined_lengths, 75),
        "combined_p90": _percentile(combined_lengths, 90),
        "combined_p95": _percentile(combined_lengths, 95),
    }

# Đa dạng phần mở đầu nhưng vẫn giữ lại format 3 mục cũ. Có thể chỉnh ba tỷ lệ này;
# tổng phải bằng 1.0. no_heading vẫn giữ ngữ nghĩa section bằng cue trong câu.
STRUCTURE_STYLE_WEIGHTS = {
    "classic_heading": 0.35,
    "alternative_heading": 0.25,
    "no_heading": 0.40,
}

# Không để LLM mặc định quay về "bệnh nhân 45 tuổi". Một phần record bỏ hẳn
# nhân khẩu học; phần còn lại nhận tuổi/giới được lấy mẫu trước trong prompt.
DEMOGRAPHIC_INCLUDE_PROBABILITY = 0.60


V2_FOCUS_AREAS = [
    {
        "key": "lab_abbrev_parentheses",
        "section_keys": ["danh_gia", "hien_tai"],
        "specialty_names": [
            "General Medicine", "Hematology - Oncology", "Endocrinology",
            "Nephrology", "Cardiovascular / Pulmonary",
        ],
        "instruction": """
GỢI Ý BỔ SUNG: LAB VIẾT TẮT + CHÚ THÍCH TRONG NGOẶC.
- Nếu phù hợp chuyên khoa, bổ sung 1-3 dòng xét nghiệm, ưu tiên: hct (hematocrit), tiểu cầu (platelets),
  hco3- (bicarbonate), ag (anion gap), bun/creatinine (ure/creatinine),
  glucose (đường huyết), lactate (acid lactat), bnp, troponin, kali.
- Toàn bộ tên/chú thích là một TÊN_XÉT_NGHIỆM; số hoặc số+đơn vị là
  KẾT_QUẢ_XÉT_NGHIỆM. Ví dụ hco3- (bicarbonate) -> TÊN_XÉT_NGHIỆM,
  20 -> KẾT_QUẢ_XÉT_NGHIỆM.
- Tuyệt đối không gán bicarbonate, anion gap, glucose, lactate, BNP, troponin
  hoặc kali thành THUỐC.
""",
    },
    {
        "key": "imaging_finding",
        "section_keys": ["danh_gia", "hien_tai"],
        "instruction": """
GỢI Ý BỔ SUNG: CHẨN ĐOÁN HÌNH ẢNH.
- Dùng một kỹ thuật X-quang, CT, MRI hoặc siêu âm Doppler làm TÊN_XÉT_NGHIỆM.
- Toàn bộ mô tả đứng sau kỹ thuật phải là một KẾT_QUẢ_XÉT_NGHIỆM trọn vẹn,
  kể cả câu phủ định như "không phát hiện gãy xương hoặc viêm xương tủy".
- Không tách các bệnh/tổn thương nằm bên trong finding thành CHẨN_ĐOÁN và
  không gán isNegated cho finding; assertions của kết quả hình ảnh luôn là [].
""",
    },
    {
        "key": "history_bullet",
        "section_keys": ["tien_su"],
        "specialty_names": [
            "General Medicine", "Cardiovascular / Pulmonary", "Endocrinology",
            "Nephrology",
        ],
        "instruction": """
GỢI Ý BỔ SUNG: BỆNH NỀN/TIỀN SỬ NỘI KHOA DẠNG BULLET.
- Dùng heading đồng nghĩa như "Tiền sử bệnh nội khoa" hoặc "Các bệnh mạn tính"
  và một vài gạch đầu dòng tự nhiên.
- Ưu tiên tăng huyết áp, đái tháo đường type 2, rối loạn lipid máu, tăng lipid
  máu, béo phì, bệnh thận mạn, suy tim, COPD, gout, nhiễm trùng tiết niệu tái phát.
- Mỗi bệnh nền là CHẨN_ĐOÁN với assertions: ["isHistorical"].
""",
    },
    {
        "key": "respiratory_emergency",
        "section_keys": ["danh_gia", "hien_tai"],
        "specialty_names": ["Cardiovascular / Pulmonary", "General Medicine"],
        "instruction": """
GỢI Ý BỔ SUNG: HÔ HẤP/CẤP CỨU/SPO2/VITALS.
- Có thể dùng triệu chứng/bệnh hô hấp như nhịp thở nhanh, thiếu oxy, khó thở,
  ho đờm vàng, chảy nước mũi, nhiễm trùng đường hô hấp trên hoặc viêm phổi kẽ.
- Ưu tiên thêm SpO2 hoặc độ bão hòa oxy: tên chỉ số là TÊN_XÉT_NGHIỆM,
  giá trị như "89% khi thở khí trời" hoặc "88-92 %" là KẾT_QUẢ_XÉT_NGHIỆM.
- Có thể thêm nhịp thở; tên và giá trị vẫn phải tách thành hai entity.
""",
    },
    {
        "key": "ulcer_foot_body_part",
        "section_keys": ["hien_tai", "danh_gia"],
        "specialty_names": ["Dermatology", "Endocrinology", "General Medicine"],
        "instruction": """
GỢI Ý BỔ SUNG: VẾT LOÉT/BÀN CHÂN/VỊ TRÍ CƠ THỂ.
- Dùng cụm đầy đủ như "loét mới ở ngón chân út bên phải", "vết loét chảy dịch
  vàng", "sưng đỏ quanh vết loét", "hoại tử đầu ngón chân cái bên trái" hoặc
  "đau bàn chân phải".
- Lấy trọn cụm biểu hiện làm TRIỆU_CHỨNG hoặc CHẨN_ĐOÁN theo ngữ cảnh.
- Không gán riêng ngón chân út, bàn chân phải, bên phải, bên trái hoặc phải/trái
  thành CHẨN_ĐOÁN.
""",
    },
    {
        "key": "historical_vs_acute_drug",
        "section_keys": ["hien_tai", "danh_gia", "tien_su"],
        "specialty_names": ["General Medicine"],
        "instruction": """
GỢI Ý BỔ SUNG: THUỐC TRƯỚC NHẬP VIỆN SO VỚI THUỐC XỬ TRÍ TẠI VIỆN.
- Khi phù hợp, đặt cạnh nhau một thuốc dùng tại nhà/trước nhập viện với
  assertions: ["isHistorical"] và một thuốc dùng một liều/xử trí tại cấp cứu
  hoặc tại viện với assertions: []. Không cần nhồi đủ nếu ngữ cảnh không tự nhiên.
- Viết rõ tín hiệu thời gian, không để hai ngữ cảnh nhập nhằng.
""",
    },
    {
        "key": "negation_list",
        "section_keys": ["hien_tai", "danh_gia"],
        "instruction": """
GỢI Ý BỔ SUNG: PHỦ ĐỊNH DẠNG DANH SÁCH.
- Có thể thêm một câu phủ nhận/không có/không thấy gồm 2-3 triệu chứng hoặc chẩn đoán
  nối bằng dấu phẩy, "và" hoặc "hoặc".
- Tách từng khái niệm riêng; tất cả đều có assertions: ["isNegated"].
- Phủ định nằm trong finding hình ảnh/lab vẫn là một KẾT_QUẢ_XÉT_NGHIỆM trọn
  vẹn với assertions: [], không áp dụng isNegated.
""",
    },
]

# Nhóm bù lỗi vòng review thứ ba. Đây chỉ là các gợi ý mềm cho prompt, không phải quota
# và không được dùng làm điều kiện reject mẫu.
V3_BARE_DRUG_FOCUS_WEIGHT = 5.0
V3_TIME_CONTEXT_FOCUS_WEIGHT = 4.0
V3_CARDIAC_TIMELINE_FOCUS_WEIGHT = 3.0

V3_FOCUS_AREAS = [
    {
        "key": "procedure_scope",
        "section_keys": ["tien_su", "hien_tai", "danh_gia"],
        "instruction": """
GỢI Ý BỔ SUNG V3: PHÂN BIỆT THỦ THUẬT VỚI CHẨN ĐOÁN.
- Có thể nhắc tự nhiên các thủ thuật như phẫu thuật thay van, chạy thận nhân tạo, ghép thận,
  nạo vét tổn thương, phaco, tán sỏi ngoài cơ thể, sinh thiết hoặc đặt stent, nhưng KHÔNG
  annotate bản thân thủ thuật vì schema không có type THỦ_THUẬT.
- Tình trạng bệnh/biến chứng vẫn annotate riêng: "ghép thận thất bại" và "suy thận mạn giai V"
  là CHẨN_ĐOÁN. Không được loại chúng chỉ vì chứa từ "ghép thận".
""",
    },
    {
        "key": "symptom_due_to_diagnosis",
        "section_keys": ["hien_tai", "danh_gia"],
        "instruction": """
GỢI Ý BỔ SUNG V3: TRIỆU CHỨNG DO CHẨN ĐOÁN.
- Sinh một cấu trúc tự nhiên như khó thở do viêm phổi, đau bụng do viêm dạ dày, đau ngực do
  nhồi máu cơ tim, giọng khàn do tổn thương dây thanh quản hoặc phù chân do suy tim.
- BẮT BUỘC tách phần biểu hiện thành TRIỆU_CHỨNG và phần nguyên nhân thành CHẨN_ĐOÁN;
  không gom toàn bộ "A do B" thành một CHẨN_ĐOÁN.
""",
    },
    {
        "key": "dynamic_labs",
        "section_keys": ["hien_tai", "danh_gia"],
        "specialty_names": [
            "General Medicine", "Nephrology", "Hematology - Oncology",
            "Cardiovascular / Pulmonary", "Endocrinology",
        ],
        "instruction": """
GỢI Ý BỔ SUNG V3: XÉT NGHIỆM ĐỘNG HỌC.
- Ưu tiên creatinine/Ure/CRP/Troponin tăng từ A lên B hoặc Kali/Hb/eGFR giảm từ A xuống B;
  có thể dùng "creatinine tăng từ 5.2 lên 6.3 mg/dl (460 - 557 umol/l)".
- Tên như "creatinine" là một TÊN_XÉT_NGHIỆM. Toàn bộ phần xu hướng từ "tăng/giảm từ..."
  đến hết giá trị, đơn vị và ngoặc quy đổi là MỘT KẾT_QUẢ_XÉT_NGHIỆM, không tách vụn số.
- Có thể thêm "photpho 8.4" hoặc "phospho máu 8.4 mg/dL", vẫn tách tên và kết quả.
- Mỗi xét nghiệm phải đủ đúng một tên và một kết quả trọn vẹn; không chỉ annotate các con số.
- Với kết quả đơn như "CRP tăng cao 15.2 mg/L", bám boundary gold:
  TÊN="CRP", KẾT_QUẢ="15.2 mg/L"; không đưa qualifier "tăng cao" vào span.
""",
    },
    {
        "key": "named_vitals_inr",
        "section_keys": ["hien_tai", "danh_gia"],
        "instruction": """
GỢI Ý BỔ SUNG V3: VITALS VÀ INR ĐỦ CẶP.
- Dùng một vài cặp tự nhiên: Nhiệt độ: 36.5 độ C; Mạch: 88 l/p; Huyết áp: 120/70 mmHg;
  Nhịp thở: 20 l/p; SpO2: 92 %.
- Tên chỉ số là TÊN_XÉT_NGHIỆM, giá trị là KẾT_QUẢ_XÉT_NGHIỆM.
- Với "INR dưới ngưỡng điều trị 1.7", annotate TÊN_XÉT_NGHIỆM="INR" và
  KẾT_QUẢ_XÉT_NGHIỆM="dưới ngưỡng điều trị 1.7" (tương tự "trên ngưỡng điều trị 3.5").
""",
    },
    {
        "key": "history_section_consistency",
        "section_keys": ["tien_su"],
        "instruction": """
GỢI Ý BỔ SUNG V3: BỆNH NỀN TRONG SECTION TIỀN SỬ.
- Liệt kê tự nhiên một vài bệnh như viêm nội tâm mạc, rối loạn chức năng tâm thất phải,
  rung nhĩ, suy thận mạn giai V, bệnh mạch vành, suy tim hoặc COPD.
- Vì chúng nằm trong Tiền sử bệnh/Tiền căn bệnh lý/Các bệnh lý mạn tính, mọi CHẨN_ĐOÁN
  khẳng định của bệnh nhân phải có isHistorical.
""",
    },
    {
        "key": "full_symptom_span",
        "section_keys": ["hien_tai", "danh_gia"],
        "instruction": """
GỢI Ý BỔ SUNG V3: SPAN TRIỆU CHỨNG TRỌN NGHĨA.
- Có thể dùng: đau vùng hạ vị bên phải và hạ vị bên trái; đau bụng vùng bụng dưới;
  đau RLQ/LLQ; đau tăng khi vận động; táo bón trở nên tồi tệ hơn; đổ mồ hôi qua đêm;
  đi ngoài ra máu.
- Lấy trọn cụm triệu chứng có vị trí/mức độ/diễn biến; không tách thành "đau bụng ở" hoặc
  chỉ riêng "hạ vị bên trái". Với "không ra máu", span là "ra máu" và có isNegated,
  không đưa chữ "không" vào span.
- Không bao giờ tạo entity một từ chỉ mức độ/hướng/thời gian như "hơi", "lên", "giây".
- Nếu cùng triệu chứng được nhắc lại ở hai vị trí thật trong văn bản thì annotate đủ cả hai
  lần; không tự lặp từ lỗi như "Khó thở nhẹ khó thở" trong cùng một cụm.
""",
    },
    {
        "key": "time_units_as_outside_context",
        "weight": V3_TIME_CONTEXT_FOCUS_WEIGHT,
        "section_keys": ["tien_su", "hien_tai", "danh_gia"],
        "instruction": """
GỢI Ý BỔ SUNG V3: THỜI ĐIỂM/THỜI LƯỢNG LÀ O, KHÔNG PHẢI ENTITY Y TẾ.
- Trong record hãy dùng tự nhiên 3-5 biểu thức thời gian khác nhau, chẳng hạn: "kéo dài
  20 giây", "sau 30 phút", "khởi phát lúc 17 giờ", "trong 2 ngày qua", "cách nhập viện
  1 tuần" hoặc "từ 3 tháng trước".
- Các số và đơn vị thời gian này đều là O: tuyệt đối không annotate "giây", "20 giây",
  "phút", "giờ", "ngày", "tuần", "tháng" hay "năm" thành CHẨN_ĐOÁN, TRIỆU_CHỨNG,
  TÊN_XÉT_NGHIỆM hoặc KẾT_QUẢ_XÉT_NGHIỆM.
- Chỉ lấy khái niệm y tế đứng cạnh chúng. Ví dụ "khó thở kéo dài 20 giây" chỉ annotate
  TRIỆU_CHỨNG="khó thở"; "thắt chặt ngực lúc 17 giờ" chỉ lấy "thắt chặt ngực".
- Không biến câu chỉ thời gian thành lab: "sau 30 phút" và "lúc 17 giờ" không phải kết quả.
- Ngoại lệ boundary đã có của THUỐC vẫn giữ nguyên: nếu tần suất/thời gian dùng nằm liền trong
  span thuốc như "ceftriaxone 1 g mỗi 8 giờ" thì toàn bộ vẫn là một THUỐC. Focus này chỉ dạy
  các mốc thời gian độc lập của triệu chứng/diễn biến là O.
""",
    },
    {
        "key": "previous_admission_drugs",
        "section_keys": ["tien_su", "hien_tai", "danh_gia"],
        "instruction": """
GỢI Ý BỔ SUNG V3: THUỐC Ở LẦN NHẬP VIỆN TRƯỚC.
- Sinh thuốc với marker rõ như "trong lần nhập viện trước", "trước đó", "tại thời điểm
  xuất viện trước", "đã dùng tại nhà" hoặc "thuốc trước nhập viện"; thuốc phải isHistorical.
- Ít nhất một thuốc trong focus này nên chỉ có tên, không kèm liều/dạng bào chế, ví dụ
  "đã dùng doxycycline tại nhà" hoặc "atenolol (uống hôm nay)". Vẫn phải annotate đúng
  "doxycycline"/"atenolol" là THUỐC isHistorical; không được chờ có số + đơn vị mới annotate.
- Marker có thể đứng sau thuốc, ví dụ "đã sử dụng ciproflagyl trong lần nhập viện trước";
  ciproflagyl vẫn phải có isHistorical.
- Không dùng riêng cụm mơ hồ "đã sử dụng" để suy ra thời điểm. Thuốc xử trí/được chỉ định
  trong lần điều trị hiện tại vẫn có assertions: [].
""",
    },
    {
        "key": "bare_drug_names_in_medication_lists",
        # Focus mới cần xuất hiện đủ trong batch nhỏ để sửa recall thuốc trần. Trọng số
        # chỉ ảnh hưởng lựa chọn bên trong phần V3, không biến thành quota/reject cứng.
        "weight": V3_BARE_DRUG_FOCUS_WEIGHT,
        "section_keys": ["tien_su", "hien_tai", "danh_gia"],
        "instruction": """
GỢI Ý BỔ SUNG V3: TÊN THUỐC KHÔNG KÈM LIỀU TRONG DANH SÁCH THUỐC.
- Sinh một danh sách ngắn gồm 2-3 thuốc, trộn thuốc có liều với thuốc chỉ có tên để model
  không phụ thuộc vào số/đơn vị. Ví dụ tự nhiên:
  "Thuốc trước khi nhập viện\nmetoprolol 25mg po bid\ndoxycycline cho viêm tuyến mồ hôi\natenolol (uống hôm nay)".
- BẮT BUỘC annotate cả thuốc không có liều: THUỐC="doxycycline" và THUỐC="atenolol".
  Không được bỏ thuốc chỉ vì phía sau không có mg/ml, dạng bào chế hoặc tần suất.
- Boundary phải dừng đúng cuối tên thuốc: trong "doxycycline cho viêm tuyến mồ hôi" chỉ lấy
  "doxycycline"; trong "atenolol (uống hôm nay)" chỉ lấy "atenolol". Không đưa "cho..."
  hoặc ngoặc mô tả thời điểm vào entity THUỐC.
- Nếu danh sách có heading "Thuốc trước khi nhập viện", "Thuốc dùng tại nhà" hoặc marker
  quá khứ rõ thì TẤT CẢ thuốc trong danh sách đều có isHistorical, kể cả "uống hôm nay".
  Nếu viết rõ "Điều trị hiện tại tại viện" thì tất cả thuốc tương ứng có assertions: [].
- Chẩn đoán chỉ định có thật trong câu vẫn annotate riêng: "viêm tuyến mồ hôi" là
  CHẨN_ĐOÁN. Không nuốt chẩn đoán vào span thuốc và không bỏ sót entity thuốc đứng trước nó.
- Có thể luân phiên các tên thuốc trần phổ biến như doxycycline, atenolol, metformin,
  amlodipine, aspirin, omeprazole, ceftriaxone hoặc paracetamol; không tự bịa hóa chất lạ.
""",
    },
    {
        "key": "long_imaging_monitor_finding",
        "section_keys": ["hien_tai", "danh_gia"],
        "specialty_names": [
            "Cardiovascular / Pulmonary", "General Medicine", "Nephrology",
            "ENT - Otolaryngology", "Orthopedic",
        ],
        "instruction": """
GỢI Ý BỔ SUNG V3: FINDING HÌNH ẢNH/MONITOR DÀI, KHÔNG TÁCH VỤN.
- Sinh một kết luận dài sau siêu âm/CT/MRI/X-quang hoặc Holter. Kỹ thuật đầy đủ như
  "chụp x-quang ngực", "siêu âm vùng cổ phải" hay "monitor Holter" là TÊN_XÉT_NGHIỆM.
- Toàn bộ finding liên tục phía sau là một KẾT_QUẢ_XÉT_NGHIỆM trọn nghĩa; không tách các từ
  "âm", "lên", kích thước hoặc tổn thương bên trong thành entity rời, và không đổi finding
  thành TRIỆU_CHỨNG/CHẨN_ĐOÁN.
- Ví dụ Holter: "nhịp xoang chiếm ưu thế, ghi nhận ngoại tâm thu nhĩ và ngoại tâm thu thất
  thường xuyên" là kết quả, không annotate riêng nhịp xoang/ngoại tâm thu thành CHẨN_ĐOÁN.
- Nếu tên kỹ thuật được nhắc lại dưới mục "Các thủ thuật đã thực hiện" thì lần nhắc mang
  nghĩa thủ thuật không annotate; chỉ annotate lần thực sự giới thiệu kết quả hình ảnh.
- Trong phần khám thực thể, không gán riêng tên cơ quan như "Phổi" hoặc "Tim" thành
  TÊN_XÉT_NGHIỆM. Thể tích hút/dẫn lưu như "0.5cc" là thông tin thủ thuật, không phải kết quả.
""",
    },
    {
        "key": "cardiac_timeline_holter_ecg",
        "weight": V3_CARDIAC_TIMELINE_FOCUS_WEIGHT,
        "section_keys": ["hien_tai", "danh_gia"],
        "specialty_names": ["Cardiovascular / Pulmonary", "General Medicine"],
        "instruction": """
GỢI Ý BỔ SUNG V3: DIỄN TIẾN TIM MẠCH, HOLTER VÀ ECG.
- Viết bệnh sử tự nhiên có 2-3 mention thật của đánh trống ngực/khó thở/thắt chặt ngực ở
  các vị trí khác nhau; annotate đủ từng mention theo thứ tự, không chỉ occurrence đầu tiên.
- Có thể dùng mốc "kéo dài 20 giây", "lúc 17 giờ", "sau 30 phút" nhưng toàn bộ số và đơn
  vị thời gian là O. Không annotate riêng "giây", "phút" hoặc "giờ".
- "monitor Holter cho thấy nhịp xoang chiếm ưu thế, ghi nhận ngoại tâm thu nhĩ và ngoại tâm
  thu thất thường xuyên": "monitor Holter" là TÊN_XÉT_NGHIỆM và toàn bộ phần sau là MỘT
  KẾT_QUẢ_XÉT_NGHIỆM; không gán nhịp xoang/ngoại tâm thu thành CHẨN_ĐOÁN.
- "ECG bình thường": ECG là TÊN_XÉT_NGHIỆM, "bình thường" là KẾT_QUẢ_XÉT_NGHIỆM.
- Trong "không liên quan đến gắng sức hoặc tư thế", gắng sức/tư thế chỉ là context và đều O.
  Nhưng cụm bệnh học đầy đủ "giảm dung nạp gắng sức" vẫn là TRIỆU_CHỨNG.
- Span triệu chứng phải dừng trước từ nối/thời gian: lấy "khó chịu vùng ngực", không lấy
  "khó chịu vùng ngực khi"; lấy "khó thở", không nuốt "kéo dài 20 giây".
- Thuốc dùng ở nhà/trước nhập viện có isHistorical; aspirin hoặc thuốc xử trí tại cấp cứu
  trong đợt hiện tại có assertions: []. Không tạo lỗi dính chữ như "atenololtrong".
""",
    },
    {
        "key": "dense_lab_bullets",
        "section_keys": ["danh_gia", "hien_tai"],
        "specialty_names": [
            "General Medicine", "Nephrology", "Hematology - Oncology",
            "Cardiovascular / Pulmonary",
        ],
        "instruction": """
GỢI Ý BỔ SUNG V3: DANH SÁCH LAB DÀY NHƯ BỆNH ÁN THẬT.
- Sinh 4-7 dòng lab ngắn, mỗi dòng vẫn phải đủ cặp tên/kết quả, ví dụ: bạch cầu: 13.9;
  neutrophil: 80%; lymphocyte: 11%; CK: 58 U/L; ALT: 92 U/L; HIV VL: đang chờ;
  soi tươi ký sinh trùng: âm tính; Chem 7: bình thường.
- "đang chờ", "âm tính" và "bình thường" là KẾT_QUẢ_XÉT_NGHIỆM. Không bỏ tên xét nghiệm,
  không annotate riêng con số mà quên tên, và không đảo thành câu khó hiểu như "bình thường chem 7".
- Khoảng một phần các dòng nên dùng thứ tự KẾT QUẢ đứng trước TÊN như bệnh án thật:
  "3.2 kali", "80% neutrophil", "11% lymphocyte", "478 tiểu cầu", "1.3 lactate",
  "0.01 troponin" hoặc "4227 BNP". Khi đó số/đơn vị đứng trước là KẾT_QUẢ_XÉT_NGHIỆM,
  từ lab đứng sau là TÊN_XÉT_NGHIỆM; danh sách entities vẫn sắp theo thứ tự xuất hiện nên
  entity KẾT_QUẢ được đặt trước entity TÊN trong riêng dòng này.
- Sinh thêm dạng giá trị rồi tên nằm trong ngoặc: "3.5 mmol/L (kali)", "138 mmol/L (natri)",
  "101 mmol/L (clo)", "2.8 mmol/L (Kali)", "135 mmol/L (Na)", "4.1 (K+)",
  "140 (Na+)", "98 (Cl-)", "8.4 mg/dL (phospho)", "0.5 mmol/L (Mg++)".
  Trong các dạng này, giá trị bên ngoài ngoặc là KẾT_QUẢ_XÉT_NGHIỆM; chỉ tên/ký hiệu nằm
  bên trong ngoặc là TÊN_XÉT_NGHIỆM, không đưa dấu ngoặc vào entity.
- Nếu có nhãn chung "điện giải đồ" hoặc "Ion đồ", nhãn chung đó cũng là
  TÊN_XÉT_NGHIỆM; sau đó vẫn annotate riêng từng giá trị và kali/natri/clo trong ngoặc.
- Mỗi entity phải là span có nghĩa hoàn chỉnh; cấm entity vụn như "âm", "thu", "nhĩ", "lên".
""",
    },
    {
        "key": "repeated_mentions_recall",
        "section_keys": ["hien_tai", "danh_gia"],
        "instruction": """
GỢI Ý BỔ SUNG V3: CÙNG KHÁI NIỆM XUẤT HIỆN NHIỀU LẦN.
- Chọn 1-2 triệu chứng hoặc chẩn đoán và nhắc lại tự nhiên đúng 2-3 lần ở các phần khác nhau,
  ví dụ lý do nhập viện, bệnh sử, danh sách triệu chứng, tiền sử và phần nhận định. Mỗi lần
  xuất hiện phải có một entity riêng đúng vị trí, kể cả text giống hệt nhau; không chỉ
  annotate lần đầu.
- Không lặp một triệu chứng quá 3 lần và không sao chép cả đoạn/template để kéo dài bệnh án.
- Assertion phải theo ngữ cảnh tại từng occurrence. Nếu muốn thay đổi từ dương tính sang
  phủ định, phải viết rõ chuyển biến thời gian như "ban đầu có nôn, hiện không còn nôn".
- Cấm mâu thuẫn vô cớ kiểu phần văn xuôi ghi "không nôn" nhưng bullet hiện tại lại ghi
  "Nôn mửa" dương tính. Cấm lặp heading hoặc dính heading với "Bệnh nhân".
""",
    },
    {
        "key": "repeated_diagnosis_mentions",
        "section_keys": ["tien_su", "hien_tai", "danh_gia"],
        "instruction": """
GỢI Ý BỔ SUNG V3: CHẨN ĐOÁN/BỆNH XUẤT HIỆN LẶP LẠI.
- Chọn một bệnh tự nhiên như suy giáp, tăng huyết áp, đái tháo đường type 2, suy tim,
  viêm phổi hoặc bệnh thận mạn và nhắc đúng 2-3 lần ở các vị trí khác nhau trong record.
- MỖI occurrence phải có một entity CHẨN_ĐOÁN riêng, sắp theo thứ tự xuất hiện; tuyệt đối
  không chỉ annotate lần đầu. Số entity cùng text phải bằng số mention thật trong input_text.
- Nếu cả các mention đều nói về cùng bệnh nền/tiền sử thì tất cả cùng có isHistorical.
  Nếu cả các mention đều mô tả chẩn đoán của đợt hiện tại thì tất cả có assertions: [].
- Không đổi assertion giữa các mention một cách vô cớ. Chỉ dùng assertion khác nhau nếu câu
  viết rõ hai chủ thể hoặc hai mốc khác nhau, và không dùng cấu trúc nhập nhằng trong focus này.
- Không copy nguyên câu hoặc heading để tạo lặp; mỗi lần nhắc phải nằm trong câu tự nhiên khác.
""",
    },
    {
        "key": "debug_single_lab_and_units",
        "section_keys": ["hien_tai", "danh_gia"],
        "instruction": """
GỢI Ý BỔ SUNG V3 TỪ DEBUG: LAB LẶP LẠI VÀ ĐƠN VỊ KHÔNG PHẢI THUỐC.
- Sinh một chỉ số được đo hai lần, ví dụ "kali là 2.4" rồi "kali vẫn giảm xuống 2.2".
  Mỗi lần phải có TÊN_XÉT_NGHIỆM="kali" và KẾT_QUẢ_XÉT_NGHIỆM là riêng giá trị
  "2.4"/"2.2" theo boundary gold; không bỏ lần đo thứ hai.
- Annotation phải bám đúng số mention thật: nếu câu viết tắt "kali lần đầu 3.8 mmol/L, sau giảm
  còn 3.5 mmol/L" thì chỉ có MỘT entity TÊN_XÉT_NGHIỆM="kali" và HAI entity kết quả.
  Chỉ tạo hai entity tên xét nghiệm khi chữ "kali" thực sự xuất hiện hai lần trong input_text,
  ví dụ "Kali lần đầu 3.8 mmol/L; sau 6 giờ kali giảm còn 3.5 mmol/L".
- Các đơn vị cân nặng/chiều dài như pound, lb, kg, inch không phải THUỐC và không annotate.
  Kali 80 mEq dùng để bổ sung vẫn là THUỐC vì đó là chất điều trị, không nhầm với đơn vị trần.
""",
    },
    {
        "key": "debug_qualitative_cardiac_tests",
        "section_keys": ["hien_tai", "danh_gia"],
        "specialty_names": ["Cardiovascular / Pulmonary", "General Medicine"],
        "instruction": """
GỢI Ý BỔ SUNG V3 TỪ DEBUG: ECG/NGHIỆM PHÁP GẮNG SỨC.
- "điện tâm đồ bình thường": TÊN="điện tâm đồ", KẾT_QUẢ="bình thường".
- "xét nghiệm gắng sức bất thường": TÊN="xét nghiệm gắng sức", KẾT_QUẢ="bất thường";
  nếu xuất hiện hai mention thật thì annotate đủ cả hai.
- Sau "Nghiệm pháp gắng sức dương tính...", toàn bộ kết luận liên tục về thiếu máu cơ tim
  và ST chênh xuống là KẾT_QUẢ_XÉT_NGHIỆM, không đổi bệnh trong finding thành CHẨN_ĐOÁN
  và không tách các từ ST/thành dưới/thành bên thành mảnh rời.
""",
    },
    {
        "key": "debug_neuro_imaging_history",
        "section_keys": ["hien_tai", "danh_gia"],
        "specialty_names": ["Neurology", "General Medicine"],
        "instruction": """
GỢI Ý BỔ SUNG V3 TỪ DEBUG: HÌNH ẢNH THẦN KINH VÀ MỐC QUÁ KHỨ.
- Có thể viết "1 tháng trước nhập viện, được chẩn đoán xuất huyết dưới nhện": chẩn đoán
  phải isHistorical dù đang nằm trong Bệnh sử hiện tại.
- Với "Chụp cắt lớp vi tính sọ não cho hình ảnh..." hoặc "Chụp kiểm tra ghi nhận...",
  kỹ thuật là TÊN_XÉT_NGHIỆM và toàn bộ finding/hướng chẩn đoán trong cùng câu là một
  KẾT_QUẢ_XÉT_NGHIỆM; không tách nang/tụ máu trong finding thành CHẨN_ĐOÁN.
- Dùng span triệu chứng đầy đủ như "mất định hướng", "đi lại không vững", "gần như ngất";
  cấm entity vụn "mất" hoặc "đi lại".
""",
    },
    {
        "key": "debug_chronic_bullets_and_current_episode",
        "section_keys": ["tien_su", "hien_tai"],
        "instruction": """
GỢI Ý BỔ SUNG V3 TỪ DEBUG: BỆNH NỀN VÀ ĐỢT HIỆN TẠI.
- Bệnh nền dòng ngắn có thể gồm đái tháo đường típ/type/tuýp/loại 2, suy tim, bệnh tim
  mạch do xơ vữa động mạch, bệnh mạch vành, tăng huyết áp phổi; trong section tiền sử
  đều là CHẨN_ĐOÁN isHistorical.
- CABG, PCI, đặt stent và phẫu thuật vẫn là thủ thuật, không annotate. Trong "bệnh tim
  mạch do xơ vữa động mạch sau CABG" chỉ lấy tên bệnh, không lấy CABG.
- "đến ED vì suy tim sung huyết cấp" là chẩn đoán của đợt hiện tại, assertions: [], không
  gán isHistorical chỉ vì câu nằm dưới mục sự kiện trước khi nhập viện.
- Tránh dữ liệu bẩn như "hệ số tống máu 50 inch"; phải viết "phân suất tống máu EF 50%".
""",
    },
    {
        "key": "debug_standalone_and_repeated_symptoms",
        "section_keys": ["hien_tai", "danh_gia"],
        "instruction": """
GỢI Ý BỔ SUNG V3 TỪ DEBUG: TRIỆU CHỨNG DÒNG TRẦN VÀ MENTION LẶP.
- Sinh các dòng không có bullet như ho, mệt mỏi, phù, sưng phù hai mắt cá chân, mất định
  hướng hoặc gần ngất; mỗi dòng vẫn phải được annotate đúng TRIỆU_CHỨNG.
- Nếu triệu chứng được nhắc lại tự nhiên ở cuối bệnh sử thì annotate từng occurrence, không
  chỉ lần đầu. Giữ span có nghĩa như "sốt lên đến 101°F" và "khó thở sau khi đi bộ vài khối".
- Khi sốt đi kèm nhiệt độ, cho phép có hoặc không có đơn vị nếu giá trị nằm trong miền °C
  hợp lý, ví dụ "sốt nhẹ 37.8", "sốt cao 39°C" hoặc "sốt lên đến 101°F". Cấm "sốt 90"
  hoặc "sốt 101" thiếu đơn vị vì không thể mặc định an toàn đó là °F.
- Không annotate mảnh trần "phân", "nhầy", "mất", "đi lại", "thành", "bên", "độ".
  Trong danh sách phủ định chỉ giữ cụm có nghĩa như phân đen, máu đỏ tươi, dịch nhầy.
- "thiếu oxy" trong lý do nhập viện/triệu chứng hiện tại là TRIỆU_CHỨNG assertions: [],
  không phải CHẨN_ĐOÁN isHistorical.
""",
    },
]

# Một nhánh riêng để V3 thỉnh thoảng sinh bệnh sử dài theo dòng thời gian. Tỷ lệ được
# điều khiển bên dưới bằng V3_VERY_LONG_FOCUS_PERCENT, không phụ thuộc trọng số các focus khác.
V3_VERY_LONG_FOCUS = {
    "key": "very_long_repeated_clinical_timeline",
    "mode": "v3_long",
    "max_completion_tokens": 3200,
    # Prompt vẫn yêu cầu rất dài; ngưỡng hard chỉ chặn output quá cụt để tránh đốt token
    # retry một record 500-1300 ký tự vốn vẫn hữu ích cho sliding-window NER.
    "min_input_chars": 350,
    # Mention lặp vẫn được yêu cầu trong prompt nhưng không hard-reject cả JSON nếu LLM
    # bỏ sót; retry một record dài chỉ vì tiêu chí augmentation này quá tốn token.
    "require_repeated_entity": False,
    "section_keys": ["hien_tai", "danh_gia"],
    "instruction": """
GỢI Ý BỔ SUNG V3: BỆNH SỬ RẤT DÀI, NHIỀU MỤC VÀ LẶP MENTION CÓ CHỦ ĐÍCH.
- Tạo 1800-3400 ký tự về MỘT ca thống nhất. Có thể dùng hoặc bỏ heading; nếu dùng thì luân
  phiên 4-6 khối như lý do nhập viện, thời điểm khởi phát, triệu chứng hiện tại, đặc điểm
  triệu chứng, diễn biến trước nhập viện, cận lâm sàng và tình trạng lúc vào viện.
- Chọn 2-3 triệu chứng trung tâm và nhắc lại tự nhiên 3-5 lần ở các vai trò khác nhau: lý do
  khám, đoạn kể diễn biến, bullet hiện tại và đánh giá cuối. Annotate MỖI occurrence thật theo
  thứ tự; không chỉ lấy lần đầu và không tạo duplicate entity nếu chữ chỉ xuất hiện một lần.
- Lặp ý phải có ích theo dòng thời gian, không lặp lỗi trong cùng cụm như "khó thở nhẹ khó
  thở", không sao chép nguyên cả câu và không dựng hai trạng thái mâu thuẫn cùng thời điểm.
- Mốc giờ/ngày/phút/giây là O. Thuốc tại nhà hoặc trước nhập viện là isHistorical; thuốc xử trí
  tại viện là []. Holter/ECG/X-quang/siêu âm là TÊN_XÉT_NGHIỆM và kết luận liên tục phía sau là
  một KẾT_QUẢ_XÉT_NGHIỆM, không tách finding thành chẩn đoán vụn.
- Heading chỉ xuất hiện một lần cho mỗi khối và phải có newline/space đúng; cấm dính chữ giữa
  hai câu. Có thể xen văn xuôi dài, bullet và ghi nhanh nhưng không được kéo dài bằng câu rác.
""",
}

# V4 chỉ dùng các đặc trưng CẤU TRÚC phổ quát rút ra từ audit input dài. Generator
# không đọc/copy data/input khi chạy, tránh bám test public hoặc làm giảm tổng quát private.
# mixed_v4 là profile độc lập: baseline + augmentation dài, không âm thầm trộn V2/V3.
V4_FOCUS_AREAS = [
    {
        "key": "long_qa_varied",
        "mode": "v4",
        "format": "qa",
        "weight": 3.0,
        "section_keys": ["tien_su", "hien_tai", "danh_gia"],
        "instruction": """
GỢI Ý BỔ SUNG V4: HỎI ĐÁP Y KHOA DÀI, ĐA DẠNG NGUỒN.
- Tạo 800-2400 ký tự và BẮT BUỘC có một cặp nhãn phân cách rõ, luân phiên giữa
  "Câu hỏi từ người dùng/Câu trả lời của bác sĩ", "Hỏi/Đáp" hoặc "Người bệnh/Bác sĩ".
  Không dùng QA ẩn hoàn toàn trong văn xuôi vì cần kiểm chứng được phân bố format.
  Người dùng được phép viết informal, lặp ý, viết tắt và sai khoảng trắng nhẹ như dữ liệu thật.
- Có 2-4 mention lặp của một triệu chứng/xét nghiệm. Annotate MỖI occurrence thật; có thể
  trộn kết quả số với kết quả định tính nhưng không sao chép câu/ví dụ từ input đánh giá.
- Phần trả lời có thể giải thích bệnh học dài. Theo mô tả tổng quát của đề, tên bệnh/triệu
  chứng/thuốc cụ thể xuất hiện rõ trong phần kiến thức VẪN là khái niệm y tế và phải annotate
  bằng span đầy đủ, assertions: [] nếu không có phủ định/gia đình/tiền sử rõ ràng.
- Phân biệt khái niệm thuộc 5 type với từ y sinh ngoài schema: protein, enzyme, gen, cơ quan,
  mô, dịch cơ thể đứng riêng, yếu tố nguy cơ và lối sống đều là O.
- Không tạo entity vụn như "Bệnh", "cơ quan", "máu", "đau" + vị trí tách rời. Phải lấy
  trọn tên bệnh hoặc triệu chứng có tính chất/vị trí khi chúng là entity.
""",
    },
    {
        "key": "long_clinical_blocks",
        "mode": "v4",
        "format": "clinical_long",
        "weight": 3.0,
        "section_keys": ["tien_su", "hien_tai", "danh_gia"],
        "instruction": """
GỢI Ý BỔ SUNG V4: BỆNH ÁN NHIỀU KHỐI RẤT DÀI.
- Tạo 1000-3200 ký tự, có thể gồm hai hoặc ba mục lâm sàng nhưng phải luân phiên tên heading;
  mỗi mục có văn xuôi xen bullet, câu cụt và kết quả cận lâm sàng.
- Cùng triệu chứng/chẩn đoán/xét nghiệm có thể lặp ở lý do nhập viện, diễn biến và đánh giá;
  annotate đủ từng mention, giữ assertion theo ngữ cảnh cục bộ của chính occurrence đó.
- Có thể đổi văn phong đột ngột giữa đoạn kể dài và ghi nhanh, nhưng các khối phải cùng ca hoặc
  được ngăn cách rõ. Không ghép thuốc-bệnh vô lý chỉ để mô phỏng nhiễu.
- Tên xét nghiệm/finding dài vẫn theo rule hiện có; thời gian như 20 giây là O; thủ thuật là O.
""",
    },
    {
        "key": "hybrid_abrupt_blocks",
        "mode": "v4",
        "format": "hybrid",
        "weight": 1.5,
        "section_keys": ["tien_su", "hien_tai", "danh_gia"],
        "instruction": """
GỢI Ý BỔ SUNG V4: FREE-FORM LAI/GHÉP KHỐI.
- Tạo 900-2800 ký tự: bắt đầu bằng Q&A hoặc đoạn tư vấn, sau đó chuyển sang một
  khối bệnh án đánh số/bullet, hoặc ngược lại. Hai khối có thể là hai ngữ cảnh độc lập nhưng
  phải có ranh giới heading/newline rõ, không dính thành một câu vô nghĩa.
- Annotate mọi khái niệm hợp lệ ở từng khối theo ngữ cảnh riêng; không truyền isHistorical,
  isFamily hay isNegated từ khối trước sang khối sau.
- Có thể có 1-2 lỗi dữ liệu thật như thiếu khoảng trắng sau dấu câu hoặc viết tắt CT/XN/HA,
  nhưng không cố tình tạo hàng loạt typo khiến entity không còn là substring có nghĩa.
- Không chèn finding hoàn toàn vô cớ vào cuối một câu trả lời; nếu có khối cận lâm sàng thì
  trình bày nó như một khối độc lập rõ ràng để model học chuyển miền, không học câu rác.
""",
    },
    {
        "key": "education_article",
        "mode": "v4",
        "format": "education",
        "weight": 1.0,
        "section_keys": ["tien_su", "hien_tai", "danh_gia"],
        "instruction": """
GỢI Ý BỔ SUNG V4: BÀI GIẢI THÍCH BỆNH DÀI.
- Tạo 1400-3600 ký tự, thỉnh thoảng dài hơn, với tiêu đề hỏi về bệnh và các mục linh hoạt
  về nguyên nhân, triệu chứng, biến chứng, xét nghiệm và điều trị; có thể xen bullet.
- Annotate trọn các tên bệnh, triệu chứng, thuốc, tên/kết quả xét nghiệm cụ thể xuất hiện trong
  bài vì nhiệm vụ tổng quát yêu cầu phát hiện khái niệm y tế xuất hiện trong free-form text.
- Không annotate cấu trúc sinh học ngoài 5 type như gen, protein, enzyme, nhiễm sắc thể, mô,
  cơ quan hoặc hồng cầu đứng riêng. Phẫu thuật/thủ thuật/phương pháp điều trị chung vẫn là O.
- Cấm span một từ chung chung/vụn. Tên bệnh, biến chứng và triệu chứng phải lấy nguyên cụm;
  không rút còn "Bệnh", "men", "mạch", "u", "máu" hoặc một từ giải phẫu đứng riêng.
""",
    },
    {
        "key": "very_long_repeated_timeline",
        "mode": "v4",
        "format": "long_timeline",
        "weight": 4.0,
        "max_completion_tokens": 3600,
        "min_input_chars": 850,
        "require_repeated_entity": False,
        "section_keys": ["hien_tai", "danh_gia"],
        "instruction": """
GỢI Ý BỔ SUNG V4: HỒ SƠ DÀI THEO DÒNG THỜI GIAN, RECALL ĐỦ MỌI MENTION.
- Tạo 2000-3800 ký tự về một ca lâm sàng nhất quán. Văn bản có 5-8 khối linh hoạt: lý do
  nhập viện, khởi phát, đoạn kể diễn biến, triệu chứng hiện tại, đặc điểm, sự kiện trước viện,
  cận lâm sàng và tình trạng cuối. Có thể xen heading, bullet và ghi nhanh; không bắt buộc mở
  đầu bằng "Tiền sử bệnh hiện tại" và không sao chép một template cố định.
- Chọn 2-3 triệu chứng chính, cho mỗi triệu chứng xuất hiện tự nhiên 3-5 lần trong các khối
  khác nhau rồi annotate đủ từng occurrence. Mỗi lần nhắc phải bổ sung vai trò/thời điểm/tính
  chất, không lặp nguyên câu và không tạo cụm lỗi kiểu "khó thở nhẹ khó thở".
- Có cả phủ định dạng danh sách và một thay đổi theo thời gian nếu hợp lý, nhưng không để cùng
  thời điểm vừa khẳng định vừa phủ định. Assertion được quyết định riêng tại từng occurrence.
- Trộn tối đa 2 kỹ thuật cận lâm sàng và 2-3 thuốc có ngữ cảnh thời điểm rõ. Finding giữ trọn
  một span kết quả; đơn vị thời gian là O; thủ thuật và sinh hoạt ngoài schema là O.
- Entity phải sắp đúng thứ tự xuất hiện và bao phủ mọi mention hợp lệ, kể cả mention ở phần
  tóm tắt cuối. Độ dài đến từ diễn biến có nghĩa, tuyệt đối không từ câu rác hoặc đoạn copy.
""",
    },
]

V4_CONTROLLED_BOUNDARY_NOISE_INSTRUCTION = """
GỢI Ý BỔ SUNG V4: NHIỄU RANH GIỚI CÓ KIỂM SOÁT.
- Tạo 2-4 chỗ thiếu khoảng trắng giống văn bản thật, ưu tiên sau dấu chấm, ví dụ cấu trúc
  "đau ngực.Hồi hộp" hoặc "viêm phổi.Ho khan". Đây là hai khái niệm độc lập: entities PHẢI
  tách ở dấu chấm, tuyệt đối không tạo một span chứa cả "đau ngực.Hồi hộp".
- Có thể tạo tối đa một lỗi dính chữ bên trong một cụm dài, chẳng hạn bỏ khoảng trắng giữa
  hai từ của một chẩn đoán; khi đó entity phải lấy nguyên cụm lỗi đúng như substring nguồn,
  không cắt thành một từ giải phẫu hoặc một mảnh vô nghĩa.
- Nhiễu chỉ nằm ở khoảng trắng. Không làm mất chữ đầu của entity, không đổi chính tả y khoa,
  không dính heading và không tạo chuỗi rác. Các entity hai phía ranh giới vẫn đủ nghĩa riêng.
"""
CONTROLLED_BOUNDARY_NOISE_RE = re.compile(
    r"(?<=[^\W\d_])[.!?](?=[^\W\d_])"
)

# V5 là profile chống overfit độc lập. quota_weight cộng đúng 600 để khi sinh 600 mẫu
# lịch focus khớp kế hoạch; với batch lớn/nhỏ hơn, lịch được scale theo cùng tỷ lệ.
V5_TARGET_CONCEPTS = [
    "đau ngực", "sốt", "đau bụng", "đau đầu", "buồn nôn", "khó thở",
    "chóng mặt", "tiểu máu", "tiểu buốt", "tăng huyết áp",
    "đái tháo đường", "sỏi thận", "chảy máu cam", "ban đỏ lan rộng",
    "đau ấn vùng mổ", "chảy nước mũi", "nhịp thở nhanh", "thiếu oxy",
    "nhiễm trùng đường hô hấp trên",
]
V5_STYLE_VARIANTS = [
    "một câu ghi nhanh không heading, có thể viết tắt nhưng vẫn hiểu được",
    "2-4 bullet ngắn kiểu bàn giao điều dưỡng, không mở đầu bằng tuổi/giới",
    "tin nhắn ngôi thứ nhất informal, nhịp câu tự nhiên và không dùng template bệnh án",
    "đoạn tái khám ngoại trú súc tích, xen một câu context O",
    "văn xuôi liền mạch có ngoặc chú thích và chuyển ý nhưng không lặp thông tin",
    "ghi chú điện thoại hoặc dặn dò sau khám, chủ thể có thể được lược bỏ",
    "dòng dữ liệu rời kiểu phiếu theo dõi; không cố thêm heading nếu không cần",
]
V5_FOCUS_AREAS = [
    {
        "key": "contrastive_assertions",
        "mode": "v5",
        "format": "contrast",
        "quota_weight": 130,
        "min_entities": 2,
        "instruction": """
NHÓM V5 — CẶP ĐỐI NGHỊCH CÙNG KHÁI NIỆM:
- Chọn một hoặc hai khái niệm phổ biến trong gợi ý và đặt chúng ở 2-4 ngữ cảnh khác nhau:
  hiện tại [], phủ định isNegated, tiền sử thật isHistorical hoặc người thân isFamily.
- Biến đổi boundary tự nhiên giữa các occurrence: thêm vị trí, mức độ, tính chất hoặc hoàn cảnh
  nằm trong triệu chứng; không lặp máy móc đúng một chuỗi.
- Mỗi assertion chỉ áp dụng cho occurrence có cue cục bộ tương ứng. Không bắt buộc đủ cả bốn
  assertion trong một record, nhưng bắt buộc có ít nhất hai trạng thái đối nghịch.
""",
    },
    {
        "key": "dense_ner_boundaries",
        "mode": "v5",
        "format": "boundary",
        "quota_weight": 110,
        "min_entities": 3,
        "instruction": """
NHÓM V5 — NER BOUNDARY DÀI VÀ SÁT NHAU:
- Tạo 3-7 entity, ưu tiên cụm triệu chứng dài có vị trí/tính chất/diễn biến như đau âm ỉ vùng
  hạ sườn phải tăng sau ăn, khó thở khi nằm tăng dần về đêm, ho khan thành từng cơn kéo dài
  hai tuần, phù mềm hai chi dưới ấn lõm hoặc cảm giác nóng rát sau xương ức.
- Có ít nhất một cụm dài và một chuỗi entity đứng sát nhau, nối bằng dấu phẩy/và/kèm theo.
- Phân biệt phần triệu chứng với hoàn cảnh O: trong "đau ngực bên trái khi đi bộ nhanh", chỉ
  lấy "đau ngực bên trái"; không nuốt hoạt động/lối sống vào span.
- Khoảng 70% giá trị của nhóm này nằm ở boundary NER; assertions tự nhiên, không nhồi family.
""",
    },
    {
        "key": "sparse_zero_entity",
        "mode": "v5",
        "format": "sparse_zero",
        "quota_weight": 15,
        "min_entities": 0,
        "sparse_variant": "zero",
        "max_completion_tokens": 500,
        "instruction": """
NHÓM V5 — RECORD THƯA KHÔNG CÓ ENTITY:
- Viết một đoạn hành chính, sinh hoạt, hướng dẫn chung hoặc trao đổi đặt lịch dài 1-4 câu mà
  không chứa thuốc, bệnh, triệu chứng hay cặp xét nghiệm cụ thể thuộc năm type.
- `entities` bắt buộc là []. Không biến tuổi, giới, nghề, thời gian, hoạt động, cơ quan giải
  phẫu, thủ thuật hay lời khuyên chung thành entity.
""",
    },
    {
        "key": "sparse_one_type",
        "mode": "v5",
        "format": "sparse_one_type",
        "quota_weight": 40,
        "min_entities": 1,
        "sparse_variant": "one_type",
        "max_completion_tokens": 600,
        "instruction": """
NHÓM V5 — RECORD THƯA CHỈ MỘT TYPE:
- Viết đoạn ngắn tự nhiên có đúng 1-3 entity và tất cả cùng một type. Luân phiên giữa chỉ
  TRIỆU_CHỨNG, chỉ THUỐC hoặc chỉ CHẨN_ĐOÁN; không tự thêm lab/thuốc/bệnh để đủ template.
- Có thể xen câu hành chính, sinh hoạt hoặc hoàn cảnh O để model học precision.
""",
    },
    {
        "key": "sparse_two_types",
        "mode": "v5",
        "format": "sparse_two_types",
        "quota_weight": 35,
        "min_entities": 2,
        "sparse_variant": "two_types",
        "max_completion_tokens": 700,
        "instruction": """
NHÓM V5 — RECORD THƯA CHỈ HAI TYPE:
- Viết đoạn ngắn có 2-4 entity và đúng hai type. Ưu tiên cặp lab TÊN/KẾT_QUẢ hoặc một cặp
  tự nhiên như TRIỆU_CHỨNG + THUỐC; không nhồi đủ năm loại.
- Xen context O hợp lý và giữ mỗi span ngắn gọn, chính xác.
""",
    },
    {
        "key": "false_cues_and_scope",
        "mode": "v5",
        "format": "scope",
        "quota_weight": 80,
        "min_entities": 3,
        "instruction": """
NHÓM V5 — CUE GIẢ VÀ PHẠM VI ASSERTION KHÓ:
- Đặt cue gần nhau trong 1-3 câu, có ít nhất ba trạng thái assertion khác nhau.
- Dùng cue giả: "không nhớ rõ thời điểm bắt đầu đau ngực" không phủ định đau ngực;
  "không dùng thuốc điều trị tăng huyết áp đều đặn" không phủ định tăng huyết áp;
  "khó thở không cải thiện sau nghỉ" vẫn là triệu chứng hiện tại.
- Có thể đặt bệnh người thân, bệnh tiền sử, triệu chứng phủ định và triệu chứng hiện tại trong
  cùng câu. Chỉ occurrence của người thân là isFamily; cue không/tiền sử không được lan sang
  các entity kế bên. Không cố tăng isFamily nếu câu không tự nhiên.
""",
    },
    {
        "key": "dirty_btc_text",
        "mode": "v5",
        "format": "dirty",
        "quota_weight": 50,
        "min_entities": 2,
        "boundary_noise": True,
        "instruction": """
NHÓM V5 — RAW TEXT BẨN GIỐNG GHI NHANH:
- Tạo 2-5 lỗi khoảng trắng có kiểm soát: dấu câu dính chữ, viết tắt, số dính đơn vị, ví dụ
  cấu trúc "BN sốt 3ngày,ho khan.Khó thở tăng", "TS THA 10n,đang dùng thuốc" hoặc
  "WBC:15.2;CRP:64mg/L;SpO2:92%". Không chép nguyên cả ví dụ và không tạo typo mất chữ.
- Raw text giữ lỗi; entity vẫn là substring chính xác. Hai khái niệm hai phía dấu câu phải là
  hai entity riêng. Không tạo một entity nuốt qua `.`, `;`, `:` hay dấu phẩy phân cách.
""",
    },
    {
        "key": "btc_medication_lists",
        "mode": "v5",
        "format": "medication_list",
        "quota_weight": 80,
        "min_entities": 7,
        "min_drugs": 5,
        "instruction": """
NHÓM V5 — DANH SÁCH THUỐC THEO GOLD BTC:
- Tạo danh sách 5-10 thuốc trước nhập viện/dùng tại nhà, có thể đánh số liên tục trong một dòng
  hoặc xuống dòng. Trộn ingredient-only với clinical drug có strength, dose form, route và frequency.
- Mỗi THUỐC phải lấy trọn regimen liên tục: tên + strength + dose form + route + frequency, ví dụ
  boundary kiểu `amlodipine 10 mg po daily` hoặc `clonazepam 0.5 mg po qam:prn`; không lấy số thứ tự.
- Tất cả thuốc thuộc danh sách trước nhập viện có isHistorical. Nếu sau thuốc có `điều trị ho`,
  `điều trị đau nhức`, `điều trị táo bón`, thì symptom/diagnosis phía sau là entity riêng và KHÔNG
  kế thừa isHistorical chỉ vì thuốc đứng trước nó.
- Không gộp hai thuốc, không annotate riêng strength/route/frequency và không bỏ thuốc thiếu liều.
- Chỉ dùng thuốc thật từ gợi ý, giữ phối hợp thuốc-chỉ định hợp lý và annotate đủ mọi item.
""",
    },
    {
        "key": "complete_occurrence_recall",
        "mode": "v5",
        "format": "recall",
        "quota_weight": 60,
        "min_entities": 5,
        "require_complete_occurrences": True,
        "instruction": """
NHÓM V5 — BAO PHỦ ĐỦ MỌI OCCURRENCE:
- Tạo record 2-6 câu hoặc bullet có ít nhất hai khái niệm lặp lại ở vị trí khác nhau, xen một
  danh sách thuốc/lab hoặc một câu có nhiều triệu chứng sát nhau.
- Annotate MỌI occurrence thật theo thứ tự, kể cả lần lặp ở tiêu đề, câu hỏi, phần giải thích,
  item cuối danh sách và lần đo lab thứ hai. Không chỉ annotate mention đầu tiên.
- Mỗi occurrence giữ assertion theo cue cục bộ của chính nó; cùng text có thể current, negated,
  historical hoặc family ở các vị trí khác nhau.
- Tự kiểm cuối: với từng entity text đã dùng, số annotation phải bằng số occurrence exact trong
  input_text, trừ khi occurrence nằm bên trong một span dài hơn đã annotate tại đúng vị trí đó.
""",
    },
]

V5_DIRTY_RECORD_PERCENT = 17.5
# Một phần V5 mô phỏng cấu trúc hỏi bệnh - trả lời dài nhưng hoàn toàn tự sinh. Đây là
# augmentation theo *kiểu văn bản*, không sao chép ViHealthQA hay input public/private.
V5_QA_RECORD_PERCENT = 25.0
V5_PRIMARY_DIRTY_PERCENT = 100 * 70 / 600
V5_EXTRA_DIRTY_NONPRIMARY_PERCENT = 100 * (
    V5_DIRTY_RECORD_PERCENT - V5_PRIMARY_DIRTY_PERCENT
) / (100 - V5_PRIMARY_DIRTY_PERCENT)
V5_DIRTY_TEXT_INSTRUCTION = """
MODIFIER V5 — NHIỄU NHẸ: record này phải có 1-3 chỗ thiếu khoảng trắng có kiểm soát.
Ưu tiên `triệu chứng.Triệu chứng`, `3ngày,ho`, `WBC:15.2;CRP:64mg/L` hoặc liều dính chữ.
Giữ raw text bẩn nhưng entities tách đúng; không để một entity chứa hai khái niệm qua dấu câu.
Chỉ dính chữ bên trong cùng một entity hoặc bên trong phần O. Không dính cue O với đầu entity
kiểu `bịchảy máu cam`, vì BIO word-level không biểu diễn được ranh giới nằm giữa một token.
"""
V5_QA_TEXT_INSTRUCTION = """
MODIFIER V5 — HỎI ĐÁP DÀI TỰ SINH:
- Viết một câu hỏi của người bệnh/người nhà và câu trả lời tư vấn của bác sĩ, khoảng 700-1800 ký tự.
  Nội dung phải mới, không chép hay diễn đạt lại một input/dataset có sẵn.
- Cùng một bệnh hoặc triệu chứng có thể được nhắc lại nhiều lần; PHẢI annotate mọi occurrence
  đúng thứ tự và đúng phạm vi cục bộ, không chỉ occurrence đầu tiên.
- Phân biệt lời kể về chính người hỏi với kiến thức chung trong câu trả lời. Tên bệnh đầy đủ trong
  phần giải thích có thể là CHẨN_ĐOÁN với assertions=[], nhưng các từ chung như `bệnh`, `máu`,
  `hồng cầu`, `protein`, `rối loạn`, tên cơ quan, tác nhân và phương pháp điều trị chung là O.
- `đã được bác sĩ khám vì X` trong chính đợt bệnh đang kể không làm X thành isHistorical.
  Chỉ dùng isHistorical khi có tiền sử/trước đây/đã từng hoặc một đợt bệnh cũ đã kết thúc rõ ràng.
- Tên thuốc chỉ annotate khi người bệnh thực sự đang/đã được điều trị bằng thuốc đó. Thực phẩm,
  chất cần tránh, thiết bị, thủ thuật và thuốc chỉ được nêu như kiến thức chung đều là O.
- Không chèn một bệnh án không liên quan vào giữa câu hỏi và câu trả lời chỉ để tăng entity.
"""
V5_DIRTY_SIGNAL_RE = re.compile(
    r"(?<=[^\W\d_])[.!?](?=[^\W\d_])|[:;,](?=\S)|(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)"
)

ENTITY_TYPES = ["THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"]
ASSERTION_TYPES = ["isNegated", "isFamily", "isHistorical", None]

# ----------------------------------------------------------------------------
# Chuyên khoa: dùng để ép bối cảnh bệnh án đa dạng, tránh việc LLM cứ quay vòng vài
# bệnh quen thuộc (tăng huyết áp, đái tháo đường type 2...) bất kể section nào.
# Mỗi entry: (tên specialty trong mtsamples, tên tiếng Việt hiển thị trong prompt,
# chuỗi chữ cái chương ICD-10 liên quan để sample diagnosis_hint đúng chuyên khoa).
# ----------------------------------------------------------------------------
SPECIALTY_CONFIG = [
    ("Cardiovascular / Pulmonary", "Tim mạch - Hô hấp", "IJ"),
    ("Gastroenterology", "Tiêu hóa", "K"),
    ("Neurology", "Thần kinh", "G"),
    ("Orthopedic", "Cơ xương khớp", "M"),
    ("Ophthalmology", "Mắt", "H"),
    ("Obstetrics / Gynecology", "Sản phụ khoa", "N"),
    ("Urology", "Tiết niệu", "N"),
    ("Nephrology", "Thận", "N"),
    ("ENT - Otolaryngology", "Tai Mũi Họng", "H"),
    ("Hematology - Oncology", "Huyết học - Ung bướu", "CD"),
    ("Dermatology", "Da liễu", "L"),
    ("Endocrinology", "Nội tiết", "E"),
    ("General Medicine", "Nội tổng quát", "ABCDEGHIJKLMN"),
]

SECTION_TYPES = [
    {
        "key": "tien_su",
        "heading": "Tiền sử bệnh",
        "headings": [
            "Tiền sử bệnh",
            "Tiền sử y khoa",
            "Tiền căn bệnh lý",
            "Bệnh sử trước đây",
            "Tiền sử bản thân và gia đình",
            "Bệnh nền",
            "Tiền căn nội khoa",
            "Thuốc và bệnh đang theo dõi",
        ],
        "desc": (
            "Mục 'Tiền sử bệnh': liệt kê thuốc đang dùng trước khi nhập viện, "
            "tiền sử bệnh nền, yếu tố nguy cơ, tiền sử gia đình. "
            "Đây là section có nhiều assertion isHistorical / isFamily nhất."
        ),
        "assertion_bias": ["isHistorical", "isFamily", None],
    },
    {
        "key": "hien_tai",
        "heading": "Tiền sử bệnh hiện tại",
        "headings": [
            "Tiền sử bệnh hiện tại",
            "Bệnh sử hiện tại",
            "Diễn biến bệnh hiện tại",
            "Quá trình bệnh lý hiện tại",
            "Lý do vào viện và bệnh sử",
            "Bệnh sử",
            "Diễn tiến trước nhập viện",
            "Lý do khám",
            "Tình trạng hiện tại",
        ],
        "desc": (
            "Mục 'Tiền sử bệnh hiện tại': lý do nhập viện, triệu chứng hiện tại, "
            "diễn biến trước khi nhập viện, có thể có phủ định triệu chứng "
            "(ví dụ 'không buồn nôn', 'không sốt')."
        ),
        "assertion_bias": ["isNegated", None, None],
    },
    {
        "key": "danh_gia",
        "heading": "Đánh giá tại bệnh viện",
        "headings": [
            "Đánh giá tại bệnh viện",
            "Khám và đánh giá tại bệnh viện",
            "Đánh giá lâm sàng và cận lâm sàng",
            "Kết quả đánh giá ban đầu",
            "Thăm khám tại bệnh viện",
            "Khám ban đầu",
            "Nhận định lâm sàng",
            "Cận lâm sàng",
            "Kết quả thăm khám",
        ],
        "desc": (
            "Mục 'Đánh giá tại bệnh viện': kết quả khám lâm sàng (vitals dạng nén "
            "kiểu 'VS98.3 12987 56 18 99RA'), kết quả xét nghiệm, kết quả chẩn đoán "
            "hình ảnh. Đây là section BẮT BUỘC phải có TÊN_XÉT_NGHIỆM và "
            "KẾT_QUẢ_XÉT_NGHIỆM (2 type dễ bị thiếu / gán sai nhất)."
        ),
        "assertion_bias": [None, None, None],
    },
]

# ----------------------------------------------------------------------------
# FEW-SHOT GOLD THẬT (từ BTC) -- neo cứng 2 rule dễ học sai nhất:
#   (a) thuốc lịch sử -> isHistorical, nhưng triệu chứng là CHỈ ĐỊNH điều trị
#       đi kèm thì KHÔNG kế thừa assertion của thuốc
#   (b) span entity tối giản, không lẫn số thứ tự/ngữ cảnh liệt kê
# (đã strip "candidates" và "position" vì 2 field đó do bước Candidate
# Retrieval / Offset Reconstruct tính, không phải LLM sinh ra)
# ----------------------------------------------------------------------------
GOLD_FEWSHOT_INPUT = (
    "Danh sách thuốc trước nhập viện chính xác và đầy đủ. "
    "1. amlodipine 10 mg po daily 2. aspirin 81 mg po daily "
    "3. metoprolol succinate xl 50 mg po daily 4. guaifenesin ml po q6h:prn điều trị ho "
    "5. nystatin oral suspension 5 ml po qid:prn điều trị đau nhức "
    "6. acetaminophen 325-650 mg po q6h:prn điều trị sốt đau 7. pravastatin 40 mg po daily "
    "8. docusate sodium 100 mg po bid điều trị táo bón 9. senna 8.6 mg po bid:prn điều trị táo bón "
    "10. clonazepam 0.5 mg po qam:prn điều trị lo âu 11. clonazepam 1.5 mg po qhs điều trị lo âu mất ngủ"
)
GOLD_FEWSHOT_OUTPUT = [
    {"text": "amlodipine 10 mg po daily", "type": "THUỐC", "assertions": ["isHistorical"]},
    {"text": "aspirin 81 mg po daily", "type": "THUỐC", "assertions": ["isHistorical"]},
    {"text": "metoprolol succinate xl 50 mg po daily", "type": "THUỐC", "assertions": ["isHistorical"]},
    {"text": "guaifenesin ml po q6h:prn", "type": "THUỐC", "assertions": ["isHistorical"]},
    {"text": "ho", "type": "TRIỆU_CHỨNG", "assertions": []},
    {"text": "nystatin oral suspension 5 ml po qid:prn", "type": "THUỐC", "assertions": ["isHistorical"]},
    {"text": "đau nhức", "type": "TRIỆU_CHỨNG", "assertions": []},
    {"text": "acetaminophen 325-650 mg po q6h:prn", "type": "THUỐC", "assertions": ["isHistorical"]},
    {"text": "sốt đau", "type": "TRIỆU_CHỨNG", "assertions": []},
    {"text": "pravastatin 40 mg po daily", "type": "THUỐC", "assertions": ["isHistorical"]},
    {"text": "docusate sodium 100 mg po bid", "type": "THUỐC", "assertions": ["isHistorical"]},
    {"text": "táo bón", "type": "TRIỆU_CHỨNG", "assertions": []},
    {"text": "senna 8.6 mg po bid:prn", "type": "THUỐC", "assertions": ["isHistorical"]},
    {"text": "táo bón", "type": "TRIỆU_CHỨNG", "assertions": []},
    {"text": "clonazepam 0.5 mg po qam:prn", "type": "THUỐC", "assertions": ["isHistorical"]},
    {"text": "lo âu", "type": "TRIỆU_CHỨNG", "assertions": []},
    {"text": "clonazepam 1.5 mg po qhs", "type": "THUỐC", "assertions": ["isHistorical"]},
    {"text": "lo âu", "type": "TRIỆU_CHỨNG", "assertions": []},
    {"text": "mất ngủ", "type": "TRIỆU_CHỨNG", "assertions": []},
]

# ----------------------------------------------------------------------------
# GOLD_FEWSHOT_2: ví dụ gold thứ 2 từ BTC (văn xuôi tự nhiên, khác cấu trúc liệt kê
# số thứ tự của GOLD_FEWSHOT_INPUT). Xác nhận 3 điều mình từng làm SAI:
#   - TÊN_XÉT_NGHIỆM giữ NGUYÊN chú thích tiếng Việt trong ngoặc nếu nguồn có sẵn
#     (vd "NEUT% (Tỷ lệ % bạch cầu trung tính)"), KHÔNG rút gọn.
#   - KẾT_QUẢ_XÉT_NGHIỆM có thể là số THUẦN không kèm đơn vị, dùng dấu PHẨY thập
#     phân theo văn phong phòng xét nghiệm VN (vd "14,43"), không bắt buộc luôn có
#     đơn vị kiểu "14.43 G/L".
#   - TRIỆU_CHỨNG giữ tính từ/cụm mô tả tính chất (màu sắc, mức độ) trong span nếu
#     đó là 1 phần bản chất mô tả triệu chứng (vd "ho đờm xanh"), khác với việc loại
#     bỏ ngữ cảnh thừa (chủ ngữ/tuổi) ở rule (b) -- 2 việc KHÔNG giống nhau.
# ----------------------------------------------------------------------------
GOLD_FEWSHOT_2_INPUT = (
    "Bệnh nhân nam 70 tuổi bị bệnh 1 tuần nay, ho đờm xanh, tức ngực, đau thượng vị, "
    "ợ hơi, được chẩn đoán mắc bệnh trào ngược dạ dày - thực quản. Bệnh nhân có tiền sử "
    "sử dụng Chlorpheniramine 0.4 MG/ML, Capsaicin 0.38 MG/ML, đã tiến hành tổng phân "
    "tích tế bào máu bằng máy lazer (tbm): WBC:14,43; NEUT% (Tỷ lệ % bạch cầu trung "
    "tính):76,4; LYPH% (Tỷ lệ bạch cầu lympho):12,8;"
)
GOLD_FEWSHOT_2_OUTPUT = [
    {"text": "bệnh trào ngược dạ dày - thực quản", "type": "CHẨN_ĐOÁN", "assertions": []},
    {"text": "ho đờm xanh", "type": "TRIỆU_CHỨNG", "assertions": []},
    {"text": "tức ngực", "type": "TRIỆU_CHỨNG", "assertions": []},
    {"text": "đau thượng vị", "type": "TRIỆU_CHỨNG", "assertions": []},
    {"text": "ợ hơi", "type": "TRIỆU_CHỨNG", "assertions": []},
    {"text": "Chlorpheniramine 0.4 MG/ML", "type": "THUỐC", "assertions": ["isHistorical"]},
    {"text": "Capsaicin 0.38 MG/ML", "type": "THUỐC", "assertions": ["isHistorical"]},
    {"text": "WBC", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
    {"text": "14,43", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
    {"text": "NEUT% (Tỷ lệ % bạch cầu trung tính)", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
    {"text": "76,4", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
    {"text": "LYPH% (Tỷ lệ bạch cầu lympho)", "type": "TÊN_XÉT_NGHIỆM", "assertions": []},
    {"text": "12,8", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []},
]


# ----------------------------------------------------------------------------
# 1. Load seed sections (10 file txt thật) để bắt chước văn phong / cấu trúc
# ----------------------------------------------------------------------------
def load_seed_sections():
    pool = {s["key"]: [] for s in SECTION_TYPES}
    if not SEED_DIR.exists():
        print(f"[!] Không thấy {SEED_DIR}, bỏ qua seed thật (vẫn gen được nhưng kém đa dạng hơn).")
        return pool

    for fp in sorted(SEED_DIR.glob("*.txt")):
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        chunks = re.split(r"\n(?=\d+\.\s+[A-ZĐÂÊÔƠƯ])", raw.strip())
        for i, chunk in enumerate(chunks):
            if i < len(SECTION_TYPES):
                pool[SECTION_TYPES[i]["key"]].append(chunk.strip())
    return pool


# ----------------------------------------------------------------------------
# 2. mtsamples: trích tên thuốc (đa từ, không phân biệt hoa/thường) + dòng vitals.
#    KHÔNG dùng nguyên câu triệu chứng/chẩn đoán tiếng Anh.
# ----------------------------------------------------------------------------
DOSE_RE = re.compile(
    r"(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?)\s?(mg|mcg|g|ml|units?|IU|mEq)\b", re.IGNORECASE
)
# Từ dừng: nếu quét ngược từ vị trí liều mà gặp từ này thì DỪNG, không coi là 1 phần tên thuốc.
# Có cả tiếng Anh (câu mtsamples) lẫn tiếng Việt (câu do LLM tự sinh xen kẽ).
DRUG_STOPWORDS_BEFORE = {
    "the", "a", "an", "of", "with", "and", "or", "to", "for", "was", "were", "is", "are",
    "given", "started", "continued", "discontinued", "daily", "since", "using", "take",
    "taking", "took", "prescribed", "patient", "dose", "dosage", "approximately", "about",
    "po", "iv", "im", "sc", "bid", "tid", "qid", "qhs", "qam", "qpm", "prn",
    "q6h", "q8h", "q12h", "q4h", "at", "she", "he", "her", "his", "on", "last",
    "year", "month", "week", "day", "then", "now", "currently", "also",
    "sử", "dụng", "dùng", "uống", "tiêm", "chỉ", "định", "kê", "đơn", "toa", "bằng",
    "điều", "trị", "bệnh", "nhân", "đã", "đang", "được", "và", "với", "mỗi", "ngày",
    "hàng", "mg", "viên", "liều",
}


def extract_drug_candidates(text, max_words_before=4):
    """
    Quét NGƯỢC từ vị trí liều lượng để gom tên thuốc đa từ (vd "metoprolol succinate 50 mg",
    "docusate sodium 100 mg", "nystatin oral suspension 5 ml") -- khác regex cũ chỉ bắt
    đúng 1 từ viết hoa đầu ngay trước liều, bỏ sót phần lớn tên thuốc đa từ/viết thường.
    """
    results = []
    for m in DOSE_RE.finditer(text):
        before = text[: m.start()]
        words = re.findall(r"[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\-]*", before)
        drug_words = []
        for w in reversed(words[-max_words_before:]):
            if w.lower() in DRUG_STOPWORDS_BEFORE:
                break
            drug_words.insert(0, w)
        if drug_words:
            results.append(f"{' '.join(drug_words)} {m.group(0)}".strip())
    return results


VITALS_RE = re.compile(r"\bVitals?:?[^\n.]{0,80}", re.IGNORECASE)


def load_mtsamples_pools(n_extra_specialties=10, rows_per_specialty=8):
    """
    drug_pool/vitals_pool: gộp từ TẤT CẢ specialty đã sample (giữ nguyên mục đích cũ).
    specialty_pool: giờ ĐẢM BẢO có đủ data cho từng chuyên khoa trong SPECIALTY_CONFIG
    (trước đây random 25/39 specialty nên có thể miss hẳn 1 chuyên khoa cần dùng để ép
    bối cảnh đa dạng) -- cộng thêm vài specialty ngẫu nhiên khác để tăng đa dạng thuốc/vitals.
    """
    if not MTSAMPLES_PATH.exists():
        print(f"[!] Không thấy {MTSAMPLES_PATH}, bỏ qua ngữ liệu mtsamples.")
        return [], [], {}

    df = pd.read_csv(MTSAMPLES_PATH)
    df["medical_specialty"] = df["medical_specialty"].str.strip()
    all_specialties = df["medical_specialty"].dropna().unique().tolist()

    curated = [name for name, _, _ in SPECIALTY_CONFIG if name in all_specialties]
    remaining = [s for s in all_specialties if s not in curated]
    random.shuffle(remaining)
    specialties = curated + remaining[:n_extra_specialties]

    drug_pool = set()
    vitals_pool = []
    specialty_pool = {}

    for spec in specialties:
        rows = df[df["medical_specialty"] == spec].sample(
            n=min(rows_per_specialty, len(df[df["medical_specialty"] == spec])),
            random_state=42,
        )
        texts = rows["transcription"].dropna().tolist()
        specialty_pool[spec] = texts
        for t in texts:
            for drug in extract_drug_candidates(str(t)):
                drug_pool.add(drug)
            for m in VITALS_RE.finditer(str(t)):
                vitals_pool.append(m.group(0).strip())

    return list(drug_pool), vitals_pool, specialty_pool


# ----------------------------------------------------------------------------
# 2b. ICD-10 XML: nguồn tên chẩn đoán thật (tiếng Anh) để gợi ý nội dung CHẨN_ĐOÁN
#     thực tế, đa dạng, và có khả năng map ngược ICD10 tốt ở bước linking sau này.
#     Chỉ lấy các chương lâm sàng phổ biến trong bệnh án nội khoa/cấp cứu -- loại các
#     chương ít gặp trong văn cảnh bệnh sử kể lại (nguyên nhân bên ngoài, mã đặc biệt...).
# ----------------------------------------------------------------------------
ICD10_CLINICAL_CHAPTERS = set("ABCDEGHIJKLMN")  # bỏ O,P,Q,R,S,T,U,V,W,X,Y,Z (ít gặp trong text kể bệnh sử)


def load_icd10_pool(max_per_chapter=150):
    """Trả về dict {chapter_letter: [(code, english_label), ...]} -- giữ theo chương để
    có thể sample đúng chương phù hợp với chuyên khoa đang chọn (xem SPECIALTY_CONFIG)."""
    if not ICD10_PATH.exists():
        print(f"[!] Không thấy {ICD10_PATH}, bỏ qua ngữ liệu ICD-10.")
        return {}

    tree = ET.parse(ICD10_PATH)
    root = tree.getroot()

    by_chapter = {}
    for cls in root.findall("Class"):
        if cls.get("kind") != "category":
            continue
        code = cls.get("code", "")
        if not code or code[0] not in ICD10_CLINICAL_CHAPTERS:
            continue
        for rubric in cls.findall("Rubric"):
            if rubric.get("kind") == "preferred":
                label = rubric.find("Label")
                if label is not None and label.text:
                    by_chapter.setdefault(code[0], []).append((code, label.text.strip()))
                break

    for chapter in by_chapter:
        random.shuffle(by_chapter[chapter])
        by_chapter[chapter] = by_chapter[chapter][:max_per_chapter]
    return by_chapter


def sample_diagnosis_for_chapters(icd10_by_chapter, chapters, k=2):
    """Gộp pool từ các chương liên quan tới 1 chuyên khoa, sample k cái. Fallback sang
    toàn bộ pool nếu chương đó rỗng (vd thiếu file ICD10 hoặc chương hiếm gặp)."""
    pool = []
    for ch in chapters:
        pool.extend(icd10_by_chapter.get(ch, []))
    if not pool:
        pool = [item for entries in icd10_by_chapter.values() for item in entries]
    return random.sample(pool, k=min(k, len(pool))) if pool else []


# ----------------------------------------------------------------------------
# 2c. RxNorm (RXNCONSO.RRF từ UMLS): nguồn tên thuốc thật đầy đủ liều+dạng bào chế,
# đa dạng và chuẩn hơn nhiều so với regex tự trích từ mtsamples. Chỉ lấy SAB=RXNORM
# (bỏ SNOMEDCT/MTHSPL... để tránh trùng lặp cùng 1 thuốc từ nhiều nguồn) và TTY thuộc
# {SCD, SBD, IN} -- lần lượt là: thuốc generic đủ liều+dạng, thuốc biệt dược, hoạt chất.
# File RRF rất lớn (hàng trăm MB, hàng triệu dòng) -- đọc theo streamline từng dòng,
# không load hết vào RAM cùng lúc.
# ----------------------------------------------------------------------------
# Ưu tiên thuốc lâm sàng có dạng/liều hoặc biệt dược. TTY=IN chứa cả hoạt chất
# và nhiều hóa chất trần hiếm dùng (vd manganese dioxide), dễ gây nhiễu linking.
RXNORM_TTY_KEEP = {"SCD", "SBD"}


def load_rxnorm_pool(max_names=5000):
    """Trả về list tên thuốc thật (string) từ RXNCONSO.RRF, đã lọc SAB=RXNORM +
    TTY phù hợp, dedupe. Đọc streaming để tránh tốn RAM với file gốc rất lớn."""
    if not RXNORM_PATH.exists():
        print(f"[!] Không thấy {RXNORM_PATH}, bỏ qua ngữ liệu RxNorm.")
        return []

    names = set()
    with open(RXNORM_PATH, encoding="utf-8", errors="ignore") as f:
        for line in f:
            cols = line.rstrip("\n").split("|")
            if len(cols) < 15:
                continue
            sab, tty, string_ = cols[11], cols[12], cols[14]
            if sab == "RXNORM" and tty in RXNORM_TTY_KEEP and string_:
                names.add(string_)
                if len(names) >= max_names * 3:  # thu thập dư rồi sample, tránh bias theo thứ tự file
                    break

    names = list(names)
    random.shuffle(names)
    return names[:max_names]


# ----------------------------------------------------------------------------
# 3. Gọi LLM qua OpenRouter
# ----------------------------------------------------------------------------
def call_llm(messages, temperature=0.9, max_tokens=1400):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    if not resp.ok:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise requests.HTTPError(f"{resp.status_code} {resp.reason} | body: {detail}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def strip_code_fence(text):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def parse_llm_json(text):
    """Parse object đầu tiên và bỏ commentary/code-fence/JSON thừa phía sau.

    ``JSONDecoder.raw_decode`` cứu được response dạng ``{...}\nGhi chú...`` hoặc hai object
    nối nhau; response rỗng/thực sự hỏng vẫn raise để retry.
    """
    cleaned = strip_code_fence(text)
    if not cleaned:
        raise json.JSONDecodeError("LLM trả content rỗng", cleaned, 0)
    object_start = cleaned.find("{")
    if object_start == -1:
        raise json.JSONDecodeError("Không tìm thấy JSON object", cleaned, 0)
    parsed, _end = json.JSONDecoder().raw_decode(cleaned[object_start:])
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("JSON gốc không phải object", cleaned, object_start)
    return parsed


# ----------------------------------------------------------------------------
# 4. Prompt builder — sinh 1 cặp input_text/output_text cho 1 section
# ----------------------------------------------------------------------------
def build_v4_generation_messages(
    section_cfg,
    drug_pool,
    icd10_by_chapter,
    specialty_pool,
    focus_cfg,
):
    """Prompt gọn cho augmentation free-form dài; không đọc hoặc copy test input."""
    available_specialties = [
        specialty for specialty in SPECIALTY_CONFIG if specialty_pool.get(specialty[0])
    ]
    if available_specialties:
        specialty_en, specialty_vi, specialty_chapters = random.choice(available_specialties)
    else:
        specialty_en, specialty_vi, specialty_chapters = (
            "General Medicine", "Nội tổng quát", "ABCDEGHIJKLMN"
        )
    diag_samples = sample_diagnosis_for_chapters(
        icd10_by_chapter, specialty_chapters, k=3
    )
    diagnosis_hint = "; ".join(label for _, label in diag_samples)
    if not diagnosis_hint:
        diagnosis_hint = "tăng huyết áp; viêm dạ dày; suy tim"
    specialty_drugs = V4_COMMON_DRUGS_BY_SPECIALTY.get(
        specialty_en, V4_COMMON_DRUGS_BY_SPECIALTY["General Medicine"]
    )
    drug_hint = ", ".join(specialty_drugs)
    format_name = focus_cfg["format"]
    boundary_noise_instruction = (
        V4_CONTROLLED_BOUNDARY_NOISE_INSTRUCTION.strip()
        if focus_cfg.get("boundary_noise")
        else (
            "Không bắt buộc tạo lỗi dính chữ ở record này. Nếu tự nhiên có thiếu khoảng trắng, "
            "entity vẫn phải dừng đúng ranh giới khái niệm."
        )
    )
    vihealthqa_profile = load_vihealthqa_style_profile()
    corpus_status = (
        f"đã đọc {vihealthqa_profile['rows']} cặp; trung vị câu hỏi "
        f"{vihealthqa_profile['question_p50']} ký tự, trung vị câu trả lời "
        f"{vihealthqa_profile['answer_p50']} ký tự; tổng Q&A p75/p90/p95 lần lượt "
        f"{vihealthqa_profile['combined_p75']}/{vihealthqa_profile['combined_p90']}/"
        f"{vihealthqa_profile['combined_p95']} ký tự"
        if vihealthqa_profile["available"]
        else "không tìm thấy corpus; dùng phân bố fallback tổng quát"
    )

    user_prompt = f"""Sinh đúng 1 object JSON làm dữ liệu NER y tế tiếng Việt FREE-FORM DÀI VÀ ĐA DẠNG.
Không sao chép văn bản đánh giá hoặc cố tái tạo một nguồn cụ thể; hãy sáng tạo ca/chủ đề mới.

CHỦ ĐỀ: {specialty_vi}. Gợi ý bệnh thật: {diagnosis_hint}.
Gợi ý thuốc phổ biến đúng chuyên khoa nếu thật sự cần: {drug_hint}. Không bắt buộc dùng thuốc;
chỉ ghép thuốc với chỉ định mà bạn chắc chắn hợp y khoa, không tự lấy thuốc lạ để chữa triệu chứng.

FORMAT V4 ĐƯỢC CHỌN: {format_name}
{focus_cfg['instruction'].strip()}

{boundary_noise_instruction}

THAM KHẢO HÌNH THỨC VIHEALTHQA (CHỈ THỐNG KÊ TỔNG HỢP): {corpus_status}.
- Chỉ dùng phân bố này để thay đổi nhịp câu và độ dài. KHÔNG chép, diễn lại gần nguyên văn,
  đưa URL, tên nguồn hoặc cố phục dựng bất kỳ question/answer nào từ corpus.
- Với Q&A thông thường có thể quanh p75-p95; thỉnh thoảng chủ động lấy long-tail 1800-3200
  ký tự. Với format long_timeline, ưu tiên đúng khoảng rất dài ghi trong focus phía trên.

SCHEMA CHỈ CÓ 5 TYPE:
- THUỐC: tên thuốc/dược chất cụ thể; span có thể gồm liều, đường dùng, tần suất liền kề.
- CHẨN_ĐOÁN: tên bệnh/chẩn đoán/tình trạng bệnh lý đầy đủ.
- TRIỆU_CHỨNG: biểu hiện/trạng thái lâm sàng đầy đủ, gồm vị trí/mức độ/tính chất cần thiết.
- TÊN_XÉT_NGHIỆM: tên chỉ số, kỹ thuật xét nghiệm/chẩn đoán hình ảnh.
- KẾT_QUẢ_XÉT_NGHIỆM: giá trị hoặc toàn bộ finding liên tục của kỹ thuật tương ứng.

ASSERTION:
- isNegated khi CHẨN_ĐOÁN/TRIỆU_CHỨNG/THUỐC bị phủ định; không đưa từ phủ định vào span.
- isHistorical khi thuộc tiền sử/trước nhập viện; isFamily khi chủ thể là người thân ruột thịt.
- Có thể multi-label nếu thật sự phù hợp, nhưng cấm đồng thời isHistorical + isNegated.
- Kết quả xét nghiệm/finding luôn assertions: []; giữ từ phủ định trong chính span kết quả.

QUY TẮC GOLD BẮT BUỘC:
1. Mỗi entity phải là substring y hệt input_text và entities sắp theo thứ tự xuất hiện.
2. Mỗi occurrence thật phải có entity riêng. Không tự tạo duplicate khi text chỉ xuất hiện một lần.
3. Không tạo span vụn/chung chung như "Bệnh", "cơ quan", "máu", "protein", "men",
   "hồng cầu", "rối loạn", tên protein/enzyme đứng riêng, "u", "hơi", "mạnh", "âm", "lên",
   "thu", "nhĩ", "thất", "giây". Nếu là entity,
   phải lấy trọn tên bệnh hoặc toàn cụm triệu chứng có vị trí/tính chất.
4. Tên bệnh/triệu chứng/thuốc cụ thể xuất hiện trong bài giải thích y khoa vẫn annotate bằng
   span đầy đủ. Nhưng gen, protein, enzyme, nhiễm sắc thể, mô/cơ quan đứng riêng, yếu tố nguy
   cơ, lối sống, tuổi, giới, nghề nghiệp và giải phẫu đơn thuần là O vì ngoài 5 type.
5. Thủ thuật/phẫu thuật/hóa trị/ghép tạng/nội soi mang nghĩa hành động là O. Một kỹ thuật giới
   thiệu kết quả như "nội soi dạ dày ghi nhận..." vẫn là TÊN_XÉT_NGHIỆM và có finding đi kèm.
6. Thời gian như 20 giây, 3 ngày, 5 năm là O. Ngoại lệ: tần suất liền trong span thuốc.
7. Lab phải tách từng cặp tên/kết quả, kể cả khi cùng chỉ số được đo nhiều lần hoặc kết quả
   là chữ như "bình thường". Không annotate tuổi hay thời gian thành kết quả xét nghiệm.
   Trong "xét nghiệm máu cho thấy bilirubin toàn phần 5.2 mg/dL", không dùng cụm chung
   "xét nghiệm máu" làm tên rồi nuốt bilirubin vào kết quả; TÊN phải là "bilirubin toàn phần"
   và KẾT_QUẢ là giá trị/qualifier tương ứng.
8. Imaging phải tách tên kỹ thuật khỏi toàn bộ finding liên tục; không dính từ phủ định vào
   tên kỹ thuật và không đổi finding thành CHẨN_ĐOÁN.
9. Thuốc phải có thật và hợp y khoa; placeholder "*******" không phải THUỐC. Không annotate
   phương pháp chung "uống thuốc". Thuốc trước nhập viện là isHistorical, thuốc hiện tại là [].
10. Trong input nhiều khối, assertion chỉ theo ngữ cảnh cục bộ; không truyền assertion qua
    heading/khối khác. Không ghép bệnh-thuốc vô lý hoặc thêm một finding rác không liên quan.
11. Cho phép typo/khoảng trắng nhẹ giống input thật, nhưng tự kiểm tra để entity vẫn nguyên vẹn.
12. Không overlap entity và không tạo entity con nằm trong một entity dài cùng occurrence.
13. Trong bài kiến thức/Q&A, assertions mặc định là [] vì đó là kiến thức chung, KHÔNG phải tiền sử
    của bệnh nhân. Chỉ dùng isHistorical khi chính occurrence có cue cục bộ rõ như "tiền sử",
    "trước đây", "đã từng mắc" hoặc "đã dùng tại nhà trước nhập viện".
14. Phân biệt bệnh với enzyme/xét nghiệm: tên enzyme hoặc chất sinh học đứng riêng là O; tên bệnh
    đầy đủ là CHẨN_ĐOÁN; chỉ dùng TÊN_XÉT_NGHIỆM khi span chứa kỹ thuật/cue thật như "xét nghiệm",
    "định lượng", "hoạt độ" hay "sàng lọc". Không biến tên bệnh thành tên xét nghiệm.
15. Tác nhân cần tránh, hóa chất gia dụng, thực phẩm, trà/thảo dược được nhắc để tư vấn và thiết bị
    hỗ trợ không phải THUỐC bệnh nhân. Chỉ annotate dược chất/thuốc cụ thể thực sự được dùng.
16. Nếu input chứa hai khái niệm bị dính qua dấu chấm, phải trả hai entity riêng. Cấm mọi entity
    chứa mẫu dấu chấm + chữ liền nhau. Cấm tách tên bệnh hai từ thành hai mảnh như "Bại" và "não".

Trả về DUY NHẤT JSON:
{{
  "input_text": "<free-form text V4 format {format_name}>",
  "entities": [
    {{"text": "<substring>", "type": "<1 trong 5 type>", "assertions": []}}
  ]
}}
"""
    return [
        {
            "role": "system",
            "content": (
                "Bạn sinh dữ liệu NER y tế tiếng Việt chính xác theo schema JSON. "
                "Ưu tiên boundary trọn nghĩa, recall đủ occurrence và không bịa entity ngoài 5 type."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def build_v5_generation_messages(
    section_cfg,
    drug_pool,
    icd10_by_chapter,
    specialty_pool,
    focus_cfg,
):
    """Prompt V5 ngắn/trung bình, ưu tiên contrast, precision và NER boundary."""
    available_specialties = [
        specialty for specialty in SPECIALTY_CONFIG if specialty_pool.get(specialty[0])
    ]
    if available_specialties:
        specialty_en, specialty_vi, _chapters = random.choice(available_specialties)
    else:
        specialty_en, specialty_vi = "General Medicine", "Nội tổng quát"

    targets = random.sample(V5_TARGET_CONCEPTS, k=min(4, len(V5_TARGET_CONCEPTS)))
    target_hint = "; ".join(targets)
    drug_hint = ", ".join(
        V4_COMMON_DRUGS_BY_SPECIALTY.get(
            specialty_en, V4_COMMON_DRUGS_BY_SPECIALTY["General Medicine"]
        )[:5]
    )
    dirty_instruction = (
        V5_DIRTY_TEXT_INSTRUCTION.strip()
        if focus_cfg.get("boundary_noise") and focus_cfg["format"] != "dirty"
        else ""
    )
    qa_instruction = (
        V5_QA_TEXT_INSTRUCTION.strip() if focus_cfg.get("qa_style") else ""
    )
    style_hint = random.choice(V5_STYLE_VARIANTS)

    user_prompt = f"""Sinh đúng 1 object JSON làm dữ liệu NER y tế tiếng Việt V5 chống overfit.
Không sao chép test/public input, không lặp template ba mục cố định và không cố nhồi đủ năm type.

NHÓM CHÍNH: {focus_cfg['format']}
{focus_cfg['instruction'].strip()}

{dirty_instruction}

{qa_instruction}

BỐI CẢNH để đa dạng hóa: {specialty_vi}. Các khái niệm ưu tiên xoay vòng: {target_hint}.
Nếu thật sự cần thuốc, chỉ dùng thuốc phổ biến phù hợp như {drug_hint}; không bịa hóa chất lạ.
VĂN PHONG BẮT BUỘC LUÂN PHIÊN Ở RECORD NÀY: {style_hint}.

SCHEMA CHỈ CÓ:
- THUỐC, CHẨN_ĐOÁN, TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM.
- Assertions chỉ gồm isNegated, isHistorical, isFamily; không có assertion thì dùng [].

GOLD BẮT BUỘC:
1. Entity là substring y hệt input_text, không overlap, sắp đúng thứ tự occurrence.
2. Span lấy trọn khái niệm nhưng không nuốt hoàn cảnh O. Vị trí/tính chất thuộc triệu chứng;
   hoạt động như đi bộ, làm việc, thức khuya và mốc thời gian độc lập là O.
3. Cue assertion có phạm vi cục bộ. "không nhớ", "không cải thiện", "không dùng đều" không
   phủ định bệnh/triệu chứng. isFamily chỉ cho bệnh của người thân; isHistorical chỉ khi occurrence
   thật sự thuộc quá khứ/tiền sử. Không truyền cue sang entity khác trong cùng câu.
   Riêng "đã được khám vì X" hoặc "đã đến viện vì X" trong đợt bệnh hiện tại thì X vẫn là [].
4. Finding sau X-quang/CT/MRI/siêu âm là KẾT_QUẢ_XÉT_NGHIỆM []; tên kỹ thuật là
   TÊN_XÉT_NGHIỆM. Phủ định trong finding nằm trong span kết quả, không dùng isNegated.
5. Lab tách từng TÊN và KẾT_QUẢ. Tuổi, thời gian, đơn vị thời gian và con số hành chính là O.
   Ví dụ `độ bão hòa oxy (SPO2) từ 88-92 % khi thở khí trời` phải lấy TÊN=`độ bão hòa oxy
   (SPO2)`, KẾT_QUẢ=`88-92 %`; `WBC:12.5;CRP:64mg/L` phải đủ bốn entity riêng.
6. Không annotate từ chung/vụn, giải phẫu đứng riêng, lối sống, exposure, thủ thuật, phương pháp
   điều trị chung hoặc mục đích dùng thuốc. Thuốc phải là dược chất cụ thể.
   Span THUỐC giữ tên + hàm lượng/liều/đường dùng/tần suất liền kề theo gold BTC; cấm bắt riêng
   `1 gram`, `500 mg` khi thiếu tên. Nếu câu có hai thuốc phối hợp thì tách hai entity thuốc.
   Với danh sách trước nhập viện, mọi thuốc là isHistorical nhưng triệu chứng/chẩn đoán chỉ định
   đứng sau `điều trị` là entity riêng và không tự kế thừa isHistorical. Không được bỏ item cuối.
7. Raw có dấu câu dính vẫn phải tách entity hai phía. Cấm entity chứa hai khái niệm kiểu
   "đau ngực.Hồi hộp". Số thập phân 15.2 vẫn là một giá trị, không coi dấu chấm là ranh giới câu.
   Không tạo span cụt kiểu `độ`, `từ`, `nhịp`, `Thiếu`, `chảy`, `bên`, `đau ấn vùng` hay
   `phù hợp với`; phải lấy trọn khái niệm có nghĩa. Heading `chẩn đoán hình ảnh` không phải tên
   một kỹ thuật xét nghiệm cụ thể.
8. Record thưa được phép có 0 entity nếu nhóm yêu cầu; tuyệt đối không thêm entity giả chỉ để
   đạt số lượng. Với nhóm khác phải tuân thủ số lượng/type ghi trong instruction.
9. Viết đa dạng: có thể một câu, ghi nhanh, đoạn tư vấn, tin nhắn, bullet ngắn hoặc văn xuôi;
   không phải lúc nào cũng bắt đầu bằng "Bệnh nhân" hay cùng tuổi/giới.

Trả về DUY NHẤT JSON:
{{
  "input_text": "<văn bản mới>",
  "entities": [
    {{"text": "<substring>", "type": "<1 trong 5 type>", "assertions": []}}
  ]
}}
"""
    return [
        {
            "role": "system",
            "content": (
                "Bạn sinh gold NER y tế tiếng Việt chính xác. Ưu tiên precision, boundary và "
                "phạm vi assertion; không ép record tổng hợp phải có đủ mọi loại entity."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def build_generation_messages(
    section_cfg,
    seed_examples,
    drug_pool,
    vitals_pool,
    icd10_by_chapter,
    specialty_pool,
    force_assertion,
    focus_cfg=None,
):
    if focus_cfg and focus_cfg.get("mode") == "v4":
        return build_v4_generation_messages(
            section_cfg, drug_pool, icd10_by_chapter, specialty_pool, focus_cfg
        )
    if focus_cfg and focus_cfg.get("mode") == "v5":
        return build_v5_generation_messages(
            section_cfg, drug_pool, icd10_by_chapter, specialty_pool, focus_cfg
        )

    v4_mode = bool(focus_cfg and focus_cfg.get("mode") == "v4")
    v4_format = focus_cfg.get("format") if v4_mode else None
    qa_mode = v4_format == "qa"
    structure_total = sum(STRUCTURE_STYLE_WEIGHTS.values())
    if abs(structure_total - 1.0) > 1e-9 or any(
        weight < 0 for weight in STRUCTURE_STYLE_WEIGHTS.values()
    ):
        raise ValueError("STRUCTURE_STYLE_WEIGHTS phải không âm và có tổng bằng 1.0")
    structure_mode = random.choices(
        tuple(STRUCTURE_STYLE_WEIGHTS),
        weights=tuple(STRUCTURE_STYLE_WEIGHTS.values()),
        k=1,
    )[0]

    if structure_mode == "classic_heading":
        heading = section_cfg["heading"]
        structure_instruction = (
            f'FORMAT MỞ ĐẦU: bắt đầu đúng bằng heading cổ điển "{heading}" ở một dòng riêng. '
            "Đây là format cũ cần tiếp tục được giữ trong một phần dữ liệu."
        )
    elif structure_mode == "alternative_heading":
        alternatives = [
            value for value in section_cfg.get("headings", [])
            if value != section_cfg["heading"]
        ]
        heading = random.choice(alternatives or [section_cfg["heading"]])
        structure_instruction = (
            f'FORMAT MỞ ĐẦU: bắt đầu bằng heading ngắn "{heading}" ở một dòng riêng; '
            f'heading này đồng nghĩa với nhóm "{section_cfg["heading"]}".'
        )
    else:
        heading = ""
        no_heading_examples = {
            "tien_su": (
                'Không dùng heading. Đi thẳng vào nội dung bằng một cue lịch sử tự nhiên như '
                '"Đang theo dõi...", "Trước đây từng...", "Thuốc dùng tại nhà gồm..." hoặc '
                'ghi nhanh "TS:". Không mặc định lặp mẫu "Bệnh nhân nữ ... có tiền sử ...".'
            ),
            "hien_tai": (
                'Không dùng heading. Đi thẳng bằng "Vào viện vì...", "Khoảng ... ngày nay...", '
                '"BN than..." hoặc một bullet triệu chứng ngắn.'
            ),
            "danh_gia": (
                'Không dùng heading. Đi thẳng bằng "Khám ghi nhận...", "Tại khoa cấp cứu...", '
                '"XN:"/"CĐHA:" hoặc các dòng kết quả ngắn.'
            ),
        }
        structure_instruction = "FORMAT MỞ ĐẦU: " + no_heading_examples[section_cfg["key"]]

    if v4_mode:
        heading = ""
        v4_structure_instructions = {
            "qa": (
                'FORMAT V4 Q&A: dùng "Câu hỏi từ người dùng:"/"Hỏi:" và phần trả lời bác sĩ.'
            ),
            "clinical_long": (
                "FORMAT V4 BỆNH ÁN DÀI: dùng 2-3 mục đánh số, văn xuôi xen bullet và ghi nhanh."
            ),
            "hybrid": (
                "FORMAT V4 LAI: ghép hai khối Q&A/bài tư vấn/bệnh án bằng heading và newline rõ."
            ),
            "education": (
                'FORMAT V4 BÀI KIẾN THỨC: tiêu đề "<bệnh> là gì?" và các mục đánh số/bullet.'
            ),
        }
        structure_instruction = v4_structure_instructions[v4_format]
    seed_snippet = random.choice(seed_examples) if seed_examples else "(không có ví dụ thật, tự sáng tạo)"
    drug_hint = ", ".join(random.sample(drug_pool, k=min(4, len(drug_pool)))) if drug_pool else "metoprolol 25mg po bid, amlodipine 10mg po daily"
    vitals_hint = random.choice(vitals_pool) if vitals_pool else "Vitals: VS 98.3 129/87 56 18 99RA"

    # Chọn 1 chuyên khoa cho lần gen này -- ưu tiên chuyên khoa có data thật trong
    # specialty_pool, để vừa lấy diagnosis_hint đúng chương ICD10 vừa có specialty_snippet
    # thật làm ngữ cảnh (khác nhau mỗi lần gọi -> tránh quay vòng vài bệnh quen thuộc).
    allowed_specialties = set(focus_cfg.get("specialty_names", [])) if focus_cfg else set()
    available_specialties = [
        specialty for specialty in SPECIALTY_CONFIG
        if specialty_pool.get(specialty[0])
        and (not allowed_specialties or specialty[0] in allowed_specialties)
    ]
    if available_specialties:
        specialty_en, specialty_vi, specialty_chapters = random.choice(available_specialties)
        specialty_snippet = random.choice(specialty_pool[specialty_en])[:800]
    else:
        specialty_en, specialty_vi, specialty_chapters = "General Medicine", "Nội tổng quát", "ABCDEGHIJKLMN"
        specialty_snippet = "(không có ngữ liệu mtsamples cho chuyên khoa này)"

    if not 0 <= DEMOGRAPHIC_INCLUDE_PROBABILITY <= 1:
        raise ValueError("DEMOGRAPHIC_INCLUDE_PROBABILITY phải nằm trong khoảng 0..1")
    if random.random() < DEMOGRAPHIC_INCLUDE_PROBABILITY:
        if specialty_en == "Obstetrics / Gynecology":
            patient_sex = "nữ"
            patient_age = random.randint(18, 49)
        else:
            patient_sex = random.choice(["nam", "nữ"])
            age_band = random.choices(
                [(18, 39), (40, 64), (65, 90)],
                weights=[0.35, 0.40, 0.25],
                k=1,
            )[0]
            patient_age = random.randint(*age_band)
        demographic_instruction = (
            f'NHÂN KHẨU HỌC CHO RECORD NÀY: nếu nhắc tuổi/giới, PHẢI dùng đúng "{patient_sex}, '
            f'{patient_age} tuổi" hoặc dạng ghi tắt tương đương "{patient_sex} {patient_age}t". '
            "Không đổi về tuổi 45 và không annotate tuổi/giới thành entity. Không nhất thiết phải "
            "mở câu bằng cụm Bệnh nhân."
        )
    else:
        demographic_instruction = (
            "NHÂN KHẨU HỌC CHO RECORD NÀY: KHÔNG ghi tuổi hoặc giới tính; đi thẳng vào bệnh sử, "
            "triệu chứng, thuốc hoặc kết quả. Không tự thêm mẫu mặc định bệnh nhân 45 tuổi."
        )

    diag_samples = sample_diagnosis_for_chapters(icd10_by_chapter, specialty_chapters, k=2)
    # Chỉ đưa tên bệnh; không đưa mã ICD vào prompt vì LLM dễ copy "(N39.4)" vào
    # input/entity, tạo span nhiễu và overlap không cần thiết cho bài NER.
    diagnosis_hint = "; ".join(label for _, label in diag_samples) if diag_samples else "hypertension, type 2 diabetes mellitus"
    # ------------------------------------------------------------------------
    # Đa dạng văn phong: data cũ toàn văn xuôi "sạch", đủ chủ vị, đúng ngữ pháp --
    # không giống bệnh án thật (bác sĩ viết vội, viết tắt, gạch đầu dòng). Random 3
    # phong cách để model quen với nhiều dạng input khác nhau, đặc biệt "gạch đầu
    # dòng" giúp tránh bug gộp cục số liệu như đã gặp trước đây (model dễ nhầm khi
    # mọi thứ dính liền trong 1 câu văn xuôi dài).
    # ------------------------------------------------------------------------
    style_mode = random.choices(
        ["van_xuoi_day_du", "viet_tat_cap_cuu", "gach_dau_dong"],
        weights=[0.4, 0.3, 0.3],
    )[0]

    if style_mode == "viet_tat_cap_cuu":
        style_instruction = (
            "PHONG CÁCH CHO LẦN NÀY: viết TẮT kiểu bác sĩ cấp cứu ghi vội, câu cụt lủn,\n"
            "bỏ bớt chủ ngữ/động từ khi có thể. Dùng các từ viết tắt thông dụng: \"BN\" (bệnh\n"
            "nhân), \"TS\" (tiền sử), \"T\" hoặc \"t\" (tuổi, viết liền sau số vd \"35t\"), \"đ/trị\"\n"
            "(điều trị), \"KQ\" (kết quả), \"XN\" (xét nghiệm), \"CĐ\" (chẩn đoán). Ví dụ văn phong\n"
            "(không copy nguyên văn, chỉ bắt chước CÁCH VIẾT): \"BN nam 35t. TS: viêm amidan tái\n"
            "phát. Cắt amidan 2018. Hiện: viêm họng cấp đ/trị amox 500mg x2.\" Entity vẫn phải là\n"
            "substring chính xác của input_text viết tắt này (span có thể ngắn hơn bình thường)."
        )
    elif style_mode == "gach_dau_dong":
        style_instruction = (
            "PHONG CÁCH CHO LẦN NÀY: dùng GẠCH ĐẦU DÒNG (-) thay vì văn xuôi liền mạch, đặc\n"
            "biệt với danh sách thuốc/triệu chứng/xét nghiệm. Ví dụ định dạng (không copy nguyên\n"
            "văn, chỉ bắt chước CẤU TRÚC):\n"
            "XN:\n"
            "- Glucose: 5.8 mmol/L\n"
            "- AST: 45 U/L\n"
            "- CRP: 12 mg/L\n"
            "Mỗi dòng vẫn phải tách TÊN_XÉT_NGHIỆM/KẾT_QUẢ_XÉT_NGHIỆM đúng theo rule vitals bên\n"
            "dưới dù format khác đi. Có thể dùng thêm dấu \"/\" (vd \"đau tăng/giảm khi vận động\")\n"
            "hoặc \"->\" (vd \"sốt cao -> dùng paracetamol\") xen kẽ cho tự nhiên."
        )
    else:
        style_instruction = (
            "PHONG CÁCH CHO LẦN NÀY: văn xuôi đầy đủ câu, đúng ngữ pháp (phong cách mặc định)."
        )

    if v4_mode:
        style_instruction = (
            "PHONG CÁCH V4: free-form text dài đa nguồn, có thể đổi giữa văn xuôi, "
            "informal, heading đánh số và bullet theo đúng format V4 đã chọn. Giữ các lỗi nhẹ "
            "có kiểm soát nhưng entity phải luôn là substring trọn nghĩa."
        )

    negation_vocab_instruction = (
        "ĐA DẠNG TỪ PHỦ ĐỊNH: đừng chỉ lặp \"không có\" -- trộn thêm các cách diễn đạt phủ định\n"
        'khác: "chưa ghi nhận", "âm tính với", "phủ nhận", "bệnh nhân bảo không bị", "không thấy",\n'
        '"chưa từng". Ví dụ: "Chưa ghi nhận sốt.", "Âm tính với lao phổi.", "Bệnh nhân phủ nhận\n'
    'đau ngực." -- entity vẫn chỉ lấy khái niệm cốt lõi (vd "sốt", "lao phổi", "đau ngực"),\n'
        "cụm phủ định (dù là từ nào) không được nằm trong text span, chỉ quyết định assertions.\n"
        "Nếu phủ định mở đầu một DANH SÁCH, assertion phải áp dụng cho TẤT CẢ phần tử: \"Không\n"
        "ghi nhận triệu chứng hạ đường huyết như vã mồ hôi, run tay hay choáng váng\" -> cả ba\n"
        "TRIỆU_CHỨNG đều có assertions: [\"isNegated\"], không chỉ phần tử đầu tiên."
    )

    include_noise = random.random() < 0.4
    noise_instruction = (
        (
            "NHIỄU SINH HOẠT (bắt buộc thêm 1 câu cho lần này): chèn 1 câu bệnh nhân kể về đời\n"
            'sống KHÔNG liên quan y khoa, KHÔNG gán entity nào cho câu này (để trống, coi như "O").\n'
            'Ví dụ: "Bệnh nhân khai dạo này áp lực công việc, hay thức khuya chạy deadline, sáng\n'
            'uống 2 ly cà phê đen." Đặt câu này TRƯỚC hoặc XEN GIỮA nội dung y khoa, không tách\n'
            "riêng ở cuối. Mục đích: dạy model KHÔNG nhầm \"thức khuya\", \"deadline\", \"cà phê đen\",\n"
            "\"áp lực công việc\" thành triệu chứng/chẩn đoán."
        )
        if include_noise
        else "Không cần thêm câu sinh hoạt ngoài lề cho lần này."
    )
    if v4_mode:
        noise_instruction = (
            "Không chèn một dòng y khoa ngẫu nhiên vô cớ. Nhiễu V4 đến từ độ dài, đổi format, "
            "lặp mention, typo nhẹ và các thuật ngữ ngoài 5 type."
        )

    force_txt = {
        "isNegated": "Bắt buộc có ít nhất 1 triệu chứng/chẩn đoán bị PHỦ ĐỊNH (ví dụ 'không sốt', 'không buồn nôn') gán assertions: ['isNegated'].",
        "isFamily": "Bắt buộc có ít nhất 1 khái niệm thuộc về NGƯỜI THÂN bệnh nhân (ví dụ 'mẹ bị tiểu đường') gán assertions: ['isFamily']. LƯU Ý: span entity CHỈ là tên chẩn đoán/triệu chứng (vd 'tiểu đường'), KHÔNG bao gồm chủ ngữ 'mẹ' hay mốc tuổi.",
        "isHistorical": "Bắt buộc có ít nhất 1 thuốc/chẩn đoán thuộc TIỀN SỬ (không phải hiện tại) gán assertions: ['isHistorical'].",
        None: "Không bắt buộc assertion đặc biệt, dùng assertions: [] nếu là thông tin hiện tại/bình thường.",
    }[force_assertion]

    # Baseline phải giữ nguyên prompt cũ. Chỉ profile quota mới được chèn block bù lỗi.
    focus_prompt_block = ""
    if focus_cfg:
        focus_label = "V4 LONG-FREEFORM" if v4_mode else "V2/V3"
        focus_prompt_block = f"""
=== GỢI Ý BỔ SUNG {focus_label} CHO RECORD NÀY ===
{focus_cfg['instruction'].strip()}
=== HẾT GỢI Ý BỔ SUNG {focus_label} ===
"""

    if v4_mode:
        document_names = {
            "qa": "đoạn hỏi đáp y khoa dài",
            "clinical_long": "bệnh án nhiều mục rất dài",
            "hybrid": "free-form text lai nhiều khối",
            "education": "bài giải thích bệnh dài",
        }
        document_request = (
            f"Hãy tạo 1 {document_names[v4_format]} bằng tiếng Việt tự nhiên và tổng quát"
        )
        section_requirement = (
            f"Nội dung chính nghiêng về {section_cfg['heading']} nhưng format tuân theo V4; "
            "không ép toàn record vào một section ngắn."
        )
        heading_requirement = (
            "- Tuân thủ đúng format V4 được chọn. Heading có thể lặp theo các khối độc lập, "
            "nhưng không được dính chữ hoặc tạo một dòng rác không liên quan."
        )
        section_semantics_requirement = (
            "- Trong bài giải thích/Q&A, vẫn annotate tên bệnh, triệu chứng, thuốc và xét nghiệm "
            "cụ thể xuất hiện rõ; chỉ để O cho từ sinh học chung hoặc nội dung ngoài 5 type."
        )
        output_text_description = f"<free-form text V4 format {v4_format}>"
    else:
        document_request = (
            f"Hãy tạo 1 đoạn bệnh án tiếng Việt TỰ NHIÊN (không dịch máy) thuộc nhóm nội dung "
            f"{section_cfg['heading']}"
        )
        section_requirement = f'Mô tả section: {section_cfg["desc"]}'
        heading_requirement = (
            '- Heading chỉ xuất hiện MỘT LẦN ở dòng đầu. Không tạo "2. Tiền sử... / 3. Tiền sử..." và\n'
            '  không dính chữ kiểu "Tiền sử bệnh hiện tạiBệnh nhân", "nôn ói.Ngoài ra".'
        )
        section_semantics_requirement = (
            f'- Dù có heading hay không, nội dung vẫn phải giữ đúng ý nghĩa lâm sàng của nhóm\n'
            f'  "{section_cfg["heading"]}". Với record tiền sử không heading, phải có cue thời gian rõ để\n'
            "  isHistorical không phụ thuộc vào việc nhìn thấy một tiêu đề cố định."
        )
        output_text_description = "<đoạn bệnh án tiếng Việt theo FORMAT MỞ ĐẦU đã yêu cầu>"

    gold_fewshot_output_str = json.dumps(GOLD_FEWSHOT_OUTPUT, ensure_ascii=False)
    gold_fewshot_2_output_str = json.dumps(GOLD_FEWSHOT_2_OUTPUT, ensure_ascii=False)

    compressed_vitals_instruction = (
        '  Với cụm vitals dạng nén (vd "VS 98.3 129/87 56 18 99RA"), LUÔN tách thành\n'
        '  TÊN_XÉT_NGHIỆM="VS" và KẾT_QUẢ_XÉT_NGHIỆM="98.3 129/87 56 18 99RA".\n'
        '  KHÔNG gộp chữ "VS"/"Vitals" vào entity kết quả.'
    )

    vitals_split_instruction = (
        "- QUY TẮC BẮT BUỘC, KHÔNG CÓ NGOẠI LỆ: mọi xét nghiệm/chỉ số CÓ TÊN RÕ RÀNG (Glucose máu,\n"
        "  HbA1c, Huyết áp, Nhịp tim, SpO2, BMI, AFB đờm, X-quang phổi, Men gan AST...) LUÔN LUÔN\n"
        "  phải tách thành 2 entity riêng: 1 TÊN_XÉT_NGHIỆM (chỉ tên chỉ số, không kèm giá trị) +\n"
        "  1 KẾT_QUẢ_XÉT_NGHIỆM (chỉ giá trị/kết quả, không kèm tên). VÍ DỤ ĐÚNG:\n"
        '  input có "Glucose máu 6.2 mmol/L" -> 2 entity: {"text":"Glucose máu","type":"TÊN_XÉT_NGHIỆM"}\n'
        '  và {"text":"6.2 mmol/L","type":"KẾT_QUẢ_XÉT_NGHIỆM"}. TUYỆT ĐỐI KHÔNG gộp thành 1 entity\n'
        '  "Glucose máu 6.2 mmol/L" -- đây là lỗi sai nghiêm trọng hay gặp nhất, PHẢI tránh.\n'
        "- KẾT_QUẢ_XÉT_NGHIỆM không chỉ là số: kết quả ĐỊNH TÍNH (vd \"(+)\", \"(-)\", \"tổn thương dạng\n"
        '  nốt\", \"bình thường\", \"không ghi nhận bất thường\") CŨNG PHẢI được trích làm entity\n'
        '  KẾT_QUẢ_XÉT_NGHIỆM, đi kèm TÊN_XÉT_NGHIỆM tương ứng đứng trước nó. VÍ DỤ: "AFB đờm (+)"\n'
        '  -> {"text":"AFB đờm","type":"TÊN_XÉT_NGHIỆM"} + {"text":"(+)","type":"KẾT_QUẢ_XÉT_NGHIỆM"}.\n'
        '  "X-quang phổi có tổn thương dạng nốt" -> {"text":"X-quang phổi","type":"TÊN_XÉT_NGHIỆM"} +\n'
        '  {"text":"tổn thương dạng nốt","type":"KẾT_QUẢ_XÉT_NGHIỆM"}. ĐỪNG bỏ sót phần kết quả này.\n'
        "- [MỚI] TÊN_XÉT_NGHIỆM: nếu nguồn viết kèm chú thích tiếng Việt trong ngoặc (vd\n"
        '  "NEUT% (Tỷ lệ % bạch cầu trung tính)"), PHẢI giữ NGUYÊN VẸN cả cụm kèm ngoặc làm\n'
        "  1 entity, KHÔNG cắt bớt chỉ lấy phần viết tắt. Thỉnh thoảng (không phải luôn luôn)\n"
        "  hãy viết bảng xét nghiệm dạng chuỗi công thức máu có chú thích kiểu này để đa dạng.\n"
        "- [MỚI] KẾT_QUẢ_XÉT_NGHIỆM KHÔNG BẮT BUỘC luôn kèm đơn vị: nếu viết dạng chuỗi liệt kê\n"
        'công thức máu kiểu "WBC:14,43; NEUT%:76,4;" (phổ biến trong phòng xét nghiệm thật),\n'
        "  kết quả là SỐ THUẦN không đơn vị, dùng DẤU PHẨY làm dấu thập phân (vd \"14,43\" không\n"
        '  phải "14.43 G/L"). Thỉnh thoảng dùng định dạng này thay vì luôn viết số+đơn vị kiểu\n'
        '  "6.2 mmol/L" -- cả 2 định dạng đều đúng, cần đa dạng, không dùng độc 1 kiểu.\n'
        f"{compressed_vitals_instruction}"
    )

    user_prompt = f"""{document_request}.

BỐI CẢNH CHUYÊN KHOA CHO LẦN NÀY: {specialty_vi}. Toàn bộ chẩn đoán/triệu chứng/thuốc/xét
nghiệm trong đoạn văn PHẢI phù hợp với chuyên khoa này (vd chuyên khoa Mắt thì viết về các
bệnh lý mắt, không viết lạc sang tim mạch). Dưới đây là 1 đoạn trích thật (tiếng Anh) từ hồ
sơ đúng chuyên khoa này để tham khảo TỪ VỰNG/BỐI CẢNH LÂM SÀNG (KHÔNG dịch nguyên văn, chỉ
lấy cảm hứng về loại bệnh/triệu chứng/xét nghiệm điển hình của chuyên khoa):
---
{specialty_snippet}
---

{style_instruction}

{demographic_instruction}

{negation_vocab_instruction}

{noise_instruction}
{focus_prompt_block}

{section_requirement}

=== VÍ DỤ GOLD THẬT #1 TỪ BAN TỔ CHỨC (dạng liệt kê số thứ tự) ===
input_text: "{GOLD_FEWSHOT_INPUT}"
output: {gold_fewshot_output_str}

=== VÍ DỤ GOLD THẬT #2 TỪ BAN TỔ CHỨC (dạng văn xuôi tự nhiên, có xét nghiệm dạng
chuỗi số liệu công thức máu) ===
input_text: "{GOLD_FEWSHOT_2_INPUT}"
output: {gold_fewshot_2_output_str}

Rule rút ra từ 2 ví dụ trên (BẮT BUỘC áp dụng khi bạn tự sinh câu mới):
(a) Thuốc trong danh sách "trước khi nhập viện" luôn có assertions: ["isHistorical"].
    NHƯNG nếu câu có dạng "<thuốc> điều trị <triệu chứng/chẩn đoán>", concept đó là CHỈ ĐỊNH
    đi kèm (lý do kê thuốc), KHÔNG tự động kế thừa assertions của thuốc -> mặc định assertions: [].
    QUAN TRỌNG: mỗi entity có assertions RIÊNG dựa theo NGỮ CẢNH CỦA CHÍNH NÓ, không phải
    "ăn theo" entity đứng cạnh. Nếu chính concept đó có dấu hiệu riêng như "tiền sử <X>",
    "tiền căn <X>", "trước đây <X>", "đã từng <X>", "từng được chẩn đoán <X>" thì PHẢI gán
    assertions: ["isHistorical"] cho chính nó, BẤT KỂ nó có đứng cạnh thuốc hay không.
    Vd "Bản thân có tiền sử thoái hóa cột sống cổ" -> "thoái hóa cột sống cổ" PHẢI có
    assertions: ["isHistorical"] vì có chữ "tiền sử" ngay trước.
(b) [SỬA] Span entity KHÔNG kéo theo NGỮ CẢNH THỪA nằm NGOÀI bản thân khái niệm: chủ ngữ
    ("mẹ", "bố", "gia đình"), số thứ tự liệt kê ("1.", "2."), mốc thời gian ("năm 60 tuổi")
    -- những thứ này chỉ ảnh hưởng đến việc chọn assertions, không đưa vào "text".
    NHƯNG khác với việc đó: tính từ/cụm mô tả TÍNH CHẤT của chính triệu chứng (màu sắc, mức
    độ, đặc điểm) LÀ MỘT PHẦN BẢN CHẤT của khái niệm, PHẢI giữ trong span khi nguồn có mô tả
    -- xem ví dụ gold #2: "ho đờm xanh" là 1 entity TRỌN VẸN (không tách "ho" + "đờm xanh"
    riêng), vì "đờm xanh" mô tả TÍNH CHẤT của chính cơn ho, không phải ngữ cảnh thừa như chủ
    ngữ/tuổi. Tương tự nếu nguồn viết "đau bụng âm ỉ", "sốt cao 39 độ", "chảy mủ vàng đục" --
    giữ nguyên cả cụm, KHÔNG rút gọn về danh từ trần trụi. Ví dụ "ngứa nhiều vùng da mặt và
    cổ" phải lấy trọn cụm, không chỉ lấy "ngứa".
    Với CHẨN_ĐOÁN, các marker "tiền sử", "tiền căn" chỉ quyết định assertion và KHÔNG nằm
    trong span: "không ghi nhận tiền sử gãy xương" -> text="gãy xương",
    assertions=["isNegated"]. Qualifier nói bệnh không biểu hiện triệu chứng cũng không phải
    tên bệnh: "loãng xương không triệu chứng" -> chỉ lấy text="loãng xương".
(c) Với PHỦ ĐỊNH ở TRIỆU_CHỨNG/CHẨN_ĐOÁN: chữ phủ định ("không", "chưa", "không có",
    "không còn") KHÔNG được nằm trong text span -- nó chỉ quyết định gán assertions:
    ["isNegated"]. Vd câu "không đau bụng" -> entity PHẢI là {{"text": "đau bụng", "type":
    "TRIỆU_CHỨNG", "assertions": ["isNegated"]}}, TUYỆT ĐỐI KHÔNG lấy "không đau bụng" làm text.
    NGOẠI LỆ QUAN TRỌNG: rule này KHÔNG áp dụng cho KẾT_QUẢ_XÉT_NGHIỆM. Với kết quả xét
    nghiệm/chẩn đoán hình ảnh, chữ phủ định là MỘT PHẦN Ý NGHĨA LÂM SÀNG của chính kết quả đó
    (kết quả âm tính khác hoàn toàn kết quả dương tính) nên PHẢI giữ nguyên cả cụm trong text,
    KHÔNG tách ra. Vd câu "không có hạch cổ bất thường" -> entity PHẢI là {{"text": "không có
    hạch cổ bất thường", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []}} -- giữ nguyên "không
    có" trong span, vì đây là toàn bộ nội dung kết luận, không phải phủ định 1 khái niệm riêng.
(d) assertions: ["isFamily"] CHỈ dùng khi chủ thể là NGƯỜI THÂN RUỘT THỊT của bệnh nhân
    (bố/mẹ/anh/chị/em/ông/bà/con). Nếu câu nói về "tiếp xúc với người bệnh X", "đồng nghiệp
    bị X", hay bất kỳ người không phải người thân nào khác, thì KHÔNG được gán isFamily,
    và trong trường hợp này cũng ĐỪNG trích concept đó thành entity luôn (vì không thuộc về
    bệnh nhân lẫn không thuộc về người thân, nằm ngoài phạm vi 3 loại assertion của đề bài).
    Một từ như "gia đình" nằm BÊN TRONG TÊN BỆNH không phải ngữ cảnh người thân. Ví dụ
    "liệt chu kỳ gia đình" là bệnh của chính bệnh nhân thì chỉ có isHistorical nếu thuộc tiền
    sử, TUYỆT ĐỐI không thêm isFamily khi không có bố/mẹ/người thân nào là chủ thể.
(e) VĂN PHONG CHẨN_ĐOÁN phải TỰ NHIÊN như bác sĩ thật ghi bệnh án, KHÔNG được dịch nguyên
    văn kiểu tên chương/mục ICD-10 (nghe như tiêu đề sách giáo khoa). Vd SAI: "Rối loạn
    chuyển hóa và dinh dưỡng trong bệnh khác". Vd ĐÚNG (bác sĩ hay viết): "rối loạn chuyển
    hóa", "rối loạn lipid máu", "suy dinh dưỡng", "rối loạn điện giải". Khi lấy cảm hứng từ
    danh sách chẩn đoán gợi ý bên dưới, hãy RÚT GỌN thành cụm ngắn bác sĩ thật hay dùng, đừng
    giữ nguyên cấu trúc câu dài kiểu phân loại ICD.
(f) Entity CÓ THỂ mang NHIỀU assertion cùng lúc nếu ngữ cảnh thật sự hợp cả 2
    nghĩa -- KHÔNG bắt buộc chỉ 1. Vd "mẹ có tiền sử ung thư vú" -> ["isFamily",
    "isHistorical"] (đúng cả 2: về người thân VÀ đã xảy ra trong quá khứ). "Không có tiền
    sử gia đình mắc tiểu đường" -> ["isFamily", "isNegated"] (đúng cả 2: về người thân VÀ
    bị phủ định). CHỈ CẤM DUY NHẤT 1 combo vì nó thật sự mâu thuẫn logic: KHÔNG BAO GIỜ
    gán cả "isHistorical" và "isNegated" cùng lúc cho 1 entity (vừa "đã xảy ra trong quá
    khứ" vừa "chưa từng xảy ra" -- vô nghĩa). Nếu ngữ cảnh có vẻ hợp cả 2, chỉ giữ
    "isNegated" (phủ định là sự thật quyết định hơn, mốc thời gian không còn ý nghĩa khi
    sự việc chưa từng xảy ra). Vd "không có tiền sử dị ứng thuốc" -> chỉ ["isNegated"],
    KHÔNG thêm isHistorical dù có chữ "tiền sử". "Dị ứng thuốc" trong trường hợp này là
    CHẨN_ĐOÁN, không phải TRIỆU_CHỨNG.
(g) KHÔNG trích các cụm MỤC ĐÍCH điều trị dạng động từ ("giảm đau", "hạ sốt", "an
    thần", "chống viêm", "lợi tiểu") làm TRIỆU_CHỨNG. Chỉ trích triệu chứng cụ thể dạng
    danh từ mà thuốc điều trị (vd "đau nhức", "táo bón", "lo âu", "mất ngủ" -- đúng như
    trong ví dụ gold ở trên).
(h) Thủ thuật/phẫu thuật (phẫu thuật X, mổ X, nội soi X, đặt stent X, sinh thiết X,
    chạy thận nhân tạo, ghép thận, nạo vét tổn thương, phaco, tán sỏi ngoài cơ thể)
    KHÔNG thuộc CHẨN_ĐOÁN và không thuộc phạm vi 5 type của đề (đề không có type THỦ_THUẬT)
    -> KHÔNG trích xuất các cụm này thành entity. Tuy nhiên tình trạng/biến chứng như "ghép
    thận thất bại" vẫn là CHẨN_ĐOÁN và phải được giữ.
(i) Nếu 1 CHẨN_ĐOÁN là bệnh nền đang được 1 THUỐC có assertions: ["isHistorical"]
    điều trị/kiểm soát trong section Tiền sử bệnh (vd "đang dùng Metformin ... điều trị đái
    tháo đường type 2"), thì CHÍNH CHẨN_ĐOÁN đó CŨNG phải gán assertions: ["isHistorical"] --
    khác với rule (a) chỉ áp dụng miễn trừ cho TRIỆU_CHỨNG là chỉ định điều trị đi kèm.
(j) TUYỆT ĐỐI KHÔNG viết câu tỉnh lược kiểu "dị ứng thuốc hay mỹ phẩm" (2 khái niệm
    dùng chung 1 từ đứng trước, chỉ viết đủ ở lần đầu, lần 2 bị lược mất). Lý do: nếu 2 khái
    niệm này được trích thành 2 entity riêng ("dị ứng thuốc" và "dị ứng mỹ phẩm"), entity thứ
    2 sẽ KHÔNG PHẢI substring thật của input_text ("dị ứng mỹ phẩm" không tồn tại liên tục
    trong câu) -> bị loại bỏ hoàn toàn khi xử lý, mất dữ liệu. Nếu định trích ra 2 entity
    riêng biệt, PHẢI viết đủ cả 2 lần: "dị ứng thuốc hay dị ứng mỹ phẩm". Quy tắc chung: bất
    kỳ 2 entity nào định trích riêng biệt thì MỖI entity phải là 1 cụm từ TRỌN VẸN, ĐỘC LẬP,
    không dùng cấu trúc tỉnh lược chia sẻ từ giữa 2 khái niệm.
=== HẾT VÍ DỤ GOLD ===

Yêu cầu bắt buộc khi sinh câu mới:
- Viết bằng tiếng Việt tự nhiên, đúng văn phong y khoa thật (tham khảo văn phong ở ví dụ thật bên dưới, KHÔNG copy nguyên văn).
- Tự kiểm tra chính tả và khoảng trắng trước khi trả JSON: không dính từ kiểu "atenololtrong",
  không để chữ thừa cuối finding kiểu "trái L", không viết nhầm "vào việc" thay cho "vào viện",
  và không lặp sát nhau kiểu "Khó thở nhẹ khó thở".
{heading_requirement}
- Câu văn PHẢI chứa NHIỀU HƠN 1 khái niệm y tế.
- Tên thuốc/liều lượng giữ nguyên tiếng Anh/Latin, có thể dùng gợi ý: {drug_hint}
- Chẩn đoán nên lấy cảm hứng từ các bệnh lý thật sau: {diagnosis_hint}. LƯU Ý: mỗi lần
  CHỈ CHỌN 1 bệnh lý duy nhất trong danh sách trên để viết vào đoạn văn, viết bằng thuật
  ngữ y khoa tiếng Việt CHUẨN và TỰ NHIÊN (không dịch từng chữ theo nghĩa đen). TUYỆT ĐỐI
  KHÔNG ghép 2 bệnh lý khác nhau thành 1 cụm từ (vd "viêm phế quản giãn phế quản" là SAI --
  đây là 2 bệnh riêng biệt bị dính vào nhau -- chỉ được chọn 1: hoặc "viêm phế quản mạn"
  hoặc "giãn phế quản").
- {force_txt}
- Đảm bảo mỗi entity trích ra ở output_text là MỘT SUBSTRING Y HỆT (nguyên văn, không sửa dấu câu/khoảng trắng) so với input_text.
- Danh sách entities PHẢI được sắp xếp đúng theo thứ tự xuất hiện từ trái sang phải trong input_text.
  Không gom theo type và không đưa chẩn đoán chính lên đầu nếu nó xuất hiện sau các entity khác.
- Exposure/lifestyle/context như "tiếp xúc với hóa chất", stress, thức khuya, uống cà phê, đi bộ,
  bơi hồ công cộng, du lịch hoặc nghề nghiệp KHÔNG thuộc 5 type -> tuyệt đối không annotate.
- Tuổi, giới tính và nghề nghiệp là thông tin nhân khẩu học, không thuộc 5 type. Ví dụ trong "BN nam
  65t", tuyệt đối không gán "65" thành KẾT_QUẢ_XÉT_NGHIỆM và không annotate "nam"/nghề nghiệp.
- Số chỉ thời lượng không phải lab: trong "tăng huyết áp 5 năm", số "5" không được annotate
  KẾT_QUẢ_XÉT_NGHIỆM. Quy tắc tương tự cho thời lượng ngày/tuần/tháng/năm.
- Tác nhân như "mỹ phẩm", "hóa chất", "thức ăn" không phải CHẨN_ĐOÁN. Trong câu "không có tiền
  sử dị ứng thuốc hay mỹ phẩm", chỉ annotate "dị ứng thuốc"; không tự dựng "dị ứng mỹ phẩm" vì
  cụm đó không tồn tại nguyên văn, và không annotate riêng "mỹ phẩm".
- Phương pháp điều trị như "liệu pháp ánh sáng", vật lý trị liệu, phẫu thuật không phải THUỐC.
  Chỉ annotate dược chất/thuốc thật; tránh hóa chất lạ không dùng lâm sàng như manganese dioxide.
- Concept chỉ xuất hiện như mục tiêu dự phòng/giảm không khẳng định bệnh đang tồn tại. Trong
  "thuốc để phòng ngừa huyết khối" không annotate "huyết khối"; trong "thuốc dùng giảm đau"
  không annotate riêng "đau". Không áp dụng rule này nếu văn bản có một mention khác khẳng định
  bệnh/triệu chứng thật của bệnh nhân; mention đó vẫn annotate tại đúng vị trí của nó.
- Nếu hai thuốc được viết ghép bằng dấu "/", phải tách thành hai THUỐC độc lập để linking RxNorm.
  Ví dụ "domperidone 10 MG / ranitidine 150 MG Oral Tablet" -> "domperidone 10 MG" và
  "ranitidine 150 MG Oral Tablet"; không gộp cả chuỗi thành một entity. Dấu "/" trong MG/ML hoặc
  x2/ngày vẫn thuộc cùng một thuốc và không được tách.
- Boundary THUỐC phải bám ví dụ gold BTC: giữ tên thuốc + hàm lượng + đường dùng + tần suất/
  thời gian dùng liền kề. Ví dụ phải lấy trọn "Lactulose 15 ml uống hàng ngày",
  "Ciprofloxacin 500 mg x 2 lần/ngày trong 7 ngày" hoặc "Hydrocortisone 1% bôi tại chỗ";
  không cắt còn tên + liều. Dừng span trước mục đích/chỉ định như "để kiểm soát...",
  "điều trị...", "giúp giảm..." và không nuốt sang thuốc tiếp theo sau "và"/dấu phẩy.
- Không viết chẩn đoán tỉnh lược kiểu "viêm gan B hoặc C". Phải viết đầy đủ "viêm gan B hoặc
  viêm gan C" và trích hai entity riêng, vì mỗi entity phải là substring thật để tính position/link ICD.
- Ngữ cảnh thuốc phải rõ: "dùng tại nhà trước nhập viện"/"thuốc trước nhập viện" -> isHistorical;
  "điều trị hiện tại"/"được chỉ định tại viện"/"đang uống hiện tại" -> assertions: []. Tránh câu
  mơ hồ chỉ viết "đang uống" mà không nói rõ trước nhập viện hay hiện tại.
- Các thuốc nằm trong cùng một vế, ví dụ "đang dùng prednisone + methotrexate", phải nhận cùng
  assertion thời gian; không được để một thuốc [] và thuốc còn lại isHistorical.
- Thuốc dùng "trong lần nhập viện trước", "trước đó", "tại thời điểm xuất viện trước", "đã
  dùng tại nhà" hoặc "thuốc trước nhập viện" phải có isHistorical. Không suy isHistorical chỉ
  từ cụm mơ hồ "đã sử dụng" nếu không có mốc trước nhập viện/quá khứ rõ ràng.
- Thuốc bệnh nhân "tự điều trị bằng" trước khi đến viện cũng là thuốc trước nhập viện và có
  isHistorical. Thuốc xử trí tại ED/cấp cứu trong chính đợt hiện tại có assertions: [].
- Marker lịch sử có thể đứng SAU tên thuốc: "sử dụng ciproflagyl trong lần nhập viện trước"
  vẫn bắt buộc gán ciproflagyl isHistorical.
- Cấu trúc "triệu chứng do chẩn đoán" phải tách hai entity. Ví dụ "giọng khàn do tổn thương
  dây thanh quản": "giọng khàn" là TRIỆU_CHỨNG, "tổn thương dây thanh quản" là CHẨN_ĐOÁN;
  không gom cả cụm thành một CHẨN_ĐOÁN.
- Trong section Tiền sử bệnh/Tiền căn bệnh lý/Các bệnh lý mạn tính, mọi bệnh nền khẳng định
  của bệnh nhân phải có isHistorical; câu phủ định vẫn ưu tiên isNegated.
- Mốc quá khứ rõ như "1 tháng trước nhập viện, được chẩn đoán X" làm X có isHistorical dù
  câu nằm trong section Bệnh sử hiện tại. Ngược lại "đến ED vì X" mô tả đợt hiện tại nên X
  có assertions: [], không tự gán isHistorical vì heading "sự kiện trước khi nhập viện".
- Mỗi occurrence thật của một concept phải có entity riêng. Dòng triệu chứng không có dấu
  bullet như "ho", "mệt mỏi", "phù" vẫn phải annotate; không chỉ bắt occurrence đầu tiên.
- Quy tắc occurrence áp dụng cả CHẨN_ĐOÁN: nếu "suy giáp" hoặc "suy tim" xuất hiện hai lần
  thật trong input_text thì output phải có đúng hai entity tương ứng theo thứ tự văn bản.
  Các mention cùng nói về bệnh nền phải nhất quán isHistorical; các mention cùng nói về đợt
  hiện tại phải nhất quán assertions: [].
- Không annotate các đơn vị trần như pound/lb/kg/inch thành THUỐC. Không annotate các mảnh
  vô nghĩa như "phân", "nhầy", "mất", "đi lại", "thành", "bên", "độ" đứng riêng.
- Nếu triệu chứng sốt có nhiệt độ thì lấy trọn số và đơn vị nếu có trong span, ví dụ
  TRIỆU_CHỨNG="sốt nhẹ 37.8", "sốt cao 39°C" hoặc "sốt lên đến 101°F". Nhiệt độ 37-43.5
  không ghi đơn vị được mặc định là °C; không sinh "sốt 90"/"sốt 101" thiếu đơn vị.
- {structure_instruction}
{section_semantics_requirement}

RULE CỨNG VỀ VITALS / XÉT NGHIỆM (bắt buộc tuân thủ, đây là lỗi hay gặp nhất):
- TÊN_XÉT_NGHIỆM PHẢI dùng thuật ngữ tiếng Việt CHUẨN, TỰ NHIÊN, ví dụ: "Glucose máu" /
  "Đường huyết" / "Glucose mao mạch" (không phải "DL glucose"), "Huyết áp", "Nhịp tim",
  "SpO2", "BMI", "HbA1c", "Men gan AST". Gợi ý định dạng số bên dưới ("{vitals_hint}")
  CHỈ để tham khảo CÁCH CÁC CON SỐ ĐƯỢC VIẾT DÍNH NHAU, TUYỆT ĐỐI KHÔNG copy nguyên các
  chữ viết tắt tiếng Anh kỳ lạ trong đó (như "DL", "T", "RA") làm tên xét nghiệm tiếng Việt.
- TUYỆT ĐỐI KHÔNG được gán type TRIỆU_CHỨNG cho bất kỳ chuỗi vitals/số liệu xét nghiệm nào.
  TRIỆU_CHỨNG chỉ dùng cho mô tả cảm giác/triệu chứng chủ quan của bệnh nhân (vd "đau ngực",
  "khó thở"), không dùng cho số đo khách quan.
- Trong khám thực thể, tên cơ quan trần như "Phổi", "Tim", "Ngực", "Bụng" không phải
  TÊN_XÉT_NGHIỆM. Lượng dịch lấy ra trong thủ thuật như "Hút 0.5cc dịch mủ" cũng không phải
  KẾT_QUẢ_XÉT_NGHIỆM và không được annotate.
- Xét nghiệm động học chỉ có một tên và một kết quả trọn vẹn. Ví dụ "creatinine tăng từ 5.2
  lên 6.3 mg/dl (460 - 557 umol/l)": TÊN_XÉT_NGHIỆM="creatinine" và một
  KẾT_QUẢ_XÉT_NGHIỆM="tăng từ 5.2 lên 6.3 mg/dl (460 - 557 umol/l)"; không tách từng số.
- Một dòng có nhiều chỉ số phải tách thành từng cặp tên/kết quả, tuyệt đối không gộp cả dòng
  làm một KẾT_QUẢ_XÉT_NGHIỆM. Ví dụ "WBC:12,5; NEUT%:78,2; LYPH%:15,3" phải tạo
  sáu entity theo thứ tự: WBC, 12,5, NEUT%, 78,2, LYPH%, 15,3.
- Không được lặp entity tên xét nghiệm nếu tên chỉ xuất hiện một lần trong input_text. Với dạng
  "kali lần đầu 3.8 mmol/L, sau giảm còn 3.5 mmol/L", annotate một tên "kali" và hai kết quả;
  với dạng có hai mention thật "Kali ...; kali ...", annotate cả hai mention và giữ đúng hoa/thường.
- Với kết quả đơn có qualifier, chỉ lấy giá trị và đơn vị theo boundary gold. Ví dụ
  "CRP tăng cao 15.2 mg/L" -> TÊN_XÉT_NGHIỆM="CRP" và
  KẾT_QUẢ_XÉT_NGHIỆM="15.2 mg/L"; để "tăng cao" ở ngoài entity.
- Mỗi vital phải đủ cặp tên/kết quả: "Nhiệt độ: 36.5 độ C", "Mạch: 88 l/p", "Huyết áp:
  120/70 mmHg", "Nhịp thở: 20 l/p", "SpO2: 92 %". Với "INR dưới ngưỡng điều trị 1.7",
  TÊN_XÉT_NGHIỆM="INR" và KẾT_QUẢ_XÉT_NGHIỆM="dưới ngưỡng điều trị 1.7".
- Dạng không có dấu hai chấm cũng phải đủ cặp: "độ bão hòa oxy 100% trên khí trời" có
  TÊN_XÉT_NGHIỆM="độ bão hòa oxy" và KẾT_QUẢ_XÉT_NGHIỆM chứa giá trị SpO2; không tách
  riêng "độ" làm tên. "Không rõ" trong Vị trí/Mức độ/Thời gian triệu chứng không phải lab.
- "điện tâm đồ bình thường" và "xét nghiệm gắng sức bất thường" phải tách tên kỹ thuật với
  kết quả định tính. Finding liên tục sau nghiệm pháp gắng sức (dương tính, thiếu máu cơ tim,
  ST chênh xuống...) là KẾT_QUẢ_XÉT_NGHIỆM, không phải CHẨN_ĐOÁN/fragment rời.
- Với "Điện tâm đồ không có dấu hiệu thiếu máu cơ tim", TÊN_XÉT_NGHIỆM="Điện tâm đồ" và
  toàn bộ "không có dấu hiệu thiếu máu cơ tim" là KẾT_QUẢ_XÉT_NGHIỆM; không annotate riêng
  "thiếu máu cơ tim" thành CHẨN_ĐOÁN.
- "Soi đáy mắt" là TÊN_XÉT_NGHIỆM. Trong "Soi đáy mắt phát hiện xuất huyết võng mạc",
  "xuất huyết võng mạc" là KẾT_QUẢ_XÉT_NGHIỆM, không phải CHẨN_ĐOÁN độc lập.
- Với chẩn đoán hình ảnh, tên kỹ thuật và kết luận phải tách đúng vai trò. Ví dụ "Siêu âm tim
  ghi nhận hở van động mạch chủ độ 2" -> TÊN_XÉT_NGHIỆM="Siêu âm tim" và
  KẾT_QUẢ_XÉT_NGHIỆM="hở van động mạch chủ độ 2"; tuyệt đối không gán "hở van động mạch chủ
  độ" làm tên xét nghiệm rồi tách riêng "2" làm kết quả.
- Quy tắc trên áp dụng cho MỌI kỹ thuật hình ảnh. Ví dụ "MRI khớp gối trái: thoái hóa khớp
  gối độ 2, tràn dịch khớp nhẹ" -> TÊN_XÉT_NGHIỆM="MRI khớp gối trái" và một
  KẾT_QUẢ_XÉT_NGHIỆM trọn vẹn="thoái hóa khớp gối độ 2, tràn dịch khớp nhẹ".
- Finding định lượng cũng phải gộp trọn với giá trị. Ví dụ "Siêu âm tim phát hiện rối loạn
  vận động vùng trước vách với phân suất tống máu EF 45%" -> TÊN_XÉT_NGHIỆM="Siêu âm tim"
  và KẾT_QUẢ_XÉT_NGHIỆM="rối loạn vận động vùng trước vách với phân suất tống máu EF 45%";
  không tách phần trước EF thành một TÊN_XÉT_NGHIỆM khác.
- Finding có cơ quan/vị trí đứng trước cũng áp dụng y hệt: "Siêu âm thận: thận phải ứ nước
  độ 2" -> TÊN_XÉT_NGHIỆM="Siêu âm thận" và KẾT_QUẢ_XÉT_NGHIỆM="thận phải ứ nước độ 2".
- Kết luận dài của siêu âm/CT/MRI/X-quang hoặc monitor Holter phải giữ thành finding trọn
  nghĩa, không tách các từ vụn như "âm", "lên", "thu", "nhĩ", kích thước hay từng tổn thương.
  "Nhịp xoang chiếm ưu thế, ghi nhận ngoại tâm thu nhĩ và ngoại tâm thu thất thường xuyên"
  sau Holter là KẾT_QUẢ_XÉT_NGHIỆM, không phải các CHẨN_ĐOÁN rời.
{vitals_split_instruction}

RULE CỨNG VỀ PHẠM VI 5 TYPE (không được lấn sang nội dung ngoài phạm vi):
- CHỈ trích xuất entity nếu nó CHẮC CHẮN thuộc 1 trong 5 type: THUỐC, CHẨN_ĐOÁN, TRIỆU_CHỨNG,
  TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM.
- Yếu tố nguy cơ / lối sống / hành vi (hút thuốc lá, uống rượu, uống cà phê, căng thẳng công
  việc, mất việc làm, nghề nghiệp...) KHÔNG PHẢI là TRIỆU_CHỨNG hay CHẨN_ĐOÁN -> KHÔNG trích
  xuất các cụm này thành entity, kể cả khi câu văn liệt kê chúng trong mục "yếu tố nguy cơ".

RULE VỀ ĐỘ ĐẦY ĐỦ (recall) -- lỗi hay gặp thứ 3:
- Khi 1 câu liệt kê NHIỀU khái niệm nối bằng "và"/"hoặc"/dấu phẩy (đặc biệt trong câu phủ định
  kiểu "không có X hoặc Y hoặc Z"), BẮT BUỘC trích xuất TẤT CẢ từng khái niệm riêng lẻ, không
  được chỉ lấy 1 phần tử rồi bỏ sót các phần tử còn lại trong cùng danh sách. Tất cả phần tử
  của danh sách phủ định này cũng PHẢI nhận assertions=["isNegated"].
- [MỚI] Nếu CÙNG 1 khái niệm xuất hiện NHIỀU LẦN trong input_text (vd nhắc lại ở phần tiền sử
  rồi nhắc lại ở phần hiện tại/điều trị), PHẢI trích xuất RIÊNG cho MỖI LẦN xuất hiện (2 entity
  cùng text, có thể khác assertions nếu ngữ cảnh khác nhau), TUYỆT ĐỐI KHÔNG chỉ trích 1 lần
  rồi bỏ qua các lần lặp lại còn lại.
- Chỉ lặp có kiểm soát: tối đa khoảng 2-3 occurrence cho một khái niệm trong một record.
  Không tạo mâu thuẫn assertion giữa phần văn xuôi và bullet nếu không có mốc chuyển biến rõ.
- Entity phải là cụm có nghĩa lâm sàng độc lập. Cấm tạo entity vụn một từ như "hơi", "âm",
  "lên", "xuống", "thu", "nhĩ", "thất", "giây" hoặc chỉ một phía của cụm phối hợp.
- Với danh sách lab, mỗi dòng phải annotate đủ TÊN_XÉT_NGHIỆM và KẾT_QUẢ_XÉT_NGHIỆM,
  kể cả kết quả chữ như "bình thường", "âm tính", "đang chờ".
- Cho phép trật tự đảo trong bệnh án: "3.2 kali" phải annotate "3.2" là
  KẾT_QUẢ_XÉT_NGHIỆM rồi "kali" là TÊN_XÉT_NGHIỆM; tương tự "80% neutrophil",
  "11% lymphocyte", "478 tiểu cầu", "1.3 lactate", "0.01 troponin", "4227 BNP".
- Dạng "3.5 mmol/L (kali)" cũng theo thứ tự đảo: "3.5 mmol/L" là KẾT_QUẢ và "kali"
  là TÊN; tương tự K+, Na+, Cl-, phospho và Mg++. Dấu ngoặc không nằm trong span entity.

RULE VỀ CẶP BỆNH-THUỐC HỢP LÝ:
- Khi viết thuốc đi kèm 1 chẩn đoán/chỉ định, CHỌN cặp bệnh-thuốc có liên hệ lâm sàng THẬT
  (vd thuốc hạ áp đi với tăng huyết áp, kháng sinh đi với nhiễm trùng, statin đi với rối loạn
  lipid máu). TRÁNH ghép ngẫu nhiên 1 thuốc không liên quan gì đến bệnh đang mô tả (vd
  oxycodone -- giảm đau sau phẫu thuật -- không nên gán làm thuốc điều trị viêm tai giữa).
- Cấm ghép Allopurinol với rối loạn lipid máu. Allopurinol chỉ nên đi với tăng acid uric
  máu/gout; nếu chỉ định là rối loạn lipid máu thì dùng statin phù hợp như Atorvastatin.
- Không đồng nhất "liệt chu kỳ gia đình" với Riley-Day: Riley-Day là rối loạn thần kinh tự
  chủ gia đình, không phải liệt chu kỳ. Không dùng phenacemide để điều trị liệt chu kỳ.
- Dùng đúng chính tả thuật ngữ y khoa: viết "sỏi thận", tuyệt đối không viết nhầm "sót thận".

Ví dụ văn phong thật (không copy, chỉ tham khảo cấu trúc):
---
{seed_snippet[:1200]}
---

Trả về DUY NHẤT 1 object JSON, đúng format:
{{
  "input_text": "{output_text_description}",
  "entities": [
    {{"text": "<substring nguyên văn của input_text, TỐI GIẢN theo rule (b)>", "type": "<1 trong 5 loại: THUỐC|CHẨN_ĐOÁN|TRIỆU_CHỨNG|TÊN_XÉT_NGHIỆM|KẾT_QUẢ_XÉT_NGHIỆM>", "assertions": [] hoặc list gồm 1-2 phần tử trong ["isNegated", "isFamily", "isHistorical"] -- xem rule (f), CHỈ CẤM riêng combo ["isHistorical", "isNegated"] cùng lúc}}
  ]
}}
Không thêm text nào khác ngoài JSON."""

    return [
        {"role": "system", "content": "Bạn là trợ lý sinh dữ liệu huấn luyện NER y tế tiếng Việt chất lượng cao, tuân thủ nghiêm ngặt schema JSON và các rule được nêu, đặc biệt là rule về span tối giản và rule phân loại vitals/xét nghiệm."},
        {"role": "user", "content": user_prompt},
    ]


# ----------------------------------------------------------------------------
# 5. Validation -- span + các rule phát hiện từ batch review thủ công
# ----------------------------------------------------------------------------
VITALS_LIKE_RE = re.compile(
    r"^\s*(VS)?\s*\d{2,3}(\.\d)?\s+\d{2,3}\D{0,3}\d{2,3}(\s+\d{1,3}){1,3}"
)
LEAKED_CONTEXT_RE = re.compile(
    r"^(bố|mẹ|ba|cha|anh|chị|em|ông|bà|gia đình|người thân)\b|"
    r"\b(năm\s+\d{1,3}\s+tuổi|\d+\s+tuổi)\s*$|"
    r"^\d+\.\s"
)
RISK_FACTOR_RE = re.compile(
    r"hút thuốc|uống rượu|uống bia|cà phê|caffeine|căng thẳng|stress|"
    r"mất việc|nghề nghiệp|áp lực công việc",
    re.IGNORECASE,
)

NEGATION_PREFIX_RE = re.compile(
    r"^(không có|không còn|chưa từng|không hề|không|chưa)\s+", re.IGNORECASE
)
FAMILY_KEYWORDS = ("bố", "mẹ", "ba", "cha", "anh", "chị", "em", "ông", "bà", "cô", "chú",
                    "bác", "gia đình", "người thân", "con")
EXPOSURE_KEYWORDS = ("tiếp xúc", "người bệnh", "đồng nghiệp", "bạn cùng phòng", "hàng xóm", "bạn bè")
HISTORICAL_MARKER_RE = re.compile(
    r"(tiền sử|tiền căn|trước đây|đã từng|từng được chẩn đoán|từng bị)\s*[:\-]?\s*$",
    re.IGNORECASE,
)

# Bắt "không có tiền sử X" / "không còn tiền sử X" khi cụm phủ định KHÔNG dính liền
# trong text span (LLM đã tự cắt span sạch theo rule (c)) -- case này fix_negation_leak()
# không bắt được vì nó chỉ nhìn PREFIX của chính text span, không nhìn ngữ cảnh câu.
NEGATED_HISTORY_CONTEXT_RE = re.compile(
    r"(không có|không còn|chưa từng|không hề)\s+tiền sử\s*[:\-]?\s*$", re.IGNORECASE
)

# Cụm "giảm đau / hạ sốt / an thần / chống viêm / lợi tiểu / kiểm soát ..." là MỤC
# ĐÍCH điều trị (cụm động từ), không phải bản thân triệu chứng -- khác với "đau nhức",
# "táo bón", "lo âu" (danh từ chỉ triệu chứng cụ thể) trong gold example.
TREATMENT_PURPOSE_RE = re.compile(
    r"^(giảm|hạ|chống|lợi|an|kiểm soát|điều hòa|ổn định|ngừa|phòng)\s", re.IGNORECASE
)

# Thủ thuật/phẫu thuật không thuộc 5 type của đề (không có THỦ_THUẬT) -- nếu LLM
# lỡ gán CHẨN_ĐOÁN cho 1 hành động phẫu thuật/thủ thuật thì loại bỏ.
PROCEDURE_RE = re.compile(
    r"^(phẫu thuật|mổ|sinh mổ|mổ lấy thai|nội soi|đặt stent|đặt catheter|sinh thiết|chọc dò|thay khớp)\b",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------------
# 6. Main generation loop, có balancing theo assertion type & entity type
# ----------------------------------------------------------------------------
# Profile v2 cũ vẫn được giữ nguyên để tái lập batch trước.
V2_FOCUS_PROBABILITY = 0.5

# Chỉ cần sửa MỘT số này khi muốn đổi tỷ lệ: 50 nghĩa là khoảng 50% mẫu nhận
# focus v3, 50% còn lại giữ baseline. Profile mixed_v3 KHÔNG trộn focus v2.
# Đây là xác suất mềm, không phải quota, nên batch nhỏ có thể lệch tỷ lệ kỳ vọng.
V3_FOCUS_PERCENT = 50

# Tỷ lệ chọn trên các lượt mixed_v3 đủ điều kiện dành riêng cho hồ sơ rất dài. Giá trị này
# phải không lớn hơn V3_FOCUS_PERCENT. Vì timeline chỉ hợp section hiện tại/đánh giá, tỷ lệ
# thực tế trên toàn batch còn phụ thuộc vòng quay section; các lượt còn lại vẫn dùng focus V3
# thường hoặc baseline, không bị ép thành văn bản dài.
V3_VERY_LONG_FOCUS_PERCENT = 8

# V4 dành 20% cho baseline đa dạng cũ và 80% cho QA/lý thuyết y khoa tự sinh.
# Không sao chép public input và không kéo profile V2/V3 vào profile này.
V4_LONG_FREEFORM_FOCUS_PERCENT = 80

# Tỷ lệ QA TRONG 80% record V4; phần còn lại là bài lý thuyết/giải thích y khoa.
V4_QA_WITHIN_FOCUS_PERCENT = 60

# 25% của phần V4 (80% batch) tương đương khoảng 20% toàn batch chủ động học lỗi
# thiếu khoảng trắng. Gold vẫn phải tách đúng ranh giới, không kéo entity qua hai câu.
V4_BOUNDARY_NOISE_WITHIN_FOCUS_PERCENT = 25


def with_optional_v4_boundary_noise(focus):
    selected = dict(focus)
    if not 0 <= V4_BOUNDARY_NOISE_WITHIN_FOCUS_PERCENT <= 100:
        raise ValueError("V4_BOUNDARY_NOISE_WITHIN_FOCUS_PERCENT phải nằm trong khoảng 0..100")
    selected["boundary_noise"] = (
        random.random() < V4_BOUNDARY_NOISE_WITHIN_FOCUS_PERCENT / 100
    )
    return selected


def _scaled_focus_counts(focus_areas, n_samples):
    """Scale quota_weight bằng largest remainder; tổng luôn đúng n_samples."""
    total_weight = sum(item["quota_weight"] for item in focus_areas)
    raw = [n_samples * item["quota_weight"] / total_weight for item in focus_areas]
    counts = [int(value) for value in raw]
    remainder = n_samples - sum(counts)
    order = sorted(
        range(len(focus_areas)),
        key=lambda idx: (raw[idx] - counts[idx], focus_areas[idx]["quota_weight"]),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def build_v5_focus_schedule(n_samples):
    """Lịch V5 theo tỷ lệ 180/150/100/100/70, cộng modifier dirty và QA.

    Đây là lịch focus đầu vào, không phải reject quota cực đoan: nếu một focus thất bại hết
    retry, vòng sinh vẫn có thể nới về baseline để không đốt token vô hạn.
    """
    if n_samples <= 0:
        return []
    counts = _scaled_focus_counts(V5_FOCUS_AREAS, n_samples)
    schedule = []
    for focus, count in zip(V5_FOCUS_AREAS, counts):
        schedule.extend(dict(focus) for _ in range(count))
    random.shuffle(schedule)

    target_dirty = round(n_samples * V5_DIRTY_RECORD_PERCENT / 100)
    already_dirty = sum(bool(item.get("boundary_noise")) for item in schedule)
    candidates = [idx for idx, item in enumerate(schedule) if not item.get("boundary_noise")]
    extra_dirty = min(max(0, target_dirty - already_dirty), len(candidates))
    for idx in random.sample(candidates, k=extra_dirty):
        schedule[idx]["boundary_noise"] = True

    target_qa = round(n_samples * V5_QA_RECORD_PERCENT / 100)
    # sparse_zero phải thật sự không chứa entity nên không phù hợp với QA lâm sàng dài.
    qa_candidates = [
        idx for idx, item in enumerate(schedule)
        if item.get("sparse_variant") != "zero"
    ]
    for idx in random.sample(qa_candidates, k=min(target_qa, len(qa_candidates))):
        schedule[idx]["qa_style"] = True
    return schedule


def choose_soft_focus(profile, section_cfg):
    """Chọn một gợi ý bù lỗi mềm; không biến nó thành quota hay điều kiện reject."""
    if profile == "baseline":
        return None

    if profile == "quota_v2":
        focus_pool = V2_FOCUS_AREAS if random.random() < V2_FOCUS_PROBABILITY else []
    elif profile == "mixed_v3":
        if not 0 <= V3_FOCUS_PERCENT <= 100:
            raise ValueError("V3_FOCUS_PERCENT phải nằm trong khoảng 0..100")
        if not 0 <= V3_VERY_LONG_FOCUS_PERCENT <= V3_FOCUS_PERCENT:
            raise ValueError(
                "V3_VERY_LONG_FOCUS_PERCENT phải nằm trong 0..V3_FOCUS_PERCENT"
            )
        focus_roll = random.random() * 100
        if focus_roll >= V3_FOCUS_PERCENT:
            return None
        if (
            focus_roll < V3_VERY_LONG_FOCUS_PERCENT
            and section_cfg["key"] in V3_VERY_LONG_FOCUS["section_keys"]
        ):
            return V3_VERY_LONG_FOCUS
        focus_pool = V3_FOCUS_AREAS
    elif profile == "mixed_v4":
        if not 0 <= V4_LONG_FREEFORM_FOCUS_PERCENT <= 100:
            raise ValueError("V4_LONG_FREEFORM_FOCUS_PERCENT phải nằm trong khoảng 0..100")
        if random.random() >= V4_LONG_FREEFORM_FOCUS_PERCENT / 100:
            return None
        if not 0 <= V4_QA_WITHIN_FOCUS_PERCENT <= 100:
            raise ValueError("V4_QA_WITHIN_FOCUS_PERCENT phải nằm trong khoảng 0..100")
        qa_focus = next(item for item in V4_FOCUS_AREAS if item["format"] == "qa")
        if random.random() < V4_QA_WITHIN_FOCUS_PERCENT / 100:
            return with_optional_v4_boundary_noise(qa_focus)
        # mixed_v4 mới dành đúng phần còn lại cho bài lý thuyết; các format lâm sàng dài
        # vẫn được giữ trong catalog để tái lập batch cũ nhưng không chen vào profile này.
        education_focus = next(
            item for item in V4_FOCUS_AREAS if item["format"] == "education"
        )
        return with_optional_v4_boundary_noise(education_focus)
    elif profile == "mixed_v5":
        # run_generation dùng lịch đã scale để batch 600 khớp kế hoạch. Nhánh này giữ API
        # tiện cho audit/test hoặc caller chọn từng focus độc lập.
        focus_pool = V5_FOCUS_AREAS
        selected = dict(random.choices(
            focus_pool,
            weights=[item["quota_weight"] for item in focus_pool],
            k=1,
        )[0])
        if (
            not selected.get("boundary_noise")
            and random.random() < V5_EXTRA_DIRTY_NONPRIMARY_PERCENT / 100
        ):
            selected["boundary_noise"] = True
        if (
            selected.get("sparse_variant") != "zero"
            and random.random() < V5_QA_RECORD_PERCENT / 100
        ):
            selected["qa_style"] = True
        return selected
    else:
        raise ValueError(f"Profile không hợp lệ: {profile}")

    compatible = [
        item for item in focus_pool
        if section_cfg["key"] in item["section_keys"]
    ]
    if not compatible:
        return None
    return random.choices(
        compatible,
        weights=[item.get("weight", 1.0) for item in compatible],
        k=1,
    )[0]


def completion_tokens_for_focus(focus_cfg):
    """Dành thêm output budget cho record dài mà không làm baseline tốn token."""
    if not focus_cfg:
        return 1400
    if "max_completion_tokens" in focus_cfg:
        return focus_cfg["max_completion_tokens"]
    if focus_cfg.get("mode") == "v4":
        return V4_MAX_COMPLETION_TOKENS
    if focus_cfg.get("mode") == "v5":
        return 2200 if focus_cfg.get("qa_style") else 1400
    return 1400


def forced_long_focus_index(profile, n_samples):
    """Batch kiểm thử đủ lớn luôn có ít nhất một mẫu dài để không bị random về 0.

    Đây chỉ là một mẫu bảo hiểm trong batch >=10; các focus còn lại vẫn được chọn mềm như cũ.
    Chọn vị trí gần giữa batch và chỉ dùng section tương thích với timeline.
    """
    if n_samples < 10 or profile not in {"mixed_v3", "mixed_v4"}:
        return None
    candidates = [
        index
        for index in range(n_samples)
        if SECTION_TYPES[index % len(SECTION_TYPES)]["key"] in {"hien_tai", "danh_gia"}
    ]
    if not candidates:
        return None
    midpoint = (n_samples - 1) / 2
    return min(candidates, key=lambda index: abs(index - midpoint))


def validate_focus_quality(record, focus_cfg):
    """QC mềm chỉ áp dụng cho focus dài chủ động, không làm baseline bị reject."""
    if not focus_cfg:
        return None
    input_text = record.get("input_text", "")
    min_chars = focus_cfg.get("min_input_chars")
    if min_chars and len(input_text) < min_chars:
        return f"focus dài cần ít nhất {min_chars} ký tự, thực tế {len(input_text)}"
    if focus_cfg.get("require_repeated_entity"):
        entity_counts = Counter(entity["text"] for entity in record.get("entities", []))
        repeated = any(
            count >= 2 and len(re.findall(re.escape(text), input_text, re.IGNORECASE)) >= 2
            for text, count in entity_counts.items()
        )
        if not repeated:
            return "focus dài cần ít nhất một entity được nhắc và annotate từ 2 lần"
    if focus_cfg.get("boundary_noise"):
        noise_re = (
            V5_DIRTY_SIGNAL_RE
            if focus_cfg.get("mode") == "v5"
            else CONTROLLED_BOUNDARY_NOISE_RE
        )
        crossing = [
            entity["text"] for entity in record.get("entities", [])
            if CONTROLLED_BOUNDARY_NOISE_RE.search(entity.get("text", ""))
        ]
        if crossing:
            return f"entity không được kéo qua ranh giới dính dấu câu: {crossing[:3]}"
        if not noise_re.search(input_text):
            # V4 dùng boundary noise như augmentation mềm. Không có noise vẫn là gold hợp lệ,
            # không đáng gọi API lại. V5 có nhóm dirty riêng nên mới giữ yêu cầu bắt buộc.
            if focus_cfg.get("mode") == "v5":
                return "focus boundary-noise cần ít nhất một chỗ thiếu khoảng trắng sau dấu câu"
    if focus_cfg.get("mode") == "v5":
        entities = record.get("entities", [])
        entity_types = {entity["type"] for entity in entities}
        assertion_states = {
            tuple(entity.get("assertions") or [])
            for entity in entities
        }
        variant = focus_cfg.get("sparse_variant")
        if variant == "zero" and entities:
            return f"sparse_zero cần entities=[], thực tế có {len(entities)}"
        if variant == "one_type" and not (
            1 <= len(entities) <= 3 and len(entity_types) == 1
        ):
            return "sparse_one_type cần 1-3 entity và đúng một type"
        if variant == "two_types" and not (
            2 <= len(entities) <= 4 and len(entity_types) == 2
        ):
            return "sparse_two_types cần 2-4 entity và đúng hai type"
        if focus_cfg.get("key") == "contrastive_assertions" and len(assertion_states) < 2:
            return "contrastive_assertions cần ít nhất hai trạng thái assertion khác nhau"
        if focus_cfg.get("key") == "false_cues_and_scope" and len(assertion_states) < 3:
            return "false_cues_and_scope cần ít nhất ba trạng thái assertion cục bộ"
        if focus_cfg.get("key") == "dense_ner_boundaries":
            if len(entities) < 3 or not any(len(entity["text"]) >= 24 for entity in entities):
                return "dense_ner_boundaries cần >=3 entity và ít nhất một span dài"
        min_drugs = focus_cfg.get("min_drugs")
        if min_drugs:
            drugs = [entity for entity in entities if entity["type"] == "THUỐC"]
            if len(drugs) < min_drugs:
                return f"medication_list cần ít nhất {min_drugs} THUỐC, thực tế {len(drugs)}"
            if any("isHistorical" not in entity.get("assertions", []) for entity in drugs):
                return "thuốc trong danh sách trước nhập viện phải có isHistorical"
            if any(
                entity["type"] == "TRIỆU_CHỨNG"
                and "isHistorical" in entity.get("assertions", [])
                for entity in entities
            ):
                return "triệu chứng chỉ định sau thuốc không được kế thừa isHistorical"
        incomplete = [
            entity["text"] for entity in entities
            if INCOMPLETE_ENTITY_END_RE.search(entity["text"])
            and len(entity["text"].split()) <= 5
        ]
        if incomplete:
            return f"v5 còn span cụt/chưa đủ nghĩa: {incomplete[:3]}"

        entity_pairs = Counter((entity["text"], entity["type"]) for entity in entities)
        for match in COMMON_LAB_PAIR_RE.finditer(input_text):
            expected = (
                (match.group("name").strip(), "TÊN_XÉT_NGHIỆM"),
                (match.group("value").strip(), "KẾT_QUẢ_XÉT_NGHIỆM"),
            )
            if any(entity_pairs[pair] == 0 for pair in expected):
                return f"v5 thiếu cặp lab đầy đủ: {match.group(0)!r}"

        # Cấm đúng các fragment đo lường từng xuất hiện trong output thật.
        measurement_fragments = {"độ", "từ", "nhịp", "thiếu"}
        bad_measurement_fragments = [
            entity["text"] for entity in entities
            if entity["text"].strip().lower() in measurement_fragments
        ]
        if bad_measurement_fragments:
            return f"v5 còn fragment đo lường: {bad_measurement_fragments[:3]}"
        if focus_cfg.get("require_complete_occurrences"):
            has_repeated_text = False
            for entity in entities:
                text = entity["text"]
                if any(
                    text in other["text"] and len(other["text"]) > len(text)
                    for other in entities
                ):
                    continue
                source_count = len(re.findall(re.escape(text), input_text))
                annotated_count = sum(candidate["text"] == text for candidate in entities)
                has_repeated_text = has_repeated_text or source_count >= 2
                if source_count != annotated_count:
                    return (
                        f"recall focus lệch occurrence của {text!r}: "
                        f"text={source_count}, entities={annotated_count}"
                    )
            if not has_repeated_text:
                return "recall focus cần ít nhất một entity text lặp từ hai occurrence"
        if focus_cfg.get("qa_style"):
            qa_markers_ok = bool(
                re.search(r"(?i)\b(?:câu hỏi|hỏi\s*:|người dùng)\b", input_text)
                and re.search(r"(?i)\b(?:câu trả lời|trả lời|bác sĩ)\b", input_text)
            )
            if len(input_text) < 600 or not qa_markers_ok:
                return "v5 qa_style cần hỏi-đáp thật sự và tối thiểu 600 ký tự"

            for entity in entities:
                text = entity["text"]
                if len(text.strip()) < 4:
                    continue
                if any(
                    text.lower() in other["text"].lower()
                    and len(other["text"]) > len(text)
                    for other in entities
                ):
                    continue
                source_count = len(re.findall(re.escape(text), input_text, re.IGNORECASE))
                annotated_count = sum(
                    candidate["text"].lower() == text.lower()
                    for candidate in entities
                )
                if source_count > annotated_count:
                    return (
                        f"v5 QA bỏ sót occurrence lặp của {text!r}: "
                        f"text={source_count}, entities={annotated_count}"
                    )
    return None


def resolve_output_path(path):
    if path.is_absolute():
        return path
    return BASE_DIR / path


def run_generation(
    profile=DEFAULT_PROFILE,
    n_samples=N_SAMPLES,
    output=None,
    reject_output=None,
):
    """Sinh dữ liệu bằng API; phần argparse/CLI nằm ở ``scripts/data_gen``."""
    if profile not in {"baseline", "quota_v2", "mixed_v3", "mixed_v4", "mixed_v5"}:
        raise ValueError(f"Profile không hợp lệ: {profile}")
    if n_samples <= 0:
        raise ValueError("Số mẫu phải lớn hơn 0.")

    default_outputs = {
        "baseline": OUTPUT_PATH,
        "quota_v2": BASE_DIR / "data" / "synthetic" / "train_500_2.jsonl",
        "mixed_v3": BASE_DIR / "data" / "synthetic" / "train_500_3.jsonl",
        "mixed_v4": BASE_DIR / "data" / "synthetic" / "train_500_4.jsonl",
        "mixed_v5": BASE_DIR / "data" / "synthetic" / "train_500_5.jsonl",
    }
    default_output = default_outputs[profile]
    default_rejects = {
        "baseline": REJECT_PATH,
        "quota_v2": BASE_DIR / "data" / "synthetic" / "reject_500_2.jsonl",
        "mixed_v3": BASE_DIR / "data" / "synthetic" / "reject_500_3.jsonl",
        "mixed_v4": BASE_DIR / "data" / "synthetic" / "reject_500_4.jsonl",
        "mixed_v5": BASE_DIR / "data" / "synthetic" / "reject_500_5.jsonl",
    }
    default_reject = default_rejects[profile]
    output_path = resolve_output_path(Path(output)) if output else default_output
    reject_path = resolve_output_path(Path(reject_output)) if reject_output else default_reject

    if API_KEY in ("Chưa có key", ""):
        print("[!] Bạn chưa điền API_KEY. Sửa biến API_KEY ở đầu file hoặc export OPENROUTER_API_KEY.")
        return
    if MODEL in ("Chọn model đi", ""):
        print("[!] Bạn chưa chọn MODEL. Sửa biến MODEL hoặc export GEN_MODEL.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reject_path.parent.mkdir(parents=True, exist_ok=True)

    print("[*] Load seed sections...")
    seed_pool = load_seed_sections()
    print("[*] Load mtsamples pools (drug + vitals + specialty context, đảm bảo đủ chuyên khoa)...")
    drug_pool, vitals_pool, specialty_pool = load_mtsamples_pools()
    print("[*] Load RxNorm pool (tên thuốc thật từ RXNCONSO.RRF, đa dạng hơn regex mtsamples)...")
    rxnorm_pool = load_rxnorm_pool()
    print(f"    -> {len(rxnorm_pool)} tên thuốc RxNorm")
    drug_pool = list(set(drug_pool) | set(rxnorm_pool))  # gộp, dedupe
    covered = [name for name, _, _ in SPECIALTY_CONFIG if specialty_pool.get(name)]
    print(f"    -> {len(drug_pool)} thuốc (gộp mtsamples + RxNorm), {len(vitals_pool)} mẫu vitals, "
          f"{len(covered)}/{len(SPECIALTY_CONFIG)} chuyên khoa có data: {covered}")
    print("[*] Load ICD-10 pool (chẩn đoán thật, nhóm theo chương lâm sàng)...")
    icd10_by_chapter = load_icd10_pool()
    print(f"    -> {sum(len(v) for v in icd10_by_chapter.values())} chẩn đoán, "
          f"{len(icd10_by_chapter)} chương")

    assertion_counter = Counter()
    entity_type_counter = Counter()
    format_counter = Counter()
    guaranteed_long_index = forced_long_focus_index(profile, n_samples)
    v5_focus_schedule = (
        build_v5_focus_schedule(n_samples) if profile == "mixed_v5" else None
    )
    suppressed_focus_indices = set()
    consecutive_failed_groups = 0

    with open(output_path, "w", encoding="utf-8") as fout, \
         open(reject_path, "w", encoding="utf-8") as freject:
        i = 0
        while i < n_samples:
            # Luôn giữ cách quay section của baseline. V2 chỉ thêm một gợi ý mềm
            # phù hợp với section hiện tại, không thay section/chuyên khoa gốc.
            section_cfg = SECTION_TYPES[i % len(SECTION_TYPES)]
            if i in suppressed_focus_indices:
                focus_cfg = None
            elif v5_focus_schedule is not None:
                focus_cfg = v5_focus_schedule[i]
            elif i == guaranteed_long_index:
                focus_cfg = (
                    V3_VERY_LONG_FOCUS
                    if profile == "mixed_v3"
                    else with_optional_v4_boundary_noise(next(
                        item for item in V4_FOCUS_AREAS
                        if item["format"] == "education"
                    ))
                )
            else:
                focus_cfg = choose_soft_focus(profile, section_cfg)
            candidates = section_cfg["assertion_bias"]
            force_assertion = min(candidates, key=lambda a: assertion_counter[a])
            # Form V4 dài cần giữ phân bố ngữ cảnh tự nhiên; không ép isFamily/isHistorical
            # vào từng record chỉ để cân counter như các sample baseline ngắn.
            effective_force_assertion = (
                None
                if focus_cfg and focus_cfg.get("mode") in {"v4", "v5"}
                else force_assertion
            )

            messages = build_generation_messages(
                section_cfg, seed_pool.get(section_cfg["key"], []),
                drug_pool, vitals_pool, icd10_by_chapter, specialty_pool, effective_force_assertion,
                focus_cfg=focus_cfg,
            )
            completion_tokens = completion_tokens_for_focus(focus_cfg)

            success = False
            max_attempts = (
                V4_MAX_RETRY_PER_SAMPLE
                if focus_cfg and focus_cfg.get("mode") in {"v4", "v5"}
                else MAX_RETRY_PER_SAMPLE
            )
            for attempt in range(1, max_attempts + 1):
                try:
                    raw = call_llm(messages, max_tokens=completion_tokens)
                    parsed = parse_llm_json(raw)
                    parsed["force_assertion"] = effective_force_assertion
                    # Metadata chỉ dùng nội bộ cho QC khi record chủ động không có heading;
                    # process_record sẽ loại field này khỏi clean JSONL cuối cùng.
                    focus_mode = focus_cfg.get("mode") if focus_cfg else None
                    is_v4_record = focus_mode == "v4"
                    is_v5_record = focus_mode == "v5"
                    # QA/lý thuyết không thừa hưởng section quay vòng của baseline. Truyền
                    # tien_su ở đây từng làm mọi bệnh trong bài kiến thức thành isHistorical.
                    parsed["_section_key_hint"] = (
                        None if is_v4_record or is_v5_record else section_cfg["key"]
                    )
                    parsed["_knowledge_context"] = bool(
                        (is_v4_record and focus_cfg.get("format") in {"qa", "education"})
                        or (is_v5_record and focus_cfg.get("qa_style"))
                    )
                    # V4 là free-form QA/lý thuyết/bệnh án lai; heading "Đánh giá" không
                    # đồng nghĩa record bắt buộc phải chứa lab. Chỉ baseline giữ QC cũ này.
                    parsed["_require_lab_pair"] = not bool(
                        focus_cfg and focus_cfg.get("mode") in {"v4", "v5"}
                    )
                    parsed["_min_entities"] = (
                        focus_cfg.get("min_entities", 2) if is_v5_record else 2
                    )

                    status, *payload = process_record(
                        parsed,
                        raw_repr=raw,
                        attempt=attempt,
                        model=MODEL,
                    )

                    if status == "reject":
                        reject_log = payload[0]
                        freject.write(json.dumps(reject_log, ensure_ascii=False) + "\n")
                        freject.flush()
                        raise ValueError(
                            f"QC reject [{reject_log['stage']}]: {reject_log['reason']}"
                        )

                    clean_record, applied_logs = payload

                    focus_quality_error = validate_focus_quality(clean_record, focus_cfg)
                    if focus_quality_error:
                        reject_log = {
                            "stage": "focus_quality_reject",
                            "reason": focus_quality_error,
                            "section": section_cfg["heading"],
                            "force_assertion": effective_force_assertion,
                            "input_text": clean_record.get("input_text", ""),
                            "raw": raw,
                            "parsed_entities": parsed.get("entities", []),
                            "cleaned_entities": clean_record.get("entities", []),
                            "attempt": attempt,
                            "model": MODEL,
                        }
                        freject.write(json.dumps(reject_log, ensure_ascii=False) + "\n")
                        freject.flush()
                        raise ValueError(
                            f"QC reject [focus_quality_reject]: {focus_quality_error}"
                        )

                    fout.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
                    fout.flush()

                    for e in clean_record["entities"]:
                            entity_type_counter[e["type"]] += 1
                            if e["assertions"]:
                                for a in e["assertions"]:
                                    assertion_counter[a] += 1
                            else:
                                assertion_counter[None] += 1
                    actual_noise_re = (
                        V5_DIRTY_SIGNAL_RE
                        if focus_cfg and focus_cfg.get("mode") == "v5"
                        else CONTROLLED_BOUNDARY_NOISE_RE
                    )
                    has_actual_boundary_noise = bool(
                        focus_cfg
                        and focus_cfg.get("boundary_noise")
                        and actual_noise_re.search(clean_record.get("input_text", ""))
                    )
                    format_counter[
                        (
                            focus_cfg.get("format", focus_cfg.get("key", "focused"))
                            + ("_boundary_noise" if has_actual_boundary_noise else "")
                        )
                        if focus_cfg else "baseline"
                    ] += 1

                    success = True
                    consecutive_failed_groups = 0
                    break
                except (json.JSONDecodeError, KeyError, ValueError, requests.HTTPError) as e:
                    print(f"    [retry {attempt}/{max_attempts}] lỗi: {e}")
                    time.sleep(1.0)
                    continue

            if success:
                i += 1
                if i % 10 == 0 or i == n_samples:
                    print(
                        f"[*] Đã sinh {i}/{n_samples} | format: {dict(format_counter)} | "
                        f"entity_type: {dict(entity_type_counter)} | assertion: {dict(assertion_counter)}"
                    )
            else:
                # Không đếm một slot thất bại như record đã sinh. Nếu chính focus dài gây lỗi,
                # thử lại cùng vị trí bằng baseline để vừa đủ số dòng mà không retry focus vô hạn.
                if focus_cfg:
                    suppressed_focus_indices.add(i)
                consecutive_failed_groups += 1
                print("    [!] Chưa có record hợp lệ; thử lại cùng vị trí với focus đã nới.")
                if consecutive_failed_groups >= 3:
                    print("    [!] Dừng sớm sau 3 nhóm retry liên tiếp để tránh tốn token vô hạn.")
                    break

            time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\n[DONE] Ghi {i}/{n_samples} record vào {output_path}")
    print(f"Phân bố entity type cuối cùng: {dict(entity_type_counter)}")
    print(f"Phân bố assertion cuối cùng: {dict(assertion_counter)}")
    print(f"Phân bố format cuối cùng: {dict(format_counter)}")
    return {
        "generated": i,
        "output": str(output_path),
        "reject_output": str(reject_path),
        "entity_type": dict(entity_type_counter),
        "assertion": dict(assertion_counter),
        "format": dict(format_counter),
    }
