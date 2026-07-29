"""
reject.py
=========
Bộ lọc chất lượng (QC filter) cho data NER y tế tiếng Việt sinh ra từ generate_data.py
(hoặc bất kỳ file .jsonl nào cùng schema {"input_text": str, "entities": [...]}).

Khác với generate_data.py (chạy TRONG lúc gen, ưu tiên "cứu" sample bằng cách retry LLM),
script này chạy SAU KHI đã có 1 file .jsonl (vd train.jsonl) -- việc của nó là dọn rác lần
cuối trước khi đưa vào huấn luyện, và ghi lại đầy đủ lý do reject để debug prompt/validator.

3 tầng rule (đúng theo yêu cầu):
  A. REJECT CẢ SAMPLE  -- lỗi nặng, không ghi ra clean.jsonl, log vào reject.jsonl
  B. REJECT 1 ENTITY    -- bỏ riêng entity đó, sample vẫn giữ nếu còn đủ tốt
  C. AUTO-FIX            -- sửa bằng rule, không reject

Thứ tự chạy (QUAN TRỌNG, xem lý do trong comment từng bước):
  1. Parse JSON + check field bắt buộc                      (A.1, A.2)
  2. Check entity có đủ text/type + type hợp lệ              (A.4)
  3. Check span (text phải là substring của input_text)      (A.3)
  4. Check assertions là list + mỗi phần tử hợp lệ enum       (A.5)
  5. AUTO-FIX (chạy trước khi check multi-assertion, vì auto-fix có thể tự sửa
     xong mới biết chắc có còn multi-assertion thật sự hay không):
       5.1 strip assertion trên TÊN_XÉT_NGHIỆM/KẾT_QUẢ_XÉT_NGHIỆM   (C.3)
       5.2 fix_negation_leak: "không đau bụng" -> "đau bụng"+isNegated (C.1)
       5.3 auto_split_merged_lab: "CRP 15 mg/L" -> "CRP" + "15 mg/L"   (C.2)
       5.4 fix_negated_history_context: "không có tiền sử X" -> isNegated,
           gỡ isHistorical nếu bị gán nhầm cả 2                        (C.4 phần 1)
       5.5 fix_missing_historical_marker: có "tiền sử X" mà thiếu
           isHistorical -> tự thêm                                     (C.4 phần 2)
  6. Check multi-assertion (>1 assertion / entity)           (A.6)
  7. Check lại lab-type có assertion hay không (lưới an toàn, đáng lẽ đã bị
     auto-fix ở bước 5.1 -- nếu vẫn còn tức là có bug, reject để soi)     (A.7)
  8. REJECT ENTITY (không reject cả sample):
       8.1 "giảm đau/điều trị/phòng ngừa/kiểm soát" bị gán TRIỆU_CHỨNG  (B.1)
       8.2 "phẫu thuật/thay khớp/nội soi..." bị gán CHẨN_ĐOÁN           (B.2)
       8.3 hút thuốc/uống rượu/stress bị gán TRIỆU_CHỨNG/CHẨN_ĐOÁN      (B.3)
       8.4 vitals dạng số bị gán TRIỆU_CHỨNG                            (B.4)
       8.5 isFamily nhưng context là "tiếp xúc với người bệnh"          (B.5)
       8.6 span kéo thêm chủ ngữ/tuổi/số thứ tự liệt kê                 (B.6)
  9. Check duplicate entity y hệt (nhiều hơn số lần thực sự xuất hiện trong text) (A.10)
  10. Check overlap span giữa các entity còn lại              (A.9)
  11. Check section "Đánh giá tại bệnh viện" thiếu lab type   (A.11)
  12. Check force_assertion (nếu record có field này)         (A.12)
  13. Check số entity còn lại sau khi lọc < 2                (A.8)

Cách dùng:
    python reject.py --input train.jsonl --clean-output clean.jsonl --reject-output reject.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# ----------------------------------------------------------------------------
# Schema hằng số -- PHẢI khớp với generate_data.py / đề bài
# ----------------------------------------------------------------------------
ENTITY_TYPES = {"THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}
LAB_TYPES = {"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}
ASSERTION_TYPES = {"isNegated", "isFamily", "isHistorical"}

# Heading section -- thứ tự PHẢI để chuỗi dài hơn trước ("Tiền sử bệnh hiện tại"
# trước "Tiền sử bệnh"), vì "Tiền sử bệnh hiện tại".startswith("Tiền sử bệnh") == True.
SECTION_HEADINGS = [
    ("Lý do vào viện và bệnh sử", "hien_tai"),
    ("Quá trình bệnh lý hiện tại", "hien_tai"),
    ("Tiền sử bệnh hiện tại", "hien_tai"),
    ("Diễn biến bệnh hiện tại", "hien_tai"),
    ("Bệnh sử hiện tại", "hien_tai"),
    ("Diễn tiến trước nhập viện", "hien_tai"),
    ("Tình trạng hiện tại", "hien_tai"),
    ("Lý do khám", "hien_tai"),
    ("Bệnh sử", "hien_tai"),
    ("Tiền sử bản thân và gia đình", "tien_su"),
    ("Tiền căn bệnh lý", "tien_su"),
    ("Tiền sử y khoa", "tien_su"),
    ("Bệnh sử trước đây", "tien_su"),
    ("Tiền sử bệnh", "tien_su"),
    ("Thuốc và bệnh đang theo dõi", "tien_su"),
    ("Tiền căn nội khoa", "tien_su"),
    ("Bệnh nền", "tien_su"),
    ("Đánh giá lâm sàng và cận lâm sàng", "danh_gia"),
    ("Khám và đánh giá tại bệnh viện", "danh_gia"),
    ("Kết quả đánh giá ban đầu", "danh_gia"),
    ("Đánh giá tại bệnh viện", "danh_gia"),
    ("Thăm khám tại bệnh viện", "danh_gia"),
    ("Nhận định lâm sàng", "danh_gia"),
    ("Kết quả thăm khám", "danh_gia"),
    ("Khám ban đầu", "danh_gia"),
    ("Cận lâm sàng", "danh_gia"),
]

# ----------------------------------------------------------------------------
# Regex dùng chung (đồng bộ với generate_data.py để 2 bên không đá nhau)
# ----------------------------------------------------------------------------
VITALS_LIKE_RE = re.compile(
    r"^\s*(VS)?\s*\d{2,3}(\.\d)?\s+\d{2,3}\D{0,3}\d{2,3}(\s+\d{1,3}){1,3}"
)
COMPRESSED_VITALS_RE = re.compile(
    r"^(?P<name>VS|Vitals)\s*"
    r"(?P<value>\d{2,3}(?:[\.,]\d)?\s+\d{2,3}/?\d{2,3}"
    r"(?:\s+\d{1,3}){2}\s+\d{2,3}(?:RA|%)?)$",
    re.IGNORECASE,
)
LEAKED_CONTEXT_RE = re.compile(
    r"^(bố|mẹ|ba|cha|anh|chị|em|ông|bà|gia đình|người thân)\b|"
    r"\b(?:BN|bệnh nhân)\b|"
    r"\b(năm\s+\d{1,3}\s+tuổi|\d+\s+tuổi)\s*$|"
    r"^\d+\.\s"
)
RISK_FACTOR_RE = re.compile(
    r"hút thuốc|uống rượu|uống bia|cà phê|caffeine|căng thẳng|stress|"
    r"mất việc|nghề nghiệp|áp lực công việc|thức khuya|đi bộ|du lịch|"
    r"đi bơi|bơi (?:ở )?hồ|hồ bơi công cộng|tiếp xúc(?: với)?(?: hóa chất)?",
    re.IGNORECASE,
)
DEMOGRAPHIC_ENTITY_RE = re.compile(
    r"^(?:(?:BN|bệnh nhân)\s+)?(?:nam|nữ)(?:\s+\d{1,3}\s*(?:t|tuổi))?$|"
    r"^(?:giới tính nam|giới tính nữ|"
    r"giáo viên|bác sĩ|điều dưỡng|công nhân|nhân viên văn phòng|lái xe|nông dân|"
    r"\d{1,3}\s*(?:t|tuổi))$",
    re.IGNORECASE,
)
NON_CONDITION_ENTITY_RE = re.compile(
    r"^(?:mỹ phẩm|hóa chất|thức ăn|thực phẩm|dầu mỡ nóng|gắng sức|tư thế)$",
    re.IGNORECASE,
)
NON_DRUG_TREATMENT_RE = re.compile(
    r"^(?:liệu pháp|quang trị liệu|vật lý trị liệu|phục hồi chức năng|phẫu thuật|thủ thuật)\b",
    re.IGNORECASE,
)
DRUG_REGIMEN_ONLY_RE = re.compile(
    r"^[A-Z][A-Z0-9+\-]{1,12}\s+phác đồ$", re.IGNORECASE
)
DRUG_DOSE_ONLY_RE = re.compile(
    r"^[<>≤≥]?\s*\d+(?:[\.,]\d+)?\s*(?:mcg|mg|g|gram|ml|mL|IU|đơn vị)"
    r"(?:\s*/\s*(?:ml|mL|kg|ngày))?$",
    re.IGNORECASE,
)
NON_CLINICAL_DRUG_RE = re.compile(
    r"^(?:manganese dioxide|băng phiến|long não|trà gừng|trà đinh hương|BiPAP|"
    r"pound|pounds|lb|lbs|kilogram|kg)$", re.IGNORECASE
)
ELLIPTICAL_DIAGNOSIS_RE = re.compile(
    r"\bviêm\s+gan\s+([A-Z])\s+(?:hoặc|hay|và)\s+([A-Z])\b",
    re.IGNORECASE,
)
COMPOUND_DRUG_SEPARATOR_RE = re.compile(r"\s+/\s+")
DRUG_ROUTE_TOKEN = (
    r"(?:uống|tiêm(?:\s+(?:tĩnh mạch|bắp|dưới da))?|truyền(?:\s+tĩnh mạch)?|"
    r"khí\s+dung|bôi(?:\s+(?:ngoài da|tại chỗ|tai))?|nhỏ(?:\s+mắt)?|"
    r"đặt(?:\s+âm đạo)?|ngậm|hít|phun|po|iv|im|sc|sq|oral)"
)
DRUG_FREQUENCY_TOKEN = (
    r"(?:(?:x\s*)?\d+(?:[\.,]\d+)?\s*(?:lần\s*/?\s*(?:ngày|tuần)|"
    r"viên\s*/?\s*ngày)|ngày\s+\d+(?:[\.,]\d+)?\s+lần|"
    r"(?:mỗi|cách)\s+(?:\d+(?:[\.,]\d+)?\s+)?(?:giờ|ngày|tuần|tháng|sáng|tối)|"
    r"hàng\s+(?:ngày|tuần|tháng|đêm)|sáng\s+hàng\s+ngày|buổi\s+(?:sáng|tối)|"
    r"khi\s+(?:cần|đau|sốt|lên\s+cơn|phù)|liều\s+duy\s+nhất|"
    r"daily|bid|tid|qid|q\d+h|qhs|qam)(?::prn)?"
)
DRUG_DURATION_TOKEN = r"(?:trong\s+\d+\s+(?:ngày|tuần|tháng)|x\s*\d+\s+ngày)"
DRUG_TRAILING_REGIMEN_RE = re.compile(
    rf"^\s+(?P<regimen>(?:{DRUG_ROUTE_TOKEN}(?:\s+{DRUG_FREQUENCY_TOKEN})?|"
    rf"{DRUG_FREQUENCY_TOKEN})(?:\s+{DRUG_DURATION_TOKEN})?(?:\s+khi\s+cần)?)",
    re.IGNORECASE,
)
DIAGNOSIS_CODE_SUFFIX_RE = re.compile(
    r"\s*\((?:ICD-?10\s*)?[A-Z]\d{2}(?:\.\d+)?\)\s*$",
    re.IGNORECASE,
)
DRUG_HOME_CONTEXT_RE = re.compile(
    r"(?:dùng|uống|sử dụng)\s+(?:tại nhà|trước nhập viện)|"
    r"(?:đã\s+)?(?:dùng|uống|sử dụng)\s+tại nhà|"
    r"(?:thuốc|đơn thuốc)\s+trước nhập viện|tiền sử\s+(?:dùng|uống|sử dụng)|"
    r"(?:trong\s+)?lần nhập viện trước|trước đó|tại thời điểm xuất viện trước|"
    r"(?:khi|lúc)\s+xuất viện trước|tự\s+điều trị\s+bằng[^\.\n]{0,120}$|"
    r"tự\s+(?:dùng|uống)[^\.\n]{0,120}$"
    r"|(?:\bTS\b\s*:?|\btiền sử\b)[^.\n]{0,120}\bđang\s+(?:dùng|uống)\s*$",
    re.IGNORECASE,
)
DRUG_CURRENT_CONTEXT_RE = re.compile(
    r"điều trị hiện tại|đang (?:dùng|uống) hiện tại|"
    r"được chỉ định(?: dùng| uống)?|(?:xử trí|dùng|uống|tiêm|khí dung).*tại (?:viện|cấp cứu)|"
    r"(?:đ/?trị|điều trị)\s*:[^.;\n]*$",
    re.IGNORECASE,
)
DRUG_POST_HISTORICAL_CONTEXT_RE = re.compile(
    r"^(?:[^.;\n]{0,80}\b)?(?:trong lần nhập viện trước|"
    r"tại thời điểm xuất viện(?: khỏi)? lần nhập viện trước|trước nhập viện|tại nhà)\b",
    re.IGNORECASE,
)
DRUG_ACTIVE_CUE_RE = re.compile(r"đang\s+(?:dùng|uống)\s*$", re.IGNORECASE)
NEGATION_PREFIX_RE = re.compile(
    r"^(bệnh nhân bảo không bị|bệnh nhân phủ nhận|bệnh nhân khai không|"
    r"chưa ghi nhận|âm tính với|phủ nhận|không thấy|"
    r"không có|không còn|không bị|chưa từng|không hề|không|chưa)\s+",
    re.IGNORECASE,
)
FAMILY_KEYWORDS = ("bố", "mẹ", "ba", "cha", "anh", "chị", "em", "ông", "bà", "cô", "chú",
                    "bác", "gia đình", "người thân", "con")
REAL_FAMILY_CONTEXT_RE = re.compile(
    r"\b(?:bố|mẹ|ba|cha|anh|chị|em|ông|bà|cô|chú|bác|con)"
    r"(?:\s+(?:bệnh nhân|của bệnh nhân))?\b|"
    r"\b(?:gia đình|người thân|họ hàng)(?:\s+bệnh nhân)?\b|"
    r"\btiền sử gia đình\b",
    re.IGNORECASE,
)
NEGATION_SCOPE_CUE_RE = re.compile(
    r"\b(?:không\s+(?:ghi nhận|có|thấy|bị|còn)|"
    r"chưa\s+(?:ghi nhận|có|thấy)|phủ nhận|âm tính với)\b",
    re.IGNORECASE,
)
NEGATION_SCOPE_BREAK_RE = re.compile(
    r"\b(?:nhưng|tuy nhiên|song|trái lại|nhập viện vì|vào viện vì|đến khám vì|"
    r"hiện(?: tại)? (?:có|ghi nhận|xuất hiện))\b",
    re.IGNORECASE,
)
EXPOSURE_KEYWORDS = ("tiếp xúc", "người bệnh", "đồng nghiệp", "bạn cùng phòng", "hàng xóm", "bạn bè")
HISTORICAL_MARKER_RE = re.compile(
    r"(tiền sử|tiền căn|trước đây|trước đó|trong lần nhập viện trước|"
    r"tại thời điểm xuất viện trước|đã từng|từng được chẩn đoán|từng bị)\s*[:\-]?\s*$",
    re.IGNORECASE,
)
NEGATED_HISTORY_CONTEXT_RE = re.compile(
    r"(không có|không còn|chưa từng|không hề)\s+tiền sử\s*[:\-]?\s*$", re.IGNORECASE
)
TREATMENT_PURPOSE_RE = re.compile(
    r"^(giảm|hạ|chống|lợi|an|kiểm soát|điều hòa|ổn định|ngừa|phòng|điều trị)\s|"
    r"^(giảm đau|điều trị|phòng ngừa|kiểm soát)$",
    re.IGNORECASE,
)
PROCEDURE_RE = re.compile(
    r"^(?:phẫu thuật|mổ|sinh mổ|mổ lấy thai|nội soi|đặt stent|đặt catheter|"
    r"sinh thiết|chọc dò|thay khớp|chạy thận nhân tạo|ghép thận|nạo vét(?:\s+tổn thương)?|"
    r"phaco|tán sỏi ngoài cơ thể|đánh giá trước ghép thận|CABG|PCI|ống nội khí quản|"
    r"đặt nội khí quản)\b",
    re.IGNORECASE,
)
PROCEDURE_COMPLICATION_RE = re.compile(
    r"^(?:ghép thận thất bại|thất bại ghép thận|biến chứng sau (?:phẫu thuật|ghép thận))\b",
    re.IGNORECASE,
)
KNOWN_DIAGNOSIS_RECALL_RE = re.compile(r"\bghép thận thất bại\b", re.IGNORECASE)
SYMPTOM_CAUSE_RE = re.compile(
    r"^(?P<symptom>giọng khàn|khó thở|đau bụng|đau ngực|phù chân)\s+do\s+"
    r"(?P<diagnosis>.+)$",
    re.IGNORECASE,
)
DYNAMIC_LAB_RESULT_RE = re.compile(
    r"(?P<result>(?:tăng|giảm)\s+từ\s+[<>≤≥]?\s*\d+(?:[\.,]\d+)?\s+"
    r"(?:lên|xuống)\s+[<>≤≥]?\s*\d+(?:[\.,]\d+)?"
    r"(?:\s*(?:mg/dl|mmol/l|g/dl|g/l|u/l|ng/ml|pg/ml|µmol/l|umol/l|%))?"
    r"(?:\s*\([^\n\)]{1,80}\))?)",
    re.IGNORECASE,
)
SIMPLE_QUALIFIED_LAB_RESULT_RE = re.compile(
    r"^(?:tăng(?:\s+(?:cao|nhẹ|vừa|rõ))?|"
    r"giảm(?:\s+(?:nhẹ|rõ|thấp))?|cao|thấp)\s+(?:còn\s+)?"
    r"(?P<value>[<>≤≥]?\s*\d+(?:[\.,]\d+)?"
    r"(?:\s*(?:mg/dl|mg/l|mmol/l|g/dl|g/l|u/l|ng/ml|pg/ml|µmol/l|umol/l|%))?)$",
    re.IGNORECASE,
)
FULL_SYMPTOM_SPAN_RE = re.compile(
    r"đau vùng hạ vị bên phải và hạ vị bên trái|"
    r"đau bụng(?: ở)? vùng bụng dưới|"
    r"đau\s+RLQ\s*/\s*LLQ|"
    r"đau tăng khi vận động|"
    r"táo bón trở nên tồi tệ hơn(?: gần đây)?|"
    r"đổ mồ hôi qua đêm|đi ngoài ra máu|"
    r"chảy máu cam|chảy nước mũi|nhịp thở nhanh|thiếu oxy|"
    r"ban đỏ(?: lan rộng| ở vị trí phẫu thuật| xuất hiện nhiều ở vị trí phẫu thuật)?|"
    r"đau khi sờ nắn vùng mổ|đau ấn vùng mổ",
    re.IGNORECASE,
)
NAMED_MEASUREMENT_RE = re.compile(
    r"(?P<name>Nhiệt độ|Mạch|Huyết áp|Nhịp thở|Nhịp tim|SPO2|"
    r"độ bão hòa oxy(?:\s*\(\s*SPO2\s*\))?)"
    r"\s*:?\s*(?:từ\s+)?(?=[<>≤≥]?\s*\d)|"
    r"(?P<plain_name>INR|photpho|phospho máu)\s+",
    re.IGNORECASE,
)
SINGLE_LAB_TREND_RE = re.compile(
    r"\b(?P<name>kali|natri|ure|creatinine|CRP|troponin|Hb|eGFR|phosph?o(?:\s+máu)?)\b"
    r"\s+(?:vẫn\s+)?(?:tăng|giảm)(?:\s+(?:cao|thấp|nhẹ|rõ|lên|xuống|còn))?\s+"
    r"(?P<value>[<>≤≥]?\s*\d+(?:[\.,]\d+)?"
    r"(?:\s*(?:mg/dl|mg/l|mmol/l|g/dl|g/l|u/l|ng/ml|pg/ml|µmol/l|umol/l|%))?)",
    re.IGNORECASE,
)
QUALITATIVE_TEST_CLAUSE_RE = re.compile(
    r"(?P<name>điện tâm đồ|ECG|xét nghiệm gắng sức|nghiệm pháp gắng sức)\s+"
    r"(?P<result>(?:bình thường|bất thường)|"
    r"(?:dương tính|âm tính)[^\.\n]*|"
    r"không\s+(?:có|ghi nhận|thấy|phát hiện)[^\.\n]*)",
    re.IGNORECASE,
)
QUALITATIVE_LAB_CLAUSE_RE = re.compile(
    r"(?P<name>xét nghiệm\s+H\.?pylori|test\s+H\.?pylori)\s*"
    r"(?P<result>dương tính|âm tính)",
    re.IGNORECASE,
)
HOLTER_NARRATIVE_CLAUSE_RE = re.compile(
    r"(?P<name>(?:monitor\s+)?Holter)\s+(?:cho thấy|ghi nhận|:)\s*"
    r"(?P<result>[^\n]+)",
    re.IGNORECASE,
)
ECG_NARRATIVE_CLAUSE_RE = re.compile(
    r"(?P<name>điện tâm đồ|ECG)\s*(?:gần nhất\s+)?"
    r"(?:ghi nhận|cho thấy|:)\s*(?P<result>[^\.\n]+)",
    re.IGNORECASE,
)
TRAILING_SYMPTOM_CONNECTOR_RE = re.compile(
    r"\s+(?:khi|vì|do|sau|trước|trong)$",
    re.IGNORECASE,
)
ST_SEGMENT_RESULT_RE = re.compile(r"Đoạn ST chênh xuống[^\.\n]*", re.IGNORECASE)
IMAGING_NARRATIVE_CLAUSE_RE = re.compile(
    r"(?P<name>(?:chụp\s+)?(?:cắt lớp vi tính(?:\s+sọ não)?|"
    r"CT(?:\s+(?!(?:không|cho|ghi|phát)\b)[^\s,:;.]+){0,4}|"
    r"MRI(?:\s+(?!(?:không|cho|ghi|phát)\b)[^\s,:;.]+){0,4}|"
    r"X[\s-]?quang(?:\s+(?!(?:không|cho|ghi|phát)\b)[^\s,:;.]+){0,4}|"
    r"siêu âm(?:\s+(?!(?:không|cho|ghi|phát)\b)[^\s,:;.]+){0,5}|"
    r"nội soi(?:\s+(?!(?:không|cho|ghi|phát)\b)[^\s,:;.]+){0,4}|soi đáy mắt|"
    r"chọc hút tủy xương)|"
    r"chụp kiểm tra)\s+(?:(?:cho hình ảnh|ghi nhận|cho thấy|phát hiện)\s+"
    r"(?P<result>[^\.\n]+)|(?P<negative_result>không\s+(?:ghi nhận|phát hiện|cho thấy)"
    r"\s+[^\.\n]+))",
    re.IGNORECASE,
)
PAST_DIAGNOSIS_CONTEXT_RE = re.compile(
    r"(?:\d+\s+(?:ngày|tuần|tháng|năm)\s+trước(?:\s+nhập viện)?|trước đây|trước đó|"
    r"đã từng|từng)\b[^\.\n]{0,120}\b(?:được chẩn đoán|chẩn đoán|mắc|bị)\s*$",
    re.IGNORECASE,
)
CURRENT_DIAGNOSIS_CONTEXT_RE = re.compile(
    r"(?:lý do (?:nhập|vào) viện\s*:|nhập viện vì|đến\s+(?:ED|khoa cấp cứu)\s+vì)"
    r"[^\.\n]{0,100}$",
    re.IGNORECASE,
)
DANGEROUS_MEDICAL_UNIT_TYPO_RE = re.compile(
    r"(?:phân suất|hệ số)\s+tống máu[^\.\n]{0,30}\b\d+(?:[\.,]\d+)?\s*inch\b",
    re.IGNORECASE,
)
TREATMENT_GOAL_CONTEXT_RE = re.compile(
    r"(?:phòng\s+ngừa|dự\s+phòng|ngăn\s+ngừa|giảm)\s*$",
    re.IGNORECASE,
)
VALUE_FIRST_LAB_RE = re.compile(
    r"(?<![\w\.,])(?P<value>[<>≤≥]?\s*\d+(?:[\.,]\d+)?"
    r"(?:\s*(?:%|mmol/l|mg/dl|g/dl|g/l|u/l|ng/ml|pg/ml|µmol/l|umol/l))?)"
    r"\s+(?P<name>kali|natri|neutrophil|lymphocyte|tiểu cầu|hct|hco3-|ag|"
    r"lactate|bnp|troponin|creatinine|ure|glucose|photpho|phospho(?:\s+máu)?)\b",
    re.IGNORECASE,
)
VALUE_PAREN_LAB_RE = re.compile(
    r"(?<![\w\.,])(?P<value>[<>≤≥]?\s*\d+(?:[\.,]\d+)?"
    r"(?:\s*(?:%|mmol/l|mg/dl|g/dl|g/l|u/l|ng/ml|pg/ml|µmol/l|umol/l))?)"
    r"\s*\(\s*(?P<name>kali|natri|clo|phospho|magie|mg\+\+|k\+|na\+?|cl-)\s*\)",
    re.IGNORECASE,
)
ELECTROLYTE_PANEL_RE = re.compile(r"\b(?:điện giải đồ|ion đồ)\b", re.IGNORECASE)
INR_INTERPRETATION_RE = re.compile(
    r"(?P<result>(?:dưới|trên) ngưỡng điều trị\s+[<>≤≥]?\s*\d+(?:[\.,]\d+)?)",
    re.IGNORECASE,
)
NOISY_FRAGMENT_RE = re.compile(
    r"^(?:hơi|âm|lên|xuống|thu|nhĩ|thất|giây|phút|giờ|ngày|tuần|tháng|năm|"
    r"phải|trái|bên phải|bên trái|từ|nhịp|thiếu|chảy|mạnh|các|bờ|tù|mềm|nhẵn|"
    r"phân|nhầy|mất|đi lại|thành|bên|độ|dương tính với|phù hợp với)$",
    re.IGNORECASE,
)
BARE_ANATOMY_TEST_RE = re.compile(r"^(?:phổi|tim|ngực|bụng|thận|gan)$", re.IGNORECASE)
PROCEDURE_VALUE_CONTEXT_RE = re.compile(
    r"(?:hút|dẫn lưu|chọc hút)\s*$", re.IGNORECASE
)
SECTION_HEADING_OCCURRENCE_RE = re.compile(
    r"(?im)^\s*(?:\d+\.\s*)?(?:Tiền sử bệnh hiện tại|Bệnh sử hiện tại|"
    r"Đánh giá tại bệnh viện|Khám tại bệnh viện|Bệnh nền|Tiền căn nội khoa|"
    r"Thuốc và bệnh đang theo dõi|Bệnh sử|Diễn tiến trước nhập viện|Lý do khám|"
    r"Tình trạng hiện tại|Khám ban đầu|Nhận định lâm sàng|Cận lâm sàng|"
    r"Kết quả thăm khám)(?=\s|Bệnh nhân|BN|$)",
)
MISSING_SENTENCE_SPACE_RE = re.compile(
    r"(?<=[a-zà-ỹ])\.(?=[A-ZĐ])|(?<=[^\W\d_])[,;](?=\S)|"
    r"(?<=[\d%])[,;](?=[^\W\d_])|"
    r"(?<=[a-zà-ỹ])(?=(?:Bệnh nhân|BN)\b)",
)
GENERIC_BIOMEDICAL_ENTITY_RE = re.compile(
    r"^(?:Bệnh|máu|rối loạn|protein|enzyme|amyloid|men\s*G6PD|"
    r"Glucose-6-Phosphate\s+Dehydrogenase|Bại|não|đầu|trán)$",
    re.IGNORECASE,
)
BARE_CELL_ENTITY_RE = re.compile(r"^(?:hồng cầu|bạch cầu|tiểu cầu)$", re.IGNORECASE)
NON_SCHEMA_SUBSTANCE_RE = re.compile(
    r"^(?:băng phiến|long não|trà gừng|trà đinh hương|tinh bột nghệ(?: tách tinh dầu)?|"
    r"BiPAP)$", re.IGNORECASE
)
GENERIC_TEST_HEADING_RE = re.compile(
    r"^(?:chẩn đoán hình ảnh|kết quả chẩn đoán hình ảnh|các kết quả khác)$",
    re.IGNORECASE,
)
INCOMPLETE_ENTITY_END_RE = re.compile(
    r"\b(?:ở|tại|vùng|bên|với|do|kèm|của|gồm|từ|thành)\s*$",
    re.IGNORECASE,
)
CURRENT_EPISODE_EXAM_CONTEXT_RE = re.compile(
    r"(?:đã\s+)?(?:được\s+)?(?:bác sĩ[^.\n]{0,30}\s+)?(?:khám|thăm khám)\s+"
    r"(?:vì|do)\s+(?:(?:các\s+)?triệu chứng\s+)?$",
    re.IGNORECASE,
)
DIAGNOSIS_DURATION_AFTER_RE = re.compile(
    r"^\s*(?:đã\s+)?\d+(?:[\.,]\d+)?\s*(?:tháng|năm)\b",
    re.IGNORECASE,
)
DIAGNOSIS_MEASUREMENT_RE = re.compile(
    r"\b\d{2,3}\s*/\s*\d{2,3}\s*(?:mmHg)?\b|\bTHẤP\b",
    re.IGNORECASE,
)
COMMON_LAB_PAIR_RE = re.compile(
    r"(?P<name>WBC|RBC|HGB|Hb|HCT(?:\s*\(\s*hematocrit\s*\))?|"
    r"NEUT%|LYMPH?%|CRP|BNP|troponin|lactate|kali|natri|creatinine|ure|"
    r"glucose|HbA1c|bạch cầu|tiểu cầu)\s*:?\s*"
    r"(?P<value>[<>≤≥]?\s*\d+(?:[\.,]\d+)?(?:\s*(?:-|–)\s*\d+(?:[\.,]\d+)?)?"
    r"\s*(?:%|G/L|T/L|g/L|g/dL|mg/L|mg/dL|mmol/L|U/L|ng/mL|pg/mL)?)",
    re.IGNORECASE,
)
ORDERED_CULTURE_TEST_RE = re.compile(
    r"\b(?P<name>cấy máu|cấy nước tiểu)\b(?=[^.\n]{0,80}\b(?:đã được gửi|được gửi|đã lấy|được chỉ định))",
    re.IGNORECASE,
)
G6PD_DISEASE_RE = re.compile(r"^(?:bệnh\s+)?thiếu\s+men\s*G6PD$", re.IGNORECASE)
EXPLICIT_PATIENT_HISTORY_RE = re.compile(
    r"(?:bệnh nhân|người bệnh|tôi|em|cháu|con tôi|trẻ)\b[^.\n]{0,70}"
    r"(?:tiền sử|trước đây|đã từng|từng mắc|từng được chẩn đoán|đã dùng)|"
    r"(?:tiền sử|trước đây|đã từng|từng mắc|từng được chẩn đoán|đã dùng)"
    r"[^.\n]{0,70}\b(?:bệnh nhân|người bệnh|tôi|em|cháu|con tôi|trẻ)",
    re.IGNORECASE,
)
MERGED_LAB_RE = re.compile(
    r"^(?P<name>[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9\s]*?)\s+"
    r"(?P<value>[\d][\d\.,/\-]*\s*"
    r"(?:g/l|mg/l|mmol/l|mg/dl|g/dl|u/l|meq/l|iu/l|ng/ml|pg/ml|mmhg|lần/phút|%)?)$",
    re.IGNORECASE,
)
NON_MERGED_LAB_RESULT_RE = re.compile(r"^nhịp\s+xoang\b", re.IGNORECASE)
MULTI_LAB_SEGMENT_RE = re.compile(
    r"^(?P<name>[^:;\n]+?)\s*:\s*"
    r"(?P<value>[<>]?\s*\d[\d\.,/\-]*(?:\s*[A-Za-zÀ-ỹ%+/\-]+(?:\^?\d+)?)?)$",
    re.IGNORECASE,
)
ECHO_TEST_RE = re.compile(r"\bsiêu âm tim\b", re.IGNORECASE)
ECHO_VALVE_FINDING_RE = re.compile(
    r"^hở van (?:động mạch chủ|hai lá|ba lá|động mạch phổi) độ$", re.IGNORECASE
)
ECHO_GRADE_RE = re.compile(r"^(?:[1-4]|i{1,3}|iv)(?:[+\-])?$", re.IGNORECASE)
IMAGING_TEST_NAME_RE = re.compile(
    r"^(?:MRI|CT|MSCT|X[\s-]?quang|siêu âm|nội soi|điện não đồ|điện cơ đồ)\b",
    re.IGNORECASE,
)
GRADED_IMAGING_FINDING_RE = re.compile(
    r"^(?:(?:hở|thoái hóa|hẹp|giãn|phì đại|tổn thương|viêm)\b|"
    r"gan\s+nhiễm\s+mỡ\b|(?:thận\s+(?:phải|trái)\s+)?ứ nước\b).{0,100}\s+độ$",
    re.IGNORECASE,
)
QUANTIFIED_IMAGING_FINDING_RE = re.compile(
    r"^(?:rối loạn vận động|giảm vận động|vô động|tăng vận động)\b.{0,180}$",
    re.IGNORECASE,
)
IMAGING_QUANT_VALUE_RE = re.compile(
    r"^[<>≤≥]?\s*\d+(?:[\.,]\d+)?\s*(?:%|mm|cm|ml|l|độ)?$",
    re.IGNORECASE,
)
HISTORICAL_SPAN_PREFIX_RE = re.compile(r"^(?:tiền sử|tiền căn)\s*[:\-]?\s+", re.IGNORECASE)
ASYMPTOMATIC_DIAGNOSIS_SUFFIX_RE = re.compile(
    r"\s+(?:không|chưa có)\s+(?:biểu hiện|triệu chứng)(?:\s+lâm sàng)?$",
    re.IGNORECASE,
)
PRURITUS_FULL_SPAN_RE = re.compile(
    r"ngứa(?:\s+(?:nhiều|dữ dội|râm ran))?(?:\s+(?:ở|tại))?\s+"
    r"(?:vùng\s+)?(?:da\s+)?"
    r"(?:mặt|cổ|đầu|ngực|bụng|lưng|tay|chân|toàn thân)"
    r"(?:\s+và\s+(?:vùng\s+)?(?:da\s+)?"
    r"(?:mặt|cổ|đầu|ngực|bụng|lưng|tay|chân|toàn thân))*",
    re.IGNORECASE,
)
FEVER_FULL_SPAN_RE = re.compile(
    r"sốt(?:\s+(?:cao|nhẹ|vừa|lên\s+(?:đến|tới)|đến))?\s+"
    r"\d+(?:[\.,]\d+)?(?:\s*(?:°\s*[CF]|độ(?:\s*[CF])?))?",
    re.IGNORECASE,
)
FEVER_VALUE_RE = re.compile(
    r"\bsốt(?:\s+(?:cao|nhẹ|vừa|lên\s+(?:đến|tới)|đến))?\s+"
    r"(?P<value>\d+(?:[\.,]\d+)?)\s*(?P<unit>°\s*[CF]|độ(?:\s*[CF])?)?",
    re.IGNORECASE,
)
ALLOPURINOL_LIPID_RE = re.compile(
    r"\ballopurinol\b[^.;\n]{0,100}\b(?:điều trị|kiểm soát)\s+"
    r"(?:rối loạn\s+(?:chuyển hóa\s+)?lipid(?:\s+máu)?|mỡ\s+máu|tăng\s+cholesterol)",
    re.IGNORECASE,
)
ALLERGY_CONDITION_RE = re.compile(r"^(?:dị ứng|quá mẫn)\s+(?:thuốc|dược phẩm)\b", re.IGNORECASE)
PERIODIC_PARALYSIS_RE = re.compile(r"\bliệt chu kỳ\b", re.IGNORECASE)
RILEY_DAY_RE = re.compile(r"\briley[\s-]?day\b", re.IGNORECASE)
PHENACEMIDE_RE = re.compile(r"\bphenacemide\b", re.IGNORECASE)
KIDNEY_STONE_TYPO_RE = re.compile(r"\bsót thận\b", re.IGNORECASE)


class SampleRejected(Exception):
    """Raise để nhảy thẳng ra ngoài với lý do reject cả sample (tầng A)."""
    def __init__(self, reason, stage):
        self.reason = reason
        self.stage = stage
        super().__init__(reason)


def detect_section(input_text):
    first_line = input_text.strip().split("\n", 1)[0].strip()
    for heading, key in SECTION_HEADINGS:
        if first_line.startswith(heading):
            return key, heading
    return None, first_line


# ----------------------------------------------------------------------------
# Bước 2: check type entity (A.4)
# ----------------------------------------------------------------------------


def normalize_and_check_type(entities):
    """A.4: type không thuộc 5 nhãn -> reject cả sample. Đồng thời chuẩn hoá field."""
    normalized = []
    for e in entities:
        if not isinstance(e, dict):
            raise SampleRejected("Entity không phải object", "schema_reject")
        text = e.get("text")
        etype = e.get("type")
        assertions = e.get("assertions", [])
        if not text or not isinstance(text, str):
            raise SampleRejected("Entity thiếu/rỗng text", "schema_reject")
        if etype not in ENTITY_TYPES:
            raise SampleRejected(f"Type không hợp lệ: '{etype}' (entity '{text}')", "schema_reject")
        normalized.append({"text": text, "type": etype, "assertions": assertions})
    return normalized


# ----------------------------------------------------------------------------
# Bước 3 (ĐỔI): span filter cấp ENTITY, không phải reject cả sample (B.7).
# Lý do đổi: 1 entity bị hallucinate (vd LLM tự suy diễn cụm bị tỉnh lược, "dị ứng
# thuốc hay mỹ phẩm" -> tự bịa "dị ứng mỹ phẩm" không tồn tại nguyên văn) không có
# nghĩa các entity KHÁC trong cùng record cũng hỏng -- reject cả record phí phạm
# 5-6 entity tốt chỉ vì 1 entity xấu. Chạy SAU autofix (case-mismatch, negation-leak)
# để không loại nhầm entity đáng lẽ sửa được.
# ----------------------------------------------------------------------------
def filter_entity_span(entities, input_text, log):
    kept = []
    for e in entities:
        if e["text"] not in input_text:
            log.append(
                f"[reject-entity-span] '{e['text']}' không phải substring của input_text -> loại"
            )
            continue
        kept.append(e)
    return kept


# ----------------------------------------------------------------------------
# Bước 4: assertion enum check (A.5)
# ----------------------------------------------------------------------------
def check_assertion_enum(entities):
    for e in entities:
        if not isinstance(e["assertions"], list):
            raise SampleRejected(f"assertions không phải list (entity '{e['text']}')", "schema_reject")
        for a in e["assertions"]:
            if a not in ASSERTION_TYPES:
                raise SampleRejected(
                    f"Assertion không hợp lệ: '{a}' (entity '{e['text']}')", "schema_reject"
                )


# ----------------------------------------------------------------------------
# Bước 5: AUTO-FIX -- không reject, chỉ sửa (tầng C)
# ----------------------------------------------------------------------------
def autofix_strip_lab_assertion(entities, log):
    fixed = []
    for e in entities:
        if e["type"] in LAB_TYPES and e["assertions"]:
            log.append(f"[autofix-strip-lab-assertion] '{e['text']}' ({e['type']}) có assertion -> xoá")
            e = {**e, "assertions": []}
        fixed.append(e)
    return fixed


def autofix_negation_leak(entities, log):
    fixed = []
    for e in entities:
        if e["type"] in ("TRIỆU_CHỨNG", "CHẨN_ĐOÁN"):
            m = NEGATION_PREFIX_RE.match(e["text"])
            if m:
                new_text = e["text"][m.end():].strip()
                if new_text:
                    log.append(f"[autofix-negation-leak] '{e['text']}' -> '{new_text}' (+isNegated)")
                    assertions = sorted(set(e["assertions"]) | {"isNegated"})
                    e = {**e, "text": new_text, "assertions": assertions}
        fixed.append(e)
    return fixed


def autofix_split_merged_lab(entities, log):
    fixed = []
    for e in entities:
        if e["type"] == "KẾT_QUẢ_XÉT_NGHIỆM":
            # Một entity kết quả đôi khi bị LLM gộp cả dòng công thức máu. Chỉ tách khi
            # có ít nhất hai segment và TẤT CẢ đều đúng dạng ``tên: giá trị`` để tránh
            # làm vỡ finding hình ảnh hoặc kết quả mô tả tự do.
            segments = [segment.strip() for segment in e["text"].split(";")]
            segment_matches = [MULTI_LAB_SEGMENT_RE.fullmatch(segment) for segment in segments]
            if len(segments) >= 2 and all(segment_matches):
                split_pairs = [
                    (match.group("name").strip(), match.group("value").strip())
                    for match in segment_matches
                ]
                log.append(
                    f"[autofix-split-multi-lab] '{e['text']}' -> {len(split_pairs)} cặp lab"
                )
                for name, value in split_pairs:
                    fixed.append({"text": name, "type": "TÊN_XÉT_NGHIỆM", "assertions": []})
                    fixed.append({"text": value, "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []})
                continue
            if NON_MERGED_LAB_RESULT_RE.match(e["text"].strip()):
                fixed.append(e)
                continue
            m = MERGED_LAB_RE.match(e["text"].strip())
            if m:
                name, value = m.group("name").strip(), m.group("value").strip()
                if name and value:
                    log.append(f"[autofix-split-merged-lab] '{e['text']}' -> TÊN_XÉT_NGHIỆM='{name}' + KẾT_QUẢ_XÉT_NGHIỆM='{value}'")
                    fixed.append({"text": name, "type": "TÊN_XÉT_NGHIỆM", "assertions": []})
                    fixed.append({"text": value, "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []})
                    continue
        fixed.append(e)
    return fixed


def autofix_negation_scope(entities, input_text, log):
    """Lan phủ định qua danh sách A, B, C trong cùng mệnh đề, không vượt từ chuyển ý."""
    fixed = []
    cursor = 0
    for entity in entities:
        start = input_text.find(entity["text"], cursor)
        if start == -1:
            start = input_text.find(entity["text"])
        if start != -1:
            cursor = start + len(entity["text"])

        if (
            start != -1
            and entity["type"] in ("TRIỆU_CHỨNG", "CHẨN_ĐOÁN")
            and "isNegated" not in entity["assertions"]
        ):
            sentence_start = max(
                input_text.rfind(".", 0, start),
                input_text.rfind(";", 0, start),
                input_text.rfind("\n", 0, start),
            ) + 1
            context_before = input_text[sentence_start:start]
            cues = list(NEGATION_SCOPE_CUE_RE.finditer(context_before))
            if cues:
                after_last_cue = context_before[cues[-1].end():]
                if not NEGATION_SCOPE_BREAK_RE.search(after_last_cue):
                    assertions = list(entity["assertions"]) + ["isNegated"]
                    log.append(
                        f"[autofix-negation-scope] '{entity['text']}' nằm trong danh sách "
                        f"phủ định -> thêm isNegated"
                    )
                    entity = {**entity, "assertions": assertions}
        fixed.append(entity)
    return fixed


def autofix_split_compressed_vitals(entities, log):
    """Thống nhất VS/Vitals nén thành tên phép đo và chuỗi kết quả riêng."""
    fixed = []
    for entity in entities:
        if entity["type"] == "KẾT_QUẢ_XÉT_NGHIỆM":
            match = COMPRESSED_VITALS_RE.fullmatch(entity["text"].strip())
            if match:
                name, value = match.group("name"), match.group("value")
                has_name = any(
                    e["type"] == "TÊN_XÉT_NGHIỆM" and e["text"].lower() == name.lower()
                    for e in entities
                )
                if not has_name:
                    fixed.append({"text": name, "type": "TÊN_XÉT_NGHIỆM", "assertions": []})
                fixed.append({"text": value, "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []})
                log.append(
                    f"[autofix-split-compressed-vitals] '{entity['text']}' -> "
                    f"TÊN_XÉT_NGHIỆM='{name}' + KẾT_QUẢ_XÉT_NGHIỆM='{value}'"
                )
                continue
        fixed.append(entity)
    return fixed


def autofix_allergy_condition_type(entities, log):
    """Dị ứng thuốc là tình trạng/chẩn đoán, không phải triệu chứng chủ quan."""
    fixed = []
    for entity in entities:
        if entity["type"] == "TRIỆU_CHỨNG" and ALLERGY_CONDITION_RE.match(entity["text"].strip()):
            log.append(
                f"[autofix-allergy-condition-type] '{entity['text']}' TRIỆU_CHỨNG -> CHẨN_ĐOÁN"
            )
            entity = {**entity, "type": "CHẨN_ĐOÁN"}
        fixed.append(entity)
    return fixed


def autofix_diagnosis_context_in_span(entities, log):
    """Bỏ marker lịch sử và qualifier không triệu chứng khỏi tên chẩn đoán."""
    fixed = []
    for entity in entities:
        if entity["type"] == "CHẨN_ĐOÁN":
            new_text = HISTORICAL_SPAN_PREFIX_RE.sub("", entity["text"].strip())
            new_text = ASYMPTOMATIC_DIAGNOSIS_SUFFIX_RE.sub("", new_text).strip()
            if new_text and new_text != entity["text"]:
                log.append(
                    f"[autofix-diagnosis-context-span] '{entity['text']}' -> '{new_text}'"
                )
                entity = {**entity, "text": new_text}
        fixed.append(entity)
    return fixed


def autofix_invalid_family_assertion(entities, input_text, log, window=80):
    """Chỉ giữ isFamily khi ngữ cảnh ngoài span thật sự nhắc đến người thân."""
    fixed = []
    cursor = 0
    for entity in entities:
        start = input_text.find(entity["text"], cursor)
        if start == -1:
            start = input_text.find(entity["text"])
        if start != -1:
            cursor = start + len(entity["text"])

        if "isFamily" in entity["assertions"] and start != -1:
            left = input_text[max(0, start - window):start]
            right = input_text[start + len(entity["text"]):start + len(entity["text"]) + window]
            # Không cho context vượt sang câu/dòng khác rồi mượn nhầm chủ thể.
            left = re.split(r"[.\n]", left)[-1]
            right = re.split(r"[.\n]", right)[0]
            outside_context = f"{left} {right}"
            if not REAL_FAMILY_CONTEXT_RE.search(outside_context):
                new_assertions = [a for a in entity["assertions"] if a != "isFamily"]
                log.append(
                    f"[autofix-invalid-family] '{entity['text']}' không có chủ thể người thân "
                    f"ngoài tên bệnh -> bỏ isFamily"
                )
                entity = {**entity, "assertions": new_assertions}
        fixed.append(entity)
    return fixed


def autofix_echo_valve_finding(entities, input_text, log):
    """Sửa lỗi tách kết luận siêu âm tim thành tên xét nghiệm + một độ rời rạc."""
    fixed = []
    consumed = set()
    lower_input = input_text.lower()

    for i, entity in enumerate(entities):
        if i in consumed:
            continue
        if entity["type"] != "TÊN_XÉT_NGHIỆM" or not ECHO_VALVE_FINDING_RE.fullmatch(entity["text"].strip()):
            fixed.append(entity)
            continue

        finding_start = lower_input.find(entity["text"].lower())
        echo_matches = list(ECHO_TEST_RE.finditer(input_text, 0, max(finding_start, 0)))
        if finding_start == -1 or not echo_matches:
            fixed.append(entity)
            continue
        echo_match = echo_matches[-1]
        if re.search(r"[.\n]", input_text[echo_match.end():finding_start]):
            fixed.append(entity)
            continue

        grade_idx = None
        grade_start = None
        for j in range(i + 1, len(entities)):
            candidate = entities[j]
            if candidate["type"] != "KẾT_QUẢ_XÉT_NGHIỆM" or not ECHO_GRADE_RE.fullmatch(candidate["text"].strip()):
                continue
            candidate_start = lower_input.find(candidate["text"].lower(), finding_start + len(entity["text"]))
            finding_end = finding_start + len(entity["text"])
            if candidate_start != -1 and not input_text[finding_end:candidate_start].strip():
                grade_idx = j
                grade_start = candidate_start
                break
        if grade_idx is None:
            fixed.append(entity)
            continue

        echo_text = input_text[echo_match.start():echo_match.end()]
        has_echo_name = any(
            e["type"] == "TÊN_XÉT_NGHIỆM" and e["text"].lower() == echo_text.lower()
            for e in entities
        )
        result_text = input_text[
            finding_start:
            grade_start + len(entities[grade_idx]["text"])
        ]
        if not has_echo_name:
            fixed.append({"text": echo_text, "type": "TÊN_XÉT_NGHIỆM", "assertions": []})
        fixed.append({"text": result_text, "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []})
        consumed.add(grade_idx)
        log.append(
            f"[autofix-echo-valve-finding] '{entity['text']}' + "
            f"'{entities[grade_idx]['text']}' -> TÊN_XÉT_NGHIỆM='{echo_text}' + "
            f"KẾT_QUẢ_XÉT_NGHIỆM='{result_text}'"
        )

    return fixed


def autofix_graded_imaging_finding(entities, input_text, log):
    """Gộp finding phân độ/định lượng và các kết luận cùng câu dưới kỹ thuật hình ảnh."""
    fixed = []
    consumed = set()
    lower_input = input_text.lower()

    for i, entity in enumerate(entities):
        if i in consumed:
            continue
        is_finding_as_name = (
            GRADED_IMAGING_FINDING_RE.fullmatch(entity["text"].strip())
            or QUANTIFIED_IMAGING_FINDING_RE.fullmatch(entity["text"].strip())
        )
        if entity["type"] != "TÊN_XÉT_NGHIỆM" or not is_finding_as_name:
            fixed.append(entity)
            continue

        finding_start = lower_input.find(entity["text"].lower())
        imaging_idx = None
        imaging_start = -1
        for candidate_idx in range(i - 1, -1, -1):
            candidate = entities[candidate_idx]
            if candidate["type"] != "TÊN_XÉT_NGHIỆM" or not IMAGING_TEST_NAME_RE.match(candidate["text"].strip()):
                continue
            candidate_start = lower_input.rfind(candidate["text"].lower(), 0, finding_start)
            if candidate_start != -1 and not re.search(
                r"[.\n]", input_text[candidate_start + len(candidate["text"]):finding_start]
            ):
                imaging_idx = candidate_idx
                imaging_start = candidate_start
                break
        if finding_start == -1 or imaging_idx is None or imaging_start == -1:
            fixed.append(entity)
            continue

        grade_idx = None
        grade_start = -1
        finding_end = finding_start + len(entity["text"])
        for candidate_idx in range(i + 1, len(entities)):
            candidate = entities[candidate_idx]
            if (
                candidate["type"] != "KẾT_QUẢ_XÉT_NGHIỆM"
                or not IMAGING_QUANT_VALUE_RE.fullmatch(candidate["text"].strip())
            ):
                continue
            candidate_start = lower_input.find(candidate["text"].lower(), finding_end)
            if candidate_start != -1 and not input_text[finding_end:candidate_start].strip():
                grade_idx = candidate_idx
                grade_start = candidate_start
                break
        if grade_idx is None:
            fixed.append(entity)
            continue

        last_idx = grade_idx
        last_end = grade_start + len(entities[grade_idx]["text"])
        # Các KẾT_QUẢ tiếp theo, ngăn bằng dấu phẩy/"và"/"kèm", vẫn là cùng một
        # kết luận hình ảnh nên giữ thành một span trọn vẹn.
        for candidate_idx in range(grade_idx + 1, len(entities)):
            candidate = entities[candidate_idx]
            if candidate["type"] != "KẾT_QUẢ_XÉT_NGHIỆM":
                break
            candidate_start = lower_input.find(candidate["text"].lower(), last_end)
            if candidate_start == -1:
                break
            separator = input_text[last_end:candidate_start]
            if not re.fullmatch(r"\s*(?:,|và|kèm(?: theo)?)\s*", separator, re.IGNORECASE):
                break
            last_idx = candidate_idx
            last_end = candidate_start + len(candidate["text"])

        result_text = input_text[finding_start:last_end]
        fixed.append({"text": result_text, "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []})
        consumed.update(range(grade_idx, last_idx + 1))
        log.append(
            f"[autofix-graded-imaging-finding] '{entity['text']}' -> "
            f"KẾT_QUẢ_XÉT_NGHIỆM='{result_text}' dưới '{entities[imaging_idx]['text']}'"
        )

    return fixed


def autofix_pruritus_full_span(entities, input_text, log):
    """Giữ mức độ/vị trí giải phẫu khi LLM chỉ lấy span trần ``ngứa``."""
    fixed = []
    for entity in entities:
        if entity["type"] == "TRIỆU_CHỨNG" and entity["text"].strip().lower() == "ngứa":
            start = input_text.lower().find(entity["text"].lower())
            if start != -1:
                match = PRURITUS_FULL_SPAN_RE.match(input_text, start)
                if match and match.group(0) != entity["text"]:
                    full_text = match.group(0)
                    log.append(f"[autofix-pruritus-full-span] '{entity['text']}' -> '{full_text}'")
                    entity = {**entity, "text": full_text}
        fixed.append(entity)
    return fixed


def autofix_fever_full_span(entities, input_text, log):
    """Giữ nhiệt độ hợp lý trong span sốt; đơn vị °C có thể được lược bỏ."""
    fixed = []
    cursor = Counter()
    lower_input = input_text.lower()
    for entity in entities:
        if entity["type"] != "TRIỆU_CHỨNG" or not entity["text"].strip().lower().startswith("sốt"):
            fixed.append(entity)
            continue
        start_at = cursor[entity["text"]]
        start = lower_input.find(entity["text"].lower(), start_at)
        if start == -1:
            start = lower_input.find(entity["text"].lower())
        if start != -1:
            cursor[entity["text"]] = start + len(entity["text"])
            match = FEVER_FULL_SPAN_RE.match(input_text, start)
            fever_value = FEVER_VALUE_RE.fullmatch(match.group(0)) if match else None
            plausible_implicit_celsius = False
            if fever_value:
                value = float(fever_value.group("value").replace(",", "."))
                plausible_implicit_celsius = (
                    fever_value.group("unit") is not None or 37.0 <= value <= 43.5
                )
            if (
                match
                and plausible_implicit_celsius
                and match.group(0) != entity["text"]
            ):
                full_text = match.group(0)
                log.append(f"[autofix-fever-full-span] '{entity['text']}' -> '{full_text}'")
                entity = {**entity, "text": full_text}
        fixed.append(entity)
    return fixed


def autofix_split_symptom_cause(entities, input_text, log):
    """Tách cấu trúc ``triệu chứng do chẩn đoán`` khi LLM gom cả cụm thành CHẨN_ĐOÁN."""
    fixed = []
    for entity in entities:
        match = SYMPTOM_CAUSE_RE.fullmatch(entity["text"]) if entity["type"] == "CHẨN_ĐOÁN" else None
        if not match:
            fixed.append(entity)
            continue

        symptom = match.group("symptom")
        diagnosis = match.group("diagnosis").strip()
        if symptom not in input_text or diagnosis not in input_text:
            fixed.append(entity)
            continue
        fixed.extend([
            {"text": symptom, "type": "TRIỆU_CHỨNG", "assertions": list(entity["assertions"])},
            {"text": diagnosis, "type": "CHẨN_ĐOÁN", "assertions": list(entity["assertions"])},
        ])
        log.append(
            f"[autofix-symptom-cause] '{entity['text']}' -> TRIỆU_CHỨNG='{symptom}' + "
            f"CHẨN_ĐOÁN='{diagnosis}'"
        )
    return fixed


def autofix_dynamic_lab_result(entities, input_text, log):
    """Gộp các số bị tách vụn trong kết quả động học tăng/giảm từ A lên/xuống B."""
    fixed = list(entities)
    additions = []
    consumed = set()
    name_cursor = Counter()

    for name_index, name_entity in enumerate(fixed):
        if name_entity["type"] != "TÊN_XÉT_NGHIỆM":
            continue
        name = name_entity["text"]
        start_at = name_cursor[name]
        name_start = input_text.find(name, start_at)
        if name_start == -1:
            name_start = input_text.find(name)
        if name_start == -1:
            continue
        name_cursor[name] = name_start + len(name)
        tail_start = name_start + len(name)
        tail = input_text[tail_start:tail_start + 180]
        match = DYNAMIC_LAB_RESULT_RE.search(tail)
        if not match or match.start() > 12:
            continue

        result_start = tail_start + match.start("result")
        result_end = tail_start + match.end("result")
        result_text = input_text[result_start:result_end]
        already_full = False
        for idx, entity in enumerate(fixed):
            if idx == name_index or entity["type"] != "KẾT_QUẢ_XÉT_NGHIỆM":
                continue
            entity_start = input_text.find(entity["text"], tail_start, result_end)
            if entity_start == result_start and entity["text"] == result_text:
                already_full = True
                break
            if entity_start != -1 and entity_start + len(entity["text"]) <= result_end:
                consumed.add(idx)
        if not already_full:
            additions.append({"text": result_text, "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []})
            log.append(f"[autofix-dynamic-lab] gộp kết quả động học -> '{result_text}'")

    return [entity for idx, entity in enumerate(fixed) if idx not in consumed] + additions


def autofix_simple_lab_gold_boundary(entities, log):
    """Thu gọn kết quả lab đơn về giá trị + đơn vị theo boundary gold BTC."""
    fixed = []
    for entity in entities:
        entity = dict(entity)
        if entity["type"] == "KẾT_QUẢ_XÉT_NGHIỆM":
            match = SIMPLE_QUALIFIED_LAB_RESULT_RE.fullmatch(entity["text"].strip())
            if match:
                old_text = entity["text"]
                entity["text"] = match.group("value").strip()
                log.append(
                    f"[autofix-lab-gold-boundary] '{old_text}' -> '{entity['text']}'"
                )
        fixed.append(entity)
    return fixed


def autofix_single_lab_trend_pairs(entities, input_text, log):
    """Bổ sung cặp lab bị miss trong dạng ``kali vẫn giảm xuống 2.2``."""
    expected = Counter()
    for match in SINGLE_LAB_TREND_RE.finditer(input_text):
        expected[(match.group("name"), "TÊN_XÉT_NGHIỆM")] += 1
        expected[(match.group("value").strip(), "KẾT_QUẢ_XÉT_NGHIỆM")] += 1

    if not expected:
        return entities

    fixed = list(entities)
    existing = Counter((entity["text"], entity["type"]) for entity in fixed)
    for (text, entity_type), count in expected.items():
        missing = max(0, count - existing[(text, entity_type)])
        for _ in range(missing):
            fixed.append({"text": text, "type": entity_type, "assertions": []})
            log.append(f"[autofix-single-lab-trend] thêm {entity_type}='{text}'")
    return fixed


def autofix_trailing_symptom_connector(entities, input_text, log):
    """Cắt từ nối bị nuốt ở cuối span triệu chứng, ví dụ ``khó chịu vùng ngực khi``."""
    fixed = []
    for entity in entities:
        if entity["type"] == "TRIỆU_CHỨNG":
            new_text = TRAILING_SYMPTOM_CONNECTOR_RE.sub("", entity["text"].strip()).strip()
            if new_text and new_text != entity["text"] and new_text in input_text:
                log.append(
                    f"[autofix-trailing-symptom-connector] '{entity['text']}' -> '{new_text}'"
                )
                entity = {**entity, "text": new_text}
        fixed.append(entity)
    return fixed


def autofix_narrative_test_clauses(entities, input_text, log):
    """Giữ nguyên finding dài sau imaging/ECG/nghiệm pháp thay vì các mảnh entity vụn."""
    clauses = []
    for pattern, tag in (
        (IMAGING_NARRATIVE_CLAUSE_RE, "imaging"),
        (QUALITATIVE_TEST_CLAUSE_RE, "qualitative-test"),
        (QUALITATIVE_LAB_CLAUSE_RE, "qualitative-lab"),
        (HOLTER_NARRATIVE_CLAUSE_RE, "holter"),
        (ECG_NARRATIVE_CLAUSE_RE, "ecg"),
    ):
        for match in pattern.finditer(input_text):
            result = match.groupdict().get("result") or match.groupdict().get("negative_result")
            clauses.append(
                (
                    match.start(),
                    match.end(),
                    match.group("name").strip(),
                    result.strip().rstrip("."),
                    tag,
                )
            )

    for match in ST_SEGMENT_RESULT_RE.finditer(input_text):
        if re.search(
            r"(?:xét nghiệm|nghiệm pháp) gắng sức",
            input_text[max(0, match.start() - 500):match.start()],
            re.IGNORECASE,
        ):
            clauses.append(
                (match.start(), match.end(), None, match.group(0).strip(), "stress-st")
            )

    if not clauses:
        return entities

    spans = assign_spans_sequential(entities, input_text)
    if spans is None:
        return entities

    drop_indexes = set()
    additions = []
    for clause_start, clause_end, name, result, tag in sorted(clauses):
        for idx, (start, end, _entity) in enumerate(spans):
            if clause_start <= start and end <= clause_end:
                drop_indexes.add(idx)
        if name:
            additions.append({"text": name, "type": "TÊN_XÉT_NGHIỆM", "assertions": []})
        additions.append(
            {"text": result, "type": "KẾT_QUẢ_XÉT_NGHIỆM", "assertions": []}
        )
        log.append(
            f"[autofix-{tag}-clause] "
            f"TÊN={name!r}, KẾT_QUẢ={result!r}"
        )

    return [
        entity for idx, entity in enumerate(entities) if idx not in drop_indexes
    ] + additions


def autofix_diagnosis_timing(entities, input_text, log, window=180):
    """Sửa assertion chẩn đoán theo cue quá khứ hoặc đợt nhập viện hiện tại rõ ràng."""
    fixed = []
    cursor = Counter()
    for entity in entities:
        if entity["type"] != "CHẨN_ĐOÁN":
            fixed.append(entity)
            continue
        start_at = cursor[entity["text"]]
        idx = input_text.find(entity["text"], start_at)
        if idx == -1:
            idx = input_text.find(entity["text"])
        if idx != -1:
            cursor[entity["text"]] = idx + len(entity["text"])
        before = input_text[max(0, idx - window):idx] if idx != -1 else ""
        after = input_text[idx + len(entity["text"]):idx + len(entity["text"]) + 30] if idx != -1 else ""
        assertions = list(entity["assertions"])
        if CURRENT_EPISODE_EXAM_CONTEXT_RE.search(before):
            if "isHistorical" in assertions:
                assertions = [value for value in assertions if value != "isHistorical"]
                log.append(
                    f"[autofix-diagnosis-current-exam] '{entity['text']}' -> bỏ isHistorical"
                )
        elif PAST_DIAGNOSIS_CONTEXT_RE.search(before) or DIAGNOSIS_DURATION_AFTER_RE.match(after):
            if "isHistorical" not in assertions:
                assertions.append("isHistorical")
                log.append(
                    f"[autofix-diagnosis-past] '{entity['text']}' -> thêm isHistorical"
                )
        elif CURRENT_DIAGNOSIS_CONTEXT_RE.search(before) and "isHistorical" in assertions:
            assertions = [value for value in assertions if value != "isHistorical"]
            log.append(
                f"[autofix-diagnosis-current] '{entity['text']}' -> bỏ isHistorical"
            )
        fixed.append({**entity, "assertions": assertions})
    return fixed


def autofix_current_hypoxia_type(entities, input_text, log):
    """Trong lý do nhập viện/triệu chứng hiện tại, ``thiếu oxy`` là TRIỆU_CHỨNG."""
    fixed = []
    cursor = 0
    for entity in entities:
        idx = input_text.lower().find(entity["text"].lower(), cursor)
        if idx == -1:
            idx = input_text.lower().find(entity["text"].lower())
        if idx != -1:
            cursor = idx + len(entity["text"])
        before = input_text[max(0, idx - 100):idx] if idx != -1 else ""
        if (
            entity["type"] == "CHẨN_ĐOÁN"
            and entity["text"].strip().lower() == "thiếu oxy"
            and re.search(
                r"(?:lý do (?:nhập|vào) viện|triệu chứng hiện tại)[^\.\n]{0,80}$",
                before,
                re.IGNORECASE,
            )
            and not re.search(r"chẩn đoán\s*$", before, re.IGNORECASE)
        ):
            entity = {
                **entity,
                "type": "TRIỆU_CHỨNG",
                "assertions": [a for a in entity["assertions"] if a != "isHistorical"],
            }
            log.append("[autofix-current-hypoxia] 'thiếu oxy' -> TRIỆU_CHỨNG, bỏ isHistorical")
        fixed.append(entity)
    return fixed


def autofix_full_known_symptom_spans(entities, input_text, log):
    """Khôi phục các span phối hợp đã được review thủ công và tránh giữ mảnh con."""
    matches = list(FULL_SYMPTOM_SPAN_RE.finditer(input_text))
    if not matches:
        return entities

    spans = assign_spans_sequential(entities, input_text)
    if spans is None:
        return entities

    match_ranges = [(match.start(), match.end(), match.group(0)) for match in matches]
    inherited = {}
    drop_indexes = set()
    covered = set()
    for index, (start, end, entity) in enumerate(spans):
        if entity["type"] != "TRIỆU_CHỨNG":
            continue
        for match_index, (target_start, target_end, target_text) in enumerate(match_ranges):
            if end <= target_start or target_end <= start:
                continue
            if entity["text"].lower() == target_text.lower() and start == target_start:
                covered.add(match_index)
                inherited[match_index] = list(entity["assertions"])
                break
            if target_start <= start and end <= target_end:
                drop_indexes.add(index)
                if entity["assertions"] and match_index not in inherited:
                    inherited[match_index] = list(entity["assertions"])
                log.append(f"[autofix-full-symptom] loại span cụt '{entity['text']}'")
                break

    fixed = [entity for index, entity in enumerate(entities) if index not in drop_indexes]
    for match_index, (start, _end, text) in enumerate(match_ranges):
        if match_index in covered:
            continue
        assertions = inherited.get(match_index, [])
        if not assertions:
            local_before = input_text[max(0, start - 60):start]
            if NEGATION_SCOPE_CUE_RE.search(local_before):
                assertions = ["isNegated"]
        fixed.append({"text": text, "type": "TRIỆU_CHỨNG", "assertions": assertions})
        log.append(f"[autofix-full-symptom] thêm span trọn vẹn '{text}'")
    return fixed


def autofix_named_measurement_pairs(entities, input_text, log):
    """Bổ sung phần tên/kết quả bị bỏ sót ở vital, INR và phospho đã định dạng rõ."""
    fixed = list(entities)
    existing = Counter((entity["text"], entity["type"]) for entity in fixed)
    expected = Counter()
    pair_ranges = []
    value_re = re.compile(
        r"[<>≤≥]?\s*\d+(?:[\.,]\d+)?"
        r"(?:\s*(?:-|–)\s*\d+(?:[\.,]\d+)?)?"
        r"(?:\s*/\s*\d+(?:[\.,]\d+)?)?"
        r"(?:\s*(?:độ\s*C|°\s*C|l/p|lần/phút|mmHg|%|mg/dL|mmol/L))?",
        re.IGNORECASE,
    )

    for match in NAMED_MEASUREMENT_RE.finditer(input_text):
        name = match.group("name") or match.group("plain_name")
        tail = input_text[match.end():match.end() + 80]
        if name.lower() == "inr":
            result_match = INR_INTERPRETATION_RE.match(tail)
        else:
            result_match = value_re.match(tail)
        if not result_match:
            continue
        result = result_match.group("result") if "result" in result_match.groupdict() else result_match.group(0)
        result = result.strip()
        expected[(name, "TÊN_XÉT_NGHIỆM")] += 1
        expected[(result, "KẾT_QUẢ_XÉT_NGHIỆM")] += 1
        pair_ranges.append(
            (
                match.start(),
                match.end() + result_match.end(),
                match.end() + result_match.start(),
                name,
                result,
            )
        )

    # Loại các mảnh LLM đặt lọt trong đúng cặp đo lường, ví dụ TÊN='độ', KQ='từ'.
    spans = assign_spans_sequential(fixed, input_text)
    if spans is not None and pair_ranges:
        kept = []
        for start, end, entity in spans:
            inside = next(
                (
                    (name, result) for pair_start, pair_end, result_start, name, result in pair_ranges
                    if (
                        pair_start <= start and end <= pair_end
                        or (entity["type"] == "KẾT_QUẢ_XÉT_NGHIỆM" and start == result_start)
                    )
                ),
                None,
            )
            if inside and (entity["text"], entity["type"]) not in {
                (inside[0], "TÊN_XÉT_NGHIỆM"),
                (inside[1], "KẾT_QUẢ_XÉT_NGHIỆM"),
            }:
                log.append(
                    f"[autofix-named-measurement] loại mảnh '{entity['text']}'"
                )
                continue
            kept.append(entity)
        fixed = kept
        existing = Counter((entity["text"], entity["type"]) for entity in fixed)

    for (text, entity_type), count in expected.items():
        missing = count - existing[(text, entity_type)]
        for _ in range(max(0, missing)):
            fixed.append({"text": text, "type": entity_type, "assertions": []})
            log.append(f"[autofix-named-measurement] thêm {entity_type}='{text}'")
    return fixed


def autofix_common_lab_pairs(entities, input_text, log):
    """Chuẩn hóa lab phổ biến viết liền/ghi nhanh thành từng cặp tên và giá trị."""
    matches = list(COMMON_LAB_PAIR_RE.finditer(input_text))
    if not matches:
        return entities

    spans = assign_spans_sequential(entities, input_text)
    fixed = []
    if spans is None:
        fixed = list(entities)
    else:
        for start, end, entity in spans:
            containing = next(
                (match for match in matches if match.start() <= start and end <= match.end()),
                None,
            )
            if containing is None:
                fixed.append(entity)
                continue
            name = containing.group("name").strip()
            value = containing.group("value").strip()
            if (entity["text"], entity["type"]) in {
                (name, "TÊN_XÉT_NGHIỆM"),
                (value, "KẾT_QUẢ_XÉT_NGHIỆM"),
            }:
                fixed.append(entity)
            else:
                log.append(f"[autofix-common-lab] loại mảnh/gộp sai '{entity['text']}'")

    existing = Counter((entity["text"], entity["type"]) for entity in fixed)
    for match in matches:
        for text, entity_type in (
            (match.group("name").strip(), "TÊN_XÉT_NGHIỆM"),
            (match.group("value").strip(), "KẾT_QUẢ_XÉT_NGHIỆM"),
        ):
            if existing[(text, entity_type)] == 0:
                fixed.append({"text": text, "type": entity_type, "assertions": []})
                log.append(f"[autofix-common-lab] thêm {entity_type}='{text}'")
            else:
                existing[(text, entity_type)] -= 1
    return fixed


def autofix_ordered_culture_tests(entities, input_text, log):
    """Cấy đã lấy/gửi là tên xét nghiệm dù kết quả còn chờ."""
    fixed = list(entities)
    existing = Counter(
        entity["text"].lower() for entity in fixed
        if entity["type"] == "TÊN_XÉT_NGHIỆM"
    )
    for match in ORDERED_CULTURE_TEST_RE.finditer(input_text):
        name = match.group("name")
        if existing[name.lower()] > 0:
            existing[name.lower()] -= 1
            continue
        fixed.append({"text": name, "type": "TÊN_XÉT_NGHIỆM", "assertions": []})
        log.append(f"[autofix-culture-test] thêm TÊN_XÉT_NGHIỆM='{name}'")
    return fixed


def autofix_value_first_lab_pairs(entities, input_text, log):
    """Khôi phục lab viết theo thứ tự giá trị trước tên, ví dụ ``3.2 kali``."""
    fixed = list(entities)
    existing = Counter((entity["text"], entity["type"]) for entity in fixed)
    expected = Counter()

    for match in VALUE_FIRST_LAB_RE.finditer(input_text):
        value = match.group("value").strip()
        name = match.group("name").strip()
        expected[(value, "KẾT_QUẢ_XÉT_NGHIỆM")] += 1
        expected[(name, "TÊN_XÉT_NGHIỆM")] += 1

    for (text, entity_type), count in expected.items():
        missing = count - existing[(text, entity_type)]
        for _ in range(max(0, missing)):
            fixed.append({"text": text, "type": entity_type, "assertions": []})
            log.append(f"[autofix-value-first-lab] thêm {entity_type}='{text}'")
    return fixed


def autofix_parenthesized_value_first_lab_pairs(entities, input_text, log):
    """Khôi phục cặp dạng ``3.5 mmol/L (kali)`` hoặc ``4.1 (K+)``."""
    fixed = list(entities)
    existing = Counter((entity["text"], entity["type"]) for entity in fixed)
    expected = Counter()

    pair_matches = list(VALUE_PAREN_LAB_RE.finditer(input_text))
    for match in pair_matches:
        value = match.group("value").strip()
        name = match.group("name").strip()
        expected[(value, "KẾT_QUẢ_XÉT_NGHIỆM")] += 1
        expected[(name, "TÊN_XÉT_NGHIỆM")] += 1

    if pair_matches:
        for panel_match in ELECTROLYTE_PANEL_RE.finditer(input_text):
            expected[(panel_match.group(0), "TÊN_XÉT_NGHIỆM")] += 1

    for (text, entity_type), count in expected.items():
        missing = count - existing[(text, entity_type)]
        for _ in range(max(0, missing)):
            fixed.append({"text": text, "type": entity_type, "assertions": []})
            log.append(f"[autofix-parenthesized-value-first-lab] thêm {entity_type}='{text}'")
    return fixed


def autofix_merge_imaging_test_name(entities, input_text, log):
    """Gộp ``chụp`` + ``x-quang ...`` thành đúng một tên kỹ thuật hình ảnh."""
    fixed = []
    consumed = set()
    for index, entity in enumerate(entities):
        if index in consumed:
            continue
        if entity["type"] == "TÊN_XÉT_NGHIỆM" and entity["text"].strip().lower() == "chụp":
            start = input_text.find(entity["text"])
            for next_index, candidate in enumerate(entities):
                if next_index == index or candidate["type"] != "TÊN_XÉT_NGHIỆM":
                    continue
                if not re.match(r"^x[\s-]?quang\b", candidate["text"], re.IGNORECASE):
                    continue
                candidate_start = input_text.find(candidate["text"], start + len(entity["text"]))
                if start != -1 and candidate_start != -1:
                    between = input_text[start + len(entity["text"]):candidate_start]
                    if between.strip() == "":
                        merged = input_text[start:candidate_start + len(candidate["text"])]
                        fixed.append({**entity, "text": merged})
                        consumed.add(next_index)
                        log.append(f"[autofix-imaging-test-name] gộp thành '{merged}'")
                        break
            else:
                fixed.append(entity)
            continue
        fixed.append(entity)
    return fixed


def autofix_history_section_assertions(entities, section_key, log):
    """Bệnh nền/thuốc khẳng định trong section tiền sử phải mang isHistorical."""
    if section_key != "tien_su":
        return entities
    fixed = []
    for entity in entities:
        assertions = list(entity["assertions"])
        if (
            entity["type"] in ("CHẨN_ĐOÁN", "THUỐC")
            and "isNegated" not in assertions
            and "isHistorical" not in assertions
        ):
            assertions.append("isHistorical")
            log.append(f"[autofix-history-section] '{entity['text']}' -> thêm isHistorical")
        fixed.append({**entity, "assertions": assertions})
    return fixed


def autofix_known_diagnosis_recall(entities, input_text, log):
    """Khôi phục biến chứng chắc chắn bị bỏ sót vì chứa tên một thủ thuật."""
    fixed = list(entities)
    existing = Counter(
        entity["text"].lower() for entity in fixed if entity["type"] == "CHẨN_ĐOÁN"
    )
    for match in KNOWN_DIAGNOSIS_RECALL_RE.finditer(input_text):
        text = match.group(0)
        key = text.lower()
        if existing[key] > 0:
            existing[key] -= 1
            continue
        fixed.append({"text": text, "type": "CHẨN_ĐOÁN", "assertions": []})
        log.append(f"[autofix-diagnosis-recall] thêm CHẨN_ĐOÁN='{text}'")
    return fixed


def autofix_negated_history_context(entities, input_text, log, window=20):
    fixed = []
    for e in entities:
        if e["type"] in ("CHẨN_ĐOÁN", "TRIỆU_CHỨNG"):
            idx = input_text.find(e["text"])
            if idx != -1:
                context_before = input_text[max(0, idx - window): idx]
                if NEGATED_HISTORY_CONTEXT_RE.search(context_before):
                    if e["assertions"] != ["isNegated"]:
                        log.append(
                            f"[autofix-negated-history] '{e['text']}' theo sau 'không có tiền sử' -> ép assertions=['isNegated']"
                        )
                    e = {**e, "assertions": ["isNegated"]}
        fixed.append(e)
    return fixed


def autofix_missing_historical_marker(entities, input_text, log, window=80):
    """isFamily + isHistorical là combo HỢP LỆ (vd 'mẹ có tiền sử ung thư vú') nên KHÔNG
    còn loại trừ isFamily ở đây nữa -- chỉ tránh add nếu context đã bị phủ định (regex dưới)."""
    fixed = []
    for e in entities:
        if e["type"] == "CHẨN_ĐOÁN" and "isHistorical" not in e["assertions"]:
            idx = input_text.find(e["text"])
            if idx != -1:
                context_before = input_text[max(0, idx - window): idx]
                if NEGATED_HISTORY_CONTEXT_RE.search(context_before):
                    fixed.append(e)
                    continue
                if HISTORICAL_MARKER_RE.search(context_before):
                    log.append(
                        f"[autofix-missing-historical] '{e['text']}' có marker 'tiền sử/trước đây' ngay trước -> thêm isHistorical"
                    )
                    e = {**e, "assertions": sorted(set(e["assertions"]) | {"isHistorical"})}
        fixed.append(e)
    return fixed


def autofix_case_mismatch(entities, input_text, log):
    """
    Entity đúng khái niệm nhưng lệch hoa/thường so với input_text (thường gặp khi
    entity nằm ở đầu câu -- input_text viết hoa đầu câu, LLM trả về chữ thường).
    Sửa lại đúng theo case thật trong input_text thay vì reject oan.
    """
    fixed = []
    lower_input = input_text.lower()
    occurrence_cache = {}
    occurrence_used = Counter()
    for e in entities:
        # Ghép lần lượt từng occurrence của chính chuỗi này. Dùng bộ đếm riêng theo text để
        # không phụ thuộc entity list đã được sort hay chưa; hai entity "kali" sẽ map đúng
        # vào "Kali" rồi "kali" nếu input có cả hai mention thật.
        key = e["text"].lower()
        if key not in occurrence_cache:
            occurrence_cache[key] = [
                match.start() for match in re.finditer(re.escape(key), lower_input)
            ]
        occurrence_index = occurrence_used[key]
        matches = occurrence_cache[key]
        idx = matches[occurrence_index] if occurrence_index < len(matches) else -1
        if idx != -1:
            real_text = input_text[idx: idx + len(e["text"])]
            occurrence_used[key] += 1
            if real_text != e["text"]:
                log.append(f"[autofix-case-mismatch] '{e['text']}' -> '{real_text}'")
                e = {**e, "text": real_text}
        fixed.append(e)
    return fixed


def autofix_excess_exact_duplicates(entities, input_text, log):
    """Bỏ annotation y hệt vượt quá số occurrence thật thay vì reject cả sample."""
    allowed = Counter()
    for entity in entities:
        key = (entity["text"], entity["type"], tuple(sorted(entity["assertions"])))
        if key not in allowed:
            allowed[key] = input_text.count(entity["text"])

    used = Counter()
    fixed = []
    for entity in entities:
        key = (entity["text"], entity["type"], tuple(sorted(entity["assertions"])))
        # Không che lỗi hallucination/span: nếu text hoàn toàn không có trong nguồn thì giữ lại
        # để span_reject xử lý. Chỉ tự bỏ phần lặp thừa của một mention có thật.
        if allowed[key] == 0:
            fixed.append(entity)
            continue
        if used[key] >= allowed[key]:
            log.append(
                f"[autofix-excess-duplicate] bỏ entity thừa '{entity['text']}' "
                f"vì input chỉ có {allowed[key]} occurrence"
            )
            continue
        used[key] += 1
        fixed.append(entity)
    return fixed


# Thứ tự ưu tiên khi 1 entity bị gán >1 assertion cùng lúc -- CHỈ giữ lại 1, theo
# đúng dạng gold thật của BTC (100% entry trong ví dụ gold chỉ có 0 hoặc 1 assertion,
# KHÔNG BAO GIỜ 2+). Lý do thứ tự:
#   isNegated cao nhất   -- phủ định là sự thật quyết định nhất (có tồn tại hay không),
#                           quan trọng hơn việc "của ai" hay "khi nào".
#   isFamily thứ nhì     -- xác định chủ thể (người thân, không phải bệnh nhân),
#                           quan trọng hơn mốc thời gian.
#   isHistorical thấp nhất -- chỉ là mốc thời gian, ít quyết định nhất trong 3 cái.
# Quyết định (ĐỔI): thà để model dự đoán NHIỀU assertion rồi hậu xử lý chọn lại lúc infer,
# còn hơn train model chỉ biết đoán 1 -- nếu BTC chấm nhiều nhãn thì train-1-nhãn sẽ hụt
# recall không cứu được. Vì vậy KHÔNG còn ép single-label nữa. Chỉ chặn đúng 1 combo THẬT
# SỰ vô nghĩa: isHistorical + isNegated cùng lúc (vừa "đã xảy ra trong quá khứ" vừa "chưa
# từng xảy ra" -- mâu thuẫn logic, không phải do đề bài cho phép nhiều nhãn). Các combo khác
# (isFamily+isNegated: "không có tiền sử gia đình mắc X"; isFamily+isHistorical: "mẹ từng
# bị X") đều hợp lệ, giữ nguyên không đụng vào.
ASSERTION_PRIORITY = ["isNegated", "isFamily", "isHistorical"]  # dùng để quyết định giữ cái nào khi có mâu thuẫn


def autofix_historical_negated_conflict(entities, log):
    """
    isHistorical + isNegated cùng lúc là NGHỊCH LÝ DUY NHẤT thật sự vô nghĩa trong 3 assertion
    (khác isFamily+isNegated hay isFamily+isHistorical đều hợp lệ). Bỏ isHistorical, giữ
    isNegated + các assertion hợp lệ khác nếu có (vd isFamily vẫn giữ nguyên nếu cùng xuất hiện).
    """
    fixed = []
    for e in entities:
        if "isHistorical" in e["assertions"] and "isNegated" in e["assertions"]:
            log.append(
                f"[autofix-historical-negated-conflict] '{e['text']}' có cả isHistorical+isNegated "
                f"(nghịch lý) -> bỏ isHistorical, giữ {[a for a in e['assertions'] if a != 'isHistorical']}"
            )
            e = {**e, "assertions": [a for a in e["assertions"] if a != "isHistorical"]}
        fixed.append(e)
    return fixed


def autofix_g6pd_disease_type(entities, log):
    """Tên bệnh thiếu men G6PD không phải tên xét nghiệm; enzyme đứng riêng bị lọc sau."""
    fixed = []
    for entity in entities:
        if G6PD_DISEASE_RE.fullmatch(entity["text"].strip()) and entity["type"] in LAB_TYPES:
            log.append(
                f"[autofix-g6pd-disease-type] '{entity['text']}' {entity['type']} -> CHẨN_ĐOÁN"
            )
            entity = {**entity, "type": "CHẨN_ĐOÁN", "assertions": []}
        fixed.append(entity)
    return fixed


def autofix_full_bai_nao_span(entities, input_text, log):
    """Không giữ hai mảnh 'Bại'/'não' khi nguồn có đầy đủ chẩn đoán 'Bại não'."""
    matches = list(re.finditer(r"\bBại\s+não\b", input_text, re.IGNORECASE))
    if not matches:
        return entities
    fixed = [
        entity for entity in entities
        if entity["text"].strip().casefold() not in {"bại", "não"}
    ]
    existing = sum(
        1 for entity in fixed
        if entity["text"].strip().casefold() == "bại não"
    )
    for match in matches[existing:]:
        fixed.append({
            "text": match.group(0), "type": "CHẨN_ĐOÁN", "assertions": []
        })
    if len(fixed) != len(entities) or existing < len(matches):
        log.append("[autofix-full-bai-nao] thay span vụn bằng CHẨN_ĐOÁN='Bại não'")
    return fixed


def autofix_knowledge_assertions(entities, input_text, log, knowledge_context=False):
    """Kiến thức chung không trở thành tiền sử nếu không có chủ thể bệnh nhân rõ."""
    if not knowledge_context:
        return entities
    fixed = []
    cursor = Counter()
    for entity in entities:
        text = entity["text"]
        idx = input_text.find(text, cursor[text])
        if idx == -1:
            idx = input_text.find(text)
        if idx != -1:
            cursor[text] = idx + len(text)
        local = input_text[max(0, idx - 120):idx] if idx != -1 else ""
        assertions = list(entity["assertions"])
        if "isHistorical" in assertions and not EXPLICIT_PATIENT_HISTORY_RE.search(local):
            assertions = [value for value in assertions if value != "isHistorical"]
            log.append(
                f"[autofix-knowledge-assertion] '{text}' là kiến thức chung -> bỏ isHistorical"
            )
        fixed.append({**entity, "assertions": assertions})
    return fixed


def autofix_drug_regimen_span(entities, input_text, log):
    """Mở rộng THUỐC qua đường dùng/tần suất liền kề theo boundary gold BTC."""
    fixed = []
    cursor = Counter()
    for entity in entities:
        if entity["type"] != "THUỐC":
            fixed.append(entity)
            continue
        text = entity["text"]
        start_at = cursor[text]
        start = input_text.find(text, start_at)
        if start == -1:
            start = input_text.find(text)
        if start == -1:
            fixed.append(entity)
            continue
        cursor[text] = start + len(text)
        tail = input_text[start + len(text):start + len(text) + 140]
        match = DRUG_TRAILING_REGIMEN_RE.match(tail)
        if not match:
            fixed.append(entity)
            continue
        end = start + len(text) + match.end("regimen")
        expanded = input_text[start:end]
        if expanded != text:
            log.append(f"[autofix-drug-regimen-span] '{text}' -> '{expanded}'")
            entity = {**entity, "text": expanded}
        fixed.append(entity)
    return fixed


def autofix_drug_dose_only_span(entities, input_text, log):
    """Khôi phục ``Tylenol 1 gram`` khi LLM chỉ gán phần liều ``1 gram``."""
    fixed = []
    occupied_drug_spans = []
    blocked = {"uống", "dùng", "liều", "tiêm", "truyền", "khí", "dung"}
    for entity in entities:
        text = entity["text"]
        if entity["type"] != "THUỐC" or not DRUG_DOSE_ONLY_RE.fullmatch(text.strip()):
            fixed.append(entity)
            if entity["type"] == "THUỐC":
                for match in re.finditer(re.escape(text), input_text):
                    if all(match.end() <= start or match.start() >= end for start, end in occupied_drug_spans):
                        occupied_drug_spans.append(match.span())
                        break
            continue
        idx = -1
        for match in re.finditer(re.escape(text), input_text):
            if all(match.end() <= start or match.start() >= end for start, end in occupied_drug_spans):
                idx = match.start()
                break
        before_start = max(0, idx - 45)
        before = input_text[before_start:idx] if idx != -1 else ""
        name_match = re.search(r"(?P<name>[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9+\-]{2,40})\s+$", before)
        if not name_match or name_match.group("name").lower() in blocked:
            fixed.append(entity)
            continue
        name = name_match.group("name")
        name_start = before_start + name_match.start("name")
        expanded = input_text[name_start:idx + len(text)]
        fixed.append({**entity, "text": expanded})
        occupied_drug_spans.append((name_start, idx + len(text)))
        log.append(f"[autofix-drug-dose-only] '{text}' -> '{expanded}'")
    return fixed


def autofix_split_compound_drugs(entities, input_text, log):
    """Tách hai thuốc ghép bằng dấu ' / ' nhưng không đụng đơn vị MG/ML hay liều x2/ngày."""
    fixed = []
    for entity in entities:
        if entity["type"] != "THUỐC" or not COMPOUND_DRUG_SEPARATOR_RE.search(entity["text"]):
            fixed.append(entity)
            continue

        parts = [part.strip() for part in COMPOUND_DRUG_SEPARATOR_RE.split(entity["text"]) if part.strip()]
        if len(parts) < 2 or any(part not in input_text for part in parts):
            fixed.append(entity)
            continue

        for part in parts:
            fixed.append({**entity, "text": part})
        log.append(
            f"[autofix-split-compound-drug] '{entity['text']}' -> "
            + " + ".join(f"'{part}'" for part in parts)
        )
    return fixed


def autofix_strip_diagnosis_code(entities, log):
    """Mã ICD là candidate downstream, không phải một phần text span chẩn đoán."""
    fixed = []
    for entity in entities:
        if entity["type"] != "CHẨN_ĐOÁN":
            fixed.append(entity)
            continue
        clean_text = DIAGNOSIS_CODE_SUFFIX_RE.sub("", entity["text"]).strip()
        if clean_text and clean_text != entity["text"]:
            log.append(f"[autofix-strip-diagnosis-code] '{entity['text']}' -> '{clean_text}'")
            fixed.append({**entity, "text": clean_text})
        else:
            fixed.append(entity)
    return fixed


def autofix_explicit_drug_timing(entities, input_text, log, section_key=None, window=220):
    """Chỉ sửa assertion khi câu có tín hiệu thời gian thuốc rõ ràng."""
    fixed = []
    search_cursor = Counter()
    for entity in entities:
        if entity["type"] != "THUỐC":
            fixed.append(entity)
            continue

        text = entity["text"]
        start_at = search_cursor[text]
        idx = input_text.find(text, start_at)
        if idx == -1:
            idx = input_text.find(text)
        if idx != -1:
            search_cursor[text] = idx + len(text)
        context = input_text[max(0, idx - window):idx] if idx != -1 else ""
        # Assertion thời gian phải theo mệnh đề chứa thuốc. Dùng toàn bộ cửa sổ 220 ký tự
        # khiến "Trước đó ..." ở câu trước kéo nhầm thuốc hiện tại sang isHistorical.
        clause_context = re.split(r"[.;]", context)[-1]
        context_after = input_text[idx + len(text):idx + len(text) + 100] if idx != -1 else ""
        assertions = list(entity["assertions"])

        if (
            DRUG_HOME_CONTEXT_RE.search(clause_context)
            or DRUG_POST_HISTORICAL_CONTEXT_RE.search(context_after)
        ):
            if "isHistorical" not in assertions:
                assertions.append("isHistorical")
                log.append(f"[autofix-drug-home-timing] '{text}' -> thêm isHistorical")
        elif DRUG_CURRENT_CONTEXT_RE.search(clause_context) and "isHistorical" in assertions:
            assertions = [value for value in assertions if value != "isHistorical"]
            log.append(f"[autofix-drug-current-timing] '{text}' -> bỏ isHistorical")
        elif DRUG_ACTIVE_CUE_RE.search(clause_context):
            if section_key == "tien_su":
                if "isHistorical" not in assertions:
                    assertions.append("isHistorical")
                    log.append(f"[autofix-drug-history-section] '{text}' -> thêm isHistorical")
            elif "isHistorical" in assertions:
                assertions = [value for value in assertions if value != "isHistorical"]
                log.append(f"[autofix-drug-active-default] '{text}' -> bỏ isHistorical, giữ assertions=[]")

        fixed.append({**entity, "assertions": assertions})
    return fixed


def autofix_drug_clause_consistency(entities, input_text, log, section_key=None, window=160):
    """Các thuốc cùng một vế 'đang dùng A + B' phải có cùng assertion thời gian."""
    fixed = [dict(entity) for entity in entities]
    for cue in re.finditer(r"\bđang\s+(?:dùng|uống)\b", input_text, re.IGNORECASE):
        clause_end_match = re.search(r"[.;\n]", input_text[cue.end():])
        clause_end = cue.end() + clause_end_match.start() if clause_end_match else len(input_text)
        members = []
        for idx, entity in enumerate(fixed):
            if entity["type"] != "THUỐC":
                continue
            pos = input_text.find(entity["text"], cue.end(), clause_end)
            if pos != -1:
                members.append(idx)
        if len(members) < 2:
            continue

        context_to_cue = input_text[max(0, cue.end() - window):cue.end()]
        historical = section_key == "tien_su" or bool(DRUG_HOME_CONTEXT_RE.search(context_to_cue))
        for idx in members:
            assertions = [value for value in fixed[idx]["assertions"] if value != "isHistorical"]
            if historical:
                assertions.append("isHistorical")
            if assertions != fixed[idx]["assertions"]:
                log.append(
                    f"[autofix-drug-clause-consistency] '{fixed[idx]['text']}' -> {assertions}"
                )
                fixed[idx]["assertions"] = assertions
    return fixed


def run_autofix_pipeline(
    entities, input_text, log, section_key=None, knowledge_context=False,
):
    entities = autofix_strip_lab_assertion(entities, log)
    entities = autofix_case_mismatch(entities, input_text, log)
    entities = autofix_narrative_test_clauses(entities, input_text, log)
    entities = autofix_drug_dose_only_span(entities, input_text, log)
    entities = autofix_drug_regimen_span(entities, input_text, log)
    entities = autofix_split_compound_drugs(entities, input_text, log)
    entities = autofix_strip_diagnosis_code(entities, log)
    entities = autofix_explicit_drug_timing(entities, input_text, log, section_key=section_key)
    entities = autofix_drug_clause_consistency(entities, input_text, log, section_key=section_key)
    entities = autofix_negation_leak(entities, log)
    entities = autofix_trailing_symptom_connector(entities, input_text, log)
    entities = autofix_negation_scope(entities, input_text, log)
    entities = autofix_simple_lab_gold_boundary(entities, log)
    entities = autofix_single_lab_trend_pairs(entities, input_text, log)
    entities = autofix_split_merged_lab(entities, log)
    entities = autofix_dynamic_lab_result(entities, input_text, log)
    entities = autofix_named_measurement_pairs(entities, input_text, log)
    entities = autofix_common_lab_pairs(entities, input_text, log)
    entities = autofix_value_first_lab_pairs(entities, input_text, log)
    entities = autofix_parenthesized_value_first_lab_pairs(entities, input_text, log)
    entities = autofix_ordered_culture_tests(entities, input_text, log)
    entities = autofix_split_compressed_vitals(entities, log)
    entities = autofix_allergy_condition_type(entities, log)
    entities = autofix_g6pd_disease_type(entities, log)
    entities = autofix_diagnosis_context_in_span(entities, log)
    entities = autofix_current_hypoxia_type(entities, input_text, log)
    entities = autofix_diagnosis_timing(entities, input_text, log)
    entities = autofix_graded_imaging_finding(entities, input_text, log)
    entities = autofix_echo_valve_finding(entities, input_text, log)
    entities = autofix_pruritus_full_span(entities, input_text, log)
    entities = autofix_fever_full_span(entities, input_text, log)
    entities = autofix_full_known_symptom_spans(entities, input_text, log)
    entities = autofix_split_symptom_cause(entities, input_text, log)
    entities = autofix_merge_imaging_test_name(entities, input_text, log)
    entities = autofix_negated_history_context(entities, input_text, log)
    entities = autofix_missing_historical_marker(entities, input_text, log)
    entities = autofix_known_diagnosis_recall(entities, input_text, log)
    entities = autofix_full_bai_nao_span(entities, input_text, log)
    entities = autofix_history_section_assertions(entities, section_key, log)
    entities = autofix_invalid_family_assertion(entities, input_text, log)
    entities = autofix_historical_negated_conflict(entities, log)
    entities = autofix_knowledge_assertions(
        entities, input_text, log, knowledge_context=knowledge_context
    )
    entities = autofix_excess_exact_duplicates(entities, input_text, log)
    return entities


def check_medical_consistency(input_text):
    """Reject các cặp thuốc-chỉ định sai rõ ràng; không tự viết lại bệnh án nguồn."""
    if len(SECTION_HEADING_OCCURRENCE_RE.findall(input_text[:300])) > 1:
        raise SampleRejected(
            "Phát hiện heading section bị lặp ở đầu record; cần chỉ giữ đúng một heading",
            "text_quality_reject",
        )
    # Thiếu khoảng trắng nhẹ là augmentation có chủ đích. Không reject toàn text ở đây;
    # check_entity_sentence_boundary bên dưới bảo đảm không entity nào nuốt qua hai câu.
    if DANGEROUS_MEDICAL_UNIT_TYPO_RE.search(input_text):
        raise SampleRejected(
            "Phát hiện đơn vị 'inch' sai trong phân suất/hệ số tống máu; cần dùng EF %",
            "medical_inconsistency_reject",
        )
    for match in FEVER_VALUE_RE.finditer(input_text):
        value = float(match.group("value").replace(",", "."))
        unit = match.group("unit")
        if not unit:
            # Trong bệnh án Việt Nam, "sốt nhẹ 37.8" thường mặc định là °C và hoàn toàn
            # hợp lệ. Số nhỏ như "sốt 3 ngày" là thời lượng; số >=30 nhưng ngoài miền
            # Celsius hợp lý (vd 90/101) vẫn bị chặn vì không thể suy ra °F an toàn.
            if 37.0 <= value <= 43.5:
                continue
            if value >= 30:
                raise SampleRejected(
                    f"Nhiệt độ sốt không đơn vị và ngoài miền °C hợp lý: '{match.group(0)}'",
                    "medical_inconsistency_reject",
                )
            continue
        normalized_unit = unit.lower().replace(" ", "")
        is_fahrenheit = "f" in normalized_unit
        plausible = 99.0 <= value <= 110.0 if is_fahrenheit else 37.0 <= value <= 43.5
        if not plausible:
            raise SampleRejected(
                f"Nhiệt độ sốt không hợp lý: '{match.group(0)}'",
                "medical_inconsistency_reject",
            )
    if ELLIPTICAL_DIAGNOSIS_RE.search(input_text):
        raise SampleRejected(
            "Chẩn đoán tỉnh lược kiểu 'viêm gan B hoặc C' không cho phép tạo hai span/position độc lập; "
            "hãy viết đầy đủ 'viêm gan B hoặc viêm gan C'",
            "elliptical_concept_reject",
        )
    if ALLOPURINOL_LIPID_RE.search(input_text):
        raise SampleRejected(
            "Allopurinol bị gán điều trị rối loạn lipid máu; cần thuốc statin hoặc chỉ định tăng acid uric máu",
            "medical_inconsistency_reject",
        )
    if PERIODIC_PARALYSIS_RE.search(input_text) and RILEY_DAY_RE.search(input_text):
        raise SampleRejected(
            "Riley-Day là rối loạn thần kinh tự chủ gia đình, không phải liệt chu kỳ",
            "medical_inconsistency_reject",
        )
    if PERIODIC_PARALYSIS_RE.search(input_text) and PHENACEMIDE_RE.search(input_text):
        raise SampleRejected(
            "Phenacemide là thuốc chống co giật, không phải điều trị phù hợp cho liệt chu kỳ",
            "medical_inconsistency_reject",
        )
    if KIDNEY_STONE_TYPO_RE.search(input_text):
        raise SampleRejected(
            "Phát hiện typo y khoa 'sót thận' (nhiều khả năng là 'sỏi thận'); reject để sinh lại thay vì tự sửa input",
            "medical_inconsistency_reject",
        )


# ----------------------------------------------------------------------------
# Bước 6-7: lưới an toàn lab-assertion (A.7). ĐÃ BỎ check_multi_assertion (A.6)
# reject-cả-sample cũ -- gold thật của BTC xác nhận assertion là SINGLE-LABEL
# (0 hoặc 1, không bao giờ 2+), nên giờ dùng autofix_collapse_to_single_assertion
# để giữ lại đúng 1 assertion theo ASSERTION_PRIORITY thay vì reject cả sample.
# ----------------------------------------------------------------------------
def check_lab_assertion_safety_net(entities):
    for e in entities:
        if e["type"] in LAB_TYPES and e["assertions"]:
            raise SampleRejected(
                f"[BUG?] Lab-type entity vẫn còn assertion sau autofix: '{e['text']}' -> {e['assertions']}",
                "schema_reject",
            )


# ----------------------------------------------------------------------------
# Bước 8: REJECT ENTITY (tầng B) -- không reject cả sample
# ----------------------------------------------------------------------------
def reject_entity_treatment_purpose(entities, log):
    cleaned = []
    for e in entities:
        if e["type"] == "TRIỆU_CHỨNG" and TREATMENT_PURPOSE_RE.match(e["text"]):
            log.append(f"[reject-entity-treatment-purpose] '{e['text']}' là mục đích điều trị -> loại")
            continue
        cleaned.append(e)
    return cleaned


def reject_entity_procedure_as_diagnosis(entities, log):
    cleaned = []
    for e in entities:
        diagnostic_endoscopy = (
            e["type"] == "TÊN_XÉT_NGHIỆM"
            and re.match(r"^nội soi\b", e["text"], re.IGNORECASE)
        )
        if PROCEDURE_RE.match(e["text"]) and not diagnostic_endoscopy and not (
            e["type"] == "CHẨN_ĐOÁN" and PROCEDURE_COMPLICATION_RE.match(e["text"])
        ):
            log.append(f"[reject-entity-procedure] '{e['text']}' là thủ thuật/phẫu thuật -> loại")
            continue
        cleaned.append(e)
    return cleaned


def reject_entity_risk_factor(entities, log):
    cleaned = []
    for e in entities:
        if e["type"] in ("TRIỆU_CHỨNG", "CHẨN_ĐOÁN") and RISK_FACTOR_RE.search(e["text"]):
            log.append(f"[reject-entity-risk-factor] '{e['text']}' là yếu tố nguy cơ/lối sống -> loại")
            continue
        cleaned.append(e)
    return cleaned


def reject_entity_treatment_goal_context(entities, input_text, log):
    """Không suy bệnh/triệu chứng hiện hữu chỉ từ mục tiêu ``phòng ngừa X``/``giảm X``."""
    cleaned = []
    cursor = Counter()
    for entity in entities:
        text = entity["text"]
        start_at = cursor[text]
        idx = input_text.find(text, start_at)
        if idx == -1:
            idx = input_text.find(text)
        if idx != -1:
            cursor[text] = idx + len(text)
        before = input_text[max(0, idx - 40):idx] if idx != -1 else ""
        before = re.split(r"[.;\n,:]", before)[-1]
        if (
            entity["type"] in ("TRIỆU_CHỨNG", "CHẨN_ĐOÁN")
            and TREATMENT_GOAL_CONTEXT_RE.search(before)
        ):
            log.append(
                f"[reject-entity-treatment-goal] '{text}' chỉ là mục đích phòng ngừa/giảm -> loại"
            )
            continue
        cleaned.append(entity)
    return cleaned


def check_diagnosis_measurement_boundary(entities):
    """Chẩn đoán không được nuốt luôn trị số huyết áp/vitals."""
    invalid = [
        entity["text"] for entity in entities
        if entity["type"] == "CHẨN_ĐOÁN"
        and DIAGNOSIS_MEASUREMENT_RE.search(entity["text"])
    ]
    if invalid:
        raise SampleRejected(
            f"CHẨN_ĐOÁN nuốt trị số đo hoặc typo chỉ số: {invalid[:3]}",
            "entity_type_reject",
        )


def reject_entity_demographic_context(entities, input_text, log):
    cleaned = []
    cursor = 0
    for entity in entities:
        text = entity["text"]
        idx = input_text.find(text, cursor)
        if idx == -1:
            idx = input_text.find(text)
        if idx != -1:
            cursor = max(cursor, idx + len(text))
        before = input_text[max(0, idx - 30):idx] if idx != -1 else ""
        after = input_text[idx + len(text):idx + len(text) + 12] if idx != -1 else ""
        numeric_age = (
            entity["type"] == "KẾT_QUẢ_XÉT_NGHIỆM"
            and re.fullmatch(r"\d{1,3}", text)
            and re.search(r"(?:\bBN|bệnh nhân)\s+(?:nam|nữ)\s*$", before, re.IGNORECASE)
            and re.match(r"\s*(?:t\b|tuổi\b)", after, re.IGNORECASE)
        )
        numeric_duration = (
            entity["type"] == "KẾT_QUẢ_XÉT_NGHIỆM"
            and re.fullmatch(r"\d+(?:[\.,]\d+)?", text)
            and re.match(r"\s*(?:giây|phút|giờ|ngày|tuần|tháng|năm)\b", after, re.IGNORECASE)
        )
        full_duration = re.fullmatch(
            r"\d+(?:[\.,]\d+)?\s*(?:giây|phút|giờ|ngày|tuần|tháng|năm)",
            text.strip(),
            re.IGNORECASE,
        )
        if DEMOGRAPHIC_ENTITY_RE.fullmatch(text) or numeric_age or numeric_duration or full_duration:
            log.append(
                f"[reject-entity-demographic] '{text}' là tuổi/thời lượng/giới/nghề nghiệp -> loại"
            )
            continue
        cleaned.append(entity)
    return cleaned


def reject_entity_non_medical_concept(entities, input_text, log):
    cleaned = []
    for entity in entities:
        text = entity["text"]
        idx = input_text.find(text)
        after = input_text[idx + len(text):idx + len(text) + 35] if idx != -1 else ""
        if GENERIC_BIOMEDICAL_ENTITY_RE.fullmatch(text):
            log.append(f"[reject-entity-generic-biomedical] '{text}' là từ/chất sinh học chung -> loại")
            continue
        if BARE_CELL_ENTITY_RE.fullmatch(text) and not (
            entity["type"] == "TÊN_XÉT_NGHIỆM"
            and re.match(r"\s*:?\s*[<>≤≥]?\s*\d", after)
        ):
            log.append(f"[reject-entity-bare-cell] '{text}' đứng riêng không phải khái niệm đích -> loại")
            continue
        if NON_SCHEMA_SUBSTANCE_RE.fullmatch(text):
            log.append(f"[reject-entity-non-schema-substance] '{text}' là tác nhân/thiết bị, không phải thuốc -> loại")
            continue
        if entity["type"] == "KẾT_QUẢ_XÉT_NGHIỆM" and text.strip().lower() == "không rõ":
            log.append(
                f"[reject-entity-unknown-context] '{text}' không phải kết quả xét nghiệm"
            )
            continue
        if entity["type"] == "TÊN_XÉT_NGHIỆM" and GENERIC_TEST_HEADING_RE.fullmatch(text):
            log.append(
                f"[reject-entity-generic-test-heading] '{text}' là heading, không phải kỹ thuật cụ thể"
            )
            continue
        if entity["type"] in ("TRIỆU_CHỨNG", "CHẨN_ĐOÁN") and NON_CONDITION_ENTITY_RE.fullmatch(text):
            log.append(f"[reject-entity-non-condition] '{text}' là tác nhân/context, không phải bệnh -> loại")
            continue
        if entity["type"] == "THUỐC" and (
            NON_DRUG_TREATMENT_RE.match(text)
            or DRUG_REGIMEN_ONLY_RE.fullmatch(text)
            or DRUG_DOSE_ONLY_RE.fullmatch(text)
            or NON_CLINICAL_DRUG_RE.fullmatch(text)
        ):
            log.append(f"[reject-entity-non-drug] '{text}' không phải thuốc lâm sàng phù hợp -> loại")
            continue
        if NOISY_FRAGMENT_RE.fullmatch(text):
            log.append(f"[reject-entity-fragment] '{text}' là mảnh từ vô nghĩa -> loại")
            continue
        if INCOMPLETE_ENTITY_END_RE.search(text) and len(text.split()) <= 5:
            log.append(f"[reject-entity-incomplete] '{text}' kết thúc bằng mảnh nối chưa đủ nghĩa -> loại")
            continue
        if entity["type"] == "TÊN_XÉT_NGHIỆM" and BARE_ANATOMY_TEST_RE.fullmatch(text):
            log.append(f"[reject-entity-bare-anatomy-test] '{text}' chỉ là cơ quan -> loại")
            continue
        cleaned.append(entity)
    return cleaned


def reject_entity_procedure_numeric_value(entities, input_text, log):
    """Không giữ lượng dịch/thể tích thao tác như 0.5cc làm kết quả xét nghiệm."""
    cleaned = []
    for entity in entities:
        if entity["type"] != "KẾT_QUẢ_XÉT_NGHIỆM":
            cleaned.append(entity)
            continue
        idx = input_text.find(entity["text"])
        before = input_text[max(0, idx - 30):idx] if idx != -1 else ""
        numeric_value = re.fullmatch(
            r"\d+(?:[\.,]\d+)?(?:\s*(?:cc|ml|mL))?", entity["text"].strip(), re.IGNORECASE
        )
        if numeric_value and PROCEDURE_VALUE_CONTEXT_RE.search(before):
            log.append(
                f"[reject-entity-procedure-value] '{entity['text']}' là lượng dịch của thủ thuật -> loại"
            )
            continue
        cleaned.append(entity)
    return cleaned


def reject_entity_vitals_mistyped(entities, log):
    cleaned = []
    for e in entities:
        if e["type"] == "TRIỆU_CHỨNG" and VITALS_LIKE_RE.match(e["text"]):
            log.append(f"[reject-entity-vitals-mistype] '{e['text']}' là vitals bị gán nhầm TRIỆU_CHỨNG -> loại")
            continue
        cleaned.append(e)
    return cleaned


def reject_entity_family_mismatch(entities, input_text, log, window=40):
    cleaned = []
    for e in entities:
        if "isFamily" in e["assertions"]:
            idx = input_text.find(e["text"])
            context_before = input_text[max(0, idx - window): idx].lower() if idx != -1 else ""
            has_family_kw = any(kw in context_before for kw in FAMILY_KEYWORDS)
            has_exposure_kw = any(kw in context_before for kw in EXPOSURE_KEYWORDS)
            if has_exposure_kw and not has_family_kw:
                log.append(
                    f"[reject-entity-family-mismatch] '{e['text']}' gán isFamily nhưng ngữ cảnh là 'tiếp xúc/người khác' -> loại"
                )
                continue
        cleaned.append(e)
    return cleaned


def reject_entity_leaked_context(entities, log):
    cleaned = []
    for e in entities:
        if LEAKED_CONTEXT_RE.search(e["text"]):
            log.append(f"[reject-entity-leaked-context] '{e['text']}' chứa chủ ngữ/tuổi/số thứ tự thừa -> loại")
            continue
        cleaned.append(e)
    return cleaned


def run_entity_reject_pipeline(entities, input_text, log):
    entities = reject_entity_treatment_purpose(entities, log)
    entities = reject_entity_treatment_goal_context(entities, input_text, log)
    entities = reject_entity_procedure_as_diagnosis(entities, log)
    entities = reject_entity_risk_factor(entities, log)
    entities = reject_entity_demographic_context(entities, input_text, log)
    entities = reject_entity_non_medical_concept(entities, input_text, log)
    entities = reject_entity_procedure_numeric_value(entities, input_text, log)
    entities = reject_entity_vitals_mistyped(entities, log)
    entities = reject_entity_family_mismatch(entities, input_text, log)
    entities = reject_entity_leaked_context(entities, log)
    return entities


def check_entity_sentence_boundary(entities):
    """Dữ liệu có thể dính câu, nhưng gold span không được kéo qua ranh giới đó."""
    crossing = [
        entity["text"] for entity in entities
        if MISSING_SENTENCE_SPACE_RE.search(entity["text"])
    ]
    if crossing:
        raise SampleRejected(
            f"Entity kéo qua hai khái niệm/câu bị dính: {crossing[:3]}",
            "entity_boundary_reject",
        )


def sort_entities_by_text_order(entities, input_text, log):
    """Sắp entity theo offset xuất hiện để position/BIO mapping ổn định."""
    test_result_ranges = []
    for pattern in (
        IMAGING_NARRATIVE_CLAUSE_RE,
        QUALITATIVE_TEST_CLAUSE_RE,
        QUALITATIVE_LAB_CLAUSE_RE,
        HOLTER_NARRATIVE_CLAUSE_RE,
        ECG_NARRATIVE_CLAUSE_RE,
    ):
        for match in pattern.finditer(input_text):
            result_group = (
                "result" if match.groupdict().get("result") is not None
                else "negative_result"
            )
            if match.groupdict().get(result_group) is not None:
                test_result_ranges.append(match.span(result_group))

    positions_by_text = {}
    for entity in entities:
        text = entity["text"]
        if text in positions_by_text:
            continue
        positions = []
        cursor = 0
        while True:
            idx = input_text.find(text, cursor)
            if idx == -1:
                break
            positions.append(idx)
            cursor = idx + max(1, len(text))
        positions_by_text[text] = positions

    used = Counter()
    decorated = []
    for original_index, entity in enumerate(entities):
        text = entity["text"]
        occurrence_key = (text, entity["type"])
        occurrence_index = used[occurrence_key]
        used[occurrence_key] += 1
        positions = positions_by_text.get(text, [])
        # Nếu span ngắn xuất hiện bên trong một entity dài ở occurrence trước, đừng map
        # nhầm nó vào đó khi còn occurrence độc lập phía sau. Ví dụ tên bệnh ngắn
        # "thiếu vitamin A" trong "thiếu vitamin A với đốm Bitot..." rồi được nhắc lại.
        shadowed_positions = []
        for position in positions:
            for other_text, other_positions in positions_by_text.items():
                if len(other_text) <= len(text):
                    continue
                if any(
                    other_position <= position
                    and position + len(text) <= other_position + len(other_text)
                    for other_position in other_positions
                ):
                    shadowed_positions.append(position)
                    break
        independent_positions = [
            position for position in positions if position not in shadowed_positions
        ]
        if independent_positions:
            positions = independent_positions
        positions_in_test_result = [
            position
            for position in positions
            if any(
                result_start <= position
                and position + len(text) <= result_end
                for result_start, result_end in test_result_ranges
            )
        ]
        if entity["type"] == "KẾT_QUẢ_XÉT_NGHIỆM" and positions_in_test_result:
            positions = positions_in_test_result
        elif entity["type"] in {"CHẨN_ĐOÁN", "TRIỆU_CHỨNG"}:
            positions_outside_test = [
                position for position in positions if position not in positions_in_test_result
            ]
            if positions_outside_test:
                positions = positions_outside_test
        if entity["type"] == "KẾT_QUẢ_XÉT_NGHIỆM" and re.fullmatch(r"\d{1,3}", text):
            non_age_positions = []
            for position in positions:
                before = input_text[max(0, position - 30):position]
                after = input_text[position + len(text):position + len(text) + 12]
                is_age = (
                    re.search(r"(?:\bBN|bệnh nhân)\s+(?:nam|nữ)\s*$", before, re.IGNORECASE)
                    and re.match(r"\s*(?:t\b|tuổi\b)", after, re.IGNORECASE)
                )
                if not is_age:
                    non_age_positions.append(position)
            if non_age_positions:
                positions = non_age_positions
        start = positions[occurrence_index] if occurrence_index < len(positions) else len(input_text)
        decorated.append((start, original_index, entity))

    sorted_entities = [entity for _, _, entity in sorted(decorated, key=lambda item: (item[0], item[1]))]
    if [id(entity) for entity in sorted_entities] != [id(entity) for entity in entities]:
        log.append("[autofix-sort-entities] sắp entities theo thứ tự xuất hiện trong input_text")
    return sorted_entities


# ----------------------------------------------------------------------------
# Bước 9: overlap check (A.9) -- gán span tuần tự theo thứ tự entities xuất
# hiện trong list (giả định list đã theo thứ tự xuất hiện trong text, đúng như
# cách generate_data.py/LLM xuất ra). Đây là heuristic, không phải offset thật.
# ----------------------------------------------------------------------------
def assign_spans_sequential(entities, input_text):
    spans = []
    cursor = 0
    for e in entities:
        idx = input_text.find(e["text"], cursor)
        if idx == -1:
            idx = input_text.find(e["text"])
        if idx == -1:
            return None
        start, end = idx, idx + len(e["text"])
        spans.append((start, end, e))
        cursor = max(cursor, end)
    return spans


def build_entity_span_metadata(entities, input_text):
    """Materialize the exact occurrence selected by QC for BIO conversion.

    Entity text alone is ambiguous when the same mention appears multiple times
    with different assertions. Keeping these offsets beside the legacy entity
    list prevents the BIO converter from assigning an annotation to a different
    occurrence later.
    """
    spans = assign_spans_sequential(entities, input_text)
    if spans is None or len(spans) != len(entities):
        raise SampleRejected(
            "Không materialize được đầy đủ entity offsets.",
            "position_materialization_reject",
        )

    metadata = []
    previous_end = -1
    for start, end, entity in spans:
        if start < previous_end or input_text[start:end] != entity["text"]:
            raise SampleRejected(
                f"Offset không xác định/overlap cho entity '{entity['text']}'.",
                "position_materialization_reject",
            )
        # Hai annotation có thể không overlap theo ký tự nhưng vẫn cùng nằm
        # trong một token số (vd tách "034" thành "03" và "4"). BIO
        # word-level không thể biểu diễn trường hợp này.
        if (
            (start > 0 and input_text[start - 1].isdigit() and input_text[start].isdigit())
            or (
                end < len(input_text)
                and input_text[end - 1].isdigit()
                and input_text[end].isdigit()
            )
        ):
            raise SampleRejected(
                f"Entity số '{entity['text']}' cắt giữa một token số.",
                "token_boundary_reject",
            )
        metadata.append({
            "char_start": start,
            "char_end": end,
            "text": entity["text"],
            "type": entity["type"],
        })
        previous_end = end
    return metadata


def resolve_nested_overlaps(entities, input_text, log):
    """Giữ span đầy đủ hơn khi một entity nằm trọn trong entity khác cùng vị trí."""
    spans = assign_spans_sequential(entities, input_text)
    if spans is None:
        return entities

    drop_indexes = set()
    for left in range(len(spans)):
        s1, e1, ent1 = spans[left]
        for right in range(left + 1, len(spans)):
            s2, e2, ent2 = spans[right]
            if e1 <= s2 or e2 <= s1:
                continue
            if s1 <= s2 and e2 <= e1 and (s1, e1) != (s2, e2):
                drop_indexes.add(right)
                log.append(
                    f"[autofix-nested-overlap] giữ '{ent1['text']}', loại span con '{ent2['text']}'"
                )
            elif s2 <= s1 and e1 <= e2 and (s1, e1) != (s2, e2):
                drop_indexes.add(left)
                log.append(
                    f"[autofix-nested-overlap] giữ '{ent2['text']}', loại span con '{ent1['text']}'"
                )

    return [entity for idx, entity in enumerate(entities) if idx not in drop_indexes]


def check_overlap(entities, input_text):
    spans = assign_spans_sequential(entities, input_text)
    if spans is None:
        return  # không xác định được offset -> bỏ qua, đã được span_reject bắt ở bước 3
    spans_sorted = sorted(spans, key=lambda s: s[0])
    for (s1, e1, ent1), (s2, e2, ent2) in zip(spans_sorted, spans_sorted[1:]):
        if s2 < e1:
            raise SampleRejected(
                f"Overlap span: '{ent1['text']}' <-> '{ent2['text']}'", "overlap_reject"
            )


# ----------------------------------------------------------------------------
# Bước 10: duplicate entity y hệt (A.10)
# ----------------------------------------------------------------------------
def check_duplicates(entities, input_text):
    key_counts = Counter()
    for e in entities:
        key = (e["text"], e["type"], tuple(sorted(e["assertions"])))
        key_counts[key] += 1
    for key, cnt in key_counts.items():
        if cnt > 1:
            text = key[0]
            occurs = input_text.count(text)
            if occurs < cnt:
                raise SampleRejected(
                    f"Duplicate entity y hệt: '{text}' lặp {cnt} lần trong entities nhưng "
                    f"chỉ xuất hiện {occurs} lần trong input_text",
                    "duplicate_reject",
                )


# ----------------------------------------------------------------------------
# Bước 11: section "Đánh giá tại bệnh viện" thiếu lab type (A.11)
# ----------------------------------------------------------------------------
def check_missing_lab_pair(entities, section_key, required=True):
    if not required or section_key != "danh_gia":
        return

    has_lab_name = any(e["type"] == "TÊN_XÉT_NGHIỆM" for e in entities)
    has_lab_result = any(e["type"] == "KẾT_QUẢ_XÉT_NGHIỆM" for e in entities)

    if not has_lab_name or not has_lab_result:
        raise SampleRejected(
            "Section 'Đánh giá tại bệnh viện' thiếu TÊN_XÉT_NGHIỆM hoặc KẾT_QUẢ_XÉT_NGHIỆM",
            "missing_lab_pair_reject",
        )


# ----------------------------------------------------------------------------
# Bước 12: force_assertion (A.12) -- chỉ áp dụng nếu record có field này
# (record thô từ pipeline gen, KHÔNG có trong train.jsonl đã strip system_prompt)
# ----------------------------------------------------------------------------
def check_force_assertion(entities, parsed_record):
    force_assertion = parsed_record.get("force_assertion")
    if not force_assertion:
        return
    has_it = any(force_assertion in e["assertions"] for e in entities)
    if not has_it:
        raise SampleRejected(
            f"force_assertion='{force_assertion}' được yêu cầu nhưng sample không có assertion này",
            "missing_force_assertion_reject",
        )


# ----------------------------------------------------------------------------
# Bước 13: quá ít entity sau khi lọc (A.8)
# ----------------------------------------------------------------------------
def check_min_entities(entities, min_entities=2):
    if len(entities) < min_entities:
        raise SampleRejected(
            f"Sau khi clean chỉ còn {len(entities)} entity (< {min_entities})", "too_few_entity_reject"
        )


# ----------------------------------------------------------------------------
# Pipeline chính -- 2 entry point:
#   - process_record(record: dict, ...)   nhận THẲNG 1 dict đã parse sẵn
#     (dùng để gọi trực tiếp trong generate_data.py, ngay sau khi LLM trả về
#     json.loads(raw) cho 1 sample, KHÔNG cần ghi/đọc file)
#   - process_sample(raw_line: str, ...)  nhận 1 dòng text thô (dùng khi lọc
#     hàng loạt từ file .jsonl, xem hàm main() bên dưới)
# ----------------------------------------------------------------------------
def process_record(record, raw_repr=None, attempt=None, model=None):
    """
    record: dict đã parse sẵn, tối thiểu có {"input_text": str, "entities": [...]}.
            Có thể có thêm "force_assertion" (nếu gọi từ generate_data.py, nơi biết
            trước force_assertion của sample đang gen) -- nếu không có thì bỏ qua
            check A.12.
    raw_repr: string thô để lưu vào field "raw" trong log reject (vd raw response
              của LLM trước khi json.loads) -- không bắt buộc, chỉ để debug.

    Trả về:
        ("keep", clean_record, applied_logs)   -- applied_logs: list các auto-fix/
                                                   reject-entity đã áp dụng (str)
        ("reject", reject_log_dict)
    """
    autofix_log = []
    entity_reject_log = []

    if not isinstance(record, dict):
        return "reject", {
            "stage": "schema_reject",
            "reason": "record không phải dict",
            "section": None,
            "force_assertion": None,
            "input_text": None,
            "raw": raw_repr,
            "parsed_entities": None,
            "cleaned_entities": None,
            "attempt": attempt,
            "model": model,
        }

    input_text = record.get("input_text")
    raw_entities = record.get("entities")

    if not input_text or not isinstance(input_text, str):
        return "reject", {
            "stage": "schema_reject",
            "reason": "Thiếu hoặc rỗng input_text",
            "section": None,
            "force_assertion": record.get("force_assertion"),
            "input_text": input_text,
            "raw": raw_repr,
            "parsed_entities": raw_entities,
            "cleaned_entities": None,
            "attempt": attempt,
            "model": model,
        }
    if raw_entities is None or not isinstance(raw_entities, list):
        return "reject", {
            "stage": "schema_reject",
            "reason": "Thiếu hoặc sai kiểu entities",
            "section": None,
            "force_assertion": record.get("force_assertion"),
            "input_text": input_text,
            "raw": raw_repr,
            "parsed_entities": raw_entities,
            "cleaned_entities": None,
            "attempt": attempt,
            "model": model,
        }

    detected_section_key, section_heading = detect_section(input_text)
    section_key_hint = record.get("_section_key_hint")
    knowledge_context = bool(record.get("_knowledge_context", False))
    section_key = (
        section_key_hint
        if section_key_hint in {"tien_su", "hien_tai", "danh_gia"}
        else detected_section_key
    )
    force_assertion = record.get("force_assertion")
    parsed_entities_snapshot = raw_entities

    try:
        # --- bước 2: type + field check ---
        entities = normalize_and_check_type(raw_entities)

        # --- bước 3: assertion enum check (chạy sớm, không cần input_text) ---
        check_assertion_enum(entities)

        # Cặp thuốc-chỉ định sai chắc chắn làm nhiễu dữ liệu dù BIO vẫn hợp lệ.
        check_medical_consistency(input_text)

        # --- bước 4: auto-fix (ĐỔI: chạy TRƯỚC span filter, để entity lệch case/
        # còn dính "không"/gộp lab... được sửa trước khi bị đánh giá substring) ---
        entities = run_autofix_pipeline(
            entities,
            input_text,
            autofix_log,
            section_key=section_key,
            knowledge_context=knowledge_context,
        )

        # --- bước 5: span filter cấp ENTITY (ĐỔI: không còn reject cả sample,
        # xem comment ở filter_entity_span) ---
        entities = filter_entity_span(entities, input_text, entity_reject_log)
        check_entity_sentence_boundary(entities)
        check_diagnosis_measurement_boundary(entities)

        # --- bước 6: lưới an toàn lab-assertion (multi-assertion reject-sample
        # cũ đã bị xoá -- xem comment ở check_lab_assertion_safety_net) ---
        check_lab_assertion_safety_net(entities)

        # --- bước 8: reject entity (không reject sample) ---
        entities = run_entity_reject_pipeline(entities, input_text, entity_reject_log)

        # BTC trả entity theo thứ tự position; sort trước duplicate/overlap và trước khi ghi output.
        entities = sort_entities_by_text_order(entities, input_text, autofix_log)
        entities = resolve_nested_overlaps(entities, input_text, entity_reject_log)

        # --- bước 9: duplicate (chạy TRƯỚC overlap: case "cùng text lặp lại nhưng
        # thực ra chỉ xuất hiện 1 lần trong input_text vì lệch hoa/thường ở lần thứ
        # 2" sẽ bị fallback tìm-lại-từ-đầu ở check_overlap tạo overlap giả -- báo lý
        # do "duplicate" ở đây rõ ràng và đúng bản chất hơn "overlap") ---
        check_duplicates(entities, input_text)

        # --- bước 10: overlap span thật (2 entity khác nhau lồng nhau) ---
        check_overlap(entities, input_text)

        # --- bước 11: missing lab pair theo section (bắt buộc có CẢ 2) ---
        check_missing_lab_pair(
            entities,
            section_key,
            required=record.get("_require_lab_pair", True),
        )

        # --- bước 12: force_assertion (bỏ qua nếu record không có field này) ---
        check_force_assertion(entities, record)

        # --- bước 13: quá ít entity ---
        check_min_entities(entities, min_entities=record.get("_min_entities", 2))

        # Khoá occurrence sau khi mọi autofix/sort/overlap check hoàn tất.
        # Lỗi tại đây phải đi qua cùng luồng reject, không làm dừng cả batch.
        entity_spans = build_entity_span_metadata(entities, input_text)

    except SampleRejected as ex:
        return "reject", {
            "stage": ex.stage,
            "reason": ex.reason,
            "section": section_heading,
            "force_assertion": force_assertion,
            "input_text": input_text,
            "raw": raw_repr,
            "parsed_entities": parsed_entities_snapshot,
            "cleaned_entities": locals().get("entities"),
            "attempt": attempt,
            "model": model,
        }

    clean_record = {
        "input_text": input_text,
        "entities": [
            {"text": e["text"], "type": e["type"], "assertions": e["assertions"]}
            for e in entities
        ],
        "entity_spans": entity_spans,
    }
    return "keep", clean_record, autofix_log + entity_reject_log


def process_sample(raw_line, attempt=None, model=None):
    """
    Wrapper cho input là 1 DÒNG TEXT THÔ (vd đọc từ file .jsonl) -- parse ra dict
    rồi gọi process_record(). Dùng trong hàm main()/CLI của file này.
    """
    try:
        parsed = json.loads(raw_line)
    except json.JSONDecodeError as e:
        return "reject", {
            "stage": "schema_reject",
            "reason": f"JSON sai format / parse lỗi: {e}",
            "section": None,
            "force_assertion": None,
            "input_text": None,
            "raw": raw_line.rstrip("\n"),
            "parsed_entities": None,
            "cleaned_entities": None,
            "attempt": attempt,
            "model": model,
        }
    return process_record(parsed, raw_repr=raw_line.rstrip("\n"), attempt=attempt, model=model)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Lọc data NER y tế tiếng Việt (reject/autofix/keep).")
    ap.add_argument("--input", default="train.jsonl", help="File .jsonl đầu vào")
    ap.add_argument("--clean-output", default="clean.jsonl", help="File .jsonl output (sample hợp lệ)")
    ap.add_argument("--reject-output", default="reject.jsonl", help="File .jsonl log các sample bị reject")
    ap.add_argument("--model", default=None, help="Tên model dùng để gen (chỉ để log vào reject.jsonl)")
    ap.add_argument("--verbose", action="store_true", help="In log autofix/reject-entity ra stdout")
    args = ap.parse_args()

    in_path = Path(args.input)
    lines = in_path.read_text(encoding="utf-8").splitlines()

    n_keep = 0
    n_reject = 0
    reject_reason_counter = Counter()

    with open(args.clean_output, "w", encoding="utf-8") as f_clean, \
         open(args.reject_output, "w", encoding="utf-8") as f_reject:

        for line_no, raw_line in enumerate(lines, start=1):
            if not raw_line.strip():
                continue

            result = process_sample(raw_line, model=args.model)

            if result[0] == "keep":
                _, clean_record, applied_logs = result
                f_clean.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
                n_keep += 1
                if args.verbose and applied_logs:
                    for l in applied_logs:
                        print(f"  [line {line_no}] {l}")
            else:
                _, reject_log = result
                reject_log["line_no"] = line_no
                f_reject.write(json.dumps(reject_log, ensure_ascii=False) + "\n")
                n_reject += 1
                reject_reason_counter[reject_log["stage"]] += 1
                if args.verbose:
                    print(f"  [line {line_no}] REJECT ({reject_log['stage']}): {reject_log['reason']}")

    total = n_keep + n_reject
    print(f"\n[DONE] {total} sample -> giữ {n_keep}, reject {n_reject}")
    if reject_reason_counter:
        print("Phân bố lý do reject (theo stage):")
        for stage, cnt in reject_reason_counter.most_common():
            print(f"  {stage}: {cnt}")
    print(f"\nClean data  -> {args.clean_output}")
    print(f"Reject log  -> {args.reject_output}")


if __name__ == "__main__":
    main()
