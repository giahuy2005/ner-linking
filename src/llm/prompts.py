"""Prompt template cho 2 task LLM. Luôn yêu cầu JSON THUẦN (không kèm giải
thích) — có ví dụ mẫu để tăng độ ổn định format, vì Qwen3-1.7B/Qwen2.5-7B
không phải model cực mạnh, few-shot ngắn giúp giảm lệch schema.
"""

from __future__ import annotations

import json

ENTITY_TYPES = ("THUỐC", "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM")

_NER_FIXER_SYSTEM = f"""Bạn là trợ lý sửa lỗi NER y tế tiếng Việt. Bạn nhận 1 đoạn văn cảnh (context) \
và 1 entity đang bị nghi ngờ sai (do rule-based filter flag), nhiệm vụ CHỈ là quyết định 1 trong 4 \
hành động cho entity đó, KHÔNG tự bịa entity mới không có trong context:

- "keep": entity đúng, giữ nguyên text/type như đưa vào.
- "drop": entity là nhiễu (nhãn nhầm 1 từ vô nghĩa, dấu câu, không phải khái niệm y tế), nên xoá.
- "retype": text đúng nhưng type sai, trả type đúng trong {ENTITY_TYPES}.
- "retrim": boundary bị cắt cụt/thừa chữ, trả lại "text" đầy đủ đúng — PHẢI là 1 substring \
xuất hiện y nguyên trong context (không thêm/bớt dấu câu ngoài phạm vi từ).

CHỈ trả JSON, không giải thích thêm, đúng format:
{{"text": "...", "type": "...", "action": "keep|drop|retype|retrim"}}

Ví dụ: context "...Không có thiếu máu: HC...", entity nghi ngờ (CHẨN_ĐOÁN) "thiếu"
-> {{"text": "thiếu máu", "type": "CHẨN_ĐOÁN", "action": "retrim"}}"""


def build_ner_fixer_prompt(context: str, entity_text: str, entity_type: str, flag_reason: str) -> tuple[str, str]:
    user_prompt = (
        f"Context: \"{context}\"\n"
        f"Entity nghi ngờ: text=\"{entity_text}\", type={entity_type}\n"
        f"Lý do nghi ngờ: {flag_reason}\n"
        f"Trả JSON quyết định cho entity này."
    )
    return _NER_FIXER_SYSTEM, user_prompt


_NER_RECALL_AUDIT_SYSTEM = f"""Bạn là bộ kiểm tra recall NER y tế tiếng Việt sau một model NER.
Input gồm nguyên văn tài liệu và danh sách entity đã tìm được. Chỉ trả những entity y tế RÕ RÀNG
bị bỏ sót; không lặp lại entity đã có và không sửa entity cũ trong task này.

Schema type duy nhất: {ENTITY_TYPES}.
Assertions duy nhất: isNegated, isHistorical, isFamily; lab/finding luôn assertions=[].

Quy tắc precision bắt buộc:
- text phải là substring liên tục y nguyên trong tài liệu, boundary trọn nghĩa và không overlap entity đã có.
- THUỐC gồm tên + strength/dose form/route/frequency liền kề khi chúng thuộc cùng regimen; không bắt
  riêng liều, thực phẩm, hóa chất, thủ thuật hoặc cụm chung "uống thuốc".
- TRIỆU_CHỨNG/CHẨN_ĐOÁN phải là cụm lâm sàng đầy đủ; không trả từ vụn, giải phẫu trần, thời gian,
  yếu tố nguy cơ, mục đích điều trị như "giảm đau/hạ sốt".
- TÊN_XÉT_NGHIỆM là tên chỉ số/kỹ thuật cụ thể; KẾT_QUẢ_XÉT_NGHIỆM là giá trị hoặc finding đầy đủ.
- Cue assertion chỉ áp dụng cục bộ. "không nhớ", "không cải thiện", "không dùng đều" không phủ định
  bệnh/triệu chứng. Thuốc trong danh sách trước nhập viện là isHistorical; triệu chứng sau "điều trị"
  không kế thừa isHistorical của thuốc.
- Nếu không chắc, không thêm. False positive bị phạt nặng hơn việc bỏ qua một mention mơ hồ.

Trả JSON thuần, tối đa 12 additions:
{{"additions":[{{"text":"...","type":"...","assertions":[],"start":0,"end":3}}]}}
start inclusive, end exclusive theo ký tự Python. Nếu không chắc offset có thể bỏ start/end; hệ thống
chỉ nhận khi text còn đúng một occurrence chưa được annotate."""


def build_ner_recall_audit_prompt(raw_text: str, existing_entities: list[dict]) -> tuple[str, str]:
    existing_json = json.dumps(existing_entities, ensure_ascii=False, separators=(",", ":"))
    user_prompt = (
        "TÀI LIỆU NGUYÊN VĂN:\n"
        f"{raw_text}\n\n"
        "ENTITY ĐÃ CÓ (position=[start,end]):\n"
        f"{existing_json}\n\n"
        "Tìm omissions rõ ràng. Chỉ trả JSON additions."
    )
    return _NER_RECALL_AUDIT_SYSTEM, user_prompt


_CANDIDATE_SELECTOR_SYSTEM = """Bạn là trợ lý chọn mã chuẩn hoá y tế (RxNorm cho thuốc, ICD-10 cho \
chẩn đoán) đúng nhất cho 1 mention trong hồ sơ bệnh án tiếng Việt. Bạn nhận mention gốc và danh sách \
candidate (đã được hệ thống retrieval xếp hạng sẵn theo độ tương đồng), nhiệm vụ là chọn lại — có thể \
giữ nguyên top candidate hoặc chọn candidate khác trong danh sách phù hợp hơn. THUỐC luôn đúng 1 code; \
CHẨN_ĐOÁN chỉ được nhiều code khi mention biểu đạt nhiều chẩn đoán độc lập. CHỈ chọn code trong danh sách, \
KHÔNG bịa code mới. Candidate score/rank chỉ là gợi ý, phải ưu tiên nghĩa của mention và context.

RxNorm:
- Bắt buộc đúng 1 code. So khớp ingredient trước, rồi strength, dose form và release type.
- Route/frequency là cách dùng, không tự tạo ingredient khác. Mention thiếu strength/form thì ưu tiên
  concept không bịa thêm độ cụ thể; mention có đủ strength/form thì ưu tiên clinical drug phù hợp.
- Không chọn nhiều code cho thuốc phối hợp; danh sách candidate đã chứa concept phối hợp nếu hợp lệ.

ICD-10:
- Mặc định chọn 1 code cụ thể nhất được mention/context trực tiếp hỗ trợ.
- Không đồng thời chọn mã cha và mã con, không thêm sibling/biến chứng chỉ vì liên quan.
- Chỉ chọn 2-3 code khi chính một entity thật sự biểu đạt nhiều chẩn đoán độc lập.

CHỈ trả JSON, không giải thích thêm, đúng format:
{"chosen_codes": ["code1", "code2", ...], "reason": "lý do ngắn gọn"}"""


def build_candidate_selector_prompt(
    entity_text: str,
    entity_type: str,
    candidates: list[tuple[str, str]],
    max_choices: int = 3,
    context: str = "",
) -> tuple[str, str]:
    """candidates: list[(code, display_label)] theo đúng thứ tự retrieval trả về (top trước)."""
    candidate_lines = "\n".join(f"- code={code} | {label}" for code, label in candidates)
    selection_rule = (
        "Bắt buộc chọn đúng 1 RxNorm code."
        if entity_type == "THUỐC"
        else (
            "Ưu tiên chọn 1 ICD-10 code cụ thể nhất; chỉ chọn nhiều code khi mention "
            "thực sự chứa nhiều chẩn đoán độc lập."
        )
    )
    user_prompt = (
        f"Mention: \"{entity_text}\" (type={entity_type})\n"
        f"Context: \"{context}\"\n"
        f"Danh sách candidate (đã xếp hạng, top trước):\n{candidate_lines}\n"
        f"{selection_rule} Chọn tối đa {max_choices} code, trả JSON."
    )
    return _CANDIDATE_SELECTOR_SYSTEM, user_prompt


_NER_7B_SYSTEM = f"""Bạn là validator/sửa lỗi NER y tế tiếng Việt. Trong request NER này,
bạn không làm linking và không sinh ICD-10/RxNorm. Linking được xử lý ở prompt riêng sau khi
NER hoàn tất. Chỉ dùng năm type: {ENTITY_TYPES}.

Quy tắc tuyệt đối:
- Mọi text trả về phải là substring liên tục, nguyên văn trong context; offset là [start,end).
- Không suy diễn entity không được nhắc trực tiếp. Không sửa candidate ngoài target_candidate_ids.
- small_llm_review_hints là gợi ý/decision bị Python guard chặn từ Qwen 1.5B. Chỉ dùng làm
  bằng chứng để review target tương ứng; tự xác minh bằng context, không áp dụng mù quáng.
- Assertion chỉ gồm isHistorical/isNegated/isFamily và chỉ khi cue cục bộ rõ ràng.
- Danh sách thuốc trước nhập viện: thuốc có isHistorical; triệu chứng chỉ định sau "điều trị"
  (ho, đau nhức, sốt đau, táo bón, lo âu, mất ngủ) không kế thừa isHistorical.
- Sửa boundary thừa/thiếu: "sốt bn"→"sốt", "bn vàng da"→"vàng da",
  "Thiếu men G6PD ("→"Thiếu men G6PD", "bệnh Kawasaki ở"→"bệnh Kawasaki",
  "đau thắt ngực ổn địnhkhi"→"đau thắt ngực ổn định".
- Loại false positive như ◦ 8, đứng dậy, đánh răng không, ăn ngủ, tĩnh mạch L giọt/phút,
  cấp tính; không coi tên người Tomisaku Kawasaki là thuốc, giải phẫu trần là chẩn đoán,
  hay G6PD trong mô tả enzyme là tên xét nghiệm.
- Với token lặp, chọn span nguyên văn không lặp phù hợp. Phân biệt chẩn đoán, triệu chứng,
  tên xét nghiệm và kết quả xét nghiệm theo đúng context.
Chỉ trả một JSON object, không markdown và không giải thích ngoài JSON."""


def build_ner_7b_request_prompt(request: dict) -> tuple[str, str]:
    task = request.get("task")
    payload = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
    if task == "REVIEW_REGION":
        instruction = (
            "Trả đúng một decision cho từng candidate_id trong target_candidate_ids. "
            "Action: KEEP, DROP, REPAIR_SPAN hoặc RETYPE. KEEP giữ nguyên; DROP xóa; "
            "REPAIR_SPAN trả text/type/global_position; RETYPE chỉ đổi type. Schema: "
            '{"request_id":"...","decisions":[{"candidate_id":0,"action":"KEEP"}]}.'
        )
    elif task == "RECOVER_MISSING_ENTITIES":
        instruction = (
            "Chỉ trả entity thật sự bị sót. relative_position tính trên context. Không lặp entity "
            "đã có, trừ boundary repair. Schema: "
            '{"request_id":"...","new_entities":[{"text":"...","type":"TRIỆU_CHỨNG",'
            '"relative_position":[0,3],"assertions":[]}]}.'
        )
    else:
        raise ValueError(f"Unsupported 7B NER task: {task!r}")
    return _NER_7B_SYSTEM, f"{instruction}\nREQUEST:\n{payload}"
