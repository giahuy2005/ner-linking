"""Prompt template cho 2 task LLM. Luôn yêu cầu JSON THUẦN (không kèm giải
thích) — có ví dụ mẫu để tăng độ ổn định format, vì Qwen3-1.7B/Qwen2.5-7B
không phải model cực mạnh, few-shot ngắn giúp giảm lệch schema.
"""

from __future__ import annotations

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


_CANDIDATE_SELECTOR_SYSTEM = """Bạn là trợ lý chọn mã chuẩn hoá y tế (RxNorm cho thuốc, ICD-10 cho \
chẩn đoán) đúng nhất cho 1 mention trong hồ sơ bệnh án tiếng Việt. Bạn nhận mention gốc và danh sách \
candidate (đã được hệ thống retrieval xếp hạng sẵn theo độ tương đồng), nhiệm vụ là chọn lại — có thể \
giữ nguyên top candidate, chọn candidate khác trong danh sách phù hợp hơn, hoặc chọn NHIỀU candidate \
nếu mention thực sự khớp nhiều mã (vd thuốc phối hợp). CHỈ được chọn code có trong danh sách đưa vào, \
KHÔNG bịa code mới.

CHỈ trả JSON, không giải thích thêm, đúng format:
{"chosen_codes": ["code1", "code2", ...], "reason": "lý do ngắn gọn"}"""


def build_candidate_selector_prompt(
    entity_text: str, entity_type: str, candidates: list[tuple[str, str]], max_choices: int = 3,
) -> tuple[str, str]:
    """candidates: list[(code, display_label)] theo đúng thứ tự retrieval trả về (top trước)."""
    candidate_lines = "\n".join(f"- code={code} | {label}" for code, label in candidates)
    user_prompt = (
        f"Mention: \"{entity_text}\" (type={entity_type})\n"
        f"Danh sách candidate (đã xếp hạng, top trước):\n{candidate_lines}\n"
        f"Chọn tối đa {max_choices} code đúng nhất, trả JSON."
    )
    return _CANDIDATE_SELECTOR_SYSTEM, user_prompt