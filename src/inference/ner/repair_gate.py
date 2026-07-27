"""Lọc sơ sơ (rule-based) output NER trước khi đưa qua linking.

Rule ở đây được rút trực tiếp từ noise thấy trong test case thật của bạn
(paste trong lúc trao đổi), KHÔNG phải rule tổng quát đoán mò:

  [TÊN_XÉT_NGHIỆM] ':' [469, 470]                -> entity chỉ có dấu câu
  [KẾT_QUẢ_XÉT_NGHIỆM] ')' [639, 640]            -> entity chỉ có dấu câu
  [TÊN_XÉT_NGHIỆM] 'Trụ niệu (–' [628, 639]       -> ngoặc mở không đóng,
                                                      tràn sang entity kế
  [TRIỆU_CHỨNG] 'chiều' [351, 356]                -> "giảm về chiều": chiều
                                                      (buổi chiều) bị bắt
                                                      nhầm thành triệu chứng
  [TRIỆU_CHỨNG] 'lan' [267, 270]                  -> "đau không lan": từ
                                                      đơn lẻ, không mang
                                                      nghĩa lâm sàng
  [CHẨN_ĐOÁN] 'thiếu' [813, 818]                  -> "Không có thiếu máu":
                                                      bị cắt cụt, mất "máu"
  [CHẨN_ĐOÁN] 'da' [978, 980]                     -> "viêm nhiễm ngoài da":
                                                      bị cắt cụt, chỉ còn
                                                      danh từ bộ phận cơ thể

Mỗi rule dưới đây map thẳng 1-1 với 1 nhóm case ở trên. Rule KHÔNG tự
đoán thêm case chưa thấy trong data thật — mở rộng danh sách khi bạn có
thêm case cụ thể, tránh over-filter làm tụt recall.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Rule 1: entity chỉ toàn dấu câu / không có 1 chữ cái hay chữ số nào
# (bắt case ':' , ')' , '-' , '–' đứng riêng thành 1 entity).
# ---------------------------------------------------------------------------
_HAS_ALNUM_RE = re.compile(r"[0-9A-Za-zÀ-ỹ]")


def _is_punct_only(text: str) -> bool:
    return not _HAS_ALNUM_RE.search(text)


# ---------------------------------------------------------------------------
# Rule 2: entity kết thúc bằng ngoặc/gạch chưa đóng ("Trụ niệu (–") — trim
# phần đuôi không cân bằng thay vì drop cả entity, vì phần đầu vẫn đúng.
# ---------------------------------------------------------------------------
_TRAILING_UNBALANCED_RE = re.compile(r"[\s]*[\(\[\{–\-]+\s*$")


def _trim_unbalanced_trailing(text: str, char_start: int, char_end: int):
    if text.count("(") <= text.count(")") and text.count("[") <= text.count("]"):
        m = _TRAILING_UNBALANCED_RE.search(text)
        if not m:
            return text, char_start, char_end
        # chỉ trim nếu phần đuôi đúng là ngoặc/gạch mồ côi (không có "(" khớp
        # với ")" nào phía trước nó trong entity) — check lại cho chắc.
    new_text = _TRAILING_UNBALANCED_RE.sub("", text)
    if new_text == text or not new_text:
        return text, char_start, char_end
    trimmed_len = len(text) - len(new_text)
    return new_text, char_start, char_end - trimmed_len


# ---------------------------------------------------------------------------
# Rule 3: entity 1 từ, thuộc loại TRIỆU_CHỨNG/CHẨN_ĐOÁN, trùng danh sách
# từ đã QUAN SÁT THẤY là false positive (từ nối/thời gian, không phải
# thuật ngữ lâm sàng khi đứng một mình).
# ---------------------------------------------------------------------------
_SINGLE_WORD_NOISE = {
    "TRIỆU_CHỨNG": {"chiều", "sáng", "lan", "và", "mà", "đã", "đang", "rồi",
                     "thì", "này", "đó", "nên", "còn", "cũng", "khi", "sau", "trước"},
    "CHẨN_ĐOÁN": {"chiều", "sáng", "và", "mà", "đã", "đang", "rồi",
                  "thì", "này", "đó", "nên", "còn", "cũng", "khi", "sau", "trước"},
}


def _is_single_word_noise(text: str, ent_type: str) -> bool:
    stripped = text.strip().lower()
    if " " in stripped or "_" in stripped:
        return False
    return stripped in _SINGLE_WORD_NOISE.get(ent_type, set())


# ---------------------------------------------------------------------------
# Rule 4: entity 1 từ CHẨN_ĐOÁN chỉ còn lại danh từ bộ phận cơ thể trần
# (không kèm bệnh danh) — dấu hiệu bị BIO cắt cụt như 'da' từ "ngoài da",
# 'thiếu' từ "thiếu máu". Đây là rule NGỜ VỰC (flag), không tự drop, vì
# 1 từ bộ phận cơ thể đứng riêng đôi khi vẫn hợp lệ tùy văn cảnh — trả về
# để bạn tự quyết định log/drop khi review.
# ---------------------------------------------------------------------------
_SUSPECT_TRUNCATED_DIAGNOSIS = {"da", "gan", "thận", "phổi", "tim", "thiếu", "suy"}


def _is_suspect_truncated(text: str, ent_type: str) -> bool:
    stripped = text.strip().lower()
    if ent_type != "CHẨN_ĐOÁN":
        return False
    return stripped in _SUSPECT_TRUNCATED_DIAGNOSIS


def filter_entities(entities: list[dict], *, drop_suspect_truncated: bool = False):
    """Áp toàn bộ rule lên list entity dict (shape giống final_results của
    engine: text/type/assertions/position=[start,end]).

    Trả về (kept, dropped) — dropped kèm lý do để bạn log/debug, KHÔNG
    silent drop.
    """
    kept: list[dict] = []
    dropped: list[dict] = []

    for ent in entities:
        text = ent["text"]
        ent_type = ent["type"]
        start, end = ent["position"]

        # Rule 1
        if _is_punct_only(text):
            dropped.append({**ent, "reason": "punct_only"})
            continue

        # Rule 2 (trim, không drop)
        new_text, new_start, new_end = _trim_unbalanced_trailing(text, start, end)
        if new_text != text:
            if not new_text.strip() or _is_punct_only(new_text):
                dropped.append({**ent, "reason": "unbalanced_bracket_emptied"})
                continue
            ent = {**ent, "text": new_text, "position": [new_start, new_end]}
            text = new_text

        # Rule 3
        if _is_single_word_noise(text, ent_type):
            dropped.append({**ent, "reason": "single_word_stopword"})
            continue

        # Rule 4 (tùy chọn, mặc định chỉ log không drop)
        if _is_suspect_truncated(text, ent_type):
            if drop_suspect_truncated:
                dropped.append({**ent, "reason": "suspect_truncated_diagnosis"})
                continue
            ent = {**ent, "flag": "suspect_truncated_diagnosis"}

        kept.append(ent)

    return kept, dropped