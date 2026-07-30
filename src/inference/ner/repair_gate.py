"""Surface-agnostic validation gate for raw NER output.

The gate only rejects structurally impossible candidates and flags low model
confidence. It intentionally contains no symptom, diagnosis, anatomy, or
private-output vocabulary; contextual decisions belong to the 1.5B/7B stages.
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


LOW_CONFIDENCE_THRESHOLD = 0.80


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

        # Very short spans are reviewed, never dropped here. This catches
        # truncated BIO pieces without maintaining a whitelist that would miss
        # valid unseen abbreviations or short symptoms.
        # ``drop_suspect_truncated`` remains in the signature for CLI/API
        # compatibility but no vocabulary-based "suspect" class is created.
        compact_length = len(re.sub(r"\s+", "", text.strip()))
        if compact_length <= 2:
            ent = {**ent, "flag": "short_span_review"}
        elif float(ent.get("score", 1.0)) < LOW_CONFIDENCE_THRESHOLD:
            ent = {**ent, "flag": "low_emission_confidence"}

        kept.append(ent)

    return kept, dropped
