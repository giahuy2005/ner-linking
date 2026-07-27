"""Parse JSON từ output LLM 1 cách khoan dung.

LLM hay bọc output trong ```json ... ```, thêm giải thích thừa trước/sau,
hoặc để trailing comma. KHÔNG silent-fail: trả None nếu parse thất bại ở
mọi candidate, để caller tự quyết định retry/fallback thay vì âm thầm
dùng dict rỗng.
"""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_BLOCK_RE = re.compile(r"[\{\[].*[\}\]]", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def extract_json(text: str) -> dict | list | None:
    candidates: list[str] = []

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidates.append(fence_match.group(1))

    block_match = _JSON_BLOCK_RE.search(text)
    if block_match:
        candidates.append(block_match.group(0))

    candidates.append(text)

    for cand in candidates:
        cleaned = _TRAILING_COMMA_RE.sub(r"\1", cand.strip())
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            continue

    return None