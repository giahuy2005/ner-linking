"""Shape JSON kỳ vọng từ mỗi task LLM.

json_guard.extract_json() parse được KHÔNG có nghĩa đúng shape (LLM có
thể trả JSON hợp lệ nhưng thiếu field/sai giá trị enum) — from_dict() ở
đây validate riêng, trả None nếu sai shape để caller fallback an toàn
thay vì crash hoặc dùng dữ liệu rác.
"""

from __future__ import annotations

from dataclasses import dataclass

_NER_FIX_ACTIONS = {"keep", "drop", "retype", "retrim"}


@dataclass
class NerFixSuggestion:
    text: str
    type: str
    action: str  # "keep" | "drop" | "retype" | "retrim"

    @classmethod
    def from_dict(cls, d: object) -> "NerFixSuggestion | None":
        if not isinstance(d, dict):
            return None
        if not all(k in d for k in ("text", "type", "action")):
            return None
        if d["action"] not in _NER_FIX_ACTIONS:
            return None
        return cls(text=str(d["text"]), type=str(d["type"]), action=str(d["action"]))


@dataclass
class CandidateSelection:
    chosen_codes: list[str]
    reason: str = ""

    @classmethod
    def from_dict(cls, d: object) -> "CandidateSelection | None":
        if not isinstance(d, dict) or "chosen_codes" not in d:
            return None
        codes = d["chosen_codes"]
        if not isinstance(codes, list):
            return None
        return cls(chosen_codes=[str(c) for c in codes], reason=str(d.get("reason", "")))