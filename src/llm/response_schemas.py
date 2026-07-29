"""Shape JSON kỳ vọng từ mỗi task LLM.

json_guard.extract_json() parse được KHÔNG có nghĩa đúng shape (LLM có
thể trả JSON hợp lệ nhưng thiếu field/sai giá trị enum) — from_dict() ở
đây validate riêng, trả None nếu sai shape để caller fallback an toàn
thay vì crash hoặc dùng dữ liệu rác.
"""

from __future__ import annotations

from dataclasses import dataclass

_NER_FIX_ACTIONS = {"keep", "drop", "retype", "retrim"}
_ENTITY_TYPES = {
    "THUỐC",
    "TRIỆU_CHỨNG",
    "CHẨN_ĐOÁN",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
}
_ASSERTION_TYPES = {"isNegated", "isHistorical", "isFamily"}


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
        if d["type"] not in _ENTITY_TYPES:
            return None
        return cls(text=str(d["text"]), type=str(d["type"]), action=str(d["action"]))


@dataclass
class NerAuditSuggestion:
    text: str
    type: str
    assertions: list[str]
    start: int | None = None
    end: int | None = None

    @classmethod
    def from_dict(cls, d: object) -> "NerAuditSuggestion | None":
        if not isinstance(d, dict):
            return None
        if not all(key in d for key in ("text", "type", "assertions")):
            return None
        text = d["text"]
        entity_type = d["type"]
        assertions = d["assertions"]
        if not isinstance(text, str) or not text.strip() or entity_type not in _ENTITY_TYPES:
            return None
        if not isinstance(assertions, list) or any(
            assertion not in _ASSERTION_TYPES for assertion in assertions
        ):
            return None
        start = d.get("start")
        end = d.get("end")
        if start is not None and type(start) is not int:
            return None
        if end is not None and type(end) is not int:
            return None
        if (start is None) != (end is None):
            return None
        return cls(
            text=text,
            type=entity_type,
            assertions=list(dict.fromkeys(assertions)),
            start=start,
            end=end,
        )


@dataclass
class NerAuditResponse:
    additions: list[NerAuditSuggestion]

    @classmethod
    def from_dict(cls, d: object) -> "NerAuditResponse | None":
        if not isinstance(d, dict) or not isinstance(d.get("additions"), list):
            return None
        additions = []
        for item in d["additions"]:
            parsed = NerAuditSuggestion.from_dict(item)
            if parsed is None:
                return None
            additions.append(parsed)
        return cls(additions=additions)


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
