"""Strict response shape for ontology candidate selection V2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CandidateSelection:
    chosen_codes: list[str]
    request_id: str = ""
    decision: str = "SELECT"
    confidence: str = "LOW"
    reason_code: str = "AMBIGUOUS"

    @classmethod
    def from_dict(cls, value: object) -> "CandidateSelection | None":
        if not isinstance(value, dict):
            return None
        codes = value.get("chosen_codes")
        if not isinstance(codes, list) or not all(isinstance(code, str) for code in codes):
            return None
        # Backward-compatible parsing is intentionally limited to old cached/test
        # rows; every newly built V2 prompt requires the fields below.
        if "decision" not in value:
            return cls(chosen_codes=list(dict.fromkeys(codes)))
        request_id = value.get("request_id")
        decision = value.get("decision")
        confidence = value.get("confidence")
        reason = value.get("reason_code")
        if not isinstance(request_id, str) or decision not in {"SELECT", "ABSTAIN", "UNRESOLVED"}:
            return None
        if confidence not in {"HIGH", "MEDIUM", "LOW"} or reason not in {
            "EXACT_MATCH", "STRUCTURED_MATCH", "CONTEXT_DISAMBIGUATION",
            "INSUFFICIENT_EVIDENCE", "AMBIGUOUS",
        }:
            return None
        if decision != "SELECT" and codes:
            return None
        return cls(list(dict.fromkeys(codes)), request_id, decision, confidence, reason)
