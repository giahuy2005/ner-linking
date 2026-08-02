"""Strict response shape for ontology candidate selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CandidateSelection:
    chosen_codes: list[str]

    @classmethod
    def from_dict(cls, value: object) -> "CandidateSelection | None":
        if not isinstance(value, dict):
            return None
        codes = value.get("chosen_codes")
        if not isinstance(codes, list) or not all(isinstance(code, str) for code in codes):
            return None
        return cls(chosen_codes=list(dict.fromkeys(codes)))
