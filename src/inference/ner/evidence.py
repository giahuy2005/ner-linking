"""Typed, serialisable evidence emitted by detailed NER inference.

The BTC output remains deliberately small.  These structures are internal and
make the CRF/span/local evidence available to candidate generation, the locked
editor, evaluation, and audit logs without leaking debug fields to submission
JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..schemas import NerEntity


@dataclass(frozen=True)
class CrfMarginalEvidence:
    decoded_tag: str
    repaired_tag: str
    probabilities: dict[str, float] = field(default_factory=dict)
    entity_mass: float = 0.0
    top_non_o_label: str | None = None
    top_non_o_probability: float = 0.0
    method: str = "exact_forward_backward"


@dataclass(frozen=True)
class SpanCandidateEvidence:
    start: int
    end: int
    type: str
    score: float
    word_start: int | None = None
    word_end: int | None = None
    source: str = "span_head"


@dataclass(frozen=True)
class LocalVerificationEvidence:
    start: int
    end: int
    type: str
    score: float
    decision: str = "UNRESOLVED"
    region_id: str | None = None


@dataclass(frozen=True)
class WordEvidence:
    index: int
    text: str
    start: int
    end: int
    line_id: int
    block_id: str | int | None = None
    crf: CrfMarginalEvidence | None = None
    span_top_label: str | None = None
    span_top_score: float = 0.0


@dataclass
class NerDetailedResult:
    raw_text_length: int
    clean_text_length: int
    crf_entities: list[NerEntity] = field(default_factory=list)
    span_candidates: list[SpanCandidateEvidence] = field(default_factory=list)
    lattice_entities: list[NerEntity] = field(default_factory=list)
    final_entities: list[NerEntity] = field(default_factory=list)
    words: list[WordEvidence] = field(default_factory=list)
    local_verifications: list[LocalVerificationEvidence] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)
    span_head_enabled: bool = False
    marginal_method: str = "exact_forward_backward"

    def validate_offsets(self, raw_text: str) -> None:
        if len(raw_text) != self.raw_text_length:
            raise ValueError("raw text length changed during detailed inference")
        for entity in self.final_entities:
            start, end = entity.position
            if not (0 <= start < end <= len(raw_text)):
                raise ValueError(f"invalid entity position: {entity.position}")
            if raw_text[start:end] != entity.text:
                raise ValueError(
                    f"entity text/offset mismatch: {entity.text!r} at {entity.position}"
                )
            if "\n" in entity.text or "\r" in entity.text:
                raise ValueError(f"entity crosses a line boundary: {entity.text!r}")

    def to_audit_dict(self) -> dict[str, Any]:
        def entity_dict(entity: NerEntity) -> dict[str, Any]:
            return {
                "text": entity.text,
                "type": entity.type,
                "assertions": list(entity.assertions),
                "position": list(entity.position),
                "score": float(entity.score),
                "flag": entity.flag,
            }

        return {
            "raw_text_length": self.raw_text_length,
            "clean_text_length": self.clean_text_length,
            "span_head_enabled": self.span_head_enabled,
            "marginal_method": self.marginal_method,
            "thresholds": dict(self.thresholds),
            "crf_entities": [entity_dict(item) for item in self.crf_entities],
            "span_candidates": [asdict(item) for item in self.span_candidates],
            "lattice_entities": [entity_dict(item) for item in self.lattice_entities],
            "final_entities": [entity_dict(item) for item in self.final_entities],
            "words": [asdict(item) for item in self.words],
            "local_verifications": [asdict(item) for item in self.local_verifications],
            "logs": list(self.logs),
        }

    @classmethod
    def from_audit_dict(cls, value: dict[str, Any]) -> "NerDetailedResult":
        """Restore a portable saved artifact for LLM-only benchmarking."""
        def entity(row: dict[str, Any]) -> NerEntity:
            return NerEntity(
                row["text"], row["type"], list(row.get("assertions", [])),
                tuple(row["position"]), float(row.get("score", 1.0)), row.get("flag"),
            )

        def word(row: dict[str, Any]) -> WordEvidence:
            crf_row = row.get("crf")
            crf = CrfMarginalEvidence(**crf_row) if isinstance(crf_row, dict) else None
            restored = dict(row); restored["crf"] = crf
            return WordEvidence(**restored)

        return cls(
            raw_text_length=int(value["raw_text_length"]),
            clean_text_length=int(value["clean_text_length"]),
            crf_entities=[entity(row) for row in value.get("crf_entities", [])],
            span_candidates=[SpanCandidateEvidence(**row) for row in value.get("span_candidates", [])],
            lattice_entities=[entity(row) for row in value.get("lattice_entities", [])],
            final_entities=[entity(row) for row in value.get("final_entities", [])],
            words=[word(row) for row in value.get("words", [])],
            local_verifications=[LocalVerificationEvidence(**row) for row in value.get("local_verifications", [])],
            thresholds={key: float(number) for key, number in value.get("thresholds", {}).items()},
            logs=list(value.get("logs", [])),
            span_head_enabled=bool(value.get("span_head_enabled", False)),
            marginal_method=str(value.get("marginal_method", "exact_forward_backward")),
        )
