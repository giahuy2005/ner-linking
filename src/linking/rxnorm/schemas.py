"""Dataclass dùng chung cho pipeline linking RxNorm. Không chứa logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedDrugMention:
    raw_text: str
    normalized_text: str
    ingredient_core: str | None = None
    strengths: list[str] = field(default_factory=list)
    strength_role: str = "missing"  # missing | single | range | ratio
    dose_forms: list[str] = field(default_factory=list)
    release_types: list[str] = field(default_factory=list)
    quantity: str | None = None
    route: str | None = None
    frequency: str | None = None
    interval_hours: int | None = None


@dataclass
class RxNormCandidate:
    rxcui: str
    tty: str
    tier: str
    name: str

    dense_score: float = 0.0
    lexical_score: float = 0.0
    final_score: float = 0.0

    active: bool = True
    historical: bool = False
    candidate_priority: int = 99

    exact_term_match: bool = False
    exact_ingredient_match: bool = False

    matched_terms: list[str] = field(default_factory=list)
    retrieval_sources: list[str] = field(default_factory=list)
    current_rxcuis: list[str] = field(default_factory=list)

    structured: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
