"""Stable candidate catalog and closed missing-entity proposals."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable

from ..schemas import NerEntity, VALID_ENTITY_TYPES
from .evidence import NerDetailedResult


def _digest(parts: Iterable[str]) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def stable_candidate_id(
    record_id: str, start: int, end: int, entity_type: str, sources: Iterable[str]
) -> str:
    source_key = ",".join(sorted(set(sources)))
    return f"cand_{_digest((record_id, str(start), str(end), entity_type, source_key))}"


def stable_proposal_id(
    record_id: str, start: int, end: int, allowed_types: Iterable[str], sources: Iterable[str]
) -> str:
    return f"prop_{_digest((record_id, str(start), str(end), ','.join(sorted(allowed_types)), ','.join(sorted(set(sources)))))}"


@dataclass
class CandidateEvidence:
    candidate_id: str
    text: str
    type: str
    position: tuple[int, int]
    sources: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    allowed_types: list[str] = field(default_factory=list)
    supports: list[str] = field(default_factory=list)
    hard_supports: list[str] = field(default_factory=list)
    negative_flags: list[str] = field(default_factory=list)
    related_candidate_ids: list[str] = field(default_factory=list)
    assertions: list[str] = field(default_factory=list)
    pre_llm_selected: bool = False

    @property
    def strong_consensus(self) -> bool:
        return "crf" in self.sources and "span_head" in self.sources and min(
            self.scores.get("crf", 0.0), self.scores.get("span_head", 0.0)
        ) >= 0.80


@dataclass
class MissingProposal:
    proposal_id: str
    text: str
    position: tuple[int, int]
    allowed_types: list[str]
    supports: list[str]
    hard_supports: list[str]
    negative_flags: list[str] = field(default_factory=list)
    auto_add_eligible: bool = False
    related_candidate_ids: list[str] = field(default_factory=list)


def _exact(raw_text: str, start: int, end: int, text: str) -> bool:
    return 0 <= start < end <= len(raw_text) and raw_text[start:end] == text and not any(
        separator in text for separator in ("\n", "\r")
    )


def build_candidate_catalog(
    record_id: str, raw_text: str, detailed: NerDetailedResult
) -> list[CandidateEvidence]:
    merged: dict[tuple[int, int, str], CandidateEvidence] = {}

    def add(entity: NerEntity, source: str, score: float) -> None:
        start, end = entity.position
        if entity.type not in VALID_ENTITY_TYPES or not _exact(raw_text, start, end, entity.text):
            return
        key = (start, end, entity.type)
        item = merged.get(key)
        if item is None:
            item = CandidateEvidence(
                candidate_id="",
                text=entity.text,
                type=entity.type,
                position=(start, end),
                allowed_types=[entity.type],
                assertions=list(entity.assertions),
            )
            merged[key] = item
        if source not in item.sources:
            item.sources.append(source)
        item.scores[source] = max(float(score), item.scores.get(source, 0.0))
        if source == "lattice":
            item.pre_llm_selected = True

    for entity in detailed.crf_entities:
        add(entity, "crf", entity.score)
    for span in detailed.span_candidates:
        entity = NerEntity(raw_text[span.start:span.end], span.type, [], (span.start, span.end), span.score)
        add(entity, "span_head", span.score)
    for entity in detailed.lattice_entities:
        add(entity, "lattice", entity.score)
    for local in detailed.local_verifications:
        entity = NerEntity(raw_text[local.start:local.end], local.type, [], (local.start, local.end), local.score)
        add(entity, "local_crf", local.score)

    for item in merged.values():
        item.sources.sort()
        item.candidate_id = stable_candidate_id(
            record_id, item.position[0], item.position[1], item.type, item.sources
        )
        item.supports = [f"{source}_candidate" for source in item.sources]
        if item.strong_consensus:
            item.hard_supports.append("crf_span_consensus")
    return sorted(merged.values(), key=lambda item: (*item.position, item.type))


_GENERIC_PREFIX = re.compile(r"^(?:bệnh|hội chứng)\s+", re.IGNORECASE)


def build_missing_proposals(
    record_id: str,
    raw_text: str,
    catalog: list[CandidateEvidence],
    *,
    trusted_seed_threshold: float = 0.95,
) -> list[MissingProposal]:
    """Generate only exact, evidence-backed proposals; never free-form spans."""
    occupied = {(item.position, item.type) for item in catalog}
    proposals: dict[tuple[int, int, str], MissingProposal] = {}
    seeds: dict[tuple[str, str], CandidateEvidence] = {}
    for item in catalog:
        if max(item.scores.values(), default=0.0) >= trusted_seed_threshold:
            seeds[(item.text.casefold(), item.type)] = item
            core = _GENERIC_PREFIX.sub("", item.text).strip()
            if len(core) >= 3:
                seeds.setdefault((core.casefold(), item.type), item)

    for (surface, entity_type), seed in seeds.items():
        if not surface:
            continue
        for match in re.finditer(re.escape(surface), raw_text.casefold()):
            start, end = match.span()
            text = raw_text[start:end]
            if ((start, end), entity_type) in occupied or not _exact(raw_text, start, end, text):
                continue
            supports = ["repeated_confirmed_surface", "exact_raw_boundary"]
            hard = ["repeated_confirmed_surface"]
            key = (start, end, entity_type)
            proposals[key] = MissingProposal(
                proposal_id=stable_proposal_id(record_id, start, end, [entity_type], supports),
                text=text,
                position=(start, end),
                allowed_types=[entity_type],
                supports=supports,
                hard_supports=hard,
                auto_add_eligible=True,
                related_candidate_ids=[seed.candidate_id],
            )
    return sorted(proposals.values(), key=lambda item: (*item.position, item.allowed_types))
