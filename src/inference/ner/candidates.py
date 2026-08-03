"""Stable candidate catalog and closed missing-entity proposals."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable

from ..schemas import (
    ASSERTION_ENTITY_TYPES,
    NerEntity,
    VALID_ENTITY_TYPES,
    normalize_assertions_for_type,
)
from .evidence import NerDetailedResult
from .editor_schemas import ReviewRegion


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
                assertions=normalize_assertions_for_type(entity.type, entity.assertions),
            )
            merged[key] = item
        if source not in item.sources:
            item.sources.append(source)
        item.scores[source] = max(float(score), item.scores.get(source, 0.0))
        if source == "lattice":
            item.pre_llm_selected = True
        if entity.flag and entity.flag not in item.negative_flags:
            item.negative_flags.append(entity.flag)

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
    catalog = sorted(merged.values(), key=lambda item: (*item.position, item.type))
    for left_index, left in enumerate(catalog):
        for right in catalog[left_index + 1:]:
            if right.position[0] >= left.position[1] + 12:
                break
            overlaps = left.position[0] < right.position[1] and left.position[1] > right.position[0]
            adjacent = 0 <= right.position[0] - left.position[1] <= 8
            if overlaps and left.type != right.type:
                left.negative_flags.append("type_disagreement")
                right.negative_flags.append("type_disagreement")
            elif overlaps and left.position != right.position:
                left.negative_flags.append("boundary_disagreement")
                right.negative_flags.append("boundary_disagreement")
            elif adjacent and left.type == right.type:
                left.negative_flags.append("possible_merge")
                right.negative_flags.append("possible_merge")
            else:
                continue
            left.related_candidate_ids.append(right.candidate_id)
            right.related_candidate_ids.append(left.candidate_id)
    for item in catalog:
        item.negative_flags = list(dict.fromkeys(item.negative_flags))
        item.related_candidate_ids = list(dict.fromkeys(item.related_candidate_ids))
    return catalog


_FUNCTION_WORDS = frozenset({
    "có", "không", "ít", "nhiều", "âm", "dương", "và", "hoặc", "của",
    "cho", "với", "trong", "ngoài", "tại", "là", "bị", "được", "còn",
})
_GENERIC_DIAGNOSIS_NOUNS = frozenset({
    "bệnh", "bệnh lý", "rối loạn", "tình trạng", "hội chứng", "tổn thương",
})
_DEVICE_OR_PROCEDURE_CUES = (
    "catheter", "picc", "stent", "đường truyền", "ống thông", "ống dẫn",
    "thông khí", "hỗ trợ thở", "áp lực dương", "thủ thuật",
)
_TEST_EQUIPMENT_CUES = (
    "kính hiển", "máy đo", "máy xét nghiệm", "thiết bị", "đầu dò", "màn hình",
)


def review_reasons(item: CandidateEvidence) -> list[str]:
    normalized = re.sub(r"\s+", " ", item.text.casefold()).strip()
    compact = "".join(char for char in normalized if char.isalnum())
    # Weak span-head audit evidence is retained in the catalog but does not
    # become an editor target. It can still be attached as context to a
    # selected target with a structural relation.
    if not item.pre_llm_selected:
        return []
    reasons = list(item.negative_flags)
    if len(compact) <= 1 and any(char.isalpha() for char in compact):
        reasons.append("one_character_alphabetic")
    if normalized in _FUNCTION_WORDS:
        reasons.append("function_word_fragment")
    if item.type == "CHẨN_ĐOÁN" and normalized in _GENERIC_DIAGNOSIS_NOUNS:
        reasons.append("generic_diagnosis_noun")
    if item.type in {"THUỐC", "CHẨN_ĐOÁN"} and any(
        cue in normalized for cue in _DEVICE_OR_PROCEDURE_CUES
    ):
        reasons.append("possible_device_or_procedure")
    if item.type == "TÊN_XÉT_NGHIỆM" and any(
        cue in normalized for cue in _TEST_EQUIPMENT_CUES
    ):
        reasons.append("possible_test_equipment")
    if item.type == "KẾT_QUẢ_XÉT_NGHIỆM" and normalized in _FUNCTION_WORDS:
        reasons.append("result_fragment_without_typed_context")
    if item.type not in ASSERTION_ENTITY_TYPES and item.assertions:
        reasons.append("assertion_not_allowed_for_type")
    score = max(item.scores.values(), default=0.0)
    if item.pre_llm_selected and score < 0.80:
        reasons.append("low_confidence_preselected")
    if (
        item.pre_llm_selected
        and "span_head" in item.sources
        and not ({"crf", "local_crf"} & set(item.sources))
    ):
        reasons.append("span_only_candidate")
    return list(dict.fromkeys(reasons))


def build_review_regions(
    record_id: str,
    raw_text: str,
    catalog: list[CandidateEvidence],
    *,
    context_radius: int = 240,
    group_distance: int = 80,
    max_candidates: int = 6,
    hard_max_candidates: int = 8,
) -> list[ReviewRegion]:
    """Build bounded primary review requests; clean consensus stays local."""
    if not 1 <= max_candidates <= hard_max_candidates <= 8:
        raise ValueError("review candidate limits must satisfy 1 <= max <= hard <= 8")
    by_id = {item.candidate_id: item for item in catalog}
    reasons_by_id = {
        item.candidate_id: review_reasons(item) for item in catalog
    }
    suspicious = [item for item in catalog if reasons_by_id[item.candidate_id]]
    assigned: set[str] = set()
    regions: list[ReviewRegion] = []
    for seed in suspicious:
        if seed.candidate_id in assigned:
            continue
        group = [seed]
        assigned.add(seed.candidate_id)
        queue = list(seed.related_candidate_ids)
        while queue and len(group) < max_candidates:
            candidate_id = queue.pop(0)
            item = by_id.get(candidate_id)
            if item is None or candidate_id in assigned or not reasons_by_id[candidate_id]:
                continue
            group.append(item)
            assigned.add(candidate_id)
            queue.extend(item.related_candidate_ids)
        for item in suspicious:
            if len(group) >= max_candidates:
                break
            if item.candidate_id in assigned:
                continue
            gap = item.position[0] - max(member.position[1] for member in group)
            if 0 <= gap <= group_distance:
                group.append(item)
                assigned.add(item.candidate_id)
        group.sort(key=lambda item: (*item.position, item.type))
        context_items = []
        target_ids = {item.candidate_id for item in group}
        for target in group:
            for candidate_id in target.related_candidate_ids:
                item = by_id.get(candidate_id)
                if item is None or candidate_id in target_ids or item in context_items:
                    continue
                if len(group) + len(context_items) >= hard_max_candidates:
                    break
                context_items.append(item)
        all_items = [*group, *context_items]
        core_start = min(item.position[0] for item in all_items)
        core_end = max(item.position[1] for item in all_items)
        context_start = max(0, core_start - context_radius)
        context_end = min(len(raw_text), core_end + context_radius)
        if context_end - context_start > 900:
            context_start = max(0, core_start - 300)
            context_end = min(len(raw_text), context_start + 900)
            if context_end < core_end:
                context_end = core_end
                context_start = max(0, context_end - 900)
        region_reasons = sorted({
            reason for item in group for reason in reasons_by_id[item.candidate_id]
        })
        regions.append(ReviewRegion(
            request_id=f"{record_id}:region:{len(regions):04d}",
            record_id=record_id,
            context=raw_text[context_start:context_end],
            context_start=context_start,
            context_end=context_end,
            target_candidate_ids=[item.candidate_id for item in group],
            context_candidate_ids=[item.candidate_id for item in context_items],
            reasons=region_reasons,
            priority=max((100 if reason in {
                "assertion_not_allowed_for_type", "boundary_disagreement",
                "type_disagreement", "one_character_alphabetic",
            } else 50) for reason in region_reasons),
            must_review=True,
        ))
    return regions


_GENERIC_PREFIX = re.compile(r"^(?:bệnh|hội chứng)\s+", re.IGNORECASE)
_WORD_CHAR_RE = re.compile(r"[^\W_]", re.UNICODE)


def _raw_token_boundaries(raw_text: str, start: int, end: int) -> bool:
    left_ok = start == 0 or not _WORD_CHAR_RE.match(raw_text[start - 1])
    right_ok = end == len(raw_text) or not _WORD_CHAR_RE.match(raw_text[end])
    return left_ok and right_ok


def _surface_is_repeatable(surface: str, seed: CandidateEvidence) -> bool:
    normalized = re.sub(r"\s+", " ", surface).strip()
    letters = [char for char in normalized if char.isalpha()]
    if len(normalized) < 2 or len(letters) < 2:
        return False
    compact = "".join(char for char in normalized if char.isalnum())
    seed_surface = re.sub(r"\s+", " ", seed.text).strip()
    abbreviation = seed_surface.isupper() and 2 <= len(compact) <= 8
    if len(compact) < 4 and not abbreviation:
        return False
    return True


def build_missing_proposals(
    record_id: str,
    raw_text: str,
    catalog: list[CandidateEvidence],
    *,
    trusted_seed_threshold: float = 0.95,
    maximum_proposals: int = 64,
) -> list[MissingProposal]:
    """Generate only exact, evidence-backed proposals; never free-form spans."""
    occupied = {(item.position, item.type) for item in catalog}
    proposals: dict[tuple[int, int, str], MissingProposal] = {}
    seeds: dict[tuple[str, str], CandidateEvidence] = {}
    for item in catalog:
        if max(item.scores.values(), default=0.0) >= trusted_seed_threshold:
            seeds[(item.text.casefold(), item.type)] = item
            core = _GENERIC_PREFIX.sub("", item.text).strip()
            if len(core) >= 4:
                seeds.setdefault((core.casefold(), item.type), item)

    for (surface, entity_type), seed in seeds.items():
        if not _surface_is_repeatable(surface, seed):
            continue
        for match in re.finditer(re.escape(surface), raw_text.casefold()):
            start, end = match.span()
            text = raw_text[start:end]
            if (
                ((start, end), entity_type) in occupied
                or not _exact(raw_text, start, end, text)
                or not _raw_token_boundaries(raw_text, start, end)
            ):
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
    ranked = sorted(
        proposals.values(),
        key=lambda item: (-len(item.text), item.position[0], item.position[1], item.allowed_types),
    )[:maximum_proposals]
    return sorted(ranked, key=lambda item: (*item.position, item.allowed_types))
