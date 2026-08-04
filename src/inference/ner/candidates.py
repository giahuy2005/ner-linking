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
_STRUCTURAL_FLAGS = frozenset({
    "boundary_disagreement", "type_disagreement", "possible_merge",
})
_SENTENCE_BREAK_RE = re.compile(r"(?:[.!?;](?:\\s|$)|[\\r\\n])")
_TOKEN_RE = re.compile(r"[^\\W_]+", re.UNICODE)


def _normalized_surface(text: str) -> str:
    return re.sub(r"\\s+", " ", text.casefold()).strip()


def _max_score(item: CandidateEvidence) -> float:
    return max((float(value) for value in item.scores.values()), default=0.0)


def _overlaps(left: CandidateEvidence, right: CandidateEvidence) -> bool:
    return left.position[0] < right.position[1] and left.position[1] > right.position[0]


def _contains(outer: CandidateEvidence, inner: CandidateEvidence) -> bool:
    return (
        outer.position[0] <= inner.position[0]
        and outer.position[1] >= inner.position[1]
        and outer.position != inner.position
    )


def _same_line(raw_text: str, left: CandidateEvidence, right: CandidateEvidence) -> bool:
    start = min(left.position[1], right.position[1])
    end = max(left.position[0], right.position[0])
    if end < start:
        start, end = min(left.position[0], right.position[0]), max(
            left.position[1], right.position[1]
        )
    between = raw_text[start:end]
    return "\\n" not in between and "\\r" not in between


def _safe_gap(raw_text: str, left: CandidateEvidence, right: CandidateEvidence) -> str | None:
    first, second = sorted((left, right), key=lambda item: item.position)
    if first.position[1] > second.position[0]:
        return ""
    value = raw_text[first.position[1]:second.position[0]]
    if _SENTENCE_BREAK_RE.search(value):
        return None
    return value


def _is_medical_abbreviation(text: str) -> bool:
    compact = "".join(char for char in text if char.isalnum())
    if not 2 <= len(compact) <= 10:
        return False
    letters = [char for char in compact if char.isalpha()]
    return bool(letters) and all(char.isupper() for char in letters)


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _looks_fragmentary(item: CandidateEvidence) -> bool:
    normalized = _normalized_surface(item.text)
    if normalized in _FUNCTION_WORDS or normalized in _GENERIC_DIAGNOSIS_NOUNS:
        return True
    if _is_medical_abbreviation(item.text):
        return False
    return _token_count(item.text) <= 1 and _max_score(item) < 0.82


def _context_boundary_is_material(
    item: CandidateEvidence,
    other: CandidateEvidence,
) -> bool:
    """Whether an audit-only boundary is strong enough to review a selected item."""
    if not _overlaps(item, other) or item.position == other.position:
        return False
    if not (_contains(other, item) or _contains(item, other)):
        return _max_score(other) >= 0.90
    if _contains(other, item):
        extension = (other.position[1] - other.position[0]) - (
            item.position[1] - item.position[0]
        )
        return extension >= 2 and (
            _max_score(other) >= 0.82
            or _max_score(other) >= _max_score(item) + 0.08
        )
    return _max_score(other) >= 0.92


def _structural_review_reasons(
    item: CandidateEvidence,
    *,
    by_id: dict[str, CandidateEvidence],
    raw_text: str,
) -> list[str]:
    reasons: list[str] = []
    item_score = _max_score(item)
    for candidate_id in item.related_candidate_ids:
        other = by_id.get(candidate_id)
        if other is None or not _same_line(raw_text, item, other):
            continue
        other_score = _max_score(other)
        if _overlaps(item, other):
            if other.pre_llm_selected:
                # A clean strong entity must not become a target merely because a
                # weak selected fragment overlaps it. Review the weak fragment and
                # keep the strong entity as context instead.
                if (
                    item.strong_consensus
                    and not other.strong_consensus
                    and other_score < 0.80
                ):
                    continue
                if item.type != other.type:
                    reasons.append("type_conflict_with_selected")
                elif item.position != other.position:
                    reasons.append("boundary_conflict_with_selected")
            elif _context_boundary_is_material(item, other):
                if _contains(other, item):
                    reasons.append("contained_by_stronger_span")
                elif item.type != other.type:
                    reasons.append("type_conflict_with_context")
                else:
                    reasons.append("boundary_conflict_with_context")
            continue

        gap = _safe_gap(raw_text, item, other)
        if gap is None or len(gap) > 12:
            continue
        same_type = item.type == other.type
        if not same_type:
            continue
        if other.pre_llm_selected:
            if (
                item.strong_consensus
                and not other.strong_consensus
                and other_score < 0.80
            ):
                continue
            if _looks_fragmentary(item) or _looks_fragmentary(other):
                reasons.append("adjacent_concept_continuation")
            elif not item.strong_consensus and not other.strong_consensus:
                reasons.append("possible_merge_with_selected")
        elif (
            _looks_fragmentary(item)
            and other_score >= max(0.82, item_score + 0.05)
        ):
            reasons.append("adjacent_concept_continuation")
    return list(dict.fromkeys(reasons))


def review_reasons(
    item: CandidateEvidence,
    *,
    by_id: dict[str, CandidateEvidence] | None = None,
    raw_text: str = "",
) -> list[str]:
    """Return target-only reasons; weak alternatives remain context/audit evidence."""
    if not item.pre_llm_selected:
        return []

    normalized = _normalized_surface(item.text)
    compact = "".join(char for char in normalized if char.isalnum())
    # Structural flags are recomputed directionally below. The old symmetric
    # flags made a clean CRF/span consensus target whenever any weak audit span
    # overlapped it, which inflated editor workload and confused MERGE decisions.
    reasons = [
        reason for reason in item.negative_flags
        if reason not in _STRUCTURAL_FLAGS
    ]
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

    score = _max_score(item)
    if score < 0.80:
        reasons.append("low_confidence_preselected")
    if (
        "span_head" in item.sources
        and not ({"crf", "local_crf"} & set(item.sources))
    ):
        reasons.append("span_only_candidate")
    if by_id is not None and raw_text:
        reasons.extend(_structural_review_reasons(item, by_id=by_id, raw_text=raw_text))

    # Clean high-confidence consensus bypasses Qwen unless it has an intrinsic
    # issue or a real conflict with another selected/stronger candidate.
    if item.strong_consensus:
        meaningful = [
            reason for reason in reasons
            if reason not in {"low_confidence_preselected", "span_only_candidate"}
        ]
        if not meaningful:
            return []
    return list(dict.fromkeys(reasons))


def _context_priority(
    candidate: CandidateEvidence,
    targets: list[CandidateEvidence],
) -> tuple[int, int, float, int, int]:
    overlaps = any(_overlaps(candidate, target) for target in targets)
    contains_target = any(_contains(candidate, target) for target in targets)
    selected_clean = candidate.pre_llm_selected and candidate.strong_consensus
    return (
        int(contains_target),
        int(overlaps),
        _max_score(candidate),
        int(selected_clean),
        candidate.position[1] - candidate.position[0],
    )


def build_review_regions(
    record_id: str,
    raw_text: str,
    catalog: list[CandidateEvidence],
    *,
    context_radius: int = 180,
    group_distance: int = 80,
    max_candidates: int = 6,
    hard_max_candidates: int = 8,
) -> list[ReviewRegion]:
    """Build structurally connected review regions with clean evidence as context."""
    if not 1 <= max_candidates <= hard_max_candidates <= 8:
        raise ValueError("review candidate limits must satisfy 1 <= max <= hard <= 8")
    by_id = {item.candidate_id: item for item in catalog}
    reasons_by_id = {
        item.candidate_id: review_reasons(item, by_id=by_id, raw_text=raw_text)
        for item in catalog
    }
    suspicious = [item for item in catalog if reasons_by_id[item.candidate_id]]
    suspicious.sort(key=lambda item: (*item.position, item.type))
    regions: list[ReviewRegion] = []
    cursor = 0

    while cursor < len(suspicious):
        group = [suspicious[cursor]]
        cursor += 1
        while cursor < len(suspicious) and len(group) < max_candidates:
            item = suspicious[cursor]
            gap = item.position[0] - group[-1].position[1]
            total_span = item.position[1] - group[0].position[0]
            # Independent targets may share one request for efficiency, but all
            # edit operations remain candidate-locked. The prompt explicitly
            # forbids merging across sentences/newlines or unrelated concepts.
            if gap > max(group_distance, 80) or total_span > 360:
                break
            group.append(item)
            cursor += 1

        target_ids = {item.candidate_id for item in group}
        context_pool: dict[str, CandidateEvidence] = {}
        for target in group:
            for candidate_id in target.related_candidate_ids:
                item = by_id.get(candidate_id)
                if item is not None and candidate_id not in target_ids:
                    context_pool[candidate_id] = item
        context_items = sorted(
            context_pool.values(),
            key=lambda item: _context_priority(item, group),
            reverse=True,
        )[: max(0, hard_max_candidates - len(group))]

        all_items = [*group, *context_items]
        core_start = min(item.position[0] for item in all_items)
        core_end = max(item.position[1] for item in all_items)
        context_start = max(0, core_start - context_radius)
        context_end = min(len(raw_text), core_end + context_radius)
        if context_end - context_start > 720:
            context_start = max(0, core_start - 240)
            context_end = min(len(raw_text), context_start + 720)
            if context_end < core_end:
                context_end = core_end
                context_start = max(0, context_end - 720)

        region_reasons = sorted({
            reason for item in group for reason in reasons_by_id[item.candidate_id]
        })
        high_priority_reasons = {
            "assertion_not_allowed_for_type", "one_character_alphabetic",
            "function_word_fragment", "generic_diagnosis_noun",
            "contained_by_stronger_span", "adjacent_concept_continuation",
            "boundary_conflict_with_selected", "type_conflict_with_selected",
            "boundary_conflict_with_context", "type_conflict_with_context",
        }
        regions.append(ReviewRegion(
            request_id=f"{record_id}:region:{len(regions):04d}",
            record_id=record_id,
            context=raw_text[context_start:context_end],
            context_start=context_start,
            context_end=context_end,
            target_candidate_ids=[item.candidate_id for item in group],
            context_candidate_ids=[item.candidate_id for item in context_items],
            reasons=region_reasons,
            priority=(
                100 if any(reason in high_priority_reasons for reason in region_reasons)
                else 50
            ),
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