"""Two-pass NER orchestration ported from notebook V11.

Pass 2 is deliberately limited to suspicious local regions.  The module is
model-agnostic so tests can supply a small callback and production can reuse
the already-loaded CRF engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable

from ..schemas import NerEntity
from ..rule.clinical import apply_clinical_rules


@dataclass(frozen=True)
class SuspiciousRegion:
    region_id: int
    start: int
    end: int
    focus_start: int
    focus_end: int
    reasons: tuple[str, ...] = ()
    priority: float = 0.0

    @property
    def position(self) -> tuple[int, int]:
        return self.start, self.end


@dataclass
class TwoPassResult:
    raw_text: str
    pass1_entities: list[NerEntity]
    regions: list[SuspiciousRegion]
    pass2_entities: list[NerEntity]
    final_entities: list[NerEntity]
    logs: list[dict] = field(default_factory=list)


_REPEATED_TOKEN_RE = re.compile(r"(?iu)\b([\wÀ-ỹ]{2,})\s+\1\b")
_STRUCTURED_LINE_RE = re.compile(
    r"(?iu)(?:^\s*[-•*]\s+|:\s*\S+|\b\d+(?:[.,]\d+)?\s*"
    r"(?:mg|mcg|g|ml|mmhg|bpm|l/ph|%|°c)\b)"
)


def _line_bounds(text: str, start: int, end: int, padding: int) -> tuple[int, int]:
    left = max(0, start - padding)
    right = min(len(text), end + padding)
    line_left = text.rfind("\n", left, start)
    line_right = text.find("\n", end, right)
    if line_left >= left:
        left = line_left + 1
    if line_right >= 0:
        right = line_right
    return left, right


def detect_suspicious_regions(
    raw_text: str,
    entities: list[NerEntity],
    *,
    score_threshold: float = 0.82,
    context_chars: int = 180,
    maximum_regions: int = 24,
) -> list[SuspiciousRegion]:
    hits: list[tuple[int, int, str, float]] = []
    for entity in entities:
        reasons = []
        if entity.flag:
            reasons.append(entity.flag)
        if entity.score < score_threshold:
            reasons.append("low_confidence")
        if re.search(r"(?i)^(?:bn|bệnh nhân)\s|\s(?:bn|ở|khi|\()$", entity.text):
            reasons.append("boundary_signal")
        if reasons:
            hits.append((*entity.position, "+".join(reasons), 2.0 + (1.0 - entity.score)))

    for match in _REPEATED_TOKEN_RE.finditer(raw_text):
        hits.append((match.start(), match.end(), "repeated_token_boundary", 3.0))

    # Adjacent predictions are often one split concept. Routing the context is
    # safe because this detector does not merge or retype anything itself.
    ordered_entities = sorted(entities, key=lambda item: item.position)
    for left_entity, right_entity in zip(ordered_entities, ordered_entities[1:]):
        gap_start, gap_end = left_entity.position[1], right_entity.position[0]
        if gap_start <= gap_end and gap_end - gap_start <= 2 \
                and raw_text[gap_start:gap_end].strip(" \t") == "":
            hits.append((left_entity.position[0], right_entity.position[1],
                         "adjacent_entity_boundary", 2.6))

    # If a trusted surface occurs more times than it was decoded, route each
    # uncovered occurrence for recovery instead of silently propagating it.
    by_surface: dict[tuple[str, str], list[NerEntity]] = {}
    for entity in entities:
        if len(entity.text.strip()) >= 3:
            by_surface.setdefault((entity.text.casefold(), entity.type), []).append(entity)
    for (surface, _entity_type), known in by_surface.items():
        for match in re.finditer(re.escape(surface), raw_text, flags=re.I):
            if not any(match.start() == item.position[0] and match.end() == item.position[1]
                       for item in known):
                hits.append((match.start(), match.end(), "repeated_surface_missing_occurrence", 2.8))

    # A structured line with no entity is an omission candidate. No clinical
    # vocabulary is used; all input documents are already in the medical task.
    cursor = 0
    occupied = [entity.position for entity in entities]
    for line in raw_text.splitlines(keepends=True):
        line_end = cursor + len(line)
        has_entity = any(cursor < end and line_end > start for start, end in occupied)
        if not has_entity and _STRUCTURED_LINE_RE.search(line):
            hits.append((cursor, line_end, "suspicious_empty_region", 2.5))
        cursor = line_end

    # Long uncovered medical gaps are notebook recovery windows.  They are
    # capped later together with all other regions.
    boundaries = sorted({0, len(raw_text), *(point for span in occupied for point in span)})
    for left, right in zip(boundaries, boundaries[1:]):
        gap = raw_text[left:right]
        if right - left >= 220 and len(gap.split()) >= 22:
            hits.append((left, right, "long_medical_gap", 2.2))

    windows = []
    for start, end, reason, priority in sorted(hits):
        left, right = _line_bounds(raw_text, start, end, context_chars)
        if windows and left <= windows[-1][1] + 40:
            old = windows[-1]
            windows[-1] = (old[0], max(old[1], right), min(old[2], start),
                           max(old[3], end), old[4] | {reason}, max(old[5], priority))
        else:
            windows.append((left, right, start, end, {reason}, priority))
    windows.sort(key=lambda item: (-item[5], item[0]))
    selected = windows[:maximum_regions]
    selected.sort(key=lambda item: item[0])
    return [
        SuspiciousRegion(i, left, right, focus_start, focus_end,
                         tuple(sorted(reasons)), priority)
        for i, (left, right, focus_start, focus_end, reasons, priority)
        in enumerate(selected)
    ]


def _merge_candidates(pass1: list[NerEntity], pass2: list[NerEntity]) -> list[NerEntity]:
    """Merge exact duplicates and resolve conflicts with recall-safe priority."""
    exact: dict[tuple[int, int, str], NerEntity] = {}
    for entity in [*pass1, *pass2]:
        key = (*entity.position, entity.type)
        previous = exact.get(key)
        if previous is None or entity.score > previous.score:
            exact[key] = entity
        elif previous is not None:
            previous.assertions = sorted(set(previous.assertions) | set(entity.assertions))
    candidates = sorted(exact.values(), key=lambda e: (e.position[0], e.position[1]))
    kept: list[NerEntity] = []
    for entity in candidates:
        conflicts = [other for other in kept if entity.position[0] < other.position[1]
                     and entity.position[1] > other.position[0]]
        if not conflicts:
            kept.append(entity)
            continue
        best = max([entity, *conflicts], key=lambda e: (e.score, e.position[1] - e.position[0]))
        if best is entity:
            kept = [other for other in kept if other not in conflicts]
            kept.append(entity)
    return sorted(kept, key=lambda e: (e.position[0], e.position[1]))


def run_two_pass_ner(
    raw_text: str,
    pass1_entities: list[NerEntity],
    predict_region: Callable[[str], list[NerEntity]],
    *,
    maximum_regions: int = 24,
) -> TwoPassResult:
    regions = detect_suspicious_regions(raw_text, pass1_entities,
                                        maximum_regions=maximum_regions)
    pass2: list[NerEntity] = []
    logs: list[dict] = []
    for region in regions:
        context = raw_text[region.start:region.end]
        try:
            local_entities = predict_region(context)
        except Exception as exc:  # pass 1 remains the safe fallback
            logs.append({"status": "pass2_error", "region_id": region.region_id,
                         "reason": str(exc)})
            continue
        for local in local_entities:
            start = region.start + local.position[0]
            end = region.start + local.position[1]
            if 0 <= start < end <= len(raw_text) and raw_text[start:end] == local.text:
                pass2.append(NerEntity(local.text, local.type, list(local.assertions),
                                       (start, end), local.score, local.flag))
        logs.append({"status": "pass2_ok", "region_id": region.region_id,
                     "entity_count": len(local_entities)})
    merged = _merge_candidates(pass1_entities, pass2)
    final, rule_logs = apply_clinical_rules(raw_text, merged)
    logs.extend(rule_logs)
    return TwoPassResult(raw_text, list(pass1_entities), regions, pass2, final, logs)
