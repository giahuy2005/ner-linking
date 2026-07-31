"""Two-pass NER orchestration ported from notebook V11.

Pass 2 is deliberately limited to suspicious local regions. The module is
model-agnostic so tests can supply a small callback and production can reuse
the already-loaded CRF engine.
"""

from __future__ import annotations

from bisect import bisect_right
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


@dataclass(frozen=True)
class _MergeCandidate:
    """Entity đã deduplicate cùng metadata nguồn dùng cho merge/log."""

    entity: NerEntity
    sources: tuple[str, ...]


@dataclass(frozen=True)
class _Solution:
    """Một nghiệm DP không overlap.

    ``weight`` dùng confidence lũy thừa ba. Nhờ vậy hai span atomic có
    confidence cao có thể thắng một span gộp hơi cao hơn, nhưng nhiều fragment
    yếu không thắng chỉ vì số lượng lớn.
    """

    weight: float = 0.0
    score_sum: float = 0.0
    covered_chars: int = 0
    indices: tuple[int, ...] = ()


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
        if (
            gap_start <= gap_end
            and gap_end - gap_start <= 2
            and raw_text[gap_start:gap_end].strip(" \t") == ""
        ):
            hits.append(
                (
                    left_entity.position[0],
                    right_entity.position[1],
                    "adjacent_entity_boundary",
                    2.6,
                )
            )

    # If a trusted surface occurs more times than it was decoded, route each
    # uncovered occurrence for recovery instead of silently propagating it.
    by_surface: dict[tuple[str, str], list[NerEntity]] = {}
    for entity in entities:
        if len(entity.text.strip()) >= 3:
            by_surface.setdefault((entity.text.casefold(), entity.type), []).append(entity)
    for (surface, _entity_type), known in by_surface.items():
        for match in re.finditer(re.escape(surface), raw_text, flags=re.I):
            if not any(
                match.start() == item.position[0] and match.end() == item.position[1]
                for item in known
            ):
                hits.append(
                    (
                        match.start(),
                        match.end(),
                        "repeated_surface_missing_occurrence",
                        2.8,
                    )
                )

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

    # Long uncovered medical gaps are notebook recovery windows. They are
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
            windows[-1] = (
                old[0],
                max(old[1], right),
                min(old[2], start),
                max(old[3], end),
                old[4] | {reason},
                max(old[5], priority),
            )
        else:
            windows.append((left, right, start, end, {reason}, priority))
    windows.sort(key=lambda item: (-item[5], item[0]))
    selected = windows[:maximum_regions]
    selected.sort(key=lambda item: item[0])
    return [
        SuspiciousRegion(
            i,
            left,
            right,
            focus_start,
            focus_end,
            tuple(sorted(reasons)),
            priority,
        )
        for i, (left, right, focus_start, focus_end, reasons, priority) in enumerate(selected)
    ]


def _clone_entity(
    entity: NerEntity,
    *,
    assertions: list[str] | None = None,
    review_hints: list[dict] | None = None,
    flag: str | None = None,
    keep_original_flag: bool = True,
) -> NerEntity:
    """Clone để merge không mutate object Pass 1/Pass 2 phục vụ debug."""

    resolved_flag = entity.flag if keep_original_flag else flag
    if keep_original_flag and flag is not None:
        resolved_flag = flag
    return NerEntity(
        text=entity.text,
        type=entity.type,
        assertions=list(entity.assertions if assertions is None else assertions),
        position=tuple(entity.position),
        score=float(entity.score),
        flag=resolved_flag,
        review_hints=list(entity.review_hints if review_hints is None else review_hints),
    )


def _merge_unique_dicts(values: list[dict]) -> list[dict]:
    """Deduplicate hint dict theo nội dung, giữ nguyên thứ tự."""

    result: list[dict] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        if any(value == old for old in result):
            continue
        result.append(dict(value))
    return result


def _duplicate_rank(entity: NerEntity, source: str) -> tuple[float, int, int, int]:
    """Chọn base exact duplicate mà không phụ thuộc thứ tự input.

    Pass 2 chỉ được ưu tiên khi score và lượng evidence bằng nhau, vì Pass 2
    nhìn context cục bộ được tạo riêng cho vùng nghi ngờ.
    """

    return (
        float(entity.score),
        len(entity.review_hints),
        len(entity.assertions),
        1 if source == "pass2" else 0,
    )


def _deduplicate_exact(
    pass1: list[NerEntity],
    pass2: list[NerEntity],
) -> tuple[list[_MergeCandidate], list[dict]]:
    grouped: dict[tuple[int, int, str], list[tuple[str, NerEntity]]] = {}
    for source, entities in (("pass1", pass1), ("pass2", pass2)):
        for entity in entities:
            key = (int(entity.position[0]), int(entity.position[1]), entity.type)
            grouped.setdefault(key, []).append((source, entity))

    candidates: list[_MergeCandidate] = []
    logs: list[dict] = []

    for key in sorted(grouped):
        versions = grouped[key]
        base_source, base_entity = max(
            versions,
            key=lambda item: _duplicate_rank(item[1], item[0]),
        )
        assertions = list(dict.fromkeys(
            assertion
            for _source, entity in versions
            for assertion in entity.assertions
        ))
        hints = _merge_unique_dicts([
            hint
            for _source, entity in versions
            for hint in entity.review_hints
            if isinstance(hint, dict)
        ])
        flags = [entity.flag for _source, entity in versions if entity.flag]
        merged_flag = base_entity.flag or (flags[0] if flags else None)
        merged_entity = _clone_entity(
            base_entity,
            assertions=assertions,
            review_hints=hints,
            flag=merged_flag,
        )
        sources = tuple(dict.fromkeys(source for source, _entity in versions))
        candidates.append(_MergeCandidate(merged_entity, sources))

        if len(versions) > 1:
            logs.append({
                "status": "two_pass_exact_duplicate_merged",
                "reason": "exact_duplicate",
                "text": merged_entity.text,
                "type": merged_entity.type,
                "position": list(merged_entity.position),
                "kept_source": base_source,
                "sources": list(sources),
                "score": float(merged_entity.score),
            })

    return candidates, logs


def _candidate_weight(candidate: _MergeCandidate) -> float:
    score = max(0.0, min(1.0, float(candidate.entity.score)))
    return score ** 3


def _solution_with(solution: _Solution, index: int, candidate: _MergeCandidate) -> _Solution:
    entity = candidate.entity
    return _Solution(
        weight=solution.weight + _candidate_weight(candidate),
        score_sum=solution.score_sum + max(0.0, float(entity.score)),
        covered_chars=solution.covered_chars + max(0, entity.position[1] - entity.position[0]),
        indices=(*solution.indices, index),
    )


def _solution_key(solution: _Solution) -> tuple[float, float, int, int, tuple[int, ...]]:
    """Tie-break deterministic và recall-safe sau objective chính."""

    return (
        round(solution.weight, 12),
        round(solution.score_sum, 12),
        len(solution.indices),
        solution.covered_chars,
        tuple(-index for index in solution.indices),
    )


def _weighted_non_overlapping_selection(
    candidates: list[_MergeCandidate],
) -> set[int]:
    """Weighted interval scheduling trên toàn bộ Pass 1 + Pass 2 candidates."""

    if not candidates:
        return set()

    ordered = sorted(
        enumerate(candidates),
        key=lambda item: (
            item[1].entity.position[1],
            item[1].entity.position[0],
            item[1].entity.type,
            -float(item[1].entity.score),
        ),
    )
    ends = [candidate.entity.position[1] for _original, candidate in ordered]
    predecessors: list[int] = []
    for ordered_index, (_original, candidate) in enumerate(ordered):
        start = candidate.entity.position[0]
        predecessors.append(bisect_right(ends, start, hi=ordered_index) - 1)

    dp: list[_Solution] = [_Solution()]
    for ordered_index, (_original, candidate) in enumerate(ordered):
        exclude = dp[ordered_index]
        predecessor = predecessors[ordered_index]
        include_base = dp[predecessor + 1]
        include = _solution_with(include_base, ordered_index, candidate)
        dp.append(include if _solution_key(include) > _solution_key(exclude) else exclude)

    selected_ordered_indices = set(dp[-1].indices)
    return {
        original_index
        for ordered_index, (original_index, _candidate) in enumerate(ordered)
        if ordered_index in selected_ordered_indices
    }


def _overlap(left: NerEntity, right: NerEntity) -> bool:
    return left.position[0] < right.position[1] and left.position[1] > right.position[0]


def _candidate_log_payload(candidate: _MergeCandidate) -> dict:
    entity = candidate.entity
    return {
        "text": entity.text,
        "type": entity.type,
        "position": list(entity.position),
        "score": float(entity.score),
        "sources": list(candidate.sources),
    }


def _merge_candidates_with_logs(
    pass1: list[NerEntity],
    pass2: list[NerEntity],
) -> tuple[list[NerEntity], list[dict]]:
    """Deduplicate exact rồi chọn tập span không overlap tối ưu toàn cục."""

    candidates, logs = _deduplicate_exact(pass1, pass2)
    selected_indices = _weighted_non_overlapping_selection(candidates)
    selected = [candidates[index] for index in sorted(selected_indices)]

    for index, candidate in enumerate(candidates):
        if index in selected_indices:
            continue
        conflicts = [
            kept
            for kept in selected
            if _overlap(candidate.entity, kept.entity)
        ]
        reason = "overlap_weighted_selection"
        if conflicts:
            types = {candidate.entity.type, *(item.entity.type for item in conflicts)}
            reason = (
                "cross_type_overlap_weighted_selection"
                if len(types) > 1
                else "same_type_overlap_weighted_selection"
            )
        logs.append({
            "status": "two_pass_merge_drop",
            "reason": reason,
            **_candidate_log_payload(candidate),
            "kept_conflicts": [_candidate_log_payload(item) for item in conflicts],
        })

    merged = sorted(
        (_clone_entity(candidate.entity) for candidate in selected),
        key=lambda entity: (entity.position[0], entity.position[1], entity.type),
    )
    logs.append({
        "status": "two_pass_merge_summary",
        "pass1_count": len(pass1),
        "pass2_count": len(pass2),
        "deduplicated_candidate_count": len(candidates),
        "selected_count": len(merged),
        "dropped_overlap_count": len(candidates) - len(merged),
    })
    return merged, logs


def _merge_candidates(pass1: list[NerEntity], pass2: list[NerEntity]) -> list[NerEntity]:
    """API tương thích cũ; merge logs dùng tại ``run_two_pass_ner``."""

    merged, _logs = _merge_candidates_with_logs(pass1, pass2)
    return merged


def run_two_pass_ner(
    raw_text: str,
    pass1_entities: list[NerEntity],
    predict_region: Callable[[str], list[NerEntity]],
    *,
    maximum_regions: int = 24,
) -> TwoPassResult:
    regions = detect_suspicious_regions(
        raw_text,
        pass1_entities,
        maximum_regions=maximum_regions,
    )
    pass2: list[NerEntity] = []
    logs: list[dict] = []
    for region in regions:
        context = raw_text[region.start:region.end]
        try:
            local_entities = predict_region(context)
        except Exception as exc:  # pass 1 remains the safe fallback
            logs.append({
                "status": "pass2_error",
                "region_id": region.region_id,
                "reason": str(exc),
            })
            continue

        accepted_count = 0
        for local in local_entities:
            start = region.start + local.position[0]
            end = region.start + local.position[1]
            if 0 <= start < end <= len(raw_text) and raw_text[start:end] == local.text:
                pass2.append(NerEntity(
                    text=local.text,
                    type=local.type,
                    assertions=list(local.assertions),
                    position=(start, end),
                    score=local.score,
                    flag=local.flag,
                    review_hints=list(local.review_hints),
                ))
                accepted_count += 1
            else:
                logs.append({
                    "status": "pass2_entity_rejected",
                    "region_id": region.region_id,
                    "reason": "invalid_global_offset_or_text",
                    "text": local.text,
                    "local_position": list(local.position),
                    "global_position": [start, end],
                })
        logs.append({
            "status": "pass2_ok",
            "region_id": region.region_id,
            "predicted_entity_count": len(local_entities),
            "accepted_entity_count": accepted_count,
        })

    merged, merge_logs = _merge_candidates_with_logs(pass1_entities, pass2)
    logs.extend(merge_logs)
    final, rule_logs = apply_clinical_rules(raw_text, merged)
    logs.extend(rule_logs)
    return TwoPassResult(
        raw_text,
        list(pass1_entities),
        regions,
        pass2,
        final,
        logs,
    )