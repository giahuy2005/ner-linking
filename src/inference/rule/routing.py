"""Build grouped REVIEW_REGION and RECOVER_MISSING_ENTITIES requests."""

from __future__ import annotations

from typing import Iterable

from ..schemas import NerEntity
from ..ner.two_pass import SuspiciousRegion


def _entity_payload(entity: NerEntity, candidate_id: int, context_start: int) -> dict:
    start, end = entity.position
    return {
        "candidate_id": candidate_id,
        "text": entity.text,
        "type": entity.type,
        "global_position": [start, end],
        "relative_position": [start - context_start, end - context_start],
        "assertions": list(entity.assertions),
        "score": round(float(entity.score), 6),
        "small_llm_review_hints": list(entity.review_hints),
        "allowed_actions": ["KEEP", "DROP", "REPAIR_SPAN", "RETYPE"],
    }


def build_handoff_requests(
    raw_text: str,
    entities: list[NerEntity],
    regions: Iterable[SuspiciousRegion],
    *,
    score_threshold: float = 0.82,
    context_chars: int = 180,
    maximum_targets_per_region: int = 8,
    maximum_review_regions: int = 15,
    maximum_recoveries: int = 12,
    request_prefix: str = "",
) -> dict:
    prefix = f"{request_prefix}-" if request_prefix else ""
    indexed = list(enumerate(entities))
    # 7B may only edit this constrained set. Small-model blocked/suggestion
    # decisions are always targets even if a deterministic boundary cleanup
    # subsequently cleared the generic confidence flag.
    targets = [
        (i, e) for i, e in indexed
        if e.flag or e.review_hints or e.score < score_threshold
    ]
    groups: list[list[tuple[int, NerEntity]]] = []
    for item in targets:
        if not groups or item[1].position[0] - groups[-1][-1][1].position[1] > 120 \
                or len(groups[-1]) >= maximum_targets_per_region:
            groups.append([item])
        else:
            groups[-1].append(item)

    reviews = []
    for group_index, group in enumerate(groups[:maximum_review_regions]):
        first = group[0][1].position[0]
        last = group[-1][1].position[1]
        context_start = max(0, first - context_chars)
        context_end = min(len(raw_text), last + context_chars)
        reviews.append({
            "task": "REVIEW_REGION",
            "request_id": f"{prefix}review-region-{group_index}-{first}-{last}",
            "context": raw_text[context_start:context_end],
            "context_global_position": [context_start, context_end],
            "target_candidate_ids": [candidate_id for candidate_id, _ in group],
            "targets": [_entity_payload(entity, candidate_id, context_start)
                        for candidate_id, entity in group],
        })

    recoveries = []
    for region in list(regions)[:maximum_recoveries]:
        existing = [
            _entity_payload(entity, candidate_id, region.start)
            for candidate_id, entity in indexed
            if entity.position[0] < region.end and entity.position[1] > region.start
        ]
        recoveries.append({
            "task": "RECOVER_MISSING_ENTITIES",
            "request_id": f"{prefix}region-{region.region_id}-{region.focus_start}-{region.focus_end}",
            "context": raw_text[region.start:region.end],
            "context_global_position": [region.start, region.end],
            "focus": {"global_position": [region.focus_start, region.focus_end],
                      "relative_position": [region.focus_start - region.start,
                                            region.focus_end - region.start]},
            "recovery_reasons": list(region.reasons),
            "existing_entities": existing,
        })
    return {
        "schema_version": "7b_handoff_v2_grouped",
        "source_text_length": len(raw_text),
        "review_target_count": len(targets),
        "review_region_count": len(reviews),
        "region_recovery_count": len(recoveries),
        "review_regions": reviews,
        "region_recoveries": recoveries,
    }
