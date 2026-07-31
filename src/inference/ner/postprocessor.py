"""Offset-safe BIO helpers synchronized with the final notebook predictor.

The current ``NerEngine`` reconciles BIO tags globally before entity extraction,
so ``merge_chunk_results`` is mainly retained for compatibility with older
chunk-per-entity inference code and unit tests.  Its conflict policy is kept
recall-safe: exact duplicates are collapsed, while non-exact same-type overlap
alternatives are preserved and flagged for downstream adjudication instead of
being deleted only because one span contains another.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

ASSERTION_PRIORITY = ["isNegated", "isFamily", "isHistorical"]

_CHUNK_CONFLICT_FLAG = "chunk_overlap_review"


def get_assertion_threshold(threshold, label: str) -> float:
    """Support one global threshold or notebook-style per-label thresholds."""
    if isinstance(threshold, dict):
        return float(threshold.get(label, 0.5))
    return float(threshold)


def clean_text_for_inference(text: str) -> str:
    """Do not mutate input; every returned position remains a raw-text offset."""
    return text


def get_label(id2label: dict, idx: int) -> str:
    if idx in id2label:
        return id2label[idx]
    return id2label[str(idx)]


def collapse_assertions(assertions: list[str]) -> list[str]:
    """Ép 1 assertion/entity theo priority nếu caller yêu cầu single-label."""
    assertions = assertions or []
    for assertion in ASSERTION_PRIORITY:
        if assertion in assertions:
            return [assertion]
    return []


def repair_bio_tags(tags: list[str]) -> list[str]:
    """Repair an invalid ``I-X`` transition into ``B-X``."""
    fixed: list[str] = []
    previous_type: str | None = None

    for tag in tags:
        if tag == "O":
            fixed.append(tag)
            previous_type = None
            continue

        if tag.startswith("B-"):
            entity_type = tag[2:]
            fixed.append(tag)
            previous_type = entity_type
            continue

        if tag.startswith("I-"):
            entity_type = tag[2:]
            fixed.append(tag if previous_type == entity_type else f"B-{entity_type}")
            previous_type = entity_type
            continue

        fixed.append("O")
        previous_type = None

    return fixed


def extract_entities_from_word_tags(
    tags: list[str],
    line_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Extract BIO entities without joining tokens across newline boundaries."""
    tags = repair_bio_tags(tags)

    entities: list[dict[str, Any]] = []
    index = 0
    token_count = len(tags)

    while index < token_count:
        tag = tags[index]
        if not tag.startswith("B-"):
            index += 1
            continue

        entity_type = tag[2:]
        start = index
        start_line = line_ids[index] if line_ids is not None else None

        end = index + 1
        while end < token_count and tags[end] == f"I-{entity_type}":
            if line_ids is not None and line_ids[end] != start_line:
                break
            end += 1

        entities.append({
            "word_start": start,
            "word_end": end,
            "type": entity_type,
        })
        index = end

    return entities


def _clone_result(result: dict) -> dict:
    """Clone mutable sidecars so chunk merge never mutates caller-owned data."""
    cloned = dict(result)
    if isinstance(result.get("assertions"), list):
        cloned["assertions"] = list(result["assertions"])
    if isinstance(result.get("review_hints"), list):
        cloned["review_hints"] = [
            dict(item) if isinstance(item, dict) else item
            for item in result["review_hints"]
        ]
    return cloned


def _score(result: dict) -> float:
    try:
        return float(result.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _span(result: dict) -> tuple[int, int]:
    return int(result["char_start"]), int(result["char_end"])


def _overlaps(left: dict, right: dict) -> bool:
    left_start, left_end = _span(left)
    right_start, right_end = _span(right)
    return left_start < right_end and left_end > right_start


def _merge_unique_strings(*groups: object) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, str) or item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged


def _merge_review_hints(*groups: object) -> list:
    merged: list = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, list):
            continue
        for item in group:
            marker = repr(sorted(item.items())) if isinstance(item, dict) else repr(item)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(dict(item) if isinstance(item, dict) else item)
    return merged


def _prefer_exact_duplicate(current: dict, challenger: dict) -> dict:
    """Pick the representative for one exact ``(start, end, type)`` key.

    Assertions and review hints from both chunk predictions are preserved.  The
    representative is selected by score, then number of review hints, then
    number of assertions.  No input dictionary is mutated.
    """
    current_clone = _clone_result(current)
    challenger_clone = _clone_result(challenger)

    current_rank = (
        _score(current_clone),
        len(current_clone.get("review_hints") or []),
        len(current_clone.get("assertions") or []),
    )
    challenger_rank = (
        _score(challenger_clone),
        len(challenger_clone.get("review_hints") or []),
        len(challenger_clone.get("assertions") or []),
    )

    winner = challenger_clone if challenger_rank > current_rank else current_clone
    winner["assertions"] = _merge_unique_strings(
        current_clone.get("assertions"), challenger_clone.get("assertions"),
    )

    merged_hints = _merge_review_hints(
        current_clone.get("review_hints"), challenger_clone.get("review_hints"),
    )
    if merged_hints:
        winner["review_hints"] = merged_hints

    return winner


def _flag_same_type_overlap_conflicts(
    results: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Preserve non-exact same-type overlap alternatives and flag them.

    Previous code sorted longer spans first and dropped every contained shorter
    span, regardless of confidence.  That destroyed a potentially correct
    atomic prediction before two-pass/clinical adjudication.  Here all distinct
    alternatives survive; each conflicted candidate receives an internal flag.
    """
    cloned = [_clone_result(result) for result in results]
    logs: list[dict] = []

    by_type: dict[str, list[int]] = defaultdict(list)
    for index, result in enumerate(cloned):
        by_type[str(result.get("type", ""))].append(index)

    for entity_type, indices in by_type.items():
        for offset, left_index in enumerate(indices):
            left = cloned[left_index]
            conflicts: list[dict] = []

            for right_index in indices[offset + 1:]:
                right = cloned[right_index]
                if not _overlaps(left, right):
                    continue

                # Exact duplicates were already collapsed before this stage.
                if _span(left) == _span(right):
                    continue

                conflicts.append({
                    "text": right.get("text"),
                    "position": list(_span(right)),
                    "score": _score(right),
                })
                right.setdefault("flag", _CHUNK_CONFLICT_FLAG)

            if not conflicts:
                continue

            left.setdefault("flag", _CHUNK_CONFLICT_FLAG)
            logs.append({
                "status": "chunk_overlap_preserved",
                "reason": "same_type_boundary_disagreement",
                "type": entity_type,
                "text": left.get("text"),
                "position": list(_span(left)),
                "score": _score(left),
                "conflicts": conflicts,
            })

    cloned.sort(key=lambda result: (
        int(result["char_start"]),
        int(result["char_end"]),
        str(result.get("type", "")),
        -_score(result),
    ))
    return cloned, logs


def drop_nested_entities(results: list[dict]) -> list[dict]:
    """Compatibility wrapper with recall-safe semantics.

    Historical behavior deleted a same-type span whenever it was contained in
    a longer span.  The function name is kept for API compatibility, but it now
    preserves distinct nested/overlapping alternatives and flags them for
    downstream review.  Exact duplicates should be collapsed by
    :func:`merge_chunk_results` before calling this function.
    """
    preserved, _logs = _flag_same_type_overlap_conflicts(results)
    return preserved


def merge_chunk_results_with_logs(
    all_results: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Merge legacy chunk-level entity predictions without silent recall loss.

    1. Collapse exact duplicate ``(char_start, char_end, type)`` predictions.
    2. Preserve non-exact same-type overlap alternatives and flag them.
    3. Return deterministic logs for A/B attribution.
    """
    best: dict[tuple[int, int, str], dict] = {}
    logs: list[dict] = []

    for raw_result in all_results:
        result = _clone_result(raw_result)
        key = (
            int(result["char_start"]),
            int(result["char_end"]),
            str(result["type"]),
        )

        previous = best.get(key)
        if previous is None:
            best[key] = result
            continue

        merged = _prefer_exact_duplicate(previous, result)
        best[key] = merged
        logs.append({
            "status": "chunk_exact_duplicate_merged",
            "position": [key[0], key[1]],
            "type": key[2],
            "kept_score": _score(merged),
        })

    preserved, conflict_logs = _flag_same_type_overlap_conflicts(
        list(best.values()),
    )
    logs.extend(conflict_logs)
    logs.append({
        "status": "chunk_merge_summary",
        "input_count": len(all_results),
        "exact_count": len(best),
        "output_count": len(preserved),
        "overlap_conflict_count": len(conflict_logs),
    })
    return preserved, logs


def merge_chunk_results(all_results: list[dict]) -> list[dict]:
    """Backward-compatible list-only wrapper."""
    merged, _logs = merge_chunk_results_with_logs(all_results)
    return merged