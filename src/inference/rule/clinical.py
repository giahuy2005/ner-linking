"""Surface-agnostic validation between NER and the 7B reviewer.

Only deterministic structure is handled here: offset validation, mechanical
boundary repair, assertion scope, overlap resolution and medication-list
layout. Clinical mentions are not recovered, deleted, or retyped from a
private-test vocabulary.
"""

from __future__ import annotations

from bisect import bisect_right
import math
import re

from ..schemas import NerEntity

ALLOWED_TYPES = {
    "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC",
    "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM",
}
ALLOWED_ASSERTIONS = {"isHistorical", "isNegated", "isFamily"}
ASSERTION_ENTITY_TYPES = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}
LAB_ENTITY_TYPES = {"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}

_DEVICE_PREFIXES = (
    "stent", "catheter", "picc", "foley", "ống dẫn mật", "ống dẫn lưu",
)
_GENERIC_NON_ENTITIES = {
    "kết quả", "xét nghiệm", "thuốc", "mẫu", "dấu hiệu", "triệu chứng",
}
_MEASUREMENT_ONLY_RE = re.compile(
    r"^\d+(?:[.,]\d+)?\s*(?:kg|fr|tuần|week|weeks|w|mg|mcg|micrograms?|"
    r"grams?|g|ml|l|iu|đơn\s+vị|units?)$",
    re.I,
)
_DOSING_ONLY_RE = re.compile(
    r"^(?:(?:iv|im|po|sc|sq|prn|bid|tid|qid|qhs|qam|daily|tĩnh\s+mạch|"
    r"tiêm|uống|truyền|liều|giảm\s+liều|đơn\s+vị|giọt|ngày|lần|mỗi|cách)"
    r"|\d+(?:[.,]\d+)?|[/x*+–-]|\s)+$",
    re.I,
)
_DRUG_HEADER_RE = re.compile(r"danh\s+sách\s+thuốc\s+trước\s+nhập\s+viện", re.I)
_NUMBERED_ITEM_RE = re.compile(r"(?:^|\s)(\d+)\.\s+", re.M)


# Negation is deliberately phrase-level. A bare "không" anywhere in the
# previous 12 tokens is too broad for Vietnamese clinical text: constructions
# such as "không thể đứng dậy do yếu chân", "không đáp ứng điều trị" and
# "không dùng thuốc ở người bị Parkinson" describe a positive deficit/action
# and must not negate a later medical mention.
_CLAUSE_BOUNDARY_RE = re.compile(
    r"(?iu)(?:[\r\n,;:.!?]+|\b(?:nhưng|tuy\s+nhiên|song|trong\s+khi|ngoại\s+trừ)\b)"
)
_NEGATION_CUE_RE = re.compile(r"(?iu)\b(?:không|chưa|phủ\s+nhận)\b")

# After the last cue, only grammatical bridge phrases are allowed before the
# entity. This intentionally prefers precision over blanket negation scope.
_DIRECT_NEGATION_SEGMENT_RE = re.compile(
    r"(?iu)^(?:không|chưa|phủ\s+nhận)"
    r"(?:\s+(?:có|còn|hề|từng|hoàn\s+toàn|ghi\s+nhận|thấy|"
    r"bất\s+kỳ|dấu\s+hiệu|triệu\s+chứng|biểu\s+hiện|bằng\s+chứng|"
    r"tình\s+trạng|tiền\s+sử|bị|mắc|xuất\s+hiện)){0,4}$"
)

# These are negative-form predicates that encode inability, failed response,
# non-adherence or another positive clinical state. They are not assertion
# negation cues for a later entity.
_NON_ASSERTION_NEGATION_SEGMENT_RE = re.compile(
    r"(?iu)^không\s+(?:"
    r"thể|nhấc|nâng|đứng|đi|vận\s+động|cử\s+động|đáp\s+ứng|cải\s+thiện|"
    r"dùng|sử\s+dụng|uống|tiêm|truyền|tuân\s+thủ|ăn|nuốt|ngủ|nói|"
    r"nghe|nghe\s+thấy|nhìn|nhìn\s+thấy|thở|tự\s+chủ"
    r")\b"
)
_POSITIVE_DEFICIT_ENTITY_RE = re.compile(
    r"(?iu)^\s*(?:"
    r"không\s+(?:thể|nhấc|nâng|đứng|đi|vận\s+động|cử\s+động|"
    r"đáp\s+ứng|cải\s+thiện|ăn|nuốt|ngủ|nói|nghe|nhìn|thở|tự\s+chủ)\b|"
    r"không\s+(?:nhìn|nghe)\s+thấy\b|không\s+thấy\s+gì\b"
    r")"
)


def _exact(raw_text: str, entity: NerEntity) -> bool:
    start, end = entity.position
    return (
        entity.type in ALLOWED_TYPES
        and 0 <= start < end <= len(raw_text)
        and raw_text[start:end] == entity.text
    )


def _copy(entity: NerEntity, *, start: int | None = None, end: int | None = None,
          text: str | None = None, entity_type: str | None = None,
          assertions: list[str] | None = None, flag: str | None = None) -> NerEntity:
    return NerEntity(
        text=entity.text if text is None else text,
        type=entity.type if entity_type is None else entity_type,
        assertions=list(entity.assertions if assertions is None else assertions),
        position=(entity.position[0] if start is None else start,
                  entity.position[1] if end is None else end),
        score=entity.score,
        flag=flag,
        review_hints=list(entity.review_hints),
    )


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split()).strip(" \t\r\n.,;:()[]{}")


def _is_hard_negative(entity: NerEntity) -> str | None:
    normalized = _normalize(entity.text)
    if normalized in _GENERIC_NON_ENTITIES:
        return "generic_non_entity"
    if normalized.isdigit():
        return "isolated_number"
    if _MEASUREMENT_ONLY_RE.fullmatch(normalized):
        return "isolated_measurement"
    if entity.type == "THUỐC" and _DOSING_ONLY_RE.fullmatch(normalized):
        return "dosing_without_ingredient"
    if entity.type in {"THUỐC", "CHẨN_ĐOÁN"} and normalized.startswith(_DEVICE_PREFIXES):
        return "device_or_procedure"
    return None


def is_linkable_entity(entity: NerEntity) -> bool:
    """Defense-in-depth gate used immediately before RxNorm/ICD retrieval."""
    return entity.type in {"THUỐC", "CHẨN_ĐOÁN"} and _is_hard_negative(entity) is None


def _numbered_section_heading(raw_text: str, position: int) -> str:
    headings = list(re.finditer(r"(?im)^\s*\d+\.\s+([^\n]+)$", raw_text[:position]))
    return _normalize(headings[-1].group(1)) if headings else ""


def _local_clause_prefix(raw_text: str, entity_start: int, *, max_chars: int = 220) -> str:
    """Return only the current clause before an entity.

    Newlines, punctuation and adversative/exception connectors stop assertion
    propagation. Comma is intentionally a boundary: BTC evidence favours sparse,
    local assertions over negating every later item in a list.
    """
    window_start = max(0, entity_start - max_chars)
    window = raw_text[window_start:entity_start]
    boundaries = list(_CLAUSE_BOUNDARY_RE.finditer(window))
    if boundaries:
        window = window[boundaries[-1].end():]
    return " ".join(window.casefold().split()).strip()


def _negation_scope_decision(local_prefix: str) -> str:
    """Classify the last local negation cue as direct, blocked or absent."""
    cues = list(_NEGATION_CUE_RE.finditer(local_prefix))
    if not cues:
        return "absent"

    segment = local_prefix[cues[-1].start():].strip()
    if _NON_ASSERTION_NEGATION_SEGMENT_RE.match(segment):
        return "blocked"
    if _DIRECT_NEGATION_SEGMENT_RE.fullmatch(segment):
        return "direct"
    return "blocked"


def _negation_separated_by_boundary(
    raw_text: str,
    entity_start: int,
    *,
    max_chars: int = 220,
) -> bool:
    """Return True when a prior cue was cut off by a clause boundary.

    This detects assertion leakage such as ``không sốt nhưng đau ngực`` or
    ``không đáp ứng điều trị, bệnh nhân vẫn sốt``.  It does not infer a new
    assertion; it only permits removal of a stale ``isNegated`` prediction.
    """
    window = raw_text[max(0, entity_start - max_chars):entity_start].casefold()
    cues = list(_NEGATION_CUE_RE.finditer(window))
    if not cues:
        return False
    boundaries = list(_CLAUSE_BOUNDARY_RE.finditer(window))
    return bool(boundaries and boundaries[-1].start() > cues[-1].start())


def _repair_assertion_scope(raw_text: str, entity: NerEntity) -> NerEntity:
    assertions = list(dict.fromkeys(
        item for item in entity.assertions if item in ALLOWED_ASSERTIONS
    ))

    # Lab entities never carry assertions, regardless of model/LLM output.
    if entity.type in LAB_ENTITY_TYPES:
        return _copy(entity, assertions=[], flag=entity.flag)

    start, _end = entity.position
    heading = _numbered_section_heading(raw_text, start)
    local_prefix = _local_clause_prefix(raw_text, start)
    normalized_entity = _normalize(entity.text)

    negated_history = bool(re.search(
        r"(?iu)\bphủ\s+nhận\s+tiền\s+sử(?:\s+\S+){0,4}\s*$",
        local_prefix,
    ))
    exception_scope = bool(re.search(r"(?iu)\bngoại\s+trừ\s*$", local_prefix))
    positive_deficit = bool(_POSITIVE_DEFICIT_ENTITY_RE.match(normalized_entity))
    negation_decision = _negation_scope_decision(local_prefix)
    scope_reset = _negation_separated_by_boundary(raw_text, start)

    # Remove a false model assertion only when the grammar clearly describes a
    # positive deficit/action or an explicit exception. Otherwise preserve the
    # model prediction and add an assertion only for a direct local cue.
    if (
        exception_scope
        or positive_deficit
        or negation_decision == "blocked"
        or scope_reset
    ):
        assertions = [item for item in assertions if item != "isNegated"]

    if negated_history:
        if "isNegated" not in assertions:
            assertions.append("isNegated")
        if "isHistorical" not in assertions:
            assertions.append("isHistorical")
    elif negation_decision == "direct" and "isNegated" not in assertions:
        assertions.append("isNegated")

    current_cue = re.search(
        r"(?iu)\b(?:hiện\s+đang|nay\s+.*?đang|hiện\s+tại|đang\s+có)\b",
        raw_text[max(0, start - 100):start],
    )
    current_section = any(marker in heading for marker in (
        "hiện tại", "đánh giá tại bệnh viện", "tình trạng hiện tại",
    ))
    if (current_cue or current_section) and not negated_history:
        assertions = [item for item in assertions if item != "isHistorical"]
    elif heading == "tiền sử bệnh" and "isHistorical" not in assertions:
        assertions.append("isHistorical")

    return _copy(
        entity,
        assertions=list(dict.fromkeys(assertions)),
        flag=entity.flag,
    )


def _is_token_char(char: str) -> bool:
    """Return True for characters that belong to one unsplittable raw token.

    Alphanumeric characters and underscore are treated as one token body. A
    deterministic boundary repair must never create a start/end between two
    such characters, because BIO word-level inference cannot justify that
    internal character boundary.
    """
    return bool(char) and (char.isalnum() or char == "_")


def _has_raw_token_boundaries(raw_text: str, start: int, end: int) -> bool:
    """Reject spans whose edge cuts through a raw alphanumeric/underscore token."""
    if not (0 <= start < end <= len(raw_text)):
        return False

    cuts_left_token = (
        start > 0
        and _is_token_char(raw_text[start - 1])
        and _is_token_char(raw_text[start])
    )
    cuts_right_token = (
        end < len(raw_text)
        and _is_token_char(raw_text[end - 1])
        and _is_token_char(raw_text[end])
    )
    return not cuts_left_token and not cuts_right_token


def _repair_boundary(raw_text: str, entity: NerEntity) -> NerEntity:
    """Apply only exact repairs that end on real raw-token boundaries.

    In particular, this function must never turn a fused token such as
    ``ổn địnhkhi`` into the internal substring ``ổn định``. If the tokenizer/BIO
    layer cannot represent a sub-token span, the full raw token is preserved and
    can be routed to the LLM reviewer instead of inventing a character boundary.
    """
    text = entity.text
    start, end = entity.position
    candidates: list[tuple[int, int]] = []

    # BIO/token offset occasionally stops inside a Unicode word (for example
    # the last accented character is omitted) or starts after its first
    # character.  A valid Vietnamese entity boundary cannot split an
    # alphanumeric token, so expansion to that token edge is deterministic
    # and does not require a medical surface dictionary.
    expanded_start, expanded_end = start, end
    first_fragment = re.match(r"[^\W\d_]+", text, flags=re.UNICODE)
    can_expand_left = bool(
        first_fragment
        and len(first_fragment.group(0)) <= 4
        and expanded_start > 0
        and raw_text[expanded_start - 1].isalpha()
        and raw_text[expanded_start].isalpha()
    )
    if can_expand_left:
        while expanded_start > 0 and raw_text[expanded_start - 1].isalnum():
            expanded_start -= 1
    last_fragment = re.search(r"[^\W\d_]+$", text, flags=re.UNICODE)
    can_expand_right = bool(
        last_fragment
        and len(last_fragment.group(0)) <= 2
        and last_fragment.group(0).islower()
        and expanded_end < len(raw_text)
        and raw_text[expanded_end - 1].isalpha()
        and raw_text[expanded_end].isalpha()
        and raw_text[expanded_end].islower()
    )
    if can_expand_right:
        while expanded_end < len(raw_text) and raw_text[expanded_end].isalnum():
            expanded_end += 1
    if (expanded_start, expanded_end) != (start, end):
        candidates.append((expanded_start, expanded_end))

    # Leading/trailing patient markers and punctuation leaked by BIO.
    trimmed = re.sub(r"^(?:bn|bệnh\s+nhân)\s+", "", text, flags=re.I)
    trimmed = re.sub(r"\s+(?:bn|ở|khi)$", "", trimmed, flags=re.I)
    trimmed = re.sub(r"\s*[\(\[\{]+\s*$", "", trimmed).strip()
    trimmed = trimmed.splitlines()[0].rstrip() if "\n" in trimmed else trimmed
    if trimmed and trimmed != text:
        local = text.find(trimmed)
        if local >= 0:
            candidates.append((start + local, start + local + len(trimmed)))

    # Prefer the second copy for duplicated prefixes.  It yields an exact span.
    tokens = list(re.finditer(r"\S+", text))
    if len(tokens) >= 2 and tokens[0].group(0).casefold() == tokens[1].group(0).casefold():
        candidates.append((start + tokens[1].start(), end))
    words = text.split()
    for width in range(1, min(4, len(words) // 2) + 1):
        if [w.casefold() for w in words[:width]] == [w.casefold() for w in words[width:2 * width]]:
            needle = " ".join(words[width:])
            local = text.find(needle)
            candidates.append((start + local, start + local + len(needle)))

    # Deliberately do NOT trim connector-looking suffixes inside a fused raw
    # token (for example ``ổn địnhkhi`` or ``atenololtrong``). Such a trim
    # invents a boundary that word-level BIO could not represent.

    valid = [
        (candidate_start, candidate_end)
        for candidate_start, candidate_end in candidates
        if _has_raw_token_boundaries(raw_text, candidate_start, candidate_end)
    ]
    if not valid:
        return entity
    # Most conservative repair: discard the fewest characters.
    new_start, new_end = max(valid, key=lambda span: span[1] - span[0])
    new_text = raw_text[new_start:new_end]
    return _copy(entity, start=new_start, end=new_end, text=new_text, flag=None)


def _safe_score(entity: NerEntity) -> float:
    """Return a finite confidence in [0, 1] for deterministic comparison."""
    try:
        score = float(entity.score)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(score, 1.0))


def _deduplicate_exact_entities(
    entities: list[NerEntity],
) -> tuple[list[NerEntity], list[dict]]:
    """Resolve only exact duplicates before interval scheduling.

    Duplicate identity is ``(start, end, type)``. The higher-confidence copy
    wins; ties prefer the copy carrying more reviewer evidence, then more
    assertions. Review hints from all duplicates are retained on the winner so
    structural evidence is not silently lost.
    """
    best_by_key: dict[tuple[int, int, str], NerEntity] = {}
    logs: list[dict] = []

    for entity in entities:
        key = (entity.position[0], entity.position[1], entity.type)
        previous = best_by_key.get(key)
        if previous is None:
            best_by_key[key] = entity
            continue

        previous_rank = (
            _safe_score(previous),
            len(previous.review_hints),
            len(previous.assertions),
        )
        entity_rank = (
            _safe_score(entity),
            len(entity.review_hints),
            len(entity.assertions),
        )
        winner, loser = (
            (entity, previous)
            if entity_rank > previous_rank
            else (previous, entity)
        )

        merged_hints: list[dict] = []
        seen_hints: set[str] = set()
        for hint in [*winner.review_hints, *loser.review_hints]:
            if not isinstance(hint, dict):
                continue
            marker = repr(sorted(hint.items(), key=lambda item: str(item[0])))
            if marker in seen_hints:
                continue
            seen_hints.add(marker)
            merged_hints.append(hint)

        best_by_key[key] = _copy(winner, flag=winner.flag)
        best_by_key[key].review_hints = merged_hints
        logs.append({
            "status": "drop",
            "reason": "exact_duplicate",
            "text": loser.text,
            "type": loser.type,
            "position": list(loser.position),
            "score": _safe_score(loser),
            "kept_score": _safe_score(winner),
        })

    unique = sorted(
        best_by_key.values(),
        key=lambda entity: (
            entity.position[0],
            entity.position[1],
            entity.type,
            -_safe_score(entity),
        ),
    )
    return unique, logs


def _interval_weight(entity: NerEntity) -> float:
    """Confidence-dominant weight used by weighted interval scheduling.

    Cubing confidence prevents several weak fragments from beating one strong
    complete span merely because there are more of them, while still allowing
    two high-confidence atomic concepts to beat one slightly higher-confidence
    merged span (for example ``lo âu`` + ``mất ngủ`` versus
    ``lo âu mất ngủ``).
    """
    score = _safe_score(entity)
    length = max(1, entity.position[1] - entity.position[0])
    return score ** 3 + min(length, 500) * 1e-9


def _plan_entity_keys(
    ordered: list[NerEntity],
    indices: tuple[int, ...],
) -> tuple[tuple[int, int, str, str], ...]:
    return tuple(
        (
            ordered[index].position[0],
            ordered[index].position[1],
            ordered[index].type,
            ordered[index].text,
        )
        for index in indices
    )


def _better_interval_plan(
    left: tuple[float, float, int, int, tuple[int, ...]],
    right: tuple[float, float, int, int, tuple[int, ...]],
    ordered: list[NerEntity],
) -> tuple[float, float, int, int, tuple[int, ...]]:
    """Choose a deterministic best DP plan.

    Priority: total interval weight, total raw confidence, fewer entities when
    mathematically tied (anti-fragmentation), larger covered length, then the
    lexicographically earlier span sequence.
    """
    epsilon = 1e-12
    if left[0] > right[0] + epsilon:
        return left
    if right[0] > left[0] + epsilon:
        return right
    if left[1] > right[1] + epsilon:
        return left
    if right[1] > left[1] + epsilon:
        return right
    if left[2] != right[2]:
        return left if left[2] < right[2] else right
    if left[3] != right[3]:
        return left if left[3] > right[3] else right
    return (
        left
        if _plan_entity_keys(ordered, left[4])
        <= _plan_entity_keys(ordered, right[4])
        else right
    )


def _resolve_overlaps_with_logs(
    entities: list[NerEntity],
) -> tuple[list[NerEntity], list[dict]]:
    """Select the globally best non-overlapping entity set with DP.

    This is true weighted interval scheduling, not confidence-first greedy.
    Exact duplicates are handled separately. Every dropped overlap is logged,
    including whether the conflict crossed entity types, so ambiguous rule
    behaviour is visible during evaluation instead of being silently hidden.
    """
    unique, logs = _deduplicate_exact_entities(entities)
    if len(unique) <= 1:
        return unique, logs

    ordered = sorted(
        unique,
        key=lambda entity: (
            entity.position[1],
            entity.position[0],
            entity.type,
            entity.text,
        ),
    )
    ends = [entity.position[1] for entity in ordered]
    previous_compatible: list[int] = []
    for index, entity in enumerate(ordered):
        previous_compatible.append(
            bisect_right(ends, entity.position[0], 0, index) - 1
        )

    # Plan fields: total_weight, total_score, entity_count, covered_length,
    # selected ordered indices.
    plans: list[tuple[float, float, int, int, tuple[int, ...]]] = [
        (0.0, 0.0, 0, 0, ())
    ]
    for index, entity in enumerate(ordered):
        base = plans[previous_compatible[index] + 1]
        include = (
            base[0] + _interval_weight(entity),
            base[1] + _safe_score(entity),
            base[2] + 1,
            base[3] + (entity.position[1] - entity.position[0]),
            (*base[4], index),
        )
        exclude = plans[index]
        plans.append(_better_interval_plan(include, exclude, ordered))

    selected_indices = set(plans[-1][4])
    selected = [ordered[index] for index in sorted(selected_indices)]
    selected_keys = {
        (entity.position[0], entity.position[1], entity.type)
        for entity in selected
    }

    for entity in ordered:
        key = (entity.position[0], entity.position[1], entity.type)
        if key in selected_keys:
            continue
        conflicts = [
            kept
            for kept in selected
            if entity.position[0] < kept.position[1]
            and entity.position[1] > kept.position[0]
        ]
        logs.append({
            "status": "drop",
            "reason": (
                "cross_type_overlap_weighted_selection"
                if any(kept.type != entity.type for kept in conflicts)
                else "same_type_overlap_weighted_selection"
            ),
            "text": entity.text,
            "type": entity.type,
            "position": list(entity.position),
            "score": _safe_score(entity),
            "interval_weight": _interval_weight(entity),
            "kept_conflicts": [
                {
                    "text": kept.text,
                    "type": kept.type,
                    "position": list(kept.position),
                    "score": _safe_score(kept),
                }
                for kept in conflicts
            ],
        })

    selected.sort(key=lambda entity: (
        entity.position[0], entity.position[1], entity.type,
    ))
    return selected, logs


def _resolve_overlaps(entities: list[NerEntity]) -> list[NerEntity]:
    """Backward-compatible list-only wrapper around the DP resolver."""
    resolved, _logs = _resolve_overlaps_with_logs(entities)
    return resolved


def pre_llm_cleanup(
    raw_text: str,
    entities: list[NerEntity],
) -> tuple[list[NerEntity], list[dict]]:
    """Semantic deterministic cleanup used only before the final 7B reviewer.

    This stage is intentionally allowed to repair boundaries/assertion scope,
    reject hard negatives and resolve overlaps.  It must not be called after
    Qwen 7B because doing so can silently overwrite a validated LLM decision.
    """
    logs: list[dict] = []
    cleaned: list[NerEntity] = []
    for entity in entities:
        if not _exact(raw_text, entity):
            logs.append({"status": "drop", "reason": "invalid_exact_span", "text": entity.text})
            continue
        drop_reason = _is_hard_negative(entity)
        if drop_reason:
            logs.append({"status": "drop", "reason": drop_reason, "text": entity.text})
            continue
        repaired = _repair_boundary(raw_text, entity)
        if repaired.position != entity.position:
            logs.append({"status": "repair", "reason": "deterministic_boundary", "before": entity.text,
                         "after": repaired.text, "position": list(repaired.position)})
        repaired = _repair_assertion_scope(raw_text, repaired)
        cleaned.append(repaired)
    # Re-check repaired candidates before overlap resolution.
    cleaned = [entity for entity in cleaned if _exact(raw_text, entity)
               and _is_hard_negative(entity) is None]
    resolved, overlap_logs = _resolve_overlaps_with_logs(cleaned)
    logs.extend(overlap_logs)
    return resolved, logs


def deterministic_cleanup(
    raw_text: str,
    entities: list[NerEntity],
) -> tuple[list[NerEntity], list[dict]]:
    """Backward-compatible alias for the pre-LLM semantic cleanup."""
    return pre_llm_cleanup(raw_text, entities)


def post_llm_validate(
    raw_text: str,
    entities: list[NerEntity],
) -> tuple[list[NerEntity], list[dict]]:
    """Validate Qwen 7B output without changing its semantics.

    Allowed operations are structural only: exact-offset/schema checks, exact
    duplicate detection, illegal-overlap detection and deterministic sorting.
    No boundary repair, assertion inference, hard-negative deletion, retyping or
    score-based overlap resolution is performed here.  Any violation raises so
    the caller can fall back to the complete pre-7B entity list for that record.
    """
    logs: list[dict] = []
    validated: list[NerEntity] = []
    seen_keys: set[tuple[int, int, str]] = set()

    for entity_index, entity in enumerate(entities):
        if not isinstance(entity, NerEntity):
            raise TypeError(
                f"post-7B entity {entity_index} must be NerEntity, "
                f"got {type(entity).__name__}"
            )
        if not _exact(raw_text, entity):
            raise ValueError(
                f"post-7B invalid exact span at entity {entity_index}: "
                f"text={entity.text!r}, position={entity.position}"
            )
        if any(
            not isinstance(assertion, str) or assertion not in ALLOWED_ASSERTIONS
            for assertion in entity.assertions
        ):
            raise ValueError(
                f"post-7B invalid assertion at entity {entity_index}: "
                f"{entity.assertions!r}"
            )
        if len(entity.assertions) != len(set(entity.assertions)):
            raise ValueError(
                f"post-7B duplicate assertion at entity {entity_index}: "
                f"{entity.assertions!r}"
            )
        if entity.type not in ASSERTION_ENTITY_TYPES and entity.assertions:
            raise ValueError(
                f"post-7B type {entity.type!r} cannot carry assertions: "
                f"{entity.assertions!r}"
            )

        key = (entity.position[0], entity.position[1], entity.type)
        if key in seen_keys:
            raise ValueError(f"post-7B exact duplicate entity: {key!r}")
        seen_keys.add(key)
        validated.append(entity)

    validated.sort(key=lambda item: (item.position[0], item.position[1], item.type))

    # The pre-7B cleanup already produces a non-overlapping list.  Therefore
    # any overlap here was introduced by a 7B edit/recovery and is rejected
    # instead of being greedily resolved after the fact.
    active_end = -1
    active_entity: NerEntity | None = None
    for entity in validated:
        if entity.position[0] < active_end:
            raise ValueError(
                "post-7B illegal overlap: "
                f"{active_entity.text!r}@{active_entity.position} with "
                f"{entity.text!r}@{entity.position}"
            )
        active_end = entity.position[1]
        active_entity = entity

    logs.append({
        "status": "post_llm_validated",
        "entity_count": len(validated),
    })
    return validated, logs


def _medication_list_entities(raw_text: str) -> list[NerEntity]:
    """Recover medication spans from a numbered pre-admission list.

    Text after ``điều trị`` is deliberately left to NER/LLMs: it may be a
    symptom or a diagnosis and must not be classified from a memorized phrase.
    """
    header = _DRUG_HEADER_RE.search(raw_text)
    if not header:
        return []
    markers = list(_NUMBERED_ITEM_RE.finditer(raw_text, header.end()))
    if not markers:
        return []
    result: list[NerEntity] = []
    for index, marker in enumerate(markers):
        item_start = marker.end()
        item_end = markers[index + 1].start() if index + 1 < len(markers) else len(raw_text)
        item = raw_text[item_start:item_end].strip()
        item_end = item_start + len(item)
        if not item:
            continue
        indication = re.search(r"\s+điều\s+trị\s+", item, flags=re.I)
        drug_text = item[:indication.start()].strip() if indication else item
        drug_end = item_start + len(drug_text)
        result.append(NerEntity(drug_text, "THUỐC", ["isHistorical"],
                                (item_start, drug_end), score=1.0))
    return result


def apply_clinical_rules(raw_text: str, entities: list[NerEntity]) -> tuple[list[NerEntity], list[dict]]:
    """Cleanup model candidates and inject only high-precision rule recovery."""
    cleaned, logs = deterministic_cleanup(raw_text, entities)
    med_entities = _medication_list_entities(raw_text)
    if med_entities:
        # Structured parsing is authoritative only for each drug span. Keep
        # NER mentions in the indication text that follows it.
        cleaned = [
            entity for entity in cleaned
            if not any(
                entity.position[0] < drug.position[1]
                and entity.position[1] > drug.position[0]
                for drug in med_entities
            )
        ]
        cleaned.extend(med_entities)
        cleaned.sort(key=lambda e: (e.position[0], e.position[1]))
        logs.append({"status": "recover", "reason": "pre_admission_medication_list",
                     "count": len(med_entities)})
    return cleaned, logs