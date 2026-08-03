"""Qwen3-8B locked candidate editor with action-level fail-safe guards."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from ...llm.batching import VersionedJsonlCache, generate_with_cache
from ...llm.json_guard import extract_json
from ..schemas import (
    ASSERTION_ENTITY_TYPES,
    NerEntity,
    normalize_assertions_for_type,
)
from .candidates import CandidateEvidence, MissingProposal
from .editor_schemas import (
    DROP_REASON_CODES,
    EditAction,
    EditOperation,
    MissingDecision,
    MissingDecisionAction,
    ReviewRegion,
)


PROMPT_VERSION = "qwen3_locked_editor_v4_strict_change_only"
_EDITOR_RESPONSE_FIELDS = frozenset({"request_id", "changes", "unresolved_ids"})


@dataclass
class EditorResult:
    entities: list[NerEntity]
    applied: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    consumed_candidate_ids: list[str] = field(default_factory=list)
    raw_response: str | None = None


def _candidate_payload(
    item: CandidateEvidence, context_start: int = 0, *, role: str = "selected_target",
) -> dict[str, Any]:
    """Compact payload containing only evidence the editor can use."""
    value: dict[str, Any] = {
        "id": item.candidate_id,
        "role": role,
        "text": item.text,
        "type": item.type,
        "local_position": [
            item.position[0] - context_start,
            item.position[1] - context_start,
        ],
    }
    optional = {
        "assertions": item.assertions,
        "sources": item.sources,
        "score": round(
            max((float(score) for score in item.scores.values()), default=0.0),
            4,
        ),
        "allowed_types": item.allowed_types,
        "flags": item.negative_flags,
    }
    value.update({key: item_value for key, item_value in optional.items() if item_value})
    if item.strong_consensus:
        value["strong_consensus"] = True
    return value


def build_editor_request(
    region: ReviewRegion,
    candidates: list[CandidateEvidence],
) -> tuple[str, str]:
    """Build the strict change-only editor request."""
    drop_reasons = "|".join(sorted(reason.value for reason in DROP_REASON_CODES))
    system = f"""Bạn là bộ biên tập NER y tế bị khóa theo candidate.
Chỉ trả các THAY ĐỔI cần thiết cho target. Target bị lược khỏi changes nghĩa là KEEP.
Context-only chỉ để tham khảo; không được tự thêm/promote/chỉnh nó, trừ khi tham gia MERGE với ít nhất một target.
Action duy nhất hợp lệ: DROP, RETYPE, REPAIR_SPAN, MERGE, UPDATE_ASSERTIONS.
Không được xuất KEEP, FLAG_UNRESOLVED hoặc confidence.
ID phải có trong payload. MERGE cần ít nhất 2 ID. Span [start,end) là local exact substring, không qua newline/câu.
Không chắc chắn: chỉ đưa ID target vào unresolved_ids. Không reasoning/markdown.
Mỗi change phải có đúng 7 field: action,candidate_ids,text,type,assertions,local_position,reason_code.
Field không dùng phải là null hoặc []. REPAIR_SPAN/MERGE phải trả type và assertions cuối cùng.
reason_code bắt buộc: RETYPE=WRONG_TYPE; REPAIR_SPAN=WRONG_BOUNDARY; MERGE=MERGE_REQUIRED; UPDATE_ASSERTIONS=ASSERTION_ERROR;
DROP chỉ dùng một trong: {drop_reasons}.
Ontology: TRIỆU_CHỨNG, CHẨN_ĐOÁN, THUỐC, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM.
Assertion isNegated/isHistorical/isFamily chỉ dùng cho TRIỆU_CHỨNG/CHẨN_ĐOÁN/THUỐC.
Output duy nhất: {{"request_id":"...","changes":[],"unresolved_ids":[]}}."""
    target_ids = set(region.target_candidate_ids)
    user = json.dumps({
        "schema_version": PROMPT_VERSION,
        "request_id": region.request_id,
        "context": region.context,
        "review_reasons": region.reasons,
        "candidates": [
            _candidate_payload(
                item,
                region.context_start,
                role=(
                    "selected_target"
                    if item.candidate_id in target_ids
                    else "context_only"
                ),
            )
            for item in candidates
        ],
        "response_schema": {
            "request_id": region.request_id,
            "changes": [],
            "unresolved_ids": [],
        },
    }, ensure_ascii=False, separators=(",", ":"))
    return system, user


def build_missing_request(
    request_id: str,
    context: str,
    context_start: int,
    proposals: list[MissingProposal],
) -> tuple[str, str]:
    system = """Bạn duyệt proposal NER y tế bị khóa theo ID.
Chỉ liệt kê proposal chắc chắn cần ADD trong additions; proposal bị lược nghĩa là REJECT/no-add.
Không chắc chắn thì đưa ID vào unresolved_ids. Không phát minh ID/text/span/type.
Assertion chỉ hợp lệ cho TRIỆU_CHỨNG/CHẨN_ĐOÁN/THUỐC. Không reasoning/markdown.
Output duy nhất: {"request_id":"...","additions":[{"proposal_id":"p","type":"CHẨN_ĐOÁN","assertions":[]}],"unresolved_ids":[]}."""
    payload = []
    for item in proposals:
        value = {
            "id": item.proposal_id,
            "text": item.text,
            "local_position": [
                item.position[0] - context_start,
                item.position[1] - context_start,
            ],
            "allowed_types": item.allowed_types,
        }
        for key, field_value in {
            "supports": item.supports,
            "hard_supports": item.hard_supports,
            "flags": item.negative_flags,
        }.items():
            if field_value:
                value[key] = field_value
        payload.append(value)
    user = json.dumps({
        "schema_version": PROMPT_VERSION,
        "request_id": request_id,
        "context": context,
        "context_global_start": context_start,
        "proposals": payload,
        "response_schema": {
            "request_id": request_id,
            "additions": [],
            "unresolved_ids": [],
        },
    }, ensure_ascii=False, separators=(",", ":"))
    return system, user


def parse_missing_response(
    raw_response: str,
) -> tuple[list[MissingDecision], list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    try:
        payload = extract_json(raw_response)
        if not isinstance(payload, dict):
            raise TypeError("missing response must be a JSON object")
        if "additions" in payload:
            additions = payload.get("additions")
            unresolved = payload.get("unresolved_ids", [])
            if not isinstance(additions, list) or not isinstance(unresolved, list):
                raise TypeError("additions and unresolved_ids must be lists")
            if not all(isinstance(item, str) for item in unresolved):
                raise TypeError("unresolved_ids must be list[str]")
            rows = [{
                **row,
                "decision": "ADD_PROPOSAL",
                "confidence": "HIGH",
                "reason_code": "VALID_MISSING_ENTITY",
            } for row in additions]
            rows.extend({
                "proposal_id": proposal_id,
                "decision": "UNRESOLVED",
                "type": None,
                "assertions": [],
                "confidence": "LOW",
                "reason_code": "AMBIGUOUS",
            } for proposal_id in unresolved)
        else:
            rows = payload.get("decisions")
            if not isinstance(rows, list):
                raise TypeError("decisions must be a list")
    except Exception as exc:
        return [], [{"reason": "invalid_json", "detail": str(exc)}]
    decisions = []
    for row in rows:
        try:
            decisions.append(MissingDecision.from_dict(row))
        except Exception as exc:
            rejected.append({
                "reason": "invalid_decision_schema",
                "detail": str(exc),
                "decision": row,
            })
    return decisions, rejected


def _parse_editor_envelope(
    raw_response: str,
    *,
    expected_request_id: str,
) -> dict[str, Any]:
    payload = extract_json(raw_response)
    if not isinstance(payload, dict):
        raise TypeError("editor response must be a JSON object")
    fields = set(payload)
    missing = _EDITOR_RESPONSE_FIELDS - fields
    extra = fields - _EDITOR_RESPONSE_FIELDS
    if missing:
        raise ValueError(f"editor response missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"editor response has unsupported fields: {sorted(extra)}")
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or request_id != expected_request_id:
        raise ValueError(
            f"request_id mismatch: expected={expected_request_id!r}, got={request_id!r}"
        )
    changes = payload.get("changes")
    unresolved_ids = payload.get("unresolved_ids")
    if not isinstance(changes, list):
        raise TypeError("changes must be a list")
    if (
        not isinstance(unresolved_ids, list)
        or not all(isinstance(item, str) for item in unresolved_ids)
    ):
        raise TypeError("unresolved_ids must be list[str]")
    if len(unresolved_ids) != len(set(unresolved_ids)):
        raise ValueError("unresolved_ids contains duplicates")
    return payload


def editor_response_error(
    raw_response: str,
    *,
    expected_request_id: str,
) -> str | None:
    """Return an envelope error for retry/audit, otherwise ``None``."""
    try:
        _parse_editor_envelope(
            raw_response,
            expected_request_id=expected_request_id,
        )
    except Exception as exc:
        return str(exc)
    return None


def editor_response_is_valid(
    raw_response: str,
    *,
    expected_request_id: str,
) -> bool:
    return editor_response_error(
        raw_response,
        expected_request_id=expected_request_id,
    ) is None


def _same_unit(raw_text: str, start: int, end: int) -> bool:
    value = raw_text[start:end]
    return (
        "\n" not in value
        and "\r" not in value
        and not any(mark in value for mark in (". ", "! ", "? "))
    )


def _to_entity(candidate: CandidateEvidence) -> NerEntity:
    return NerEntity(
        candidate.text,
        candidate.type,
        normalize_assertions_for_type(candidate.type, candidate.assertions),
        candidate.position,
        max(candidate.scores.values(), default=1.0),
    )


def _entity_key(entity: NerEntity) -> tuple[int, int, str]:
    return entity.position[0], entity.position[1], entity.type


def _entity_audit(entity: NerEntity) -> dict[str, Any]:
    return {
        "text": entity.text,
        "type": entity.type,
        "assertions": list(entity.assertions),
        "position": list(entity.position),
    }


def apply_editor_response(
    raw_text: str,
    candidates: list[CandidateEvidence],
    raw_response: str,
    *,
    context_start: int = 0,
    validation_candidates: list[CandidateEvidence] | None = None,
    target_candidate_ids: list[str] | None = None,
    expected_request_id: str = "",
    baseline_entities: list[NerEntity] | None = None,
) -> EditorResult:
    """Apply one strict change-only response without promoting omitted context."""
    target_ids = set(target_candidate_ids or [item.candidate_id for item in candidates])
    by_id = {item.candidate_id: item for item in candidates}
    validation_candidates = validation_candidates or candidates

    if baseline_entities is None:
        baseline_entities = [
            _to_entity(item) for item in candidates if item.pre_llm_selected
        ]
    baseline_by_key = {_entity_key(item): item for item in baseline_entities}
    entity_map = {
        item.candidate_id: baseline_by_key[_entity_key(_to_entity(item))]
        for item in candidates
        if _entity_key(_to_entity(item)) in baseline_by_key
    }
    region_baseline = list({
        _entity_key(entity): entity for entity in entity_map.values()
    }.values())
    result = EditorResult(
        entities=sorted(region_baseline, key=lambda item: (*item.position, item.type)),
        raw_response=raw_response,
    )

    try:
        payload = _parse_editor_envelope(
            raw_response,
            expected_request_id=expected_request_id,
        )
        raw_actions = payload["changes"]
        unresolved_ids = payload["unresolved_ids"]
    except Exception as exc:
        result.rejected.append({
            "reason": "invalid_response_envelope",
            "detail": str(exc),
        })
        result.unresolved.extend(sorted(target_ids))
        return result

    for candidate_id in unresolved_ids:
        if candidate_id not in target_ids:
            result.rejected.append({
                "reason": "unknown_or_non_target_unresolved_id",
                "candidate_id": candidate_id,
            })
        else:
            result.unresolved.append(candidate_id)

    parsed: list[EditOperation] = []
    for raw_action in raw_actions:
        try:
            parsed.append(EditOperation.from_dict(raw_action))
        except Exception as exc:
            result.rejected.append({
                "reason": "invalid_action_schema",
                "detail": str(exc),
                "action": raw_action,
            })

    eligible_for_duplicate_check = [
        operation for operation in parsed
        if all(candidate_id in by_id for candidate_id in operation.candidate_ids)
        and bool(set(operation.candidate_ids) & target_ids)
    ]
    counts: dict[str, int] = {}
    for operation in eligible_for_duplicate_check:
        for candidate_id in operation.candidate_ids:
            counts[candidate_id] = counts.get(candidate_id, 0) + 1

    consumed: set[str] = set()
    additions: list[NerEntity] = []
    unresolved_set = set(result.unresolved)
    for operation in parsed:
        ids = operation.candidate_ids
        errors: list[str] = []
        if any(candidate_id not in by_id for candidate_id in ids):
            errors.append("unknown_candidate_id")
        if not errors and not (set(ids) & target_ids):
            errors.append("context_only_change_without_target")
        if any(counts.get(candidate_id, 0) > 1 for candidate_id in ids):
            errors.append("duplicate_actions_for_candidate")
        if set(ids) & unresolved_set:
            errors.append("candidate_also_marked_unresolved")

        if not errors:
            if operation.action == EditAction.MERGE:
                current_target_ids = [
                    candidate_id for candidate_id in ids
                    if candidate_id in target_ids and candidate_id in entity_map
                ]
                if not current_target_ids:
                    errors.append("merge_has_no_current_target_entity")
            elif ids[0] not in entity_map:
                errors.append("target_not_in_current_entities")

        if errors:
            result.rejected.append({
                "reason": errors,
                "action": asdict(operation),
            })
            continue

        targets = [by_id[candidate_id] for candidate_id in ids]
        name = operation.action
        replacement: NerEntity | None = None
        before_entities = [
            _entity_audit(entity_map[candidate_id])
            for candidate_id in ids
            if candidate_id in entity_map
        ]

        if name == EditAction.DROP:
            # The strict schema already requires an explicit non-entity reason.
            # Do not silently veto a valid DROP merely because the deterministic
            # router did not attach a duplicate semantic flag.
            pass
        elif name == EditAction.RETYPE:
            target = targets[0]
            current = entity_map[ids[0]]
            if operation.type == current.type:
                errors.append("retype_is_noop")
            elif (
                target.strong_consensus
                and operation.type not in target.allowed_types
                and "type_disagreement" not in target.negative_flags
            ):
                errors.append("strong_candidate_retype_without_competing_evidence")
            else:
                final_type = operation.type or current.type
                replacement = NerEntity(
                    current.text,
                    final_type,
                    normalize_assertions_for_type(final_type, current.assertions),
                    current.position,
                    current.score,
                )
        elif name in {EditAction.REPAIR_SPAN, EditAction.MERGE}:
            assert operation.local_position is not None and operation.text is not None
            start = context_start + operation.local_position[0]
            end = context_start + operation.local_position[1]
            if not (0 <= start < end <= len(raw_text)) or raw_text[start:end] != operation.text:
                errors.append("invalid_exact_span")
            elif not _same_unit(raw_text, start, end):
                errors.append("crosses_structural_boundary")
            elif (
                name == EditAction.REPAIR_SPAN
                and not (
                    start < targets[0].position[1]
                    and end > targets[0].position[0]
                )
            ):
                errors.append("repair_does_not_overlap_target")
            elif name == EditAction.REPAIR_SPAN and any(
                other.candidate_id not in ids
                and start < other.position[1]
                and end > other.position[0]
                for other in validation_candidates
            ):
                errors.append("repair_would_swallow_other_candidate_use_merge")
            elif name == EditAction.MERGE and not all(
                start <= item.position[0] and end >= item.position[1]
                for item in targets
            ):
                errors.append("merge_does_not_cover_targets")
            elif name == EditAction.MERGE and any(
                right.position[0] - left.position[1] > 32
                for left, right in zip(
                    sorted(targets, key=lambda item: item.position),
                    sorted(targets, key=lambda item: item.position)[1:],
                )
            ):
                errors.append("merge_targets_not_adjacent")
            else:
                final_type = operation.type or targets[0].type
                replacement = NerEntity(
                    operation.text,
                    final_type,
                    normalize_assertions_for_type(final_type, operation.assertions),
                    (start, end),
                    1.0,
                )
        elif name == EditAction.UPDATE_ASSERTIONS:
            current = entity_map[ids[0]]
            if current.type not in ASSERTION_ENTITY_TYPES:
                errors.append("assertions_not_allowed_for_type")
            else:
                replacement = NerEntity(
                    current.text,
                    current.type,
                    normalize_assertions_for_type(
                        current.type,
                        operation.assertions,
                    ),
                    current.position,
                    current.score,
                )

        if errors:
            result.rejected.append({
                "reason": errors,
                "action": asdict(operation),
            })
            continue

        consumed.update(ids)
        if replacement is not None:
            additions.append(replacement)
        result.applied.append({
            "action": name.value,
            "candidate_ids": list(ids),
            "entities_before": before_entities,
            "entity_after": _entity_audit(replacement) if replacement else None,
        })

    final = [
        entity for candidate_id, entity in entity_map.items()
        if candidate_id not in consumed
    ]
    final.extend(additions)
    unique = {_entity_key(item): item for item in final}
    accepted: list[NerEntity] = []
    for entity in sorted(
        unique.values(),
        key=lambda item: (
            -float(item.score),
            -(item.position[1] - item.position[0]),
            *item.position,
            item.type,
        ),
    ):
        if any(
            entity.position[0] < old.position[1]
            and entity.position[1] > old.position[0]
            for old in accepted
        ):
            result.rejected.append({
                "reason": "post_editor_overlap_resolution",
                "entity": _entity_audit(entity),
            })
            continue
        accepted.append(entity)
    result.entities = sorted(accepted, key=lambda item: (*item.position, item.type))
    result.consumed_candidate_ids = sorted(consumed)
    result.unresolved = list(dict.fromkeys(result.unresolved))
    for entity in result.entities:
        if (
            raw_text[entity.position[0]:entity.position[1]] != entity.text
            or any(char in entity.text for char in "\r\n")
        ):
            raise ValueError("editor produced invalid exact offset")
    return result


def apply_missing_decisions(
    raw_text: str,
    existing: list[NerEntity],
    proposals: list[MissingProposal],
    decisions: list[MissingDecision],
) -> EditorResult:
    result = EditorResult(entities=list(existing))
    by_id = {item.proposal_id: item for item in proposals}
    seen: set[str] = set()
    for decision in decisions:
        if decision.proposal_id not in by_id or decision.proposal_id in seen:
            result.rejected.append({
                "reason": "unknown_or_duplicate_proposal_id",
                "proposal_id": decision.proposal_id,
            })
            continue
        seen.add(decision.proposal_id)
        proposal = by_id[decision.proposal_id]
        if decision.decision != MissingDecisionAction.ADD_PROPOSAL:
            if decision.decision == MissingDecisionAction.UNRESOLVED:
                result.unresolved.append(decision.proposal_id)
            continue
        start, end = proposal.position
        overlaps = [
            item for item in result.entities
            if start < item.position[1] and end > item.position[0]
        ]
        errors = []
        if decision.type not in proposal.allowed_types:
            errors.append("type_not_allowed")
        if not proposal.auto_add_eligible or not proposal.hard_supports:
            errors.append("insufficient_support")
        if proposal.negative_flags:
            errors.append("structural_negative")
        if overlaps:
            errors.append("unsafe_overlap")
        if raw_text[start:end] != proposal.text or any(
            char in proposal.text for char in "\r\n"
        ):
            errors.append("invalid_exact_span")
        if errors:
            result.rejected.append({
                "reason": errors,
                "proposal_id": proposal.proposal_id,
            })
            continue
        final_type = decision.type or proposal.allowed_types[0]
        entity = NerEntity(
            proposal.text,
            final_type,
            normalize_assertions_for_type(final_type, decision.assertions),
            proposal.position,
            1.0,
        )
        result.entities.append(entity)
        result.applied.append({
            "action": "ADD_PROPOSAL",
            "proposal_id": proposal.proposal_id,
        })
    result.unresolved.extend(sorted(set(by_id) - seen))
    result.unresolved = list(dict.fromkeys(result.unresolved))
    result.entities.sort(key=lambda item: (*item.position, item.type))
    return result