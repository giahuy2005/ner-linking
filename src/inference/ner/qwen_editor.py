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
    EditAction,
    EditOperation,
    MissingDecision,
    MissingDecisionAction,
    ReasonCode,
    ReviewRegion,
)


PROMPT_VERSION = "qwen3_locked_editor_v3_change_only_speed"
_EXPLICIT_DROP_REASONS = {
    ReasonCode.NON_ENTITY_ANATOMY, ReasonCode.NON_ENTITY_PERSON,
    ReasonCode.NON_ENTITY_SPECIALTY, ReasonCode.NON_ENTITY_SPECIMEN,
    ReasonCode.NON_ENTITY_ACTIVITY, ReasonCode.NON_ENTITY_MECHANISM,
    ReasonCode.GENERIC_BIOMEDICAL, ReasonCode.FUNCTION_WORD_OR_FRAGMENT,
    ReasonCode.PROCEDURE_NOT_TEST,
}


@dataclass
class EditorResult:
    entities: list[NerEntity]
    applied: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    raw_response: str | None = None


def _candidate_payload(
    item: CandidateEvidence, context_start: int = 0, *, role: str = "selected_target",
) -> dict[str, Any]:
    """Compact payload containing only evidence the editor can use."""
    value: dict[str, Any] = {
        "id": item.candidate_id, "role": role, "text": item.text,
        "type": item.type,
        "local_position": [item.position[0] - context_start, item.position[1] - context_start],
    }
    optional = {
        "assertions": item.assertions, "sources": item.sources,
        "score": round(max((float(score) for score in item.scores.values()), default=0.0), 4),
        "allowed_types": item.allowed_types, "flags": item.negative_flags,
    }
    value.update({key: item_value for key, item_value in optional.items() if item_value})
    if item.strong_consensus:
        value["strong_consensus"] = True
    return value


def build_editor_request(region: ReviewRegion, candidates: list[CandidateEvidence]) -> tuple[str, str]:
    """Build the compact V3 change-only request."""
    system = """Bạn là bộ biên tập NER y tế bị khóa theo candidate.
Chỉ trả các THAY ĐỔI cần thiết cho target. Candidate bị lược khỏi changes nghĩa là KEEP.
Context-only chỉ để tham khảo; không được chỉnh hoặc thêm nó, trừ khi tham gia MERGE với target.
Action hợp lệ trong changes: DROP, RETYPE, REPAIR_SPAN, MERGE, UPDATE_ASSERTIONS.
ID phải có trong payload. Span [start,end) là local exact substring, không qua newline/câu.
Không chắc chắn: đưa ID target vào unresolved_ids. Không reasoning/markdown.
Mỗi change có action,candidate_ids,text,type,assertions,local_position,reason_code; field không dùng là null/[].
Ontology: TRIỆU_CHỨNG, CHẨN_ĐOÁN, THUỐC, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM.
Assertion isNegated/isHistorical/isFamily chỉ dùng cho triệu chứng/chẩn đoán/thuốc.
Output duy nhất: {"request_id":"...","changes":[],"unresolved_ids":[]}."""
    target_ids = set(region.target_candidate_ids)
    user = json.dumps({
        "schema_version": PROMPT_VERSION,
        "request_id": region.request_id,
        "context": region.context,
        "review_reasons": region.reasons,
        "candidates": [
            _candidate_payload(
                item, region.context_start,
                role="selected_target" if item.candidate_id in target_ids else "context_only",
            )
            for item in candidates
        ],
        "response_schema": {
            "request_id": region.request_id, "changes": [], "unresolved_ids": [],
        },
    }, ensure_ascii=False, separators=(",", ":"))
    return system, user


def build_missing_request(
    request_id: str, context: str, context_start: int, proposals: list[MissingProposal]
) -> tuple[str, str]:
    system = """Bạn duyệt proposal NER y tế bị khóa theo ID.
Chỉ liệt kê proposal chắc chắn cần ADD trong additions; proposal bị lược nghĩa là REJECT/no-add.
Không chắc chắn thì đưa ID vào unresolved_ids. Không phát minh ID/text/span/type.
Assertion chỉ hợp lệ cho TRIỆU_CHỨNG/CHẨN_ĐOÁN/THUỐC. Không reasoning/markdown.
Output duy nhất: {"request_id":"...","additions":[{"proposal_id":"p","type":"CHẨN_ĐOÁN","assertions":[]}],"unresolved_ids":[]}."""
    payload = []
    for item in proposals:
        value = {
            "id": item.proposal_id, "text": item.text,
            "local_position": [item.position[0] - context_start, item.position[1] - context_start],
            "allowed_types": item.allowed_types,
        }
        for key, field_value in {
            "supports": item.supports, "hard_supports": item.hard_supports,
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
            "request_id": request_id, "additions": [], "unresolved_ids": [],
        },
    }, ensure_ascii=False, separators=(",", ":"))
    return system, user


def parse_missing_response(raw_response: str) -> tuple[list[MissingDecision], list[dict[str, Any]]]:
    rejected = []
    try:
        payload = extract_json(raw_response)
        if "additions" in payload:
            additions = payload.get("additions")
            unresolved = payload.get("unresolved_ids", [])
            if not isinstance(additions, list) or not isinstance(unresolved, list):
                raise TypeError("additions and unresolved_ids must be lists")
            rows = [{
                **row, "decision": "ADD_PROPOSAL", "confidence": "HIGH",
                "reason_code": "VALID_MISSING_ENTITY",
            } for row in additions]
            rows.extend({
                "proposal_id": proposal_id, "decision": "UNRESOLVED",
                "type": None, "assertions": [], "confidence": "LOW",
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
            rejected.append({"reason": "invalid_decision_schema", "detail": str(exc), "decision": row})
    return decisions, rejected


def _same_unit(raw_text: str, start: int, end: int) -> bool:
    value = raw_text[start:end]
    return "\n" not in value and "\r" not in value and not any(mark in value for mark in (". ", "! ", "? "))


def _to_entity(candidate: CandidateEvidence) -> NerEntity:
    return NerEntity(
        candidate.text,
        candidate.type,
        normalize_assertions_for_type(candidate.type, candidate.assertions),
        candidate.position,
        max(candidate.scores.values(), default=1.0),
    )


def apply_editor_response(
    raw_text: str,
    candidates: list[CandidateEvidence],
    raw_response: str,
    *,
    context_start: int = 0,
    validation_candidates: list[CandidateEvidence] | None = None,
    target_candidate_ids: list[str] | None = None,
) -> EditorResult:
    """Apply change-only V3 responses; old full-action cache rows remain readable."""
    target_ids = set(target_candidate_ids or [item.candidate_id for item in candidates])
    originals = [
        _to_entity(item) for item in candidates
        if item.pre_llm_selected and item.candidate_id in target_ids
    ]
    result = EditorResult(entities=originals, raw_response=raw_response)
    by_id = {item.candidate_id: item for item in candidates}
    validation_candidates = validation_candidates or candidates
    try:
        payload = extract_json(raw_response)
        change_only = "changes" in payload
        raw_actions = payload.get("changes") if change_only else payload.get("actions")
        if not isinstance(raw_actions, list):
            raise TypeError("changes/actions must be a list")
        unresolved_ids = payload.get("unresolved_ids", []) if change_only else []
        if not isinstance(unresolved_ids, list) or not all(isinstance(item, str) for item in unresolved_ids):
            raise TypeError("unresolved_ids must be list[str]")
    except Exception as exc:
        result.rejected.append({"reason": "invalid_json", "detail": str(exc)})
        result.unresolved.extend(sorted(target_ids))
        return result

    for candidate_id in unresolved_ids:
        if candidate_id not in target_ids:
            result.rejected.append({"reason": "unknown_or_non_target_unresolved_id", "candidate_id": candidate_id})
        elif candidate_id not in result.unresolved:
            result.unresolved.append(candidate_id)

    parsed: list[EditOperation] = []
    for raw_action in raw_actions:
        try:
            parsed.append(EditOperation.from_dict(raw_action))
        except Exception as exc:
            result.rejected.append({"reason": "invalid_action_schema", "detail": str(exc), "action": raw_action})

    counts: dict[str, int] = {}
    for operation in parsed:
        for candidate_id in operation.candidate_ids:
            counts[candidate_id] = counts.get(candidate_id, 0) + 1

    entity_map = {
        item.candidate_id: _to_entity(item)
        for item in candidates if item.pre_llm_selected
    }
    consumed: set[str] = set()
    additions: list[NerEntity] = []
    for operation in parsed:
        ids = operation.candidate_ids
        errors = []
        if any(candidate_id not in by_id for candidate_id in ids):
            errors.append("unknown_candidate_id")
        if not errors and not (set(ids) & target_ids):
            errors.append("context_only_change_without_target")
        if any(counts.get(candidate_id, 0) > 1 for candidate_id in ids):
            errors.append("duplicate_actions_for_candidate")
        if errors:
            result.rejected.append({"reason": errors, "action": asdict(operation)})
            continue
        targets = [by_id[candidate_id] for candidate_id in ids]
        name = operation.action
        replacement: NerEntity | None = None

        if name == EditAction.DROP:
            if any(item.strong_consensus for item in targets):
                if operation.reason_code not in _EXPLICIT_DROP_REASONS:
                    errors.append("strong_consensus_drop_requires_explicit_non_target_reason")
                elif not any(item.negative_flags for item in targets):
                    errors.append("strong_consensus_drop_requires_structural_or_semantic_evidence")
        elif name == EditAction.RETYPE:
            target = targets[0]
            if operation.type == target.type:
                errors.append("retype_is_noop")
            elif target.strong_consensus and operation.type not in target.allowed_types:
                errors.append("strong_candidate_retype_without_competing_evidence")
            else:
                final_type = operation.type or target.type
                replacement = NerEntity(
                    target.text, final_type,
                    normalize_assertions_for_type(final_type, target.assertions),
                    target.position, max(target.scores.values(), default=1.0),
                )
        elif name in {EditAction.REPAIR_SPAN, EditAction.MERGE}:
            assert operation.local_position is not None and operation.text is not None
            start = context_start + operation.local_position[0]
            end = context_start + operation.local_position[1]
            if not (0 <= start < end <= len(raw_text)) or raw_text[start:end] != operation.text:
                errors.append("invalid_exact_span")
            elif not _same_unit(raw_text, start, end):
                errors.append("crosses_structural_boundary")
            elif name == EditAction.REPAIR_SPAN and not (start < targets[0].position[1] and end > targets[0].position[0]):
                errors.append("repair_does_not_overlap_target")
            elif name == EditAction.REPAIR_SPAN and any(
                other.candidate_id not in ids
                and start < other.position[1] and end > other.position[0]
                for other in validation_candidates
            ):
                errors.append("repair_would_swallow_other_candidate_use_merge")
            elif name == EditAction.MERGE and not all(start <= item.position[0] and end >= item.position[1] for item in targets):
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
                    operation.text, final_type,
                    normalize_assertions_for_type(final_type, operation.assertions),
                    (start, end), 1.0,
                )
        elif name == EditAction.UPDATE_ASSERTIONS:
            target = targets[0]
            if target.type not in ASSERTION_ENTITY_TYPES:
                errors.append("assertions_not_allowed_for_type")
            else:
                replacement = NerEntity(
                    target.text, target.type,
                    normalize_assertions_for_type(target.type, operation.assertions),
                    target.position, max(target.scores.values(), default=1.0),
                )
        elif name == EditAction.FLAG_UNRESOLVED:
            result.unresolved.extend(ids)

        if errors:
            result.rejected.append({"reason": errors, "action": asdict(operation)})
            continue
        if name == EditAction.KEEP and ids[0] not in entity_map:
            entity_map[ids[0]] = _to_entity(targets[0])
        if name == EditAction.DROP:
            consumed.update(ids)
        elif replacement is not None:
            consumed.update(ids)
            additions.append(replacement)
        result.applied.append({"action": name.value, "candidate_ids": ids})

    if not change_only:
        covered = set(counts)
        result.unresolved.extend(sorted(target_ids - covered))
    final = [entity for candidate_id, entity in entity_map.items() if candidate_id not in consumed]
    final.extend(additions)
    unique = {(item.position[0], item.position[1], item.type): item for item in final}
    accepted = []
    for entity in sorted(
        unique.values(),
        key=lambda item: (-float(item.score), -(item.position[1] - item.position[0]), *item.position, item.type),
    ):
        if any(
            entity.position[0] < old.position[1]
            and entity.position[1] > old.position[0]
            for old in accepted
        ):
            result.rejected.append({
                "reason": "post_editor_overlap_resolution",
                "entity": {"text": entity.text, "type": entity.type, "position": list(entity.position)},
            })
            continue
        accepted.append(entity)
    result.entities = sorted(accepted, key=lambda item: (*item.position, item.type))
    for entity in result.entities:
        if raw_text[entity.position[0]:entity.position[1]] != entity.text or any(c in entity.text for c in "\r\n"):
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
            result.rejected.append({"reason": "unknown_or_duplicate_proposal_id", "proposal_id": decision.proposal_id})
            continue
        seen.add(decision.proposal_id)
        proposal = by_id[decision.proposal_id]
        if decision.decision != MissingDecisionAction.ADD_PROPOSAL:
            if decision.decision == MissingDecisionAction.UNRESOLVED:
                result.unresolved.append(decision.proposal_id)
            continue
        start, end = proposal.position
        overlaps = [item for item in result.entities if start < item.position[1] and end > item.position[0]]
        errors = []
        if decision.type not in proposal.allowed_types:
            errors.append("type_not_allowed")
        if not proposal.auto_add_eligible or not proposal.hard_supports:
            errors.append("insufficient_support")
        if proposal.negative_flags:
            errors.append("structural_negative")
        if overlaps:
            errors.append("unsafe_overlap")
        if raw_text[start:end] != proposal.text or any(c in proposal.text for c in "\r\n"):
            errors.append("invalid_exact_span")
        if errors:
            result.rejected.append({"reason": errors, "proposal_id": proposal.proposal_id})
            continue
        final_type = decision.type or proposal.allowed_types[0]
        entity = NerEntity(
            proposal.text, final_type,
            normalize_assertions_for_type(final_type, decision.assertions),
            proposal.position, 1.0,
        )
        result.entities.append(entity)
        result.applied.append({"action": "ADD_PROPOSAL", "proposal_id": proposal.proposal_id})
    result.unresolved.extend(sorted(set(by_id) - seen))
    result.entities.sort(key=lambda item: (*item.position, item.type))
    return result
