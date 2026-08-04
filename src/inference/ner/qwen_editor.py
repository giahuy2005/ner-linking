"""Qwen3-8B locked candidate editor with action-level fail-safe guards."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any

from ...llm.batching import VersionedJsonlCache, generate_with_cache
from ...llm.json_guard import extract_json
from ..rule.clinical import repair_assertions_only
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


PROMPT_VERSION = "qwen3_locked_editor_v9_final_ner_scope_safe"
_EDITOR_RESPONSE_FIELDS = frozenset({"request_id", "changes", "unresolved_ids"})
_MISSING_RESPONSE_FIELDS = frozenset({"request_id", "additions", "unresolved_ids"})


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
Chỉ trả các THAY ĐỔI cần thiết cho selected_target. Target bị lược khỏi changes nghĩa là KEEP.
Context-only chỉ để tham khảo; không được tự thêm/promote/chỉnh nó, trừ khi tham gia MERGE với ít nhất một target.
Action duy nhất hợp lệ: DROP, RETYPE, REPAIR_SPAN, MERGE, UPDATE_ASSERTIONS.
Không được xuất KEEP, FLAG_UNRESOLVED, confidence, reasoning hoặc markdown. Mỗi candidate chỉ nên xuất hiện trong một change; nếu MERGE hợp lệ thì không đồng thời DROP/REPAIR cùng candidate đó.
ID phải có trong payload. MERGE cần ít nhất 2 ID. Span [start,end) là local exact substring, không qua newline/câu.
MERGE chỉ dùng khi các mảnh tạo thành MỘT mention y tế bị tách. Không merge hai triệu chứng/chẩn đoán độc lập chỉ vì chúng gần nhau hoặc ngăn bởi dấu phẩy/"và"/"hoặc".
Nếu candidate là giải phẫu, mẫu bệnh phẩm, cơ chế, hoạt động hoặc khái niệm ngoài 5 loại ontology thì dùng DROP với reason phù hợp; KHÔNG RETYPE sang loại ngoài ontology.
Không chắc chắn: chỉ đưa ID selected_target vào unresolved_ids.
Mỗi change phải có đúng 7 field: action,candidate_ids,text,type,assertions,local_position,reason_code.
Field không dùng phải là null hoặc []. REPAIR_SPAN/MERGE phải trả type và assertions cuối cùng.
assertions BẮT BUỘC là JSON list[str], chỉ được chứa đúng các chuỗi "isNegated", "isHistorical", "isFamily"; không được chứa object, boolean hoặc key type/text. Không có assertion thì dùng [].
Assertion scope:
- isNegated chỉ khi phủ định trực tiếp đúng entity; "không thể/không nhấc/không đáp ứng" mô tả thiếu hụt dương tính, không phải phủ định entity.
- Cue phủ định nằm sau một phần dương tính như "viêm kết mạc ... không ghèn" không được phủ định toàn span.
- isHistorical chỉ cho tình trạng/thuốc quá khứ của bệnh nhân hoặc section tiền sử; danh sách kiến thức/yếu tố nguy cơ không phải tiền sử bệnh nhân.
- isFamily chỉ khi entity thuộc người thân hoặc section tiền sử gia đình; bạn bè, đồng nghiệp, người cùng đơn vị không phải gia đình.
reason_code bắt buộc: RETYPE=WRONG_TYPE; REPAIR_SPAN=WRONG_BOUNDARY; MERGE=MERGE_REQUIRED; UPDATE_ASSERTIONS=ASSERTION_ERROR;
DROP chỉ dùng một trong: {drop_reasons}.
Ontology duy nhất: TRIỆU_CHỨNG, CHẨN_ĐOÁN, THUỐC, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM.
Ví dụ DROP hợp lệ: {{"action":"DROP","candidate_ids":["id"],"text":null,"type":null,"assertions":[],"local_position":null,"reason_code":"FUNCTION_WORD_OR_FRAGMENT"}}.
Ví dụ MERGE hợp lệ: {{"action":"MERGE","candidate_ids":["id1","id2"],"text":"mention exact","type":"TRIỆU_CHỨNG","assertions":[],"local_position":[10,23],"reason_code":"MERGE_REQUIRED"}}.
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


def build_editor_retry_request(
    original_prompt: tuple[str, str],
    invalid_response: str,
    validation_error: str,
) -> tuple[str, str]:
    """Build one corrective retry prompt instead of repeating a deterministic error."""
    system, user = original_prompt
    try:
        payload = json.loads(user)
    except json.JSONDecodeError:
        payload = {"original_request": user}
    payload["retry_correction"] = {
        "validation_error": validation_error,
        "previous_response": invalid_response[:1800],
        "instruction": (
            "Sửa đúng lỗi validation và trả lại duy nhất JSON envelope. "
            "assertions phải là list[str]; type chỉ thuộc ontology; "
            "unresolved_ids chỉ chứa selected_target."
        ),
    }
    return (
        system + "\nĐây là lần retry duy nhất. Không lặp lại response sai; sửa đúng validation_error.",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )


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


def build_missing_retry_request(
    original_prompt: tuple[str, str],
    invalid_response: str,
    validation_error: str,
) -> tuple[str, str]:
    """Build one corrective retry for malformed missing-proposal JSON."""
    system, user = original_prompt
    try:
        payload = json.loads(user)
    except json.JSONDecodeError:
        payload = {"original_request": user}
    payload["retry_correction"] = {
        "validation_error": validation_error,
        "previous_response": invalid_response[:1600],
        "instruction": (
            "Trả duy nhất JSON envelope đúng request_id. additions phải là list object; "
            "proposal_id phải thuộc payload; assertions phải là list[str]."
        ),
    }
    return (
        system + "\nĐây là lần retry duy nhất. Sửa đúng validation_error, không thêm reasoning.",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )


def _parse_missing_envelope(
    raw_response: str,
    *,
    expected_request_id: str = "",
    allowed_proposal_ids: set[str] | None = None,
) -> dict[str, Any]:
    payload = extract_json(raw_response)
    if not isinstance(payload, dict):
        raise TypeError("missing response must be a JSON object")
    fields = set(payload)
    missing = _MISSING_RESPONSE_FIELDS - fields
    extra = fields - _MISSING_RESPONSE_FIELDS
    if missing:
        raise ValueError(f"missing response missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"missing response has unsupported fields: {sorted(extra)}")
    request_id = payload.get("request_id")
    if not isinstance(request_id, str):
        raise TypeError("request_id must be a string")
    if expected_request_id and request_id != expected_request_id:
        raise ValueError(
            f"request_id mismatch: expected={expected_request_id!r}, got={request_id!r}"
        )
    additions = payload.get("additions")
    unresolved = payload.get("unresolved_ids")
    if not isinstance(additions, list):
        raise TypeError("additions must be a list")
    if not isinstance(unresolved, list) or not all(isinstance(item, str) for item in unresolved):
        raise TypeError("unresolved_ids must be list[str]")
    if len(unresolved) != len(set(unresolved)):
        raise ValueError("unresolved_ids contains duplicates")
    seen: set[str] = set()
    for index, row in enumerate(additions):
        if not isinstance(row, dict):
            raise TypeError(f"additions[{index}] must be an object")
        decision = MissingDecision.from_dict({
            **row,
            "decision": "ADD_PROPOSAL",
            "confidence": "HIGH",
            "reason_code": "VALID_MISSING_ENTITY",
        })
        if decision.proposal_id in seen:
            raise ValueError(f"duplicate proposal_id: {decision.proposal_id}")
        seen.add(decision.proposal_id)
        if allowed_proposal_ids is not None and decision.proposal_id not in allowed_proposal_ids:
            raise ValueError(f"unknown proposal_id: {decision.proposal_id}")
    if allowed_proposal_ids is not None:
        unknown = set(unresolved) - allowed_proposal_ids
        if unknown:
            raise ValueError(f"unknown unresolved proposal IDs: {sorted(unknown)}")
    conflict = seen & set(unresolved)
    if conflict:
        raise ValueError(f"proposal both added and unresolved: {sorted(conflict)}")
    return payload


def missing_response_error(
    raw_response: str,
    *,
    expected_request_id: str = "",
    allowed_proposal_ids: set[str] | None = None,
) -> str | None:
    try:
        _parse_missing_envelope(
            raw_response,
            expected_request_id=expected_request_id,
            allowed_proposal_ids=allowed_proposal_ids,
        )
    except Exception as exc:
        return str(exc)
    return None


def missing_response_is_valid(
    raw_response: str,
    *,
    expected_request_id: str = "",
    allowed_proposal_ids: set[str] | None = None,
) -> bool:
    return missing_response_error(
        raw_response,
        expected_request_id=expected_request_id,
        allowed_proposal_ids=allowed_proposal_ids,
    ) is None


def parse_missing_response(
    raw_response: str,
    *,
    expected_request_id: str = "",
    allowed_proposal_ids: set[str] | None = None,
) -> tuple[list[MissingDecision], list[dict[str, Any]]]:
    try:
        payload = _parse_missing_envelope(
            raw_response,
            expected_request_id=expected_request_id,
            allowed_proposal_ids=allowed_proposal_ids,
        )
    except Exception as exc:
        return [], [{"reason": "invalid_json", "detail": str(exc)}]

    rows = [{
        **row,
        "decision": "ADD_PROPOSAL",
        "confidence": "HIGH",
        "reason_code": "VALID_MISSING_ENTITY",
    } for row in payload["additions"]]
    rows.extend({
        "proposal_id": proposal_id,
        "decision": "UNRESOLVED",
        "type": None,
        "assertions": [],
        "confidence": "LOW",
        "reason_code": "AMBIGUOUS",
    } for proposal_id in payload["unresolved_ids"])

    decisions: list[MissingDecision] = []
    rejected: list[dict[str, Any]] = []
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
    allowed_candidate_ids: set[str] | None = None,
    target_candidate_ids: set[str] | None = None,
) -> str | None:
    """Validate only the envelope and closed operation schema.

    Semantic/action conflicts are intentionally handled operation-by-operation
    in ``apply_editor_response``. One bad or duplicate action must not force a
    retry that discards other valid actions in the same response.
    """
    del allowed_candidate_ids, target_candidate_ids
    try:
        payload = _parse_editor_envelope(
            raw_response,
            expected_request_id=expected_request_id,
        )
        for index, raw_action in enumerate(payload["changes"]):
            try:
                EditOperation.from_dict(raw_action)
            except Exception as exc:
                raise ValueError(f"changes[{index}] invalid: {exc}") from exc
    except Exception as exc:
        return str(exc)
    return None

def editor_response_is_valid(
    raw_response: str,
    *,
    expected_request_id: str,
    allowed_candidate_ids: set[str] | None = None,
    target_candidate_ids: set[str] | None = None,
) -> bool:
    return editor_response_error(
        raw_response,
        expected_request_id=expected_request_id,
        allowed_candidate_ids=allowed_candidate_ids,
        target_candidate_ids=target_candidate_ids,
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


@dataclass
class _OperationPlan:
    operation: EditOperation
    consumed_ids: set[str]
    replacement: NerEntity | None
    before_entities: list[dict[str, Any]]
    priority: tuple[float, ...]
    realigned_from: list[int] | None = None
    realigned_to: list[int] | None = None


def _operation_base_priority(action: EditAction) -> int:
    return {
        EditAction.MERGE: 500,
        EditAction.REPAIR_SPAN: 400,
        EditAction.RETYPE: 300,
        EditAction.UPDATE_ASSERTIONS: 200,
        EditAction.DROP: 100,
    }[action]


def _find_exact_occurrences(
    raw_text: str,
    text: str,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    cursor = raw_text.find(text, start, end)
    while cursor >= 0:
        output.append((cursor, cursor + len(text)))
        cursor = raw_text.find(text, cursor + 1, end)
    return output


def _span_validation_errors(
    raw_text: str,
    start: int,
    end: int,
    *,
    operation: EditOperation,
    targets: list[CandidateEvidence],
    validation_candidates: list[CandidateEvidence],
) -> list[str]:
    errors: list[str] = []
    if not (0 <= start < end <= len(raw_text)) or raw_text[start:end] != operation.text:
        return ["invalid_exact_span"]
    if not _same_unit(raw_text, start, end):
        errors.append("crosses_structural_boundary")
    if operation.action == EditAction.REPAIR_SPAN:
        target = targets[0]
        if not (start < target.position[1] and end > target.position[0]):
            errors.append("repair_does_not_overlap_target")
        # Audit-only alternatives must not block a safe boundary repair. Only a
        # selected entity that would actually be swallowed requires MERGE.
        if any(
            other.candidate_id not in operation.candidate_ids
            and other.pre_llm_selected
            and start < other.position[1]
            and end > other.position[0]
            for other in validation_candidates
        ):
            errors.append("repair_would_swallow_selected_candidate_use_merge")
    elif operation.action == EditAction.MERGE:
        if not all(
            start <= item.position[0] and end >= item.position[1]
            for item in targets
        ):
            errors.append("merge_does_not_cover_targets")
        ordered = sorted(targets, key=lambda item: item.position)
        if any(
            right.position[0] - left.position[1] > 32
            for left, right in zip(ordered, ordered[1:])
        ):
            errors.append("merge_targets_not_adjacent")
        if (
            len(ordered) >= 2
            and all(item.type == "THUỐC" and item.strong_consensus for item in ordered)
            and any(
                re.search(r"\b(?:va|và|and|hoac|hoặc)\b|[+,;]", _fold_surface(
                    raw_text[left.position[1]:right.position[0]]
                ))
                for left, right in zip(ordered, ordered[1:])
            )
        ):
            errors.append("merge_independent_strong_drugs")
        final_type = operation.type or targets[0].type
        if final_type != "THUỐC":
            strong_atomic = []
            for item in ordered:
                item_score = max((float(value) for value in item.scores.values()), default=0.0)
                temporary = NerEntity(item.text, item.type, [], item.position, item_score)
                if (
                    item.type == final_type
                    and item_score >= 0.88
                    and _unsafe_fragment_reason(temporary) is None
                ):
                    strong_atomic.append(item)
            non_overlapping = []
            for item in strong_atomic:
                if not non_overlapping or item.position[0] >= non_overlapping[-1].position[1]:
                    non_overlapping.append(item)
            if len(non_overlapping) >= 2 and all(
                re.fullmatch(
                    r"(?iu)[\s,;/]*(?:(?:và|va|hay|hoặc|hoac|kèm|kem)[\s,;/]*)?",
                    raw_text[left.position[1]:right.position[0]],
                )
                for left, right in zip(non_overlapping, non_overlapping[1:])
            ):
                errors.append("merge_independent_strong_concepts")
        for other in validation_candidates:
            other_score = max((float(value) for value in other.scores.values()), default=0.0)
            is_numeric_measurement = bool(re.search(
                r"\d\s*(?:mg|mcg|g|ml|l|mmol|meq|iu|%)(?:\s*/\s*\d*\s*[a-z]+)?",
                _fold_surface(other.text),
            ))
            if (
                other.type == "KẾT_QUẢ_XÉT_NGHIỆM"
                and final_type != other.type
                and other_score >= 0.80
                and is_numeric_measurement
                and start <= other.position[0]
                and end >= other.position[1]
            ):
                errors.append("merge_would_absorb_typed_measurement")
                break
    return errors


def _resolve_operation_span(
    raw_text: str,
    operation: EditOperation,
    targets: list[CandidateEvidence],
    validation_candidates: list[CandidateEvidence],
    *,
    context_start: int,
    context_end: int | None,
) -> tuple[int | None, int | None, list[str], list[int] | None, list[int] | None]:
    assert operation.local_position is not None and operation.text is not None
    original_start = context_start + operation.local_position[0]
    original_end = context_start + operation.local_position[1]
    direct_errors = _span_validation_errors(
        raw_text,
        original_start,
        original_end,
        operation=operation,
        targets=targets,
        validation_candidates=validation_candidates,
    )
    if not direct_errors:
        return original_start, original_end, [], None, None

    # Realign only by an exact, unique occurrence inside the review context.
    # No fuzzy matching is allowed because offsets are a scoring invariant.
    search_start = max(0, context_start)
    search_end = min(
        len(raw_text),
        context_end if context_end is not None else max(
            [item.position[1] for item in targets] + [original_end]
        ) + 256,
    )
    valid_occurrences: list[tuple[int, int]] = []
    for start, end in _find_exact_occurrences(
        raw_text,
        operation.text,
        search_start,
        search_end,
    ):
        if not _span_validation_errors(
            raw_text,
            start,
            end,
            operation=operation,
            targets=targets,
            validation_candidates=validation_candidates,
        ):
            valid_occurrences.append((start, end))
    if len(valid_occurrences) == 1:
        start, end = valid_occurrences[0]
        return (
            start,
            end,
            [],
            [original_start, original_end],
            [start, end],
        )
    if len(valid_occurrences) > 1:
        return None, None, ["ambiguous_exact_span_realign"], None, None
    return None, None, direct_errors, None, None


def _plan_operation(
    raw_text: str,
    operation: EditOperation,
    *,
    by_id: dict[str, CandidateEvidence],
    entity_map: dict[str, NerEntity],
    target_ids: set[str],
    validation_candidates: list[CandidateEvidence],
    context_start: int,
    context_end: int | None,
) -> tuple[_OperationPlan | None, list[str]]:
    ids = operation.candidate_ids
    errors: list[str] = []
    if any(candidate_id not in by_id for candidate_id in ids):
        return None, ["unknown_candidate_id"]
    if not (set(ids) & target_ids):
        return None, ["context_only_change_without_target"]
    if operation.action == EditAction.MERGE:
        current_target_ids = [
            candidate_id
            for candidate_id in ids
            if candidate_id in target_ids and candidate_id in entity_map
        ]
        if not current_target_ids:
            errors.append("merge_has_no_current_target_entity")
    elif ids[0] not in entity_map:
        errors.append("target_not_in_current_entities")
    if errors:
        return None, errors

    targets = [by_id[candidate_id] for candidate_id in ids]
    replacement: NerEntity | None = None
    realigned_from = realigned_to = None
    before_entities = [
        _entity_audit(entity_map[candidate_id])
        for candidate_id in ids
        if candidate_id in entity_map
    ]

    if operation.action == EditAction.DROP:
        pass
    elif operation.action == EditAction.RETYPE:
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
                max(current.score, 1.0),
            )
    elif operation.action in {EditAction.REPAIR_SPAN, EditAction.MERGE}:
        start, end, span_errors, realigned_from, realigned_to = _resolve_operation_span(
            raw_text,
            operation,
            targets,
            validation_candidates,
            context_start=context_start,
            context_end=context_end,
        )
        errors.extend(span_errors)
        if not errors:
            assert start is not None and end is not None and operation.text is not None
            final_type = operation.type or targets[0].type
            replacement = NerEntity(
                operation.text,
                final_type,
                normalize_assertions_for_type(final_type, operation.assertions),
                (start, end),
                1.0,
                "qwen_merge" if operation.action == EditAction.MERGE else "qwen_repair",
            )
    elif operation.action == EditAction.UPDATE_ASSERTIONS:
        current = entity_map[ids[0]]
        if current.type not in ASSERTION_ENTITY_TYPES:
            errors.append("assertions_not_allowed_for_type")
        else:
            replacement = NerEntity(
                current.text,
                current.type,
                normalize_assertions_for_type(current.type, operation.assertions),
                current.position,
                max(current.score, 1.0),
            )

    consumed_ids = {candidate_id for candidate_id in ids if candidate_id in entity_map}
    if replacement is not None:
        overlapping_unconsumed = [
            candidate_id
            for candidate_id, entity in entity_map.items()
            if candidate_id not in consumed_ids
            and replacement.position[0] < entity.position[1]
            and replacement.position[1] > entity.position[0]
        ]
        if operation.action == EditAction.MERGE and overlapping_unconsumed:
            # Qwen often names a longer context candidate but omits a selected
            # fragment fully contained by the same replacement. Auto-consume
            # only weak/contested contained entities from this review region;
            # never swallow an independent strong-consensus entity.
            auto_consumed: set[str] = set()
            for candidate_id in overlapping_unconsumed:
                entity = entity_map[candidate_id]
                candidate = by_id.get(candidate_id)
                fully_contained = (
                    replacement.position[0] <= entity.position[0]
                    and replacement.position[1] >= entity.position[1]
                )
                structurally_related = bool(
                    candidate is not None
                    and (
                        candidate_id in target_ids
                        or set(candidate.related_candidate_ids) & set(ids)
                        or any(
                            candidate_id in item.related_candidate_ids
                            for item in targets
                        )
                    )
                )
                contested = bool(
                    candidate is not None
                    and (
                        not candidate.strong_consensus
                        or candidate.negative_flags
                    )
                )
                if fully_contained and structurally_related and contested:
                    auto_consumed.add(candidate_id)
            consumed_ids.update(auto_consumed)
            overlapping_unconsumed = [
                candidate_id for candidate_id in overlapping_unconsumed
                if candidate_id not in auto_consumed
            ]
            if auto_consumed:
                before_entities.extend(
                    _entity_audit(entity_map[candidate_id])
                    for candidate_id in sorted(auto_consumed)
                )
        if overlapping_unconsumed:
            errors.append("replacement_overlaps_unconsumed_entity")

    if errors:
        return None, errors

    target_count = len(consumed_ids & target_ids)
    replacement_span = (
        replacement.position[1] - replacement.position[0]
        if replacement is not None
        else 0
    )
    target_start = min((item.position[0] for item in targets), default=0)
    target_end = max((item.position[1] for item in targets), default=0)
    extra_chars = max(0, replacement_span - (target_end - target_start))
    priority = (
        float(_operation_base_priority(operation.action)),
        float(target_count),
        float(len(consumed_ids)),
        float(-extra_chars),
        float(replacement_span),
    )
    return _OperationPlan(
        operation=operation,
        consumed_ids=consumed_ids,
        replacement=replacement,
        before_entities=before_entities,
        priority=priority,
        realigned_from=realigned_from,
        realigned_to=realigned_to,
    ), []


def apply_editor_response(
    raw_text: str,
    candidates: list[CandidateEvidence],
    raw_response: str,
    *,
    context_start: int = 0,
    context_end: int | None = None,
    validation_candidates: list[CandidateEvidence] | None = None,
    target_candidate_ids: list[str] | None = None,
    expected_request_id: str = "",
    baseline_entities: list[NerEntity] | None = None,
) -> EditorResult:
    """Apply valid actions independently and resolve duplicate actions safely."""
    target_ids = set(target_candidate_ids or [item.candidate_id for item in candidates])
    by_id = {item.candidate_id: item for item in candidates}
    validation_candidates = validation_candidates or candidates

    if baseline_entities is None:
        baseline_entities = [_to_entity(item) for item in candidates if item.pre_llm_selected]
    baseline_by_key = {_entity_key(item): item for item in baseline_entities}
    entity_map = {
        item.candidate_id: baseline_by_key[_entity_key(_to_entity(item))]
        for item in candidates
        if _entity_key(_to_entity(item)) in baseline_by_key
    }
    region_baseline = list({_entity_key(entity): entity for entity in entity_map.values()}.values())
    result = EditorResult(
        entities=sorted(region_baseline, key=lambda item: (*item.position, item.type)),
        raw_response=raw_response,
    )

    try:
        payload = _parse_editor_envelope(raw_response, expected_request_id=expected_request_id)
    except Exception as exc:
        result.rejected.append({"reason": "invalid_response_envelope", "detail": str(exc)})
        result.unresolved.extend(sorted(target_ids))
        return result

    unresolved_ids = payload["unresolved_ids"]
    for candidate_id in unresolved_ids:
        if candidate_id not in target_ids:
            result.rejected.append({
                "reason": "unknown_or_non_target_unresolved_id",
                "candidate_id": candidate_id,
            })
        else:
            result.unresolved.append(candidate_id)

    plans: list[_OperationPlan] = []
    for action_index, raw_action in enumerate(payload["changes"]):
        try:
            operation = EditOperation.from_dict(raw_action)
        except Exception as exc:
            result.rejected.append({
                "reason": "invalid_action_schema",
                "detail": str(exc),
                "action_index": action_index,
                "action": raw_action,
            })
            continue
        plan, errors = _plan_operation(
            raw_text,
            operation,
            by_id=by_id,
            entity_map=entity_map,
            target_ids=target_ids,
            validation_candidates=validation_candidates,
            context_start=context_start,
            context_end=context_end,
        )
        if plan is None:
            result.rejected.append({
                "reason": errors,
                "action_index": action_index,
                "action": asdict(operation),
            })
            continue
        plans.append(plan)

    # Greedy set packing after full validation: MERGE/REPAIR wins over a DROP
    # or duplicate lower-information action touching the same candidate.
    selected: list[_OperationPlan] = []
    claimed_ids: set[str] = set()
    for plan in sorted(plans, key=lambda item: item.priority, reverse=True):
        conflict = claimed_ids & plan.consumed_ids
        if conflict:
            result.rejected.append({
                "reason": "superseded_by_higher_priority_action",
                "conflicting_candidate_ids": sorted(conflict),
                "action": asdict(plan.operation),
            })
            continue
        selected.append(plan)
        claimed_ids.update(plan.consumed_ids)

    consumed: set[str] = set()
    additions: list[NerEntity] = []
    for plan in selected:
        consumed.update(plan.consumed_ids)
        if plan.replacement is not None:
            additions.append(plan.replacement)
        audit_row = {
            "action": plan.operation.action.value,
            "candidate_ids": list(plan.operation.candidate_ids),
            "entities_before": plan.before_entities,
            "entity_after": _entity_audit(plan.replacement) if plan.replacement else None,
        }
        if plan.realigned_to is not None:
            audit_row["local_position_realigned_from"] = plan.realigned_from
            audit_row["global_position_realigned_to"] = plan.realigned_to
        result.applied.append(audit_row)

    final = [
        entity
        for candidate_id, entity in entity_map.items()
        if candidate_id not in consumed
    ]
    final.extend(additions)
    result.entities = sorted(
        {_entity_key(item): item for item in final}.values(),
        key=lambda item: (*item.position, item.type),
    )
    result.consumed_candidate_ids = sorted(consumed)
    result.unresolved = [
        candidate_id
        for candidate_id in dict.fromkeys(result.unresolved)
        if candidate_id not in consumed
    ]
    for entity in result.entities:
        if raw_text[entity.position[0]:entity.position[1]] != entity.text or any(
            char in entity.text for char in "\r\n"
        ):
            raise ValueError("editor produced invalid exact offset")
    return result


_FOLD_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_FUNCTION_FRAGMENTS = frozenset({
    "cac", "co", "con", "cua", "duoc", "hang", "hay", "it", "la",
    "mot", "nhieu", "nhung", "phai", "sau", "tai", "theo", "trong",
    "tu", "va", "voi", "hoac", "khong",
})
_DIAGNOSIS_FRAGMENTS = frozenset({
    "benh", "viem", "ton", "mien", "xoang", "gan", "tuy", "da", "chi",
    "virus", "fibrin", "gen", "enzyme", "hong cau", "da day", "phe nang", "ho",
})
_TEST_FRAGMENTS = frozenset({
    "xet", "chup", "do", "duong", "mau", "phan", "dich", "chi", "gan", "tuy",
})
_SYMPTOM_FRAGMENTS = frozenset({"chi", "da", "duoc", "cac", "nhieu", "sau", "ton", "mien"})
_DRUG_NOISE_TOKENS = frozenset({
    "ngay", "hang", "lieu", "cham", "moi", "tuan", "lan", "cach", "uong",
    "tiem", "truyen", "po", "iv", "im", "bid", "tid", "qid", "prn",
    "mg", "mcg", "g", "ml", "l", "gram", "microgram", "microgam",
    "giot", "phut", "sui", "vien", "ong", "goi", "chai", "kem",
})
_DRUG_PROCEDURE_PREFIX_RE = re.compile(
    r"^(?:tho|loc mau|phau thuat|dat |chup |xet nghiem|truyen dich\b)"
)
_COORDINATION_RE = re.compile(r"(?:\b(?:va|hoac|hay|kem)\b|[,;/()]|\s[-–—]\s)")


def _fold_surface(text: str) -> str:
    value = unicodedata.normalize("NFD", text.casefold()).replace("đ", "d")
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", value).strip(" \t\r\n.,;:()[]{}")


def _surface_tokens(text: str) -> list[str]:
    return [token for token in _FOLD_TOKEN_RE.split(_fold_surface(text)) if token]


def _medical_abbreviation(text: str) -> bool:
    compact = "".join(char for char in text if char.isalnum())
    letters = [char for char in compact if char.isalpha()]
    return bool(
        2 <= len(compact) <= 10
        and letters
        and all(char.isupper() for char in letters)
    )


def _unsafe_fragment_reason(entity: NerEntity) -> str | None:
    folded = _fold_surface(entity.text)
    tokens = _surface_tokens(entity.text)
    if _medical_abbreviation(entity.text):
        return None
    if len(tokens) == 1:
        token = tokens[0]
        if token in _FUNCTION_FRAGMENTS:
            return "function_word_fragment"
        if entity.type == "CHẨN_ĐOÁN" and token in _DIAGNOSIS_FRAGMENTS:
            return "generic_or_non_diagnostic_single_token"
        if entity.type == "TÊN_XÉT_NGHIỆM" and token in _TEST_FRAGMENTS:
            return "incomplete_test_name"
        if entity.type == "TRIỆU_CHỨNG" and token in _SYMPTOM_FRAGMENTS:
            return "non_symptom_single_token"
    if entity.type == "CHẨN_ĐOÁN" and folded in _DIAGNOSIS_FRAGMENTS:
        return "generic_or_anatomical_diagnosis_fragment"
    if entity.type == "THUỐC":
        meaningful = [
            token
            for token in tokens
            if token not in _DRUG_NOISE_TOKENS and not token.isdigit()
        ]
        if entity.text.lstrip().startswith("/") or not meaningful:
            return "dose_or_administration_fragment"
        if _DRUG_PROCEDURE_PREFIX_RE.match(folded):
            return "procedure_or_supportive_care_not_drug"
    return None



def _candidate_max_score(item: CandidateEvidence | None) -> float:
    if item is None:
        return 0.0
    return max((float(value) for value in item.scores.values()), default=0.0)


def _line_bounds_for_span(raw_text: str, start: int, end: int) -> tuple[int, int]:
    left = raw_text.rfind("\n", 0, start) + 1
    right = raw_text.find("\n", end)
    if right < 0:
        right = len(raw_text)
    return left, right


def _clause_bounds_for_span(raw_text: str, start: int, end: int) -> tuple[int, int]:
    line_start, line_end = _line_bounds_for_span(raw_text, start, end)
    left_part = raw_text[line_start:start]
    right_part = raw_text[end:line_end]
    left_matches = list(re.finditer(r"[;:.!?]", left_part))
    clause_start = line_start + (left_matches[-1].end() if left_matches else 0)
    right_match = re.search(r"[;:.!?]", right_part)
    clause_end = end + (right_match.start() if right_match else len(right_part))
    return clause_start, clause_end



_SUBTYPE_MODIFIER_RE = re.compile(
    r"^(?:vo can|man tinh|man tinh|cap tinh|toan bo|toan the|"
    r"lan toa|khu tru|nguyen phat|thu phat|tai phat)$"
)


def _recover_subtype_modifier_spans(
    raw_text: str,
    entities: list[NerEntity],
    catalogue: list[CandidateEvidence],
) -> tuple[list[NerEntity], list[dict[str, Any]]]:
    """Restore the medical head before an isolated subtype modifier.

    The rule is evidence-bound: the complete span must already exist as an
    exact catalogue candidate, end at the modifier boundary, stay within one
    clause, contain no coordination, and add only a short medical head.  The
    modifier's selected type is preserved, which resolves mixed CRF/span-head
    disagreements such as a symptom head plus a diagnosis subtype modifier.
    """
    selected = sorted(entities, key=lambda item: (*item.position, item.type))
    consumed: set[tuple[int, int, str]] = set()
    additions: list[NerEntity] = []
    audit: list[dict[str, Any]] = []

    for modifier in selected:
        modifier_key = _entity_key(modifier)
        if modifier_key in consumed or modifier.type != "CHẨN_ĐOÁN":
            continue
        folded_modifier = _fold_surface(modifier.text)
        if not _SUBTYPE_MODIFIER_RE.fullmatch(folded_modifier):
            continue

        proposals: list[tuple[int, float, CandidateEvidence]] = []
        for candidate in catalogue:
            if candidate.position[1] != modifier.position[1]:
                continue
            if not (candidate.position[0] < modifier.position[0]):
                continue
            if raw_text[candidate.position[0]:candidate.position[1]] != candidate.text:
                continue
            if not _same_unit(raw_text, candidate.position[0], candidate.position[1]):
                continue
            if re.search(r"[,;/]|\b(?:va|hoac|hay|kem)\b", _fold_surface(candidate.text)):
                continue
            tokens = _surface_tokens(candidate.text)
            modifier_tokens = _surface_tokens(modifier.text)
            added_count = len(tokens) - len(modifier_tokens)
            if not (1 <= added_count <= 4 and 2 <= len(tokens) <= 7):
                continue
            if not candidate.text.endswith(modifier.text):
                continue
            score = _candidate_max_score(candidate)
            if score < 0.45:
                continue
            if not set(candidate.negative_flags) & {"boundary_disagreement", "type_disagreement"}:
                continue
            proposals.append((candidate.position[0], score, candidate))

        if not proposals:
            continue
        # Prefer the closest complete head, then stronger model evidence.
        _start, _score, winner = max(
            proposals,
            key=lambda row: (row[0], row[1]),
        )
        contained = [
            item for item in selected
            if winner.position[0] <= item.position[0]
            and winner.position[1] >= item.position[1]
            and len(_surface_tokens(item.text)) <= 3
        ]
        assertions = list(dict.fromkeys(
            assertion
            for item in contained
            for assertion in item.assertions
        ))
        replacement = NerEntity(
            winner.text,
            modifier.type,
            normalize_assertions_for_type(modifier.type, assertions or modifier.assertions),
            winner.position,
            max(float(modifier.score or 0.0), _candidate_max_score(winner)),
            modifier.flag,
        )
        for item in contained:
            consumed.add(_entity_key(item))
        consumed.add(modifier_key)
        additions.append(replacement)
        audit.append({
            "reason": "subtype_modifier_head_completion",
            "entities_before": [_entity_audit(item) for item in contained] or [_entity_audit(modifier)],
            "entity_after": _entity_audit(replacement),
            "candidate_id": winner.candidate_id,
        })

    output = [item for item in selected if _entity_key(item) not in consumed]
    output.extend(additions)
    dedup = {_entity_key(item): item for item in output}
    return sorted(dedup.values(), key=lambda item: (*item.position, item.type)), audit


def _recover_single_boundary_expansions(
    raw_text: str,
    entities: list[NerEntity],
    catalogue: list[CandidateEvidence],
) -> tuple[list[NerEntity], list[dict[str, Any]]]:
    """Expand one short boundary fragment to a stronger exact catalogue span.

    The replacement must share a boundary, stay inside one clause, have clearly
    stronger evidence, and must not swallow another selected entity.  This is
    intended for cases such as a generic head word followed by its complement;
    it does not use a private vocabulary or fuzzy offsets.
    """
    selected = list(entities)
    selected_keys = {_entity_key(item) for item in selected}
    evidence_by_key = {
        (item.position[0], item.position[1], item.type): item for item in catalogue
    }
    output: list[NerEntity] = []
    audit: list[dict[str, Any]] = []
    used_replacements: set[tuple[int, int, str]] = set()
    for entity in selected:
        entity_tokens = _surface_tokens(entity.text)
        base_evidence = evidence_by_key.get(_entity_key(entity))
        # Prefer original model evidence when available.  Entities restored
        # from JSON/editor may carry score=1.0 as a placeholder, which must
        # not block a clearly stronger span-head boundary candidate.
        base_score = (
            _candidate_max_score(base_evidence)
            if base_evidence is not None
            else float(entity.score or 0.0)
        )
        if len(entity_tokens) > 2 or _medical_abbreviation(entity.text):
            output.append(entity)
            continue
        candidates = []
        for candidate in catalogue:
            if candidate.type != entity.type or candidate.position == entity.position:
                continue
            if not (
                candidate.position[0] <= entity.position[0]
                and candidate.position[1] >= entity.position[1]
                and (
                    candidate.position[0] == entity.position[0]
                    or candidate.position[1] == entity.position[1]
                )
            ):
                continue
            if candidate.position[1] - candidate.position[0] > 72:
                continue
            if any(ch in candidate.text for ch in "\r\n,;"):
                continue
            candidate_score = _candidate_max_score(candidate)
            if candidate_score < max(0.88, base_score + 0.06):
                continue
            if "boundary_disagreement" not in candidate.negative_flags:
                continue
            if not _same_unit(raw_text, candidate.position[0], candidate.position[1]):
                continue
            if any(
                other is not entity
                and other.position[0] < candidate.position[1]
                and other.position[1] > candidate.position[0]
                and not (
                    candidate.position[0] <= other.position[0]
                    and candidate.position[1] >= other.position[1]
                    and len(_surface_tokens(other.text)) <= 1
                )
                for other in selected
            ):
                continue
            candidates.append((candidate_score, candidate))
        if not candidates:
            output.append(entity)
            continue
        _score, winner = max(candidates, key=lambda row: (row[0], len(row[1].text)))
        key = (winner.position[0], winner.position[1], winner.type)
        if key in used_replacements or key in selected_keys:
            output.append(entity)
            continue
        replacement = NerEntity(
            winner.text,
            winner.type,
            normalize_assertions_for_type(winner.type, entity.assertions),
            winner.position,
            max(float(entity.score or 0.0), _candidate_max_score(winner)),
        )
        used_replacements.add(key)
        output.append(replacement)
        audit.append({
            "reason": "single_fragment_boundary_expansion",
            "entity_before": _entity_audit(entity),
            "entity_after": _entity_audit(replacement),
            "candidate_id": winner.candidate_id,
        })
    dedup = {_entity_key(item): item for item in output}
    return sorted(dedup.values(), key=lambda item: (*item.position, item.type)), audit


def _restore_dominant_contained_mention(
    raw_text: str,
    entities: list[NerEntity],
    catalogue: list[CandidateEvidence],
) -> tuple[list[NerEntity], list[dict[str, Any]]]:
    """Trim a weak wrapper around one dominant exact medical mention."""
    by_key = {(item.position[0], item.position[1], item.type): item for item in catalogue}
    wrapper_tokens = {
        "ho", "chu quan", "co", "bi", "tinh trang", "dau hieu", "trieu chung",
    }
    output: list[NerEntity] = []
    audit: list[dict[str, Any]] = []
    for outer in entities:
        outer_evidence = by_key.get(_entity_key(outer))
        outer_score = _candidate_max_score(outer_evidence)
        weak_wrapper = bool(
            outer.flag == "qwen_merge"
            or (
                outer_evidence is not None
                and not outer_evidence.pre_llm_selected
                and outer_score < 0.84
            )
        )
        if not weak_wrapper or outer.type not in {"CHẨN_ĐOÁN", "TRIỆU_CHỨNG"}:
            output.append(outer)
            continue
        candidates = []
        for item in catalogue:
            if item.type != outer.type or item.position == outer.position:
                continue
            if not (
                outer.position[0] <= item.position[0]
                and outer.position[1] >= item.position[1]
            ):
                continue
            score = _candidate_max_score(item)
            if score < 0.90 or _unsafe_fragment_reason(
                NerEntity(item.text, item.type, [], item.position, score)
            ) is not None:
                continue
            candidates.append((score, item))
        if not candidates:
            output.append(outer)
            continue
        _score, winner = max(
            candidates,
            key=lambda row: (
                sum(char.isalnum() for char in row[1].text),
                row[0],
            ),
        )
        before = _fold_surface(raw_text[outer.position[0]:winner.position[0]])
        after = _fold_surface(raw_text[winner.position[1]:outer.position[1]])
        leftovers = " ".join(value for value in (before, after) if value).strip(" ,;/")
        outer_alnum = sum(char.isalnum() for char in outer.text)
        winner_alnum = sum(char.isalnum() for char in winner.text)
        if (
            outer_alnum <= 0
            or winner_alnum / outer_alnum < 0.58
            or leftovers not in wrapper_tokens
        ):
            output.append(outer)
            continue
        replacement = NerEntity(
            winner.text,
            winner.type,
            normalize_assertions_for_type(winner.type, outer.assertions),
            winner.position,
            max(float(outer.score or 0.0), _candidate_max_score(winner)),
        )
        output.append(replacement)
        audit.append({
            "reason": "restore_dominant_contained_mention",
            "entity_before": _entity_audit(outer),
            "entity_after": _entity_audit(replacement),
        })
    dedup = {_entity_key(item): item for item in output}
    return sorted(dedup.values(), key=lambda item: (*item.position, item.type)), audit


def _restore_independent_catalogue_mentions(
    raw_text: str,
    entities: list[NerEntity],
    catalogue: list[CandidateEvidence],
) -> tuple[list[NerEntity], list[dict[str, Any]]]:
    """Undo only evidence-backed Qwen merges of independent atomic mentions.

    The fallback is deliberately conservative.  For symptoms it requires the
    in-memory ``qwen_merge`` flag.  For diagnoses, a weak span-only outer may be
    repaired from two complete strong inner diagnoses.  THUỐC is excluded
    because immutable concatenated medication spans may carry multiple RxCUIs.
    """
    catalogue_by_key = {
        (item.position[0], item.position[1], item.type): item for item in catalogue
    }
    diagnosis_head = re.compile(
        r"(?iu)^(?:viem|nhiem|ung thu|u|suy|roi loan|hoi chung|ap xe|"
        r"tang huyet ap|ha huyet ap|thieu mau|xuat huyet|loet|nang|"
        r"nhồi mau|nhoi mau|tram cam|stress)\b"
    )
    output: list[NerEntity] = []
    audit: list[dict[str, Any]] = []
    for outer in entities:
        if outer.type not in {"CHẨN_ĐOÁN", "TRIỆU_CHỨNG"}:
            output.append(outer)
            continue
        outer_evidence = catalogue_by_key.get(_entity_key(outer))
        outer_score = _candidate_max_score(outer_evidence)
        eligible_outer = outer.flag == "qwen_merge"
        if outer.type == "CHẨN_ĐOÁN":
            eligible_outer = eligible_outer or bool(
                outer_evidence is not None
                and not outer_evidence.pre_llm_selected
                and outer_score < 0.84
                and "boundary_disagreement" in outer_evidence.negative_flags
            )
        if not eligible_outer:
            output.append(outer)
            continue

        rows: list[tuple[float, CandidateEvidence]] = []
        for item in catalogue:
            if item.type != outer.type or item.position == outer.position:
                continue
            if not (
                outer.position[0] <= item.position[0]
                and outer.position[1] >= item.position[1]
            ):
                continue
            score = _candidate_max_score(item)
            temporary = NerEntity(item.text, item.type, [], item.position, score)
            if score < 0.88 or _unsafe_fragment_reason(temporary) is not None:
                continue
            if _medical_abbreviation(item.text) and len(item.text.strip()) <= 3:
                continue
            if outer.type == "TRIỆU_CHỨNG":
                if not (item.pre_llm_selected or item.strong_consensus):
                    continue
            else:
                if not (item.pre_llm_selected or item.strong_consensus):
                    if score < 0.95 or diagnosis_head.search(_fold_surface(item.text)) is None:
                        continue
            rows.append((score, item))

        chosen: list[CandidateEvidence] = []
        for _score, item in sorted(rows, key=lambda row: (row[1].position[0], -row[0], row[1].position[1])):
            if any(
                item.position[0] < old.position[1] and item.position[1] > old.position[0]
                for old in chosen
            ):
                continue
            chosen.append(item)
        if len(chosen) < 2:
            output.append(outer)
            continue
        chosen.sort(key=lambda item: item.position)
        gaps = [raw_text[left.position[1]:right.position[0]] for left, right in zip(chosen, chosen[1:])]
        if not all(
            re.fullmatch(
                r"(?iu)[\s,;/]*(?:(?:và|va|hay|hoặc|hoac|kèm|kem)[\s,;/]*)?",
                gap,
            )
            for gap in gaps
        ):
            output.append(outer)
            continue
        outer_alnum = sum(char.isalnum() for char in outer.text)
        covered_alnum = sum(sum(char.isalnum() for char in item.text) for item in chosen)
        if outer_alnum <= 0 or covered_alnum / outer_alnum < 0.74:
            output.append(outer)
            continue
        replacements = [
            NerEntity(
                item.text,
                item.type,
                normalize_assertions_for_type(item.type, outer.assertions),
                item.position,
                max(float(outer.score or 0.0), _candidate_max_score(item)),
                "restored_atomic_mention",
            )
            for item in chosen
        ]
        output.extend(replacements)
        audit.append({
            "reason": "restore_independent_catalogue_mentions",
            "entity_before": _entity_audit(outer),
            "entities_after": [_entity_audit(item) for item in replacements],
        })
    dedup = {_entity_key(item): item for item in output}
    return sorted(dedup.values(), key=lambda item: (*item.position, item.type)), audit


def _contextual_role_cleanup(
    raw_text: str,
    entities: list[NerEntity],
    catalogue: list[CandidateEvidence],
) -> tuple[list[NerEntity], list[dict[str, Any]]]:
    """Drop/retype role-confused entities using local grammatical evidence."""
    catalogue_by_key = {
        (item.position[0], item.position[1], item.type): item for item in catalogue
    }
    output: list[NerEntity] = []
    audit: list[dict[str, Any]] = []
    for entity in entities:
        start, end = entity.position
        line_start, line_end = _line_bounds_for_span(raw_text, start, end)
        clause_start, clause_end = _clause_bounds_for_span(raw_text, start, end)
        line = _fold_surface(raw_text[line_start:line_end])
        clause = _fold_surface(raw_text[clause_start:clause_end])
        left = _fold_surface(raw_text[max(line_start, start - 72):start])
        right = _fold_surface(raw_text[end:min(line_end, end + 96)])
        folded = _fold_surface(entity.text)
        evidence = catalogue_by_key.get(_entity_key(entity))
        # Editor-created/restored entities may carry score=1.0 as a neutral
        # placeholder.  For contextual role decisions, prefer the original
        # catalogue confidence whenever it exists; otherwise that placeholder
        # would suppress low-confidence retyping such as CK-MB -> test.
        score = (
            _candidate_max_score(evidence)
            if evidence is not None
            else float(entity.score or 0.0)
        )

        if entity.type == "KẾT_QUẢ_XÉT_NGHIỆM":
            physical_exam_context = _fold_surface(
                raw_text[max(0, line_start - 160):line_end]
            )
            if folded in {"deu", "ro", "mem"} and re.search(
                r"\b(?:kham|cac bo phan|toan than|lam sang|tim|phoi|bung)\b",
                physical_exam_context,
            ):
                audit.append({"reason": "physical_exam_modifier_not_lab_result", "entity": _entity_audit(entity)})
                continue
            if folded == "khong":
                audit.append({"reason": "result_function_word", "entity": _entity_audit(entity)})
                continue
            if folded in {"tang", "giam", "thay doi", "deu", "ro", "mem"}:
                nearby_test = any(
                    item.type == "TÊN_XÉT_NGHIỆM"
                    and item.position[0] < clause_end
                    and item.position[1] > clause_start
                    for item in entities
                )
                explicit_test = bool(re.search(
                    r"\b(?:xet nghiem|ket qua|dinh luong|do nong do|chi so|men|troponin|creatinin|ure|huyet)",
                    clause,
                ))
                numeric = bool(re.search(r"\d", clause))
                if not nearby_test and not explicit_test and not numeric:
                    audit.append({"reason": "result_modifier_without_test", "entity": _entity_audit(entity)})
                    continue

        if entity.type == "TÊN_XÉT_NGHIỆM":
            if re.fullmatch(r"\d+(?:[./]\d+)?", folded):
                audit.append({"reason": "isolated_numeric_not_test_name", "entity": _entity_audit(entity)})
                continue
            physical_heads = {"tim", "phoi", "bung", "gan", "lach", "than", "da", "niem mac"}
            if folded in physical_heads and re.search(
                r"\b(?:kham|cac bo phan|toan than|lam sang)\b", _fold_surface(raw_text[max(0, line_start - 160):line_end])
            ):
                audit.append({"reason": "physical_exam_anatomy_not_test", "entity": _entity_audit(entity)})
                continue
            test_cue = bool(re.search(
                r"\b(?:xet nghiem|test|dinh luong|kiem tra|sang loc|ket qua|"
                r"cay|chup|sieu am|noi soi|dien tam do|ecg|mri|ct|phan tich)\b",
                clause,
            ))
            tight_context = _fold_surface(
                raw_text[max(clause_start, start - 44):min(clause_end, end + 36)]
            )
            tight_test_cue = bool(re.search(
                r"\b(?:xet nghiem|test|dinh luong|kiem tra|sang loc|ket qua|"
                r"cay|chup|sieu am|noi soi|dien tam do|ecg|mri|ct)\b",
                tight_context,
            ))
            numeric_surrounding = _fold_surface(
                raw_text[max(clause_start, start - 44):start]
                + " "
                + raw_text[end:min(clause_end, end + 36)]
            )
            numeric_near = bool(re.search(
                r"(?:[:=<>]\s*\d|\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|ml|l|"
                r"mmol|meq|iu|%|mmhg|bpm|u/l|ng/ml|pg/ml)\b)",
                numeric_surrounding,
            ))
            person_role = bool(re.search(
                r"\b(?:bac si|bs|doctor|primary care|cham soc chinh)\b",
                clause,
            ))
            specimen_pattern = bool(
                re.search(r"\b(?:lay|thu|mau)\b", left)
                and re.search(r"\b(?:de|cho)\s+(?:phan tich|xet nghiem)\b", right)
            )
            mechanism_pattern = bool(re.search(
                r"\b(?:dong vai tro|bao ve|bi pha huy|giam sut|hoat tinh cua|"
                r"so luong va hoat tinh|cau tao|co che|trong te bao|trong hong cau)\b",
                clause,
            ))
            if person_role and not test_cue:
                audit.append({"reason": "person_role_not_test", "entity": _entity_audit(entity)})
                continue
            if specimen_pattern:
                audit.append({"reason": "specimen_not_test", "entity": _entity_audit(entity)})
                continue
            if mechanism_pattern and not (tight_test_cue or numeric_near):
                audit.append({"reason": "biological_object_not_test", "entity": _entity_audit(entity)})
                continue

        if entity.type == "CHẨN_ĐOÁN":
            # A parenthesized English expansion immediately following a
            # symptom is the same symptom mention, not a new diagnosis.
            previous_symptoms = [
                item for item in entities
                if item.type == "TRIỆU_CHỨNG"
                and item.position[1] <= start
                and 0 <= start - item.position[1] <= 4
            ]
            if start > 0 and raw_text[start - 1:start] == "(" and raw_text[end:end + 1] == ")" and previous_symptoms:
                previous = max(previous_symptoms, key=lambda item: item.position[1])
                replacement = NerEntity(
                    entity.text, "TRIỆU_CHỨNG", list(previous.assertions),
                    entity.position, entity.score, entity.flag,
                )
                output.append(replacement)
                audit.append({
                    "reason": "parenthetical_symptom_alias_retype",
                    "entity_before": _entity_audit(entity),
                    "entity_after": _entity_audit(replacement),
                })
                continue
            if folded.startswith("khong ") and re.search(r"\b(?:phan loai|chia thanh)\b", _fold_surface(raw_text[max(0, start - 220):start])):
                audit.append({"reason": "classification_descriptor_not_diagnosis", "entity": _entity_audit(entity)})
                continue
            if re.search(r"\bcham soc\s*$", left) and len(_surface_tokens(entity.text)) <= 4:
                audit.append({"reason": "care_object_not_diagnosis", "entity": _entity_audit(entity)})
                continue
            treatment_context = _fold_surface(raw_text[max(0, start - 180):min(len(raw_text), end + 180)])
            if _medical_abbreviation(entity.text) and re.search(
                r"\b(?:lieu phap|quang tri lieu|chieu|duoc su dung|lan dieu tri|psoralen)\b",
                treatment_context,
            ) and not re.search(r"\b(?:chan doan|mac|bi)\b", clause):
                audit.append({"reason": "treatment_abbreviation_not_diagnosis", "entity": _entity_audit(entity)})
                continue
            if re.match(r"^(?:voi|cua|trong|tai|theo)\b", folded):
                audit.append({"reason": "leading_function_phrase_not_diagnosis", "entity": _entity_audit(entity)})
                continue
            if re.match(r"^benh hoc(?:\s|$)", folded):
                audit.append({"reason": "generic_pathology_descriptor", "entity": _entity_audit(entity)})
                continue
            if re.fullmatch(r"(?:benh|tinh trang)\s+di truyen(?:\s+(?:lan|troi))?", folded):
                audit.append({"reason": "generic_inheritance_descriptor", "entity": _entity_audit(entity)})
                continue
            if re.match(r"^(?:dot bien|bien the)\s+gen\b", folded):
                audit.append({"reason": "genetic_finding_without_diagnosis_label", "entity": _entity_audit(entity)})
                continue
            abbreviation = _medical_abbreviation(entity.text)
            lab_neighbors = any(
                item.type == "TÊN_XÉT_NGHIỆM"
                and item is not entity
                and line_start <= item.position[0] < line_end
                and abs(item.position[0] - start) <= 120
                for item in entities
            )
            assay_shape = bool(re.search(r"[-/+0-9]", entity.text))
            if abbreviation and assay_shape and score < 0.75 and (
                lab_neighbors or re.search(r"[↑↓]|\b(?:xet nghiem|men tim|troponin)\b", line)
            ):
                replacement = NerEntity(
                    entity.text,
                    "TÊN_XÉT_NGHIỆM",
                    [],
                    entity.position,
                    entity.score,
                    entity.flag,
                )
                output.append(replacement)
                audit.append({
                    "reason": "low_confidence_assay_abbreviation_retype",
                    "entity_before": _entity_audit(entity),
                    "entity_after": _entity_audit(replacement),
                })
                continue
        output.append(entity)
    dedup = {_entity_key(item): item for item in output}
    return sorted(dedup.values(), key=lambda item: (*item.position, item.type)), audit


def _entity_quality(
    entity: NerEntity,
    *,
    catalogue_by_key: dict[tuple[int, int, str], CandidateEvidence],
) -> float:
    candidate = catalogue_by_key.get(_entity_key(entity))
    score = float(entity.score or 0.0)
    if candidate is not None:
        score = max(score, max((float(value) for value in candidate.scores.values()), default=0.0))
        if candidate.strong_consensus:
            score += 0.20
    score += min(0.18, 0.04 * max(0, len(_surface_tokens(entity.text)) - 1))
    score += min(0.10, 0.002 * len(entity.text))
    return score


def _invalid_outer_overlap(
    outer: NerEntity,
    inners: list[NerEntity],
) -> bool:
    folded = _fold_surface(outer.text)
    if re.match(r"^(?:danh rang|an uong|di lai|tap luyen)\b", folded):
        return True
    if folded.startswith("ho ") and any(inner.position[0] > outer.position[0] for inner in inners):
        return True
    # A long drug span that absorbs a full negated/toxicity clause is not a
    # medication mention. Keep the contained diagnosis rather than the clause.
    if outer.type == "THUỐC" and re.search(r"\b(?:khong gay|gay doc|gay quai thai)\b", folded):
        return True
    # Result and test mentions should not be fused into one nested span.
    if outer.type == "KẾT_QUẢ_XÉT_NGHIỆM" and any(
        inner.type == "TÊN_XÉT_NGHIỆM" for inner in inners
    ):
        return True
    # Alternatives/coordinated diagnoses are not one mention when at least two
    # independent contained entities already exist.
    non_overlapping_inners = sorted(inners, key=lambda item: item.position)
    independent_count = 0
    cursor = -1
    for inner in non_overlapping_inners:
        if inner.position[0] >= cursor:
            independent_count += 1
            cursor = inner.position[1]
    if independent_count >= 2 and re.search(r"(?:\b(?:hoac|hay|va)\b|/)", folded):
        return True
    return False


def _resolve_global_overlaps(
    entities: list[NerEntity],
    *,
    catalogue: list[CandidateEvidence],
) -> tuple[list[NerEntity], list[dict[str, Any]]]:
    """Resolve only true overlap components with conservative containment rules."""
    if not entities:
        return [], []
    catalogue_by_key = {
        (item.position[0], item.position[1], item.type): item
        for item in catalogue
    }
    ordered = sorted(entities, key=lambda item: (*item.position, item.type))
    components: list[list[NerEntity]] = []
    current: list[NerEntity] = []
    current_end = -1
    for entity in ordered:
        if not current or entity.position[0] < current_end:
            current.append(entity)
            current_end = max(current_end, entity.position[1])
        else:
            components.append(current)
            current = [entity]
            current_end = entity.position[1]
    if current:
        components.append(current)

    selected: list[NerEntity] = []
    audit: list[dict[str, Any]] = []
    for component in components:
        if len(component) == 1:
            selected.extend(component)
            continue
        # Prefer a semantically complete containing span unless it is a clear
        # wrapper/activity/alternative. This avoids the old failure where a
        # dosage fragment beat the full drug mention or "tăng" beat "men gan tăng".
        ranked_outers = sorted(
            component,
            key=lambda item: (
                item.position[1] - item.position[0],
                _entity_quality(item, catalogue_by_key=catalogue_by_key),
            ),
            reverse=True,
        )
        chosen: list[NerEntity] = []
        remaining = list(component)
        for outer in ranked_outers:
            if outer not in remaining:
                continue
            inners = [
                item for item in remaining
                if item is not outer
                and outer.position[0] <= item.position[0]
                and outer.position[1] >= item.position[1]
            ]
            if inners and _invalid_outer_overlap(outer, inners):
                remaining.remove(outer)
                continue
            chosen.append(outer)
            remaining = [
                item for item in remaining
                if item is outer
                or not (
                    outer.position[0] < item.position[1]
                    and outer.position[1] > item.position[0]
                )
            ]
            if outer in remaining:
                remaining.remove(outer)
        # Any crossing spans left after containment handling: choose the best
        # evidence one, never emit overlapping final entities.
        for item in sorted(
            remaining,
            key=lambda entity: _entity_quality(entity, catalogue_by_key=catalogue_by_key),
            reverse=True,
        ):
            if any(
                item.position[0] < kept.position[1]
                and item.position[1] > kept.position[0]
                for kept in chosen
            ):
                continue
            chosen.append(item)
        chosen = sorted(chosen, key=lambda item: (*item.position, item.type))
        selected.extend(chosen)
        chosen_keys = {_entity_key(item) for item in chosen}
        for entity in component:
            if _entity_key(entity) in chosen_keys:
                continue
            conflicts = [
                kept for kept in chosen
                if entity.position[0] < kept.position[1]
                and entity.position[1] > kept.position[0]
            ]
            audit.append({
                "reason": "global_overlap_resolution",
                "entity": _entity_audit(entity),
                "kept_conflicts": [_entity_audit(item) for item in conflicts],
            })
    return sorted(selected, key=lambda item: (*item.position, item.type)), audit

def _recover_catalogue_compositions(
    raw_text: str,
    entities: list[NerEntity],
    catalogue: list[CandidateEvidence],
) -> tuple[list[NerEntity], list[dict[str, Any]]]:
    """Recover a semantically complete span from split low-confidence pieces.

    This is a general boundary reconciliation step, not a surface rule. A
    candidate must exactly cover at least two selected pieces, have structural
    conflict evidence, and must not swallow multiple strong independent spans.
    """
    selected = list(entities)
    selected_keys = {_entity_key(item) for item in selected}
    proposals: list[tuple[float, CandidateEvidence, list[NerEntity]]] = []
    for candidate in catalogue:
        key = (candidate.position[0], candidate.position[1], candidate.type)
        if key in selected_keys or not candidate.text or len(candidate.text) > 96:
            continue
        if raw_text[candidate.position[0]:candidate.position[1]] != candidate.text:
            continue
        score = max((float(value) for value in candidate.scores.values()), default=0.0)
        if score < 0.50 or not set(candidate.negative_flags) & {
            "possible_merge", "boundary_disagreement", "type_disagreement",
        }:
            continue
        contained = [
            entity for entity in selected
            if candidate.position[0] <= entity.position[0]
            and candidate.position[1] >= entity.position[1]
        ]
        if len(contained) < 2:
            continue
        contained_evidence: list[CandidateEvidence | None] = []
        strong_count = 0
        for entity in contained:
            evidence = next((
                item for item in catalogue
                if item.position == entity.position and item.type == entity.type
            ), None)
            contained_evidence.append(evidence)
            if evidence is not None and evidence.strong_consensus:
                strong_count += 1
        protected_count = sum(
            evidence is not None and max(
                (float(value) for value in evidence.scores.values()), default=0.0
            ) >= 0.90
            for evidence in contained_evidence
        )
        if strong_count >= 2:
            continue
        if protected_count >= 1 and len({entity.type for entity in contained}) >= 2:
            continue
        # Reconcile token/boundary fragments, not two already-complete
        # synonymous concepts (e.g. a diagnosis followed by its expansion).
        fragment_like_count = sum(
            len(_surface_tokens(entity.text)) == 1 or len(entity.text.strip()) <= 4
            for entity in contained
        )
        if fragment_like_count == 0 and len(contained) < 3:
            continue
        type_counts: dict[str, int] = {}
        for entity in contained:
            type_counts[entity.type] = type_counts.get(entity.type, 0) + 1
        majority = max(type_counts.values())
        if type_counts.get(candidate.type, 0) < majority:
            continue
        # Do not fuse two high-confidence independent concepts separated by
        # punctuation; low-confidence token fragments remain eligible.
        if re.search(r"[,;]", candidate.text) and sum(float(item.score or 0.0) >= 0.75 for item in contained) >= 2:
            continue
        ordered_contained = sorted(contained, key=lambda item: item.position)
        gaps = " ".join(
            _fold_surface(raw_text[left.position[1]:right.position[0]])
            for left, right in zip(ordered_contained, ordered_contained[1:])
        )
        if re.search(r"\b(?:sau khi|truoc khi|vi|do|nhung|mac du|het)\b", gaps):
            continue
        contained_scores: list[float] = []
        for entity, evidence in zip(contained, contained_evidence):
            if evidence is not None:
                contained_scores.append(max(
                    (float(value) for value in evidence.scores.values()),
                    default=float(entity.score or 0.0),
                ))
            else:
                contained_scores.append(float(entity.score or 0.0))
        average = sum(contained_scores) / len(contained_scores)
        if score + 0.12 < average:
            continue
        completeness = len(contained) + min(2.0, len(candidate.text) / 32.0)
        proposals.append((score + 0.05 * completeness, candidate, contained))

    audit: list[dict[str, Any]] = []
    used: set[tuple[int, int, str]] = set()
    replacements: list[NerEntity] = []
    for _priority, candidate, contained in sorted(proposals, key=lambda row: row[0], reverse=True):
        contained_keys = {_entity_key(item) for item in contained}
        if used & contained_keys:
            continue
        replacement = NerEntity(
            candidate.text,
            candidate.type,
            normalize_assertions_for_type(
                candidate.type,
                [value for item in contained for value in item.assertions],
            ),
            candidate.position,
            max(1.0, max((float(item.score or 0.0) for item in contained), default=0.0)),
        )
        used.update(contained_keys)
        replacements.append(replacement)
        audit.append({
            "reason": "catalogue_composition_recovery",
            "candidate_id": candidate.candidate_id,
            "entities_before": [_entity_audit(item) for item in contained],
            "entity_after": _entity_audit(replacement),
        })
    output = [item for item in selected if _entity_key(item) not in used]
    output.extend(replacements)
    return sorted(output, key=lambda item: (*item.position, item.type)), audit



def _restore_strong_contained_drugs(
    raw_text: str,
    entities: list[NerEntity],
    catalogue: list[CandidateEvidence],
) -> tuple[list[NerEntity], list[dict[str, Any]]]:
    """Replace unsafe long drug wrappers with exact strong contained mentions.

    This is evidence-driven: no drug name is hard-coded. It handles explanatory
    clauses ("X không gây...") and independently administered drugs joined by
    conjunctions, while preserving genuine branded formulations introduced by
    a nearby ``Brand:`` label.
    """
    by_key = {(item.position[0], item.position[1], item.type): item for item in catalogue}
    output: list[NerEntity] = []
    audit: list[dict[str, Any]] = []
    clause_re = re.compile(
        r"\b(?:khong gay|gay doc|gay quai thai|do dau|van con|nhung|"
        r"dong thoi|sau khi|truoc khi|de dieu tri|co the su dung)\b"
    )
    coordination_re = re.compile(r"(?:\b(?:va|hoac|dong thoi|kem)\b|,)")

    for entity in entities:
        if entity.type != "THUỐC":
            output.append(entity)
            continue
        folded = _fold_surface(entity.text)
        outer_evidence = by_key.get(_entity_key(entity))
        inner_rows: list[tuple[float, CandidateEvidence]] = []
        for item in catalogue:
            if item.type != "THUỐC" or item.position == entity.position:
                continue
            if not (entity.position[0] <= item.position[0] and entity.position[1] >= item.position[1]):
                continue
            score = max((float(value) for value in item.scores.values()), default=0.0)
            if not (item.strong_consensus or (item.pre_llm_selected and score >= 0.90)):
                continue
            if raw_text[item.position[0]:item.position[1]] != item.text:
                continue
            inner_rows.append((score, item))
        if not inner_rows:
            output.append(entity)
            continue

        # Keep only a non-overlapping set of strongest exact contained drugs.
        chosen_evidence: list[CandidateEvidence] = []
        for _score, item in sorted(inner_rows, key=lambda row: (-row[0], row[1].position)):
            if any(
                item.position[0] < kept.position[1] and item.position[1] > kept.position[0]
                for kept in chosen_evidence
            ):
                continue
            chosen_evidence.append(item)
        chosen_evidence.sort(key=lambda item: item.position)

        replacement: list[CandidateEvidence] = []
        marker = clause_re.search(folded)
        if marker is not None:
            # Strong candidate sharing the outer start is the medication prefix;
            # clause candidates are span-head-only and never enter chosen_evidence.
            prefix_rows = [
                item for item in chosen_evidence
                if item.position[0] == entity.position[0]
            ]
            if prefix_rows:
                replacement = [max(
                    prefix_rows,
                    key=lambda item: max((float(value) for value in item.scores.values()), default=0.0),
                )]
        elif coordination_re.search(folded) and len(chosen_evidence) >= 2:
            before = raw_text[max(0, entity.position[0] - 32):entity.position[0]]
            introduced_by_brand = bool(re.search(r"[^\n:]{2,24}:\s*$", before))
            outer_score = max(
                (float(value) for value in (outer_evidence.scores if outer_evidence else {}).values()),
                default=0.0,
            )
            outer_is_consensus = bool(outer_evidence and outer_evidence.strong_consensus)
            if not introduced_by_brand and not outer_is_consensus:
                replacement = chosen_evidence

        if not replacement:
            output.append(entity)
            continue
        replacement_entities = [
            NerEntity(
                item.text, item.type,
                normalize_assertions_for_type(item.type, entity.assertions),
                item.position,
                max((float(value) for value in item.scores.values()), default=entity.score),
            )
            for item in replacement
        ]
        output.extend(replacement_entities)
        audit.append({
            "reason": "restore_strong_contained_drugs",
            "entity_before": _entity_audit(entity),
            "entities_after": [_entity_audit(item) for item in replacement_entities],
        })
    dedup = {_entity_key(item): item for item in output}
    return sorted(dedup.values(), key=lambda item: (*item.position, item.type)), audit

def finalize_entities_after_editor(
    raw_text: str,
    entities: list[NerEntity],
    catalogue: list[CandidateEvidence],
) -> tuple[list[NerEntity], list[dict[str, Any]]]:
    """Conservative record-level cleanup after all editor/recovery regions."""
    unique = {_entity_key(item): item for item in entities}
    restored, restore_audit = _restore_strong_contained_drugs(
        raw_text, list(unique.values()), catalogue
    )
    subtype_completed, subtype_audit = _recover_subtype_modifier_spans(
        raw_text, restored, catalogue
    )
    expanded, expansion_audit = _recover_single_boundary_expansions(
        raw_text, subtype_completed, catalogue
    )
    composed, composition_audit = _recover_catalogue_compositions(
        raw_text, expanded, catalogue
    )
    dominant, dominant_audit = _restore_dominant_contained_mention(
        raw_text, composed, catalogue
    )
    atomic, atomic_audit = _restore_independent_catalogue_mentions(
        raw_text, dominant, catalogue
    )
    contextual, contextual_audit = _contextual_role_cleanup(
        raw_text, atomic, catalogue
    )
    kept: list[NerEntity] = []
    audit: list[dict[str, Any]] = [
        *restore_audit,
        *subtype_audit,
        *expansion_audit,
        *composition_audit,
        *dominant_audit,
        *atomic_audit,
        *contextual_audit,
    ]
    for entity in sorted(contextual, key=lambda item: (*item.position, item.type)):
        reason = _unsafe_fragment_reason(entity)
        if reason is not None:
            audit.append({
                "reason": "deterministic_fragment_cleanup",
                "detail": reason,
                "entity": _entity_audit(entity),
            })
            continue
        kept.append(entity)
    kept, overlap_audit = _resolve_global_overlaps(kept, catalogue=catalogue)
    audit.extend(overlap_audit)
    kept, assertion_audit = repair_assertions_only(raw_text, kept)
    audit.extend(assertion_audit)
    for entity in kept:
        if raw_text[entity.position[0]:entity.position[1]] != entity.text:
            raise ValueError("finalizer produced invalid exact offset")
    return kept, audit


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
        errors: list[str] = []
        if decision.type not in proposal.allowed_types:
            errors.append("type_not_allowed")
        if not proposal.auto_add_eligible or not proposal.hard_supports:
            errors.append("insufficient_support")
        if proposal.negative_flags:
            errors.append("structural_negative")
        if overlaps:
            if any(
                item.position == proposal.position and item.type == decision.type
                for item in overlaps
            ):
                errors.append("duplicate_existing_entity")
            elif any(
                item.type == decision.type
                and item.position[0] <= start
                and item.position[1] >= end
                for item in overlaps
            ):
                errors.append("covered_by_existing_entity")
            else:
                errors.append("unsafe_overlap")
        if raw_text[start:end] != proposal.text or any(char in proposal.text for char in "\r\n"):
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
            "entity_after": _entity_audit(entity),
        })
    result.unresolved.extend(sorted(set(by_id) - seen))
    result.unresolved = list(dict.fromkeys(result.unresolved))
    result.entities.sort(key=lambda item: (*item.position, item.type))
    return result