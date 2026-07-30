"""Batched 7B NER review/recovery with strict validation and fallback."""

from __future__ import annotations

import logging
from typing import Any

from ...llm.json_guard import extract_json
from ...llm.prompts import build_ner_7b_request_prompt
from ..rule.clinical import ALLOWED_ASSERTIONS, ALLOWED_TYPES, deterministic_cleanup
from ..schemas import NerEntity

LOGGER = logging.getLogger(__name__)
_ACTIONS = {"KEEP", "DROP", "REPAIR_SPAN", "RETYPE"}


def _span(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return None
    return int(value[0]), int(value[1])


def _assertions(value: Any, fallback: list[str]) -> list[str] | None:
    if value is None:
        return list(fallback)
    # LLM may return structurally valid JSON with objects/numbers nested in the
    # assertion list.  Check type before set membership so malformed output is
    # rejected instead of crashing on an unhashable dict/list.
    if not isinstance(value, list) or any(
        not isinstance(item, str) or item not in ALLOWED_ASSERTIONS
        for item in value
    ):
        return None
    return list(dict.fromkeys(value))


def _validate_response(request: dict, parsed: Any) -> tuple[dict | None, str | None]:
    if not isinstance(parsed, dict):
        return None, "response_not_object"
    if parsed.get("request_id") != request.get("request_id"):
        return None, "request_id_mismatch"
    if request["task"] == "REVIEW_REGION":
        decisions = parsed.get("decisions")
        if not isinstance(decisions, list):
            return None, "decisions_not_list"
        wanted = request["target_candidate_ids"]
        if any(not isinstance(item, dict) or not isinstance(item.get("candidate_id"), int)
               or isinstance(item.get("candidate_id"), bool) for item in decisions):
            return None, "invalid_candidate_id"
        returned = [item["candidate_id"] for item in decisions]
        if len(decisions) != len(wanted) or sorted(returned) != sorted(wanted):
            return None, "missing_or_extra_decision"
        if any(
            not isinstance(item.get("action"), str)
            or item.get("action") not in _ACTIONS
            for item in decisions
        ):
            return None, "invalid_action"
        target_by_id = {
            target["candidate_id"]: target for target in request.get("targets", [])
            if isinstance(target, dict) and isinstance(target.get("candidate_id"), int)
        }
        if any(
            item["action"] not in target_by_id.get(item["candidate_id"], {}).get(
                "allowed_actions", []
            )
            for item in decisions
        ):
            return None, "action_not_allowed_for_target"
    else:
        if not isinstance(parsed.get("new_entities"), list):
            return None, "new_entities_not_list"
    return parsed, None


def _generate_requests(requests: list[dict], llm, *, batch_size: int, retry_rounds: int) -> tuple[dict[str, dict], list[dict]]:
    pending = list(requests)
    accepted: dict[str, dict] = {}
    logs: list[dict] = []
    for attempt in range(retry_rounds + 1):
        if not pending:
            break
        prompts = [build_ner_7b_request_prompt(request) for request in pending]
        try:
            if hasattr(llm, "generate_batch"):
                outputs = llm.generate_batch(prompts, batch_size=batch_size)
            else:
                outputs = [llm.generate(system, user) for system, user in prompts]
        except Exception as exc:
            logs.extend({"status": "batch_error", "request_id": request["request_id"],
                         "attempt": attempt, "reason": str(exc)} for request in pending)
            continue
        if len(outputs) != len(pending):
            logs.extend({"status": "batch_error", "request_id": request["request_id"],
                         "attempt": attempt, "reason": "output_count_mismatch"} for request in pending)
            continue
        retry: list[dict] = []
        for request, output in zip(pending, outputs):
            parsed, error = _validate_response(request, extract_json(output))
            if error:
                logs.append({"status": "response_rejected", "request_id": request["request_id"],
                             "attempt": attempt, "reason": error})
                retry.append(request)
            else:
                accepted[request["request_id"]] = parsed
                logs.append({"status": "response_accepted", "request_id": request["request_id"],
                             "attempt": attempt})
        pending = retry
    for request in pending:
        logs.append({"status": "fallback", "request_id": request["request_id"],
                     "reason": "retry_exhausted_keep_pre_7b"})
    return accepted, logs


def _apply_review(raw_text: str, entities: list[NerEntity], request: dict,
                  response: dict, logs: list[dict]) -> list[NerEntity]:
    result = list(entities)
    context_span = _span(request.get("context_global_position"))
    target_ids = set(request["target_candidate_ids"])
    for decision in response["decisions"]:
        candidate_id = decision["candidate_id"]
        if candidate_id not in target_ids or not 0 <= candidate_id < len(result):
            logs.append({"status": "decision_rejected", "request_id": request["request_id"],
                         "candidate_id": candidate_id, "reason": "non_target_candidate"})
            continue
        original = result[candidate_id]
        action = decision["action"]
        if action == "KEEP":
            logs.append({"status": "decision_applied", "candidate_id": candidate_id, "action": action})
            continue
        if action == "DROP":
            result[candidate_id] = None
            logs.append({"status": "decision_applied", "candidate_id": candidate_id, "action": action})
            continue
        new_type = decision.get("type", original.type)
        if not isinstance(new_type, str) or new_type not in ALLOWED_TYPES:
            logs.append({"status": "decision_rejected", "candidate_id": candidate_id,
                         "reason": "invalid_type"})
            continue
        assertions = _assertions(decision.get("assertions"), original.assertions)
        if assertions is None:
            logs.append({"status": "decision_rejected", "candidate_id": candidate_id,
                         "reason": "invalid_assertions"})
            continue
        if action == "RETYPE":
            result[candidate_id] = NerEntity(original.text, new_type, assertions,
                                             original.position, original.score, None)
            logs.append({"status": "decision_applied", "candidate_id": candidate_id, "action": action})
            continue
        new_span = _span(decision.get("global_position"))
        new_text = decision.get("text")
        if new_span is None or not isinstance(new_text, str) or context_span is None:
            reason = "invalid_repair_schema"
        else:
            start, end = new_span
            ctx_start, ctx_end = context_span
            near = start < original.position[1] + 80 and end > original.position[0] - 80
            exact = 0 <= start < end <= len(raw_text) and raw_text[start:end] == new_text
            reason = None if ctx_start <= start < end <= ctx_end and near and exact else "invalid_repair_span"
        if reason:
            logs.append({"status": "decision_rejected", "candidate_id": candidate_id, "reason": reason})
            continue
        result[candidate_id] = NerEntity(new_text, new_type, assertions, new_span, original.score, None)
        logs.append({"status": "decision_applied", "candidate_id": candidate_id, "action": action,
                     "before": original.text, "after": new_text})
    return result


def _apply_recovery(raw_text: str, entities: list[NerEntity], request: dict,
                    response: dict, logs: list[dict]) -> list[NerEntity]:
    result = list(entities)
    context_span = _span(request.get("context_global_position"))
    if context_span is None:
        return result
    context_start, context_end = context_span
    for suggestion in response["new_entities"]:
        if not isinstance(suggestion, dict):
            continue
        relative = _span(suggestion.get("relative_position"))
        text = suggestion.get("text")
        entity_type = suggestion.get("type")
        assertions = _assertions(suggestion.get("assertions", []), [])
        if (
            relative is None
            or not isinstance(text, str)
            or not isinstance(entity_type, str)
            or entity_type not in ALLOWED_TYPES
            or assertions is None
        ):
            logs.append({"status": "recovery_rejected", "request_id": request["request_id"],
                         "reason": "invalid_schema"})
            continue
        start, end = context_start + relative[0], context_start + relative[1]
        exact = (0 <= relative[0] < relative[1] <= context_end - context_start
                 and raw_text[start:end] == text)
        if not exact:
            logs.append({"status": "recovery_rejected", "request_id": request["request_id"],
                         "reason": "invalid_exact_span", "text": text})
            continue
        overlaps = [e for e in result if e is not None
                    and start < e.position[1] and end > e.position[0]]
        if any(e.position == (start, end) and e.type == entity_type for e in overlaps):
            logs.append({"status": "recovery_rejected", "reason": "exact_duplicate", "text": text})
            continue
        # Boundary recovery may replace one same-type contained fragment only.
        replaceable = [e for e in overlaps if e.type == entity_type
                       and start <= e.position[0] and end >= e.position[1]]
        if overlaps and len(replaceable) != len(overlaps):
            logs.append({"status": "recovery_rejected", "reason": "unsafe_overlap", "text": text})
            continue
        result = [e for e in result if e is None or e not in replaceable]
        result.append(NerEntity(text, entity_type, assertions, (start, end), 0.5, None))
        logs.append({"status": "recovery_applied", "request_id": request["request_id"],
                     "text": text, "position": [start, end]})
    return result


def review_entities_batch(raw_texts_by_id: dict[str, str], entities_by_id: dict[str, list[NerEntity]],
                          handoffs_by_id: dict[str, dict], llm, *, batch_size: int = 4,
                          retry_rounds: int = 1) -> tuple[dict[str, list[NerEntity]], list[dict]]:
    """Run all requests in GPU batches; invalid requests fall back independently."""
    request_owner: dict[str, str] = {}
    request_map: dict[str, dict] = {}
    requests = []
    for record_id, handoff in handoffs_by_id.items():
        for request in [*handoff.get("review_regions", []), *handoff.get("region_recoveries", [])]:
            request = dict(request)
            if request["request_id"] in request_owner:
                raise ValueError(
                    f"duplicate request_id across records: {request['request_id']!r}; "
                    "build handoffs with request_prefix=record_id"
                )
            request_owner[request["request_id"]] = record_id
            request_map[request["request_id"]] = request
            requests.append(request)
    responses, logs = _generate_requests(requests, llm, batch_size=batch_size,
                                         retry_rounds=retry_rounds)
    result = {record_id: list(entities) for record_id, entities in entities_by_id.items()}
    for request_id, response in responses.items():
        record_id = request_owner[request_id]
        request = request_map[request_id]
        if request["task"] == "REVIEW_REGION":
            result[record_id] = _apply_review(raw_texts_by_id[record_id], result[record_id],
                                              request, response, logs)
        else:
            result[record_id] = _apply_recovery(raw_texts_by_id[record_id], result[record_id],
                                                request, response, logs)
    for record_id, entities in result.items():
        entities = [entity for entity in entities if entity is not None]
        result[record_id], cleanup_logs = deterministic_cleanup(raw_texts_by_id[record_id], entities)
        logs.extend({"record_id": record_id, **item} for item in cleanup_logs)
    for item in logs:
        LOGGER.info("7b_ner %s", item)
    return result, logs
