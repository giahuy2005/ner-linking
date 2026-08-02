"""Qwen3-8B locked candidate editor with action-level fail-safe guards."""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ...llm.json_guard import extract_json
from ..schemas import ASSERTION_ENTITY_TYPES, NerEntity
from .candidates import CandidateEvidence, MissingProposal
from .editor_schemas import EditAction, EditOperation, MissingDecision, MissingDecisionAction, ReasonCode


PROMPT_VERSION = "qwen3_locked_editor_v1"
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


class VersionedJsonlCache:
    """Append-only cache whose key includes model/task/prompt and full payload."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.values: dict[str, str] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                        self.values[str(row["key"])] = str(row["response"])
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        continue

    @staticmethod
    def make_key(model_id: str, task: str, prompt: tuple[str, str], config_hash: str = "") -> str:
        payload = json.dumps({
            "model_id": model_id, "task": task, "prompt_version": PROMPT_VERSION,
            "system": prompt[0], "user": prompt[1], "config_hash": config_hash,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def put(self, key: str, response: str) -> None:
        if key in self.values:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")
        self.values[key] = response


def generate_with_cache(
    llm, prompts: list[tuple[str, str]], *, batch_size: int, model_id: str,
    task: str, cache: VersionedJsonlCache | None = None,
) -> list[str]:
    results: list[str | None] = [None] * len(prompts)
    pending_indexes, pending_prompts, keys = [], [], []
    for index, prompt in enumerate(prompts):
        key = VersionedJsonlCache.make_key(model_id, task, prompt)
        cached = cache.get(key) if cache else None
        if cached is None:
            pending_indexes.append(index); pending_prompts.append(prompt); keys.append(key)
        else:
            results[index] = cached
    if pending_prompts:
        generated = llm.generate_batch(pending_prompts, batch_size=batch_size)
        if len(generated) != len(pending_prompts):
            generated = [""] * len(pending_prompts)
        for index, key, response in zip(pending_indexes, keys, generated):
            results[index] = response
            if cache:
                cache.put(key, response)
    return [item or "" for item in results]


def _candidate_payload(item: CandidateEvidence) -> dict[str, Any]:
    value = asdict(item)
    value["position"] = list(item.position)
    value["strong_consensus"] = item.strong_consensus
    return value


def build_editor_request(request_id: str, context: str, context_start: int, candidates: list[CandidateEvidence]) -> tuple[str, str]:
    system = (
        "You are a locked Vietnamese medical NER candidate editor. Return one JSON object only. "
        "Decide every supplied candidate exactly once. Never invent IDs, spans, text, types, or raw text. "
        "Allowed actions: KEEP, DROP, RETYPE, REPAIR_SPAN, MERGE, UPDATE_ASSERTIONS, FLAG_UNRESOLVED. "
        "Non-target classes include anatomy-only, person, specialty, specimen, activity/lifestyle, "
        "mechanism/etiology, generic biomedical concept, procedure-not-test, and function fragments."
    )
    user = json.dumps({
        "schema_version": PROMPT_VERSION,
        "request_id": request_id,
        "context": context,
        "context_global_start": context_start,
        "candidates": [_candidate_payload(item) for item in candidates],
        "response_schema": {"request_id": request_id, "actions": []},
    }, ensure_ascii=False, separators=(",", ":"))
    return system, user


def build_missing_request(
    request_id: str, context: str, context_start: int, proposals: list[MissingProposal]
) -> tuple[str, str]:
    system = (
        "You are a locked missing-entity adjudicator. Return JSON only. "
        "For every supplied proposal_id choose ADD_PROPOSAL, REJECT, or UNRESOLVED. "
        "Never invent IDs, text, positions, types, or spans."
    )
    payload = []
    for item in proposals:
        value = asdict(item)
        value["position"] = list(item.position)
        value["local_position"] = [
            item.position[0] - context_start, item.position[1] - context_start,
        ]
        payload.append(value)
    user = json.dumps({
        "schema_version": PROMPT_VERSION,
        "request_id": request_id,
        "context": context,
        "context_global_start": context_start,
        "proposals": payload,
        "response_schema": {"request_id": request_id, "decisions": []},
    }, ensure_ascii=False, separators=(",", ":"))
    return system, user


def parse_missing_response(raw_response: str) -> tuple[list[MissingDecision], list[dict[str, Any]]]:
    rejected = []
    try:
        payload = extract_json(raw_response)
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
    return NerEntity(candidate.text, candidate.type, list(candidate.assertions), candidate.position, max(candidate.scores.values(), default=1.0))


def apply_editor_response(
    raw_text: str,
    candidates: list[CandidateEvidence],
    raw_response: str,
    *,
    context_start: int = 0,
) -> EditorResult:
    """Validate/apply actions independently. Invalid JSON keeps every candidate."""
    originals = [_to_entity(item) for item in candidates if item.pre_llm_selected]
    result = EditorResult(entities=originals, raw_response=raw_response)
    by_id = {item.candidate_id: item for item in candidates}
    try:
        payload = extract_json(raw_response)
        raw_actions = payload.get("actions")
        if not isinstance(raw_actions, list):
            raise TypeError("actions must be a list")
    except Exception as exc:
        result.rejected.append({"reason": "invalid_json", "detail": str(exc)})
        result.unresolved.extend(by_id)
        return result

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
                replacement = NerEntity(target.text, operation.type or target.type, list(target.assertions), target.position, max(target.scores.values(), default=1.0))
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
                for other in candidates
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
                replacement = NerEntity(operation.text, operation.type or targets[0].type, list(operation.assertions), (start, end), 1.0)
        elif name == EditAction.UPDATE_ASSERTIONS:
            target = targets[0]
            if target.type not in ASSERTION_ENTITY_TYPES:
                errors.append("assertions_not_allowed_for_type")
            else:
                replacement = NerEntity(target.text, target.type, list(operation.assertions), target.position, max(target.scores.values(), default=1.0))
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

    covered = set(counts)
    result.unresolved.extend(sorted(set(by_id) - covered))
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
        entity = NerEntity(proposal.text, decision.type or proposal.allowed_types[0], list(decision.assertions), proposal.position, 1.0)
        result.entities.append(entity)
        result.applied.append({"action": "ADD_PROPOSAL", "proposal_id": proposal.proposal_id})
    result.unresolved.extend(sorted(set(by_id) - seen))
    result.entities.sort(key=lambda item: (*item.position, item.type))
    return result
