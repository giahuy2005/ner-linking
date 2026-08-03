"""Closed schemas for Qwen3 NER editing and proposal decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..schemas import ASSERTION_ENTITY_TYPES, VALID_ASSERTIONS, VALID_ENTITY_TYPES


class EditAction(str, Enum):
    KEEP = "KEEP"
    DROP = "DROP"
    RETYPE = "RETYPE"
    REPAIR_SPAN = "REPAIR_SPAN"
    MERGE = "MERGE"
    UPDATE_ASSERTIONS = "UPDATE_ASSERTIONS"
    FLAG_UNRESOLVED = "FLAG_UNRESOLVED"


class ReasonCode(str, Enum):
    VALID_ENTITY = "VALID_ENTITY"
    WRONG_TYPE = "WRONG_TYPE"
    WRONG_BOUNDARY = "WRONG_BOUNDARY"
    MERGE_REQUIRED = "MERGE_REQUIRED"
    ASSERTION_ERROR = "ASSERTION_ERROR"
    NON_ENTITY_ANATOMY = "NON_ENTITY_ANATOMY"
    NON_ENTITY_PERSON = "NON_ENTITY_PERSON"
    NON_ENTITY_SPECIALTY = "NON_ENTITY_SPECIALTY"
    NON_ENTITY_SPECIMEN = "NON_ENTITY_SPECIMEN"
    NON_ENTITY_ACTIVITY = "NON_ENTITY_ACTIVITY"
    NON_ENTITY_MECHANISM = "NON_ENTITY_MECHANISM"
    GENERIC_BIOMEDICAL = "GENERIC_BIOMEDICAL"
    FUNCTION_WORD_OR_FRAGMENT = "FUNCTION_WORD_OR_FRAGMENT"
    PROCEDURE_NOT_TEST = "PROCEDURE_NOT_TEST"
    AMBIGUOUS = "AMBIGUOUS"


class MissingDecisionAction(str, Enum):
    ADD_PROPOSAL = "ADD_PROPOSAL"
    REJECT = "REJECT"
    UNRESOLVED = "UNRESOLVED"


class MissingReasonCode(str, Enum):
    VALID_MISSING_ENTITY = "VALID_MISSING_ENTITY"
    NOT_AN_ENTITY = "NOT_AN_ENTITY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class ReviewRegion:
    request_id: str
    record_id: str
    context: str
    context_start: int
    context_end: int
    candidate_ids: list[str]
    reasons: list[str] = field(default_factory=list)
    priority: int = 0
    must_review: bool = True

    def __post_init__(self) -> None:
        if not self.request_id or not self.record_id:
            raise ValueError("review region IDs cannot be empty")
        if not 0 <= self.context_start < self.context_end:
            raise ValueError("invalid review context span")
        if not self.candidate_ids or len(self.candidate_ids) > 8:
            raise ValueError("review region must contain 1..8 candidates")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("review region candidate IDs must be unique")


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} must be list[str]")
    if len(value) != len(set(value)):
        raise ValueError(f"{field_name} contains duplicates")
    return list(value)


def _position(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2 or any(type(item) is not int for item in value):
        raise TypeError("local_position must be [int, int] or null")
    if not 0 <= value[0] < value[1]:
        raise ValueError("local_position must be a non-empty half-open span")
    return value[0], value[1]


@dataclass(frozen=True)
class EditOperation:
    action: EditAction
    candidate_ids: list[str]
    text: str | None
    type: str | None
    assertions: list[str]
    local_position: tuple[int, int] | None
    confidence: str
    reason_code: ReasonCode

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EditOperation":
        if not isinstance(value, dict):
            raise TypeError("edit operation must be an object")
        action = EditAction(value.get("action"))
        ids = _string_list(value.get("candidate_ids"), "candidate_ids")
        if not ids:
            raise ValueError("candidate_ids cannot be empty")
        text = value.get("text")
        if text is not None and not isinstance(text, str):
            raise TypeError("text must be string or null")
        entity_type = value.get("type")
        if entity_type is not None and entity_type not in VALID_ENTITY_TYPES:
            raise ValueError("invalid entity type")
        assertions = _string_list(value.get("assertions", []), "assertions")
        if set(assertions) - VALID_ASSERTIONS:
            raise ValueError("invalid assertion")
        confidence = value.get("confidence")
        if confidence not in {"HIGH", "MEDIUM", "LOW"}:
            raise ValueError("invalid confidence")
        reason = ReasonCode(value.get("reason_code"))
        operation = cls(action, ids, text, entity_type, assertions, _position(value.get("local_position")), confidence, reason)
        operation._validate_action_shape()
        return operation

    def _validate_action_shape(self) -> None:
        if self.action == EditAction.MERGE and len(self.candidate_ids) < 2:
            raise ValueError("MERGE requires at least two candidate IDs")
        if self.action != EditAction.MERGE and len(self.candidate_ids) != 1:
            raise ValueError(f"{self.action.value} requires exactly one candidate ID")
        expected_reason = {
            EditAction.RETYPE: ReasonCode.WRONG_TYPE,
            EditAction.REPAIR_SPAN: ReasonCode.WRONG_BOUNDARY,
            EditAction.MERGE: ReasonCode.MERGE_REQUIRED,
            EditAction.UPDATE_ASSERTIONS: ReasonCode.ASSERTION_ERROR,
        }.get(self.action)
        if expected_reason is not None and self.reason_code != expected_reason:
            raise ValueError(f"{self.action.value} requires reason {expected_reason.value}")
        if self.action == EditAction.DROP and self.confidence != "HIGH":
            raise ValueError("DROP requires HIGH confidence")
        if self.action in {EditAction.KEEP, EditAction.DROP, EditAction.FLAG_UNRESOLVED}:
            if self.text is not None or self.type is not None or self.local_position is not None:
                raise ValueError(f"{self.action.value} cannot mutate text, type, or position")
        if self.action == EditAction.UPDATE_ASSERTIONS:
            if self.text is not None or self.type is not None or self.local_position is not None:
                raise ValueError("UPDATE_ASSERTIONS can only change assertions")
        if self.action == EditAction.RETYPE and self.type is None:
            raise ValueError("RETYPE requires type")
        if self.action in {EditAction.REPAIR_SPAN, EditAction.MERGE}:
            if not self.text or self.local_position is None or self.type is None:
                raise ValueError(f"{self.action.value} requires text, type and local_position")


@dataclass(frozen=True)
class MissingDecision:
    proposal_id: str
    decision: MissingDecisionAction
    type: str | None
    assertions: list[str] = field(default_factory=list)
    confidence: str = "LOW"
    reason_code: MissingReasonCode = MissingReasonCode.AMBIGUOUS

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MissingDecision":
        if not isinstance(value, dict):
            raise TypeError("missing decision must be an object")
        proposal_id = value.get("proposal_id")
        if not isinstance(proposal_id, str) or not proposal_id:
            raise ValueError("proposal_id is required")
        decision = MissingDecisionAction(value.get("decision"))
        entity_type = value.get("type")
        if entity_type is not None and entity_type not in VALID_ENTITY_TYPES:
            raise ValueError("invalid entity type")
        assertions = _string_list(value.get("assertions", []), "assertions")
        if set(assertions) - VALID_ASSERTIONS:
            raise ValueError("invalid assertion")
        confidence = value.get("confidence", "LOW")
        if confidence not in {"HIGH", "MEDIUM", "LOW"}:
            raise ValueError("invalid confidence")
        reason = MissingReasonCode(value.get("reason_code", "AMBIGUOUS"))
        if decision == MissingDecisionAction.ADD_PROPOSAL:
            if entity_type is None or confidence != "HIGH" or reason != MissingReasonCode.VALID_MISSING_ENTITY:
                raise ValueError("ADD_PROPOSAL requires type, HIGH confidence and VALID_MISSING_ENTITY")
            if entity_type not in ASSERTION_ENTITY_TYPES and assertions:
                raise ValueError("assertions not allowed for selected type")
        return cls(proposal_id, decision, entity_type, assertions, confidence, reason)
