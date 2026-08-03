"""Prompt builder for the whitelisted ontology selector V2."""

from __future__ import annotations

import json
from typing import Any

SELECTOR_PROMPT_VERSION = "qwen3_linking_selector_v2"

_SYSTEM = """Bạn chọn mã RxNorm hoặc ICD-10 từ whitelist cho một mention y tế.
Không được phát minh mã. THUỐC tối đa 1 mã. CHẨN_ĐOÁN tối đa allowed_max mã và chỉ 2 khi mention phối hợp rõ.
Nếu bằng chứng yếu hoặc mơ hồ thì ABSTAIN/UNRESOLVED. Không markdown, không reasoning.
Chỉ xuất đúng JSON schema:
{"request_id":"...","decision":"SELECT|ABSTAIN|UNRESOLVED","chosen_codes":["code"],"confidence":"HIGH|MEDIUM|LOW","reason_code":"EXACT_MATCH|STRUCTURED_MATCH|CONTEXT_DISAMBIGUATION|INSUFFICIENT_EVIDENCE|AMBIGUOUS"}"""


def build_candidate_selector_prompt(
    entity_text: str,
    entity_type: str,
    candidates: list[Any],
    max_choices: int = 2,
    context: str = "",
    request_id: str = "request",
) -> tuple[str, str]:
    payload = {
        "schema_version": SELECTOR_PROMPT_VERSION,
        "request_id": request_id,
        "entity": {"text": entity_text, "type": entity_type, "local_context": context},
        "allowed_max": max_choices,
        "candidates": candidates,
        "response_schema": {
            "request_id": request_id, "decision": "SELECT", "chosen_codes": ["code"],
            "confidence": "HIGH", "reason_code": "STRUCTURED_MATCH",
        },
    }
    return _SYSTEM, json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
