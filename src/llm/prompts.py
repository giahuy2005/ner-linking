"""Prompt builder for the compact whitelisted ontology selector V3."""

from __future__ import annotations

import json
from typing import Any

SELECTOR_PROMPT_VERSION = "qwen3_linking_selector_v6_immutable_multicode"

_SYSTEM = """Bạn chọn mã RxNorm hoặc ICD-10 chỉ từ whitelist đã được retrieval/reranker hỗ trợ.
Không phát minh mã. Không dùng mã có hard_conflicts.
Ưu tiên mã khớp đúng mức độ cụ thể của mention; không chọn subtype khi mention thiếu qualifier.
Với RxNorm: mọi hoạt chất, strength, dose form, release và route ghi rõ phải tương thích.
Mention có nhiều thuốc chỉ chọn concept phối hợp chứa đủ mọi hoạt chất; không chọn một thành phần riêng.
Nếu mention chỉ có hoạt chất, ưu tiên ingredient concept thay vì sản phẩm strength/form cụ thể.
Nếu mention chung và whitelist có mã không đặc hiệu/phù hợp hơn, ưu tiên mã đó.
THUỐC tối đa 1 mã. CHẨN_ĐOÁN tối đa allowed_max mã và chỉ 2 khi mention phối hợp rõ.
Không chọn đồng thời mã cha và mã con. Nếu bằng chứng chưa đủ thì ABSTAIN/UNRESOLVED.
Không markdown, không reasoning. Chỉ xuất đúng JSON schema:
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
        "entity": {
            "text": entity_text,
            "type": entity_type,
            "local_context": context[-320:] if context else "",
        },
        "allowed_max": max_choices,
        "candidates": candidates,
        "response_schema": {
            "request_id": request_id,
            "decision": "SELECT",
            "chosen_codes": ["code"],
            "confidence": "HIGH",
            "reason_code": "STRUCTURED_MATCH",
        },
    }
    return _SYSTEM, json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), default=str
    )