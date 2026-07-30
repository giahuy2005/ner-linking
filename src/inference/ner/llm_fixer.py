"""Dùng Qwen2.5-1.5B để sửa entity bị repair_gate flag và audit omissions.

Repair chỉ gọi theo entity nghi ngờ; recall audit gọi một lần/document rồi
lọc đề xuất bằng exact substring, offset, overlap và schema deterministic.

Lifecycle load/unload model do CALLER quản lý (không tự load() trong
module này) — đúng ý đồ "load 1 lần, sửa cả batch, rồi unload" của bạn,
xem cli.py/pipeline.py chỗ gọi.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ...llm.json_guard import extract_json
from ...llm.prompts import build_ner_fixer_prompt, build_ner_recall_audit_prompt
from ...llm.response_schemas import NerAuditResponse, NerFixSuggestion
from ..schemas import NerEntity
from . import repair_gate

if TYPE_CHECKING:
    from ...llm.backend import LocalLLM

CONTEXT_RADIUS = 60  # số ký tự lấy thêm mỗi bên entity làm context cho LLM
MAX_AUDIT_DOCUMENT_CHARS = 12000
MAX_AUDIT_ADDITIONS = 12
LAB_TYPES = {"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}


def _guarded_drop_allowed(entity: NerEntity) -> bool:
    """Recall-first DROP gate matching the notebook's small-model policy."""
    compact_length = len("".join(entity.text.split()))
    return bool(
        (entity.flag == "suspect_truncated_diagnosis"
         and entity.score < 0.35 and compact_length < 6)
        or (entity.flag == "low_emission_confidence"
            and entity.score < 0.10 and compact_length < 4)
    )


def _with_review_hint(entity: NerEntity, hint: dict, *, clear_flag: bool = False) -> NerEntity:
    return NerEntity(
        text=entity.text,
        type=entity.type,
        assertions=list(entity.assertions),
        position=entity.position,
        score=entity.score,
        flag=None if clear_flag else entity.flag,
        review_hints=[*entity.review_hints, hint],
    )


def _get_context(raw_text: str, position: tuple[int, int], radius: int = CONTEXT_RADIUS) -> str:
    start, end = position
    ctx_start = max(0, start - radius)
    ctx_end = min(len(raw_text), end + radius)
    return raw_text[ctx_start:ctx_end]


def _locate_span(raw_text: str, position: tuple[int, int], new_text: str, radius: int = CONTEXT_RADIUS):
    """Tìm lại offset của new_text (LLM trả về, action=retrim) trong vùng
    lân cận entity cũ trên raw_text — CHỈ chấp nhận nếu tìm thấy y nguyên,
    không đoán mò/fuzzy match để tránh entity trôi sang câu khác."""
    start, end = position
    ctx_start = max(0, start - radius)
    ctx_end = min(len(raw_text), end + radius)
    window = raw_text[ctx_start:ctx_end]

    matches = []
    cursor = 0
    while True:
        idx = window.find(new_text, cursor)
        if idx == -1:
            break
        abs_start = ctx_start + idx
        matches.append((abs_start, abs_start + len(new_text)))
        cursor = idx + 1

    if not matches:
        return None
    # Một thuật ngữ có thể lặp trong cùng context. Chọn occurrence gần span
    # gốc nhất thay vì window.find() luôn lấy occurrence đầu tiên.
    return min(matches, key=lambda span: (abs(span[0] - start), abs(span[1] - end)))


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and end > other_start for other_start, other_end in occupied)


def _resolve_audit_span(raw_text: str, suggestion, occupied: list[tuple[int, int]]):
    """Resolve only exact, currently-uncovered occurrences.

    LLM character arithmetic is often off for Vietnamese text. A supplied offset
    is accepted only when exact; without a valid offset we accept text only when
    exactly one uncovered occurrence exists, so repeated mentions cannot drift.
    """
    if suggestion.start is not None and suggestion.end is not None:
        span = (suggestion.start, suggestion.end)
        if (
            0 <= span[0] < span[1] <= len(raw_text)
            and raw_text[span[0]:span[1]] == suggestion.text
            and not _overlaps(span, occupied)
        ):
            return span

    matches = []
    cursor = 0
    while True:
        index = raw_text.find(suggestion.text, cursor)
        if index < 0:
            break
        span = (index, index + len(suggestion.text))
        if not _overlaps(span, occupied):
            matches.append(span)
        cursor = index + 1
    return matches[0] if len(matches) == 1 else None


def _validated_audit_entity(raw_text: str, suggestion, occupied: list[tuple[int, int]]):
    if suggestion.type in LAB_TYPES and suggestion.assertions:
        return None
    span = _resolve_audit_span(raw_text, suggestion, occupied)
    if span is None:
        return None
    candidate = {
        "text": raw_text[span[0]:span[1]],
        "type": suggestion.type,
        "assertions": suggestion.assertions,
        "position": [span[0], span[1]],
    }
    kept, dropped = repair_gate.filter_entities([candidate])
    if dropped or not kept or kept[0].get("flag") is not None:
        return None
    return NerEntity(
        text=candidate["text"],
        type=candidate["type"],
        assertions=list(candidate["assertions"]),
        position=span,
        score=0.5,
        flag=None,
    )


def audit_missing_entities(
    raw_text: str,
    entities: list[NerEntity],
    llm: LocalLLM,
    *,
    max_document_chars: int = MAX_AUDIT_DOCUMENT_CHARS,
) -> list[NerEntity]:
    """Ask 1.5B for high-precision omissions, then validate deterministically."""
    if len(raw_text) > max_document_chars:
        print(
            f"[llm_audit] bỏ audit document dài {len(raw_text)} > {max_document_chars} ký tự",
            file=sys.stderr,
        )
        return entities

    existing_payload = [
        {
            "text": entity.text,
            "type": entity.type,
            "assertions": list(entity.assertions),
            "position": list(entity.position),
        }
        for entity in entities
    ]
    system_prompt, user_prompt = build_ner_recall_audit_prompt(raw_text, existing_payload)
    try:
        raw_output = llm.generate(system_prompt, user_prompt)
        response = NerAuditResponse.from_dict(extract_json(raw_output))
    except Exception as exc:
        print(f"[llm_audit] lỗi gọi/parse LLM: {exc} -> không thêm entity", file=sys.stderr)
        return entities
    if response is None:
        print("[llm_audit] LLM trả sai schema -> không thêm entity", file=sys.stderr)
        return entities

    return _merge_audit_response(raw_text, entities, response)


def _merge_audit_response(
    raw_text: str,
    entities: list[NerEntity],
    response: NerAuditResponse,
) -> list[NerEntity]:
    occupied = [entity.position for entity in entities]
    additions = []
    for suggestion in response.additions[:MAX_AUDIT_ADDITIONS]:
        entity = _validated_audit_entity(raw_text, suggestion, occupied)
        if entity is None:
            continue
        additions.append(entity)
        occupied.append(entity.position)

    return sorted([*entities, *additions], key=lambda entity: entity.position)


def audit_missing_entities_batch(
    raw_texts_by_id: dict[str, str],
    entities_by_id: dict[str, list[NerEntity]],
    llm: LocalLLM,
    *,
    batch_size: int = 4,
    max_document_chars: int = MAX_AUDIT_DOCUMENT_CHARS,
) -> dict[str, list[NerEntity]]:
    """Batch the one-audit-call-per-document pass for local GPU throughput."""
    results = {record_id: list(entities) for record_id, entities in entities_by_id.items()}
    record_ids = []
    prompts = []
    for record_id, raw_text in raw_texts_by_id.items():
        if len(raw_text) > max_document_chars:
            continue
        entities = results.get(record_id, [])
        existing_payload = [
            {
                "text": entity.text,
                "type": entity.type,
                "assertions": list(entity.assertions),
                "position": list(entity.position),
            }
            for entity in entities
        ]
        record_ids.append(record_id)
        prompts.append(build_ner_recall_audit_prompt(raw_text, existing_payload))

    try:
        if hasattr(llm, "generate_batch"):
            raw_outputs = llm.generate_batch(prompts, batch_size=batch_size)
        else:
            raw_outputs = [llm.generate(system, user) for system, user in prompts]
        if len(raw_outputs) != len(prompts):
            raise ValueError(
                f"batch returned {len(raw_outputs)} outputs for {len(prompts)} prompts"
            )
    except Exception as exc:
        print(f"[llm_audit] batch generation lỗi: {exc} -> giữ nguyên batch", file=sys.stderr)
        return results

    for record_id, raw_output in zip(record_ids, raw_outputs):
        try:
            response = NerAuditResponse.from_dict(extract_json(raw_output))
        except Exception as exc:
            print(f"[llm_audit] lỗi parse record '{record_id}': {exc} -> giữ nguyên", file=sys.stderr)
            continue
        if response is None:
            continue
        results[record_id] = _merge_audit_response(
            raw_texts_by_id[record_id],
            results[record_id],
            response,
        )
    return results


def fix_flagged_entities(
    raw_text: str,
    entities: list[NerEntity],
    llm: LocalLLM,
    *,
    context_radius: int = CONTEXT_RADIUS,
) -> list[NerEntity]:
    """Trả list entity mới: entity không bị flag giữ nguyên; entity bị
    flag được LLM quyết định keep/drop/retype/retrim. Parse lỗi hoặc
    LLM trả sai shape -> GIỮ NGUYÊN entity gốc (không silent drop), in
    cảnh báo ra stderr để bạn review log."""
    fixed: list[NerEntity] = []

    for ent in entities:
        if ent.flag is None:
            fixed.append(ent)
            continue

        context = _get_context(raw_text, ent.position, context_radius)
        system_prompt, user_prompt = build_ner_fixer_prompt(context, ent.text, ent.type, ent.flag)

        try:
            raw_output = llm.generate(system_prompt, user_prompt)
            parsed = extract_json(raw_output)
            suggestion = NerFixSuggestion.from_dict(parsed)
        except Exception as exc:
            print(f"[llm_fixer] lỗi gọi LLM cho '{ent.text}': {exc} -> giữ nguyên entity", file=sys.stderr)
            fixed.append(ent)
            continue

        if suggestion is None:
            print(f"[llm_fixer] LLM trả sai shape cho '{ent.text}': {raw_output!r} -> giữ nguyên entity", file=sys.stderr)
            fixed.append(ent)
            continue

        if suggestion.action == "drop":
            if _guarded_drop_allowed(ent):
                continue
            print(
                f"[llm_fixer] chặn DROP không đủ bằng chứng cho '{ent.text}', chuyển tiếp 7B",
                file=sys.stderr,
            )
            fixed.append(_with_review_hint(ent, {
                "requested_action": "DROP",
                "status": "blocked_unsafe_drop",
                "guard_reason": ent.flag,
            }))
            continue

        if suggestion.action == "keep":
            fixed.append(NerEntity(text=ent.text, type=ent.type, assertions=ent.assertions,
                                    position=ent.position, score=ent.score, flag=None,
                                    review_hints=list(ent.review_hints)))
            continue

        if suggestion.action == "retype":
            fixed.append(_with_review_hint(ent, {
                "requested_action": "RETYPE_SUGGEST",
                "status": "suggestion_only",
                "original_type": ent.type,
                "suggested_type": suggestion.type,
            }))
            continue

        if suggestion.action == "retrim":
            # Notebook V9+: small LLM never mutates boundaries; it only routes
            # the proposed repair to the constrained 7B target batch.
            fixed.append(_with_review_hint(ent, {
                "requested_action": "BOUNDARY_REVIEW_SUGGESTED",
                "status": "suggestion_only",
                "suggested_text": suggestion.text,
                "suggested_type": suggestion.type,
            }))
            continue

        # action lạ (không thuộc 4 giá trị) đã bị NerFixSuggestion.from_dict chặn ở trên -> không tới đây
        fixed.append(ent)

    return fixed


def fix_flagged_entities_batch(
    raw_texts_by_id: dict[str, str],
    entities_by_id: dict[str, list[NerEntity]],
    llm: LocalLLM,
    *,
    batch_size: int = 4,
    context_radius: int = CONTEXT_RADIUS,
) -> dict[str, list[NerEntity]]:
    """Batch the guarded Qwen2.5-1.5B fixer across the complete input set.

    Only candidates already marked suspicious are sent to the small model.
    Invalid JSON/schema, inexact retrims and overlapping repairs all fall back
    to the original entity, preserving the notebook's recall-first policy.
    """
    results = {record_id: list(entities) for record_id, entities in entities_by_id.items()}
    jobs: list[tuple[str, int, NerEntity]] = []
    prompts = []
    for record_id, entities in results.items():
        raw_text = raw_texts_by_id[record_id]
        for index, entity in enumerate(entities):
            if entity.flag is None:
                continue
            context = _get_context(raw_text, entity.position, context_radius)
            prompts.append(build_ner_fixer_prompt(
                context, entity.text, entity.type, entity.flag
            ))
            jobs.append((record_id, index, entity))

    if not prompts:
        return results
    try:
        if hasattr(llm, "generate_batch"):
            raw_outputs = llm.generate_batch(prompts, batch_size=batch_size)
        else:
            raw_outputs = [llm.generate(system, user) for system, user in prompts]
        if len(raw_outputs) != len(prompts):
            raise ValueError(
                f"batch returned {len(raw_outputs)} outputs for {len(prompts)} prompts"
            )
    except Exception as exc:
        print(f"[llm_fixer] batch generation lỗi: {exc} -> giữ nguyên batch", file=sys.stderr)
        return results

    replacements: dict[str, dict[int, NerEntity | None]] = {}
    for (record_id, index, entity), raw_output in zip(jobs, raw_outputs):
        try:
            suggestion = NerFixSuggestion.from_dict(extract_json(raw_output))
        except Exception as exc:
            print(
                f"[llm_fixer] lỗi parse '{entity.text}': {exc} -> giữ nguyên",
                file=sys.stderr,
            )
            continue
        if suggestion is None:
            continue

        replacement: NerEntity | None = entity
        if suggestion.action == "drop":
            if _guarded_drop_allowed(entity):
                replacement = None
            else:
                # Preserve the flag so the rebuilt handoff routes it to 7B.
                replacement = _with_review_hint(entity, {
                    "requested_action": "DROP",
                    "status": "blocked_unsafe_drop",
                    "guard_reason": entity.flag,
                })
        elif suggestion.action == "keep":
            replacement = NerEntity(
                entity.text, entity.type, list(entity.assertions),
                entity.position, entity.score, None, list(entity.review_hints),
            )
        elif suggestion.action == "retype":
            replacement = _with_review_hint(entity, {
                "requested_action": "RETYPE_SUGGEST",
                "status": "suggestion_only",
                "original_type": entity.type,
                "suggested_type": suggestion.type,
            })
        elif suggestion.action == "retrim":
            replacement = _with_review_hint(entity, {
                "requested_action": "BOUNDARY_REVIEW_SUGGESTED",
                "status": "suggestion_only",
                "suggested_text": suggestion.text,
                "suggested_type": suggestion.type,
            })
        replacements.setdefault(record_id, {})[index] = replacement

    for record_id, by_index in replacements.items():
        results[record_id] = [
            by_index.get(index, entity)
            for index, entity in enumerate(results[record_id])
            if by_index.get(index, entity) is not None
        ]
    return results
