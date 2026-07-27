"""Dùng LocalLLM (NER_FIXER_CONFIG) sửa CÁC ENTITY BỊ repair_gate FLAG
nghi ngờ (vd 'suspect_truncated_diagnosis') — KHÔNG gửi toàn bộ entity
của document, chỉ gửi phần cần nghi vấn, để tiết kiệm gọi model 7-tỷ-lần
mỗi record.

Lifecycle load/unload model do CALLER quản lý (không tự load() trong
module này) — đúng ý đồ "load 1 lần, sửa cả batch, rồi unload" của bạn,
xem cli.py/pipeline.py chỗ gọi.
"""

from __future__ import annotations

import sys

from ...llm.backend import LocalLLM
from ...llm.json_guard import extract_json
from ...llm.prompts import build_ner_fixer_prompt
from ...llm.response_schemas import NerFixSuggestion
from ..schemas import NerEntity

CONTEXT_RADIUS = 60  # số ký tự lấy thêm mỗi bên entity làm context cho LLM


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

    idx = window.find(new_text)
    if idx == -1:
        return None
    abs_start = ctx_start + idx
    abs_end = abs_start + len(new_text)
    return abs_start, abs_end


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
            continue

        if suggestion.action == "keep":
            fixed.append(NerEntity(text=ent.text, type=ent.type, assertions=ent.assertions,
                                    position=ent.position, score=ent.score, flag=None))
            continue

        if suggestion.action == "retype":
            fixed.append(NerEntity(text=ent.text, type=suggestion.type, assertions=ent.assertions,
                                    position=ent.position, score=ent.score, flag=None))
            continue

        if suggestion.action == "retrim":
            span = _locate_span(raw_text, ent.position, suggestion.text, context_radius)
            if span is None:
                print(f"[llm_fixer] retrim '{suggestion.text}' không tìm thấy quanh vị trí gốc "
                      f"'{ent.text}' -> giữ nguyên entity", file=sys.stderr)
                fixed.append(ent)
                continue
            fixed.append(NerEntity(text=suggestion.text, type=suggestion.type, assertions=ent.assertions,
                                    position=span, score=ent.score, flag=None))
            continue

        # action lạ (không thuộc 4 giá trị) đã bị NerFixSuggestion.from_dict chặn ở trên -> không tới đây
        fixed.append(ent)

    return fixed