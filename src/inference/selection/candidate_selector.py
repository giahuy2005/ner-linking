"""Dùng LocalLLM (CANDIDATE_SELECTOR_CONFIG) chọn lại candidate đúng nhất
trong list mà RxNormLinker/Icd10Linker đã retrieval + rerank sẵn.

LUÔN validate code LLM chọn PHẢI nằm trong list candidate gốc đưa vào —
không tin LLM tự bịa code không có trong danh sách (hallucination). Nếu
LLM lỗi hoặc chọn toàn code không hợp lệ, fallback về top candidate của
linker (KHÔNG trả rỗng — rỗng tệ hơn dùng lại kết quả linker gốc).
"""

from __future__ import annotations

import sys

from ...llm.backend import LocalLLM
from ...llm.json_guard import extract_json
from ...llm.prompts import build_candidate_selector_prompt
from ...llm.response_schemas import CandidateSelection


def _display(cand, key_priority: tuple[str, ...] = ("code", "rxcui")) -> tuple[str, str] | None:
    """(code, label) cho 1 candidate — dùng chung logic duck-type với
    pipeline._extract_codes (RxNormCandidate dataclass field .rxcui, dict
    ICD-10 field "code"), thêm label để LLM có ngữ cảnh chọn."""
    code = None
    for key in key_priority:
        if hasattr(cand, key):
            code = str(getattr(cand, key))
            break
        if isinstance(cand, dict) and key in cand:
            code = str(cand[key])
            break
    if code is None:
        return None

    label = getattr(cand, "name", None) or (cand.get("matched_term") if isinstance(cand, dict) else None) or ""
    return code, str(label)


def select_candidates(
    entity_text: str,
    entity_type: str,
    candidates: list,
    llm: LocalLLM,
    *,
    top_k_context: int = 10,
    max_choices: int = 3,
) -> list[str]:
    """candidates: list gốc trả về từ linker (RxNormCandidate hoặc dict
    ICD-10, ĐÃ sort theo score của linker — top trước). Trả list code đã
    được LLM chọn lại, thứ tự theo LLM ưu tiên."""
    if not candidates:
        return []

    display_pairs = []
    valid_codes = []
    for cand in candidates[:top_k_context]:
        pair = _display(cand)
        if pair is None:
            continue
        display_pairs.append(pair)
        valid_codes.append(pair[0])

    if not display_pairs:
        return []

    fallback_codes = valid_codes[:max_choices]

    system_prompt, user_prompt = build_candidate_selector_prompt(
        entity_text, entity_type, display_pairs, max_choices=max_choices,
    )

    try:
        raw_output = llm.generate(system_prompt, user_prompt)
        parsed = extract_json(raw_output)
        selection = CandidateSelection.from_dict(parsed)
    except Exception as exc:
        print(f"[candidate_selector] lỗi gọi LLM cho '{entity_text}': {exc} -> dùng top candidate linker", file=sys.stderr)
        return fallback_codes

    if selection is None:
        print(f"[candidate_selector] LLM trả sai shape cho '{entity_text}' -> dùng top candidate linker", file=sys.stderr)
        return fallback_codes

    # chỉ giữ code THẬT SỰ có trong danh sách đưa vào, chặn hallucination
    valid_set = set(valid_codes)
    chosen = [c for c in selection.chosen_codes if c in valid_set]

    if not chosen:
        print(f"[candidate_selector] LLM chọn toàn code không hợp lệ cho '{entity_text}' "
              f"(trả: {selection.chosen_codes}) -> dùng top candidate linker", file=sys.stderr)
        return fallback_codes

    return chosen[:max_choices]