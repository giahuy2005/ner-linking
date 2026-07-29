"""Dùng LocalLLM (CANDIDATE_SELECTOR_CONFIG) chọn lại candidate đúng nhất
trong list mà RxNormLinker/Icd10Linker đã retrieval + rerank sẵn.

LUÔN validate code LLM chọn PHẢI nằm trong list candidate gốc đưa vào —
không tin LLM tự bịa code không có trong danh sách (hallucination). Nếu
LLM lỗi hoặc chọn toàn code không hợp lệ, fallback về top candidate của
linker (KHÔNG trả rỗng — rỗng tệ hơn dùng lại kết quả linker gốc).
"""

from __future__ import annotations

import re
import sys
import unicodedata
from typing import TYPE_CHECKING

from ...llm.json_guard import extract_json
from ...llm.prompts import build_candidate_selector_prompt
from ...llm.response_schemas import CandidateSelection

if TYPE_CHECKING:
    from ...llm.backend import LocalLLM


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
    details = [str(label)]
    if isinstance(cand, dict):
        if cand.get("term_type"):
            details.append(f"term_type={cand['term_type']}")
        if cand.get("score") is not None:
            details.append(f"score={float(cand['score']):.4f}")
        if cand.get("language"):
            details.append(f"lang={cand['language']}")
    else:
        if getattr(cand, "tty", None):
            details.append(f"TTY={cand.tty}")
        details.append(f"rule_score={float(getattr(cand, 'final_score', 0.0)):.4f}")
        features = getattr(cand, "features", {}) or {}
        for key in ("ingredient_relation", "strength_relation", "form_relation", "release_relation"):
            if features.get(key):
                details.append(f"{key}={features[key]}")
    return code, " | ".join(details)


def _normalize_exact(text: str) -> str:
    value = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", value).strip(" \t\r\n.,;:()[]{}")


def _high_confidence_top(entity_text: str, entity_type: str, candidate) -> bool:
    """Skip 7B only for deterministic exact cases; ambiguous cases still use it."""
    if entity_type == "CHẨN_ĐOÁN" and isinstance(candidate, dict):
        matched = candidate.get("matched_term")
        return isinstance(matched, str) and _normalize_exact(matched) == _normalize_exact(entity_text)
    if entity_type != "THUỐC" or isinstance(candidate, dict):
        return False
    features = getattr(candidate, "features", {}) or {}
    return bool(
        getattr(candidate, "exact_term_match", False)
        and features.get("ingredient_relation") == "exact"
        and features.get("strength_relation") not in {"order_dose_mismatch"}
        and features.get("form_relation") not in {"mismatch"}
        and features.get("release_relation") not in {"mismatch"}
    )


def _prepare_selection(
    entity_text: str,
    entity_type: str,
    candidates: list,
    *,
    top_k_context: int,
    max_choices: int,
    context: str,
):
    if not candidates:
        return None
    display_pairs = []
    valid_codes = []
    for candidate in candidates[:top_k_context]:
        pair = _display(candidate)
        if pair is not None:
            display_pairs.append(pair)
            valid_codes.append(pair[0])
    if not display_pairs:
        return None
    choice_limit = 1 if entity_type == "THUỐC" else max_choices
    fallback = valid_codes[:choice_limit]
    prompt = None
    if not _high_confidence_top(entity_text, entity_type, candidates[0]):
        prompt = build_candidate_selector_prompt(
            entity_text,
            entity_type,
            display_pairs,
            max_choices=choice_limit,
            context=context,
        )
    return {
        "entity_text": entity_text,
        "valid_codes": valid_codes,
        "choice_limit": choice_limit,
        "fallback": fallback,
        "prompt": prompt,
    }


def _finish_selection(raw_output: str, prepared: dict) -> list[str]:
    try:
        selection = CandidateSelection.from_dict(extract_json(raw_output))
    except Exception as exc:
        print(
            f"[candidate_selector] lỗi parse cho '{prepared['entity_text']}': {exc} -> fallback",
            file=sys.stderr,
        )
        return prepared["fallback"]
    if selection is None:
        return prepared["fallback"]
    valid_set = set(prepared["valid_codes"])
    chosen = list(dict.fromkeys(code for code in selection.chosen_codes if code in valid_set))
    return chosen[:prepared["choice_limit"]] if chosen else prepared["fallback"]


def select_candidates(
    entity_text: str,
    entity_type: str,
    candidates: list,
    llm: LocalLLM,
    *,
    top_k_context: int = 10,
    max_choices: int = 3,
    context: str = "",
) -> list[str]:
    """candidates: list gốc trả về từ linker (RxNormCandidate hoặc dict
    ICD-10, ĐÃ sort theo score của linker — top trước). Trả list code đã
    được LLM chọn lại, thứ tự theo LLM ưu tiên."""
    prepared = _prepare_selection(
        entity_text,
        entity_type,
        candidates,
        top_k_context=top_k_context,
        max_choices=max_choices,
        context=context,
    )
    if prepared is None:
        return []
    if prepared["prompt"] is None:
        return prepared["fallback"][:1]

    try:
        raw_output = llm.generate(*prepared["prompt"])
    except Exception as exc:
        print(f"[candidate_selector] lỗi gọi LLM cho '{entity_text}': {exc} -> dùng top candidate linker", file=sys.stderr)
        return prepared["fallback"]
    return _finish_selection(raw_output, prepared)


def select_candidates_many(
    items: list[dict],
    llm: LocalLLM,
    *,
    top_k_context: int = 10,
    max_choices: int = 3,
    batch_size: int = 4,
) -> list[list[str]]:
    """Batch only ambiguous selections; exact matches return without 7B."""
    results: list[list[str] | None] = [None] * len(items)
    pending_indexes = []
    prepared_items = []
    prompts = []
    for index, item in enumerate(items):
        prepared = _prepare_selection(
            item["entity_text"],
            item["entity_type"],
            item["candidates"],
            top_k_context=top_k_context,
            max_choices=max_choices,
            context=item.get("context", ""),
        )
        if prepared is None:
            results[index] = []
        elif prepared["prompt"] is None:
            results[index] = prepared["fallback"][:1]
        else:
            pending_indexes.append(index)
            prepared_items.append(prepared)
            prompts.append(prepared["prompt"])

    if prompts:
        try:
            if hasattr(llm, "generate_batch"):
                raw_outputs = llm.generate_batch(prompts, batch_size=batch_size)
            else:
                raw_outputs = [llm.generate(*prompt) for prompt in prompts]
            if len(raw_outputs) != len(prompts):
                raise ValueError(
                    f"batch returned {len(raw_outputs)} outputs for {len(prompts)} prompts"
                )
        except Exception as exc:
            print(f"[candidate_selector] batch lỗi: {exc} -> fallback", file=sys.stderr)
            raw_outputs = [""] * len(prompts)
        for index, prepared, raw_output in zip(pending_indexes, prepared_items, raw_outputs):
            results[index] = _finish_selection(raw_output, prepared)

    return [result or [] for result in results]
