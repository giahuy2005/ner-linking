"""Select final ontology codes from a retrieved, evidence-rich whitelist.

The selector never invents codes. ICD retrieval/reranking is the source of
support evidence; Qwen only resolves genuine ambiguity between supported
candidates. A Qwen-selected whitelist code is not passed through the obsolete
lexical hard gate a second time.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import unicodedata
from dataclasses import asdict, is_dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...llm.batching import VersionedJsonlCache, generate_with_cache
from ...llm.json_guard import extract_json
from ...llm.prompts import SELECTOR_PROMPT_VERSION, build_candidate_selector_prompt
from ...llm.response_schemas import CandidateSelection

if TYPE_CHECKING:
    from ...llm.backend import LocalLLM

LAST_SELECTION_AUDIT: list[dict] = []
LAST_SELECTION_WORKLOAD: dict = {}

_SUPPORT_ORDER = {"exact": 4, "strong": 3, "medium": 2, "weak": 1, "rejected": 0}

_UNSAFE_SINGLE_DIAGNOSIS_TOKENS = frozenset({
    "benh", "viem", "ton", "mien", "xoang", "gan", "tuy", "da", "chi",
    "virus", "fibrin", "gen", "enzyme", "mau", "dich",
})

_NON_SEMANTIC_TOKENS = {
    "benh", "hoi", "chung", "thuoc", "type", "typ", "khong", "xac", "dinh",
    "va", "hoac", "kem", "mg", "mcg", "g", "ml", "po", "iv", "im", "bid",
    "tid", "qid", "daily", "dac", "hieu", "khac", "chua", "phan", "loai",
    "vi", "tri", "he", "do", "cua", "nos", "unspecified", "other",
}


def _pctl(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return int(ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)])


def _display(candidate, key_priority: tuple[str, ...] = ("code", "rxcui")) -> tuple[str, str] | None:
    code = None
    for key in key_priority:
        if hasattr(candidate, key):
            code = str(getattr(candidate, key))
            break
        if isinstance(candidate, dict) and key in candidate:
            code = str(candidate[key])
            break
    if code is None:
        return None
    label = (
        getattr(candidate, "name", None)
        or (candidate.get("matched_term") if isinstance(candidate, dict) else None)
        or ""
    )
    return code, str(label)


def _candidate_payload(candidate) -> dict[str, Any]:
    displayed = _display(candidate)
    if displayed is None:
        return {}
    code, label = displayed
    if is_dataclass(candidate):
        raw = asdict(candidate)
        features = raw.get("features", {}) or {}
        value = {
            "code": code,
            "name": raw.get("name", label),
            "tty": raw.get("tty"),
            "tier": raw.get("tier"),
            "active": raw.get("active"),
            "support_level": raw.get("support_level"),
            "score": raw.get("final_score"),
            "exact_flags": {
                "term": raw.get("exact_term_match"),
                "ingredient": raw.get("exact_ingredient_match"),
            },
            "relations": {
                key: features[key]
                for key in (
                    "ingredient_relation", "strength_relation",
                    "form_relation", "release_relation",
                )
                if features.get(key)
            },
            "hard_conflicts": list(raw.get("rejection_reasons", []) or []),
        }
        return {key: item for key, item in value.items() if item not in (None, "", [], {})}

    raw = candidate if isinstance(candidate, dict) else {}
    matched_terms = [
        str(value)
        for value in (raw.get("matched_terms", []) or [])
        if isinstance(value, str)
    ][:4]
    value = {
        "code": code,
        "matched_term": raw.get("matched_term", label),
        "matched_terms": matched_terms,
        "term_type": raw.get("term_type"),
        "support_level": raw.get("support_level"),
        "support_score": raw.get("support_score"),
        "support_rank": raw.get("support_rank"),
        "dense_score": raw.get("aggregate_score", raw.get("score")),
        "exact_alias_source": raw.get("exact_alias_source"),
        "mention_coverage": raw.get("mention_coverage"),
        "candidate_coverage": raw.get("candidate_coverage"),
        "lexical_similarity": raw.get("lexical_similarity"),
        "extra_tokens": list(raw.get("extra_tokens", []) or [])[:6],
        "over_specific": bool(raw.get("over_specific", False)),
        "hierarchy_has_descendants": bool(raw.get("hierarchy_has_descendants", False)),
        "hard_conflicts": list(raw.get("hard_conflicts", []) or []),
    }
    return {
        key: item
        for key, item in value.items()
        if item not in (None, "", [], {}, False)
    }


def _normalize_exact(text: str) -> str:
    value = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", value).strip(" \t\r\n.,;:()[]{}")


def _fold_ascii(text: str) -> str:
    value = unicodedata.normalize("NFD", _normalize_exact(text)).replace("đ", "d").replace("Đ", "D")
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _fold_ascii(text))
        if token not in _NON_SEMANTIC_TOKENS and not token.isdigit()
    }


def _is_medical_abbreviation(text: str) -> bool:
    compact = "".join(char for char in text if char.isalnum())
    letters = [char for char in compact if char.isalpha()]
    return bool(
        2 <= len(compact) <= 10
        and letters
        and all(char.isupper() for char in letters)
    )


def _unsafe_linking_fragment(entity_text: str, entity_type: str) -> str | None:
    """Block ontology linking for obvious NER fragments, not valid short diseases."""
    if entity_type != "CHẨN_ĐOÁN" or _is_medical_abbreviation(entity_text):
        return None
    tokens = [
        token for token in re.findall(r"[a-z0-9]+", _fold_ascii(entity_text))
        if not token.isdigit()
    ]
    if len(tokens) == 1 and tokens[0] in _UNSAFE_SINGLE_DIAGNOSIS_TOKENS:
        return "unsafe_single_token_diagnosis_fragment"
    return None


def _lexical_support(entity_text: str, label: str) -> tuple[float, float]:
    mention = _fold_ascii(entity_text)
    candidate = _fold_ascii(label)
    mention_tokens = _meaningful_tokens(entity_text)
    candidate_tokens = _meaningful_tokens(label)
    coverage = (
        len(mention_tokens & candidate_tokens) / len(mention_tokens)
        if mention_tokens else 0.0
    )
    return coverage, SequenceMatcher(None, mention, candidate).ratio()


def _hard_candidate_conflict(candidate) -> bool:
    if isinstance(candidate, dict):
        return bool(candidate.get("hard_conflicts")) or candidate.get("support_level") == "rejected"
    return bool(getattr(candidate, "rejection_reasons", []) or [])


def _candidate_supported(entity_text: str, entity_type: str, candidate) -> bool:
    """Whether a candidate may enter the selector whitelist.

    Extra label tokens are a soft specificity feature, never a standalone hard
    rejection. Hard conflicts are the only unconditional rejection.
    """
    displayed = _display(candidate)
    if displayed is None or _hard_candidate_conflict(candidate):
        return False
    _code, label = displayed

    if entity_type == "CHẨN_ĐOÁN" and isinstance(candidate, dict):
        support_level = str(candidate.get("support_level", ""))
        if support_level in {"exact", "strong", "medium"}:
            return True
        if candidate.get("exact_alias_source"):
            return True
        if candidate.get("normalized_exact_match"):
            return True
        # Backward-compatible evidence for old retrieval rows without V3 fields.
        score = float(candidate.get("aggregate_score", candidate.get("score", 0.0)) or 0.0)
        coverage = candidate.get("mention_coverage")
        similarity = candidate.get("lexical_similarity")
        if coverage is None or similarity is None:
            coverage, similarity = _lexical_support(entity_text, label)
        return bool(
            (score >= 0.74 and float(coverage) >= 0.50)
            or (score >= 0.84 and float(similarity) >= 0.48)
        )

    if entity_type == "THUỐC" and not isinstance(candidate, dict):
        # RxNormRuleReranker is the single source of truth. The selector must
        # not create a second, contradictory lexical reranker.
        level = str(getattr(candidate, "support_level", "weak"))
        return bool(level in {"exact", "strong", "medium"})

    if isinstance(candidate, dict) and not any(
        key in candidate for key in ("score", "features", "final_score", "support_level")
    ):
        return True
    return False


def _selection_key(candidate, raw_rank: int) -> tuple:
    if isinstance(candidate, dict):
        level = _SUPPORT_ORDER.get(str(candidate.get("support_level", "weak")), 1)
        exact = int(bool(candidate.get("exact_alias_source") or candidate.get("normalized_exact_match")))
        over_specific = int(bool(candidate.get("over_specific")))
        score = float(candidate.get("support_score", candidate.get("aggregate_score", candidate.get("score", 0.0))) or 0.0)
        dense = float(candidate.get("aggregate_score", candidate.get("score", 0.0)) or 0.0)
        return (-exact, -level, over_specific, -score, -dense, raw_rank)
    support = _SUPPORT_ORDER.get(str(getattr(candidate, "support_level", "weak")), 1)
    exact = int(bool(getattr(candidate, "exact_term_match", False)))
    score = float(getattr(candidate, "final_score", 0.0) or 0.0)
    return (-exact, -support, 0, -score, -float(getattr(candidate, "dense_score", 0.0) or 0.0), raw_rank)


def _rank_supported_candidates(entity_text: str, entity_type: str, candidates: list) -> list[tuple[int, Any]]:
    supported = [
        (rank, candidate)
        for rank, candidate in enumerate(candidates, 1)
        if _candidate_supported(entity_text, entity_type, candidate)
    ]
    supported.sort(key=lambda pair: _selection_key(pair[1], pair[0]))
    return supported


def _explicitly_coordinated_diagnoses(text: str) -> bool:
    normalized = _normalize_exact(text)
    return bool(re.search(r"(?:\b(?:và|hoặc|kèm)\b|[+;])", normalized))


def _code_key(code: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", code).upper()


def _hierarchy_related(left: str, right: str) -> bool:
    a, b = _code_key(left), _code_key(right)
    return bool(a != b and (a.startswith(b) or b.startswith(a)))


def _deterministic_decision(entity_type: str, supported: list[tuple[int, Any]]) -> tuple[list[str], str]:
    if not supported:
        return [], "no_supported_candidate"
    top = supported[0][1]
    displayed = _display(top)
    if displayed is None:
        return [], "no_displayable_candidate"
    code, _label = displayed

    if entity_type == "CHẨN_ĐOÁN" and isinstance(top, dict):
        level = str(top.get("support_level", "weak"))
        exact_unique = bool(
            top.get("exact_alias_source") == "configured_exact_alias"
            or (
                top.get("exact_alias_source") == "metadata_exact_unique"
                and top.get("exact_text_quality", True)
            )
        )
        ambiguous_exact_alias = top.get("exact_alias_source") == "metadata_exact_ambiguous"
        normalized_exact = bool(
            (top.get("normalized_exact_match") or top.get("catalogue_exact_match"))
            and not ambiguous_exact_alias
        )
        over_specific = bool(top.get("over_specific"))
        top_score = float(top.get("support_score", 0.0) or 0.0)
        second_score = (
            float(supported[1][1].get("support_score", 0.0) or 0.0)
            if len(supported) > 1 and isinstance(supported[1][1], dict)
            else 0.0
        )
        margin = top_score - second_score
        if exact_unique:
            return [code], "exact_unique_alias"
        if normalized_exact and not over_specific:
            return [code], "normalized_exact"
        strong_shape = bool(
            int(top.get("mention_token_count", 0) or 0) >= 2
            and (
                top.get("phrase_containment")
                or top.get("abbreviation_support")
                or top.get("technical_containment")
                or top.get("catalogue_exact_match")
                or top.get("normalized_exact_match")
            )
        )
        if level == "strong" and not over_specific and strong_shape and (
            len(supported) == 1 or margin >= 0.055
        ):
            return [code], "strong_supported_margin"
        if len(supported) == 1 and level == "medium":
            return [], "single_medium_abstain"
        return [], "ambiguous_supported_candidates"

    # RxNorm deterministic bypass follows the structured reranker only.
    level = str(getattr(top, "support_level", "weak"))
    margin = float(getattr(top, "top1_margin", 0.0) or 0.0)
    exact_count = sum(
        str(getattr(candidate, "support_level", "weak")) == "exact"
        for _rank, candidate in supported
    )
    if level == "exact" and exact_count == 1:
        return [code], "rxnorm_exact_unique"
    if level == "strong" and (
        len(supported) == 1 or margin >= 0.055
    ):
        return [code], "rxnorm_strong_margin"
    if len(supported) == 1 and level == "medium":
        return [], "single_medium_abstain"
    return [], "ambiguous_supported_candidates"


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
    unsafe_reason = _unsafe_linking_fragment(entity_text, entity_type)
    if unsafe_reason is not None:
        return {
            "entity_text": entity_text,
            "entity_type": entity_type,
            "choice_limit": 1,
            "fallback": [],
            "prompt": None,
            "request_id": "",
            "valid_codes": [],
            "candidates_by_code": {},
            "decision_reason": unsafe_reason,
            "supported_count": 0,
            "shortlist_codes": [],
            "raw_ranks": {},
        }
    choice_limit = 1
    if entity_type == "CHẨN_ĐOÁN" and max_choices >= 2 and _explicitly_coordinated_diagnoses(entity_text):
        choice_limit = 2

    supported_ranked = _rank_supported_candidates(entity_text, entity_type, candidates)
    fallback, decision_reason = _deterministic_decision(entity_type, supported_ranked)
    shortlist_pairs = supported_ranked[: max(1, top_k_context)]
    shortlist = [candidate for _rank, candidate in shortlist_pairs]
    valid_codes: list[str] = []
    candidates_by_code: dict[str, Any] = {}
    candidate_payloads: list[dict[str, Any]] = []
    raw_ranks: dict[str, int] = {}
    for raw_rank, candidate in shortlist_pairs:
        displayed = _display(candidate)
        payload = _candidate_payload(candidate)
        if displayed is None or not payload:
            continue
        code = displayed[0]
        if code in candidates_by_code:
            continue
        valid_codes.append(code)
        candidates_by_code[code] = candidate
        candidate_payloads.append(payload)
        raw_ranks[code] = raw_rank

    if not valid_codes:
        return {
            "entity_text": entity_text,
            "entity_type": entity_type,
            "choice_limit": choice_limit,
            "fallback": [],
            "prompt": None,
            "request_id": "",
            "valid_codes": [],
            "candidates_by_code": {},
            "decision_reason": "no_supported_candidate",
            "supported_count": 0,
            "shortlist_codes": [],
            "raw_ranks": {},
        }

    request_id = hashlib.sha1(
        f"{entity_type}|{entity_text}|{context}|{'|'.join(valid_codes)}".encode("utf-8")
    ).hexdigest()[:16]
    prompt = None
    if not fallback and len(valid_codes) >= 2:
        prompt = build_candidate_selector_prompt(
            entity_text,
            entity_type,
            candidate_payloads,
            max_choices=choice_limit,
            context=context,
            request_id=request_id,
        )
    return {
        "entity_text": entity_text,
        "entity_type": entity_type,
        "choice_limit": choice_limit,
        "fallback": fallback,
        "prompt": prompt,
        "request_id": request_id,
        "valid_codes": valid_codes,
        "candidates_by_code": candidates_by_code,
        "decision_reason": decision_reason,
        "supported_count": len(supported_ranked),
        "shortlist_codes": valid_codes,
        "raw_ranks": raw_ranks,
    }


def _apply_hierarchy_guard(codes: list[str]) -> list[str]:
    output: list[str] = []
    for code in codes:
        if any(_hierarchy_related(code, kept) for kept in output):
            continue
        output.append(code)
    return output


def _selector_response_valid(raw_output: str, prepared: dict) -> bool:
    try:
        selection = CandidateSelection.from_dict(extract_json(raw_output))
    except Exception:
        return False
    if selection is None or selection.request_id != prepared["request_id"]:
        return False
    if len(selection.chosen_codes) > prepared["choice_limit"]:
        return False
    valid_set = set(prepared["valid_codes"])
    return all(code in valid_set for code in selection.chosen_codes)


def _finish_selection(raw_output: str, prepared: dict) -> list[str]:
    try:
        selection = CandidateSelection.from_dict(extract_json(raw_output))
    except Exception as exc:
        print(
            f"[candidate_selector] parse error for {prepared['entity_text']!r}: {exc}",
            file=sys.stderr,
        )
        return list(prepared["fallback"])
    if selection is None or selection.request_id != prepared["request_id"]:
        return list(prepared["fallback"])
    if len(selection.chosen_codes) > prepared["choice_limit"]:
        return list(prepared["fallback"])

    valid_set = set(prepared["valid_codes"])
    chosen = list(dict.fromkeys(code for code in selection.chosen_codes if code in valid_set))
    # Do not reapply the older lexical support gate here. The prompt whitelist
    # already contains only supported candidates. Only true hard conflicts and
    # hierarchy/max-count invariants remain.
    chosen = [
        code
        for code in chosen
        if not _hard_candidate_conflict(prepared["candidates_by_code"][code])
    ]
    chosen = _apply_hierarchy_guard(chosen)
    return chosen[: prepared["choice_limit"]]


def select_supported_top_candidates(
    entity_text: str,
    entity_type: str,
    candidates: list,
    *,
    max_choices: int = 2,
) -> list[str]:
    supported = _rank_supported_candidates(entity_text, entity_type, candidates)
    selected, _reason = _deterministic_decision(entity_type, supported)
    return _apply_hierarchy_guard(selected)[:max_choices]


def select_candidates(
    entity_text: str,
    entity_type: str,
    candidates: list,
    llm: LocalLLM,
    *,
    top_k_context: int = 10,
    max_choices: int = 2,
    context: str = "",
) -> list[str]:
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
        return list(prepared["fallback"])
    try:
        raw_output = llm.generate(*prepared["prompt"])
    except Exception as exc:
        print(
            f"[candidate_selector] LLM error for {entity_text!r}: {exc} -> fallback",
            file=sys.stderr,
        )
        return list(prepared["fallback"])
    return _finish_selection(raw_output, prepared)


def select_candidates_many(
    items: list[dict],
    llm: LocalLLM,
    *,
    top_k_context: int = 10,
    max_choices: int = 2,
    batch_size: int = 4,
    cache_path: str | Path | None = None,
    model_id: str = "Qwen/Qwen3-8B",
) -> list[list[str]]:
    """Batch genuine ambiguities and isolate oversized/failed prompts."""
    global LAST_SELECTION_AUDIT, LAST_SELECTION_WORKLOAD
    results: list[list[str] | None] = [None] * len(items)
    audit = [
        {
            "status": "pending",
            "cache_hit": False,
            "selected_codes": [],
            "decision_reason": "",
            "supported_count": 0,
            "shortlist_codes": [],
            "raw_ranks": {},
        }
        for _ in items
    ]

    prepared_rows: list[tuple[int, dict]] = []
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
            audit[index]["status"] = "retrieval_empty"
            continue
        audit[index].update({
            "decision_reason": prepared["decision_reason"],
            "supported_count": prepared["supported_count"],
            "shortlist_codes": list(prepared["shortlist_codes"]),
            "raw_ranks": dict(prepared["raw_ranks"]),
        })
        if prepared["prompt"] is None:
            results[index] = list(prepared["fallback"])
            if results[index]:
                audit[index]["status"] = "deterministic_bypass"
            elif prepared["decision_reason"] == "single_medium_abstain":
                audit[index]["status"] = "single_medium_abstain"
            elif prepared["decision_reason"] == "unsafe_single_token_diagnosis_fragment":
                audit[index]["status"] = "unsafe_fragment_abstain"
            else:
                audit[index]["status"] = "unsupported_abstain"
            audit[index]["selected_codes"] = list(results[index])
        else:
            prepared_rows.append((index, prepared))

    selector_events: list[dict] = []
    prompt_token_counts: list[int] = []
    safe_rows: list[tuple[int, dict]] = []
    # Count each prompt separately if the batch count raises. One oversized
    # prompt must never terminate selection for every other entity.
    if prepared_rows:
        prompts = [prepared["prompt"] for _index, prepared in prepared_rows]
        try:
            counts = llm.count_prompt_tokens(prompts) if hasattr(llm, "count_prompt_tokens") else [0] * len(prompts)
            safe_rows = prepared_rows
            prompt_token_counts = list(counts)
        except Exception as batch_exc:
            print(f"[candidate_selector] token-count batch error: {batch_exc}; isolating rows", file=sys.stderr)
            for index, prepared in prepared_rows:
                try:
                    count = (
                        llm.count_prompt_tokens([prepared["prompt"]])[0]
                        if hasattr(llm, "count_prompt_tokens") else 0
                    )
                except Exception as row_exc:
                    results[index] = list(prepared["fallback"])
                    audit[index].update({
                        "status": "prompt_too_long_fallback",
                        "prompt_error": str(row_exc),
                        "selected_codes": list(results[index]),
                    })
                    continue
                safe_rows.append((index, prepared))
                prompt_token_counts.append(int(count))

    LAST_SELECTION_WORKLOAD = {
        "selector_count": len(safe_rows),
        "selector_input_tokens_p50": _pctl(prompt_token_counts, 0.50),
        "selector_input_tokens_p95": _pctl(prompt_token_counts, 0.95),
        "selector_input_tokens_max": max(prompt_token_counts, default=0),
        "selector_oversized_count": len(prepared_rows) - len(safe_rows),
    }
    print(f"[Qwen:workload-before-selector] {LAST_SELECTION_WORKLOAD}", file=sys.stderr, flush=True)

    if safe_rows:
        prompts = [prepared["prompt"] for _index, prepared in safe_rows]
        cache = VersionedJsonlCache(cache_path) if cache_path is not None else None
        if cache:
            generation_config = {
                "max_new_tokens": 128,
                "max_batch_tokens": None,
                "batch_size": batch_size,
                "min_batch_size": 1,
                "dynamic_batching": True,
            }
            for position, (index, prepared) in enumerate(safe_rows):
                key = VersionedJsonlCache.make_key(
                    model_id,
                    "linking_selector",
                    prepared["prompt"],
                    prompt_version=SELECTOR_PROMPT_VERSION,
                    generation_config=generation_config,
                )
                cached = cache.get(key)
                audit[index]["cache_hit"] = bool(
                    cached is not None and _selector_response_valid(cached, prepared)
                )
        try:
            raw_outputs = generate_with_cache(
                llm,
                prompts,
                batch_size=batch_size,
                model_id=model_id,
                task="linking_selector",
                prompt_version=SELECTOR_PROMPT_VERSION,
                cache=cache,
                max_new_tokens=128,
                prompt_token_counts=prompt_token_counts or None,
                progress_callback=selector_events.append,
                response_validator=lambda position, raw: _selector_response_valid(
                    raw, safe_rows[position][1]
                ),
            )
        except Exception as exc:
            print(f"[candidate_selector] batch error: {exc} -> per-row fallback", file=sys.stderr)
            raw_outputs = [""] * len(safe_rows)

        for (index, prepared), raw_output in zip(safe_rows, raw_outputs):
            results[index] = _finish_selection(raw_output, prepared)
            audit[index].update({
                "status": "selector_selected" if results[index] else "selector_abstained",
                "selected_codes": list(results[index] or []),
                "raw_response": raw_output,
            })

    output_tokens = [
        int(value)
        for event in selector_events
        for value in event.get("output_tokens_by_row", [])
    ]
    LAST_SELECTION_WORKLOAD.update({
        "selector_microbatch_count": len(selector_events),
        "selector_output_tokens_p50": _pctl(output_tokens, 0.50),
        "selector_output_tokens_p95": _pctl(output_tokens, 0.95),
    })
    print(f"[Qwen:workload-after-selector] {LAST_SELECTION_WORKLOAD}", file=sys.stderr, flush=True)

    LAST_SELECTION_AUDIT = audit
    select_candidates_many.last_audit = audit
    return [result or [] for result in results]