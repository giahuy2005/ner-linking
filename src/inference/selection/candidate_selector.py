"""Use the loaded Qwen3-8B instance to select from retrieved candidates.

LUÔN validate code LLM chọn PHẢI nằm trong list candidate gốc đưa vào —
không tin LLM tự bịa code không có trong danh sách (hallucination). Nếu
LLM lỗi hoặc chọn toàn code không hợp lệ thì trả rỗng, trừ exact match đã
được kiểm chứng. Với metric phạt mã thừa/sai, abstain an toàn hơn ép mã rác.
"""

from __future__ import annotations

import re
import sys
import unicodedata
import hashlib
import json
from dataclasses import asdict, is_dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING

from ...llm.json_guard import extract_json
from ...llm.prompts import build_candidate_selector_prompt
from ...llm.response_schemas import CandidateSelection

if TYPE_CHECKING:
    from ...llm.backend import LocalLLM

LAST_SELECTION_AUDIT: list[dict] = []


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


def _candidate_payload(candidate) -> dict:
    displayed = _display(candidate)
    if displayed is None:
        return {}
    code, label = displayed
    if is_dataclass(candidate):
        raw = asdict(candidate)
        return {
            "code": code, "name": raw.get("name", label), "tty": raw.get("tty"),
            "tier": raw.get("tier"), "active": raw.get("active"),
            "historical": raw.get("historical"), "current_rxcuis": raw.get("current_rxcuis", []),
            "exact_flags": {"term": raw.get("exact_term_match"), "ingredient": raw.get("exact_ingredient_match")},
            "support_level": raw.get("support_level"), "score": raw.get("final_score"),
            "features": raw.get("features", {}), "matched_terms": raw.get("matched_terms", []),
            "query_sources": raw.get("retrieval_sources", []), "hard_conflicts": raw.get("rejection_reasons", []),
        }
    raw = candidate if isinstance(candidate, dict) else {}
    return {
        "code": code, "preferred_or_matched_term": raw.get("matched_term", label),
        "language": raw.get("language"), "term_type": raw.get("term_type"),
        "score": raw.get("score"), "matched_terms": raw.get("matched_terms", []),
        "query_sources": raw.get("query_sources", []), "aggregate_components": raw.get("aggregate_components", {}),
        "exact_alias_source": raw.get("exact_alias_source"), "chapter_support": raw.get("chapter_support"),
        "lexical_coverage": raw.get("lexical_coverage"), "over_specific": raw.get("over_specific", False),
    }


class SelectorJsonlCache:
    def __init__(self, path: str | Path):
        self.path = Path(path); self.values = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line); self.values[str(row["key"])] = str(row["response"])
                except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                    continue

    @staticmethod
    def key(model_id: str, prompt: tuple[str, str], config_hash: str = "") -> str:
        value = json.dumps({"version": "selector_v2", "model": model_id, "prompt": prompt,
                            "config": config_hash}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def get(self, key): return self.values.get(key)

    def put(self, key, response):
        if key in self.values: return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")
        self.values[key] = response


def _normalize_exact(text: str) -> str:
    value = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", value).strip(" \t\r\n.,;:()[]{}")


_NON_SEMANTIC_TOKENS = {
    "benh", "hoi", "chung", "thuoc", "type", "typ", "khong", "xac", "dinh",
    "va", "hoac", "kem",
    "va", "hoac", "kem",
    "mg", "mcg", "g", "ml", "po", "iv", "im", "bid", "tid", "qid", "daily",
}


def _fold_ascii(text: str) -> str:
    value = unicodedata.normalize("NFD", _normalize_exact(text))
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", _fold_ascii(text))
        if token not in _NON_SEMANTIC_TOKENS and not token.isdigit()
    }


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


def _candidate_supported(entity_text: str, entity_type: str, candidate) -> bool:
    """Precision gate after 7B selection.

    Retrieval rank alone is not evidence.  Production candidates carry score/
    feature metadata; low-score candidates with no lexical or ingredient
    agreement are rejected.  Metadata-free candidates remain accepted for
    backwards-compatible custom linkers and unit-test doubles.
    """
    displayed = _display(candidate)
    if displayed is None:
        return False
    _code, label = displayed

    if entity_type == "CHẨN_ĐOÁN" and isinstance(candidate, dict):
        matched_term = str(candidate.get("matched_term") or "")
        score_value = candidate.get("score")
        if score_value is None:
            return True
        if matched_term and _normalize_exact(matched_term) == _normalize_exact(entity_text):
            return True
        score = float(score_value or 0.0)
        coverage, similarity = _lexical_support(entity_text, matched_term)
        mention_tokens = _meaningful_tokens(entity_text)
        candidate_tokens = _meaningful_tokens(matched_term)
        # Production ICD candidates carry aggregated matched-term evidence. A
        # label that adds an unsupported disease qualifier is over-specific
        # even when it contains every mention token (for example an anatomic
        # subtype). Keep it in retrieval, but do not emit it deterministically.
        if candidate.get("matched_terms") and candidate_tokens - mention_tokens:
            return False
        compact = re.sub(r"\W+", "", entity_text, flags=re.UNICODE)
        abbreviation = compact.isupper() and 2 <= len(compact) <= 8
        if len(mention_tokens) == 1 and len(candidate_tokens) > 1 and not abbreviation:
            # A generic one-word diagnosis matching one token of a longer ICD
            # label is underspecified; do not manufacture the missing disease.
            return score >= 0.84 and similarity >= 0.68
        return bool(
            (score >= 0.68 and (coverage >= 0.50 or similarity >= 0.72))
            or (score >= 0.82 and (coverage > 0.0 or similarity >= 0.45))
            or (abbreviation and score >= 0.92)
        )

    if entity_type == "THUỐC" and not isinstance(candidate, dict):
        features = getattr(candidate, "features", {}) or {}
        has_evidence = bool(features) or hasattr(candidate, "final_score")
        if not has_evidence:
            return True
        ingredient = features.get("ingredient_relation")
        if getattr(candidate, "exact_term_match", False) and ingredient == "exact":
            return True
        names = [str(getattr(candidate, "name", "") or "")]
        names.extend(str(value) for value in getattr(candidate, "matched_terms", []) or [])
        coverage, similarity = max(
            (_lexical_support(entity_text, name) for name in names if name),
            default=(0.0, 0.0),
        )
        final_score = float(getattr(candidate, "final_score", 0.0) or 0.0)
        no_conflict = all(features.get(key) not in {"mismatch", "order_dose_mismatch", "conflict"}
                          for key in ("strength_relation", "form_relation", "release_relation"))
        return bool(
            ingredient == "exact" and no_conflict and final_score >= 0.42
            or ingredient == "partial" and coverage >= 0.50 and final_score >= 0.50
            or coverage >= 0.75 and similarity >= 0.55 and final_score >= 0.48
        )

    # Legacy/custom dictionary candidates without confidence metadata.
    if isinstance(candidate, dict) and not any(
        key in candidate for key in ("score", "features", "final_score")
    ):
        return True
    return False


def _explicitly_coordinated_diagnoses(text: str) -> bool:
    """Allow two ICD codes only when the mention explicitly coordinates concepts."""
    normalized = _normalize_exact(text)
    return bool(re.search(r"(?:\b(?:và|hoặc|kèm)\b|[+;])", normalized))


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
    candidates_by_code = {}
    for candidate in candidates[:top_k_context]:
        pair = _display(candidate)
        if pair is not None:
            display_pairs.append(_candidate_payload(candidate))
            valid_codes.append(pair[0])
            candidates_by_code.setdefault(pair[0], candidate)
    if not display_pairs:
        return None
    choice_limit = 1
    if entity_type == "CHẨN_ĐOÁN" and max_choices >= 2 \
            and _explicitly_coordinated_diagnoses(entity_text):
        choice_limit = 2
    supported_candidates = [candidate for candidate in candidates[:top_k_context]
                            if _candidate_supported(entity_text, entity_type, candidate)]
    high_confidence = (
        _high_confidence_top(entity_text, entity_type, candidates[0])
        and _candidate_supported(entity_text, entity_type, candidates[0])
    )
    if not high_confidence and supported_candidates:
        top = supported_candidates[0]
        support = getattr(top, "support_level", None) if not isinstance(top, dict) else top.get("support_level")
        margin = getattr(top, "top1_margin", None) if not isinstance(top, dict) else top.get("top1_margin")
        if len(supported_candidates) == 1 and support in {"exact", "strong"}:
            high_confidence = True
        elif len(supported_candidates) == 1 and isinstance(top, dict) and float(top.get("score", 0.0) or 0.0) >= .72:
            high_confidence = True
        elif support in {"exact", "strong"} and margin is not None and float(margin) >= .08:
            high_confidence = True
        elif isinstance(top, dict) and len(supported_candidates) >= 2:
            first_score = float(top.get("aggregate_score", top.get("score", 0.0)) or 0.0)
            second = supported_candidates[1]
            second_score = float(second.get("aggregate_score", second.get("score", 0.0)) or 0.0)
            if first_score >= .80 and first_score - second_score >= .08:
                high_confidence = True
    fallback = valid_codes[:1] if high_confidence else []
    prompt = None
    request_id = hashlib.sha1(
        f"{entity_type}|{entity_text}|{context}".encode("utf-8")
    ).hexdigest()[:16]
    # Qwen is only useful for a real ambiguity between at least two supported
    # candidates. One weak candidate or an empty supported set must abstain.
    if not high_confidence and len(supported_candidates) >= 2:
        prompt = build_candidate_selector_prompt(
            entity_text,
            entity_type,
            display_pairs,
            max_choices=choice_limit,
            context=context,
            request_id=request_id,
        )
    return {
        "entity_text": entity_text,
        "valid_codes": valid_codes,
        "candidates_by_code": candidates_by_code,
        "entity_type": entity_type,
        "choice_limit": choice_limit,
        "fallback": fallback,
        "prompt": prompt,
        "request_id": request_id,
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
    if selection.request_id and selection.request_id != prepared.get("request_id", selection.request_id):
        return prepared["fallback"]
    if selection.request_id and len(selection.chosen_codes) > prepared["choice_limit"]:
        return prepared["fallback"]
    chosen = list(dict.fromkeys(code for code in selection.chosen_codes if code in valid_set))
    supported = [
        code for code in chosen
        if _candidate_supported(
            prepared["entity_text"],
            prepared["entity_type"],
            prepared["candidates_by_code"][code],
        )
    ]
    return supported[:prepared["choice_limit"]]


def select_supported_top_candidates(
    entity_text: str,
    entity_type: str,
    candidates: list,
    *,
    max_choices: int = 2,
) -> list[str]:
    """Conservative deterministic output for runs without the 7B selector."""
    limit = 1
    if entity_type == "CHẨN_ĐOÁN" and max_choices >= 2 \
            and _explicitly_coordinated_diagnoses(entity_text):
        limit = 2
    selected = []
    for candidate in candidates:
        pair = _display(candidate)
        if pair is None or not _candidate_supported(entity_text, entity_type, candidate):
            continue
        if pair[0] not in selected:
            selected.append(pair[0])
        if len(selected) >= limit:
            break
    return selected


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
    max_choices: int = 2,
    batch_size: int = 4,
    cache_path: str | Path | None = None,
    model_id: str = "Qwen/Qwen3-8B",
) -> list[list[str]]:
    """Batch only ambiguous selections; exact matches return without 7B."""
    global LAST_SELECTION_AUDIT
    results: list[list[str] | None] = [None] * len(items)
    audit = [{"status": "pending", "cache_hit": False, "selected_codes": []} for _ in items]
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
            audit[index]["status"] = "retrieval_empty"
        elif prepared["prompt"] is None:
            results[index] = prepared["fallback"][:1]
            audit[index]["status"] = "deterministic_bypass" if prepared["fallback"] else "unsupported_abstain"
            audit[index]["selected_codes"] = list(results[index])
        else:
            pending_indexes.append(index)
            prepared_items.append(prepared)
            prompts.append(prepared["prompt"])

    if prompts:
        cache = SelectorJsonlCache(cache_path) if cache_path is not None else None
        pending_prompts, pending_meta, cached_rows = [], [], []
        for position, prompt in enumerate(prompts):
            key = SelectorJsonlCache.key(model_id, prompt)
            value = cache.get(key) if cache else None
            if value is None:
                pending_prompts.append(prompt); pending_meta.append((position, key))
            else:
                cached_rows.append((position, value))
                audit[pending_indexes[position]]["cache_hit"] = True
        try:
            if hasattr(llm, "generate_batch"):
                generated = llm.generate_batch(pending_prompts, batch_size=batch_size) if pending_prompts else []
            else:
                generated = [llm.generate(*prompt) for prompt in pending_prompts]
            if len(generated) != len(pending_prompts):
                raise ValueError(
                    f"batch returned {len(generated)} outputs for {len(pending_prompts)} prompts"
                )
            raw_outputs = [""] * len(prompts)
            for position, value in cached_rows: raw_outputs[position] = value
            for (position, key), value in zip(pending_meta, generated):
                raw_outputs[position] = value
                if cache: cache.put(key, value)
        except Exception as exc:
            print(f"[candidate_selector] batch lỗi: {exc} -> fallback", file=sys.stderr)
            raw_outputs = [""] * len(prompts)
        for index, prepared, raw_output in zip(pending_indexes, prepared_items, raw_outputs):
            results[index] = _finish_selection(raw_output, prepared)
            audit[index].update({
                "status": "selector_selected" if results[index] else "selector_abstained",
                "selected_codes": list(results[index] or []), "raw_response": raw_output,
            })

    LAST_SELECTION_AUDIT = audit
    select_candidates_many.last_audit = audit
    return [result or [] for result in results]
