"""Candidate generation: parsed mention -> dict[rxcui, RxNormCandidate].

Nhiệm vụ duy nhất: không để mất gold (recall cao). KHÔNG quyết định
candidate nào đúng nhất — việc đó thuộc về reranker.py.
"""

from __future__ import annotations

from typing import Any
import unicodedata

import numpy as np
try:
    import torch
    import torch.nn.functional as F
except ImportError:  # Lightweight repository/path tests do not need inference deps.
    torch = None
    F = None
_inference_mode = torch.inference_mode if torch is not None else (lambda: (lambda function: function))
from rapidfuzz import fuzz
try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = AutoTokenizer = None

from . import config
from .repository import RxNormRepository
from .schemas import ParsedDrugMention, RxNormCandidate
from .parser import normalize_text, build_query_variants
from ..sapbert_encoder import resolve_model_source

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_project_path(path_str: str) -> str:
    """Compatibility wrapper around the cross-platform SapBERT resolver."""
    source, _is_local = resolve_model_source(
        str(path_str),
        project_root=_PROJECT_ROOT,
    )
    return source

class SentenceEncoder:
    """Bọc model + tokenizer để encode query thành dense vector.

    Logic pooling/normalize lấy nguyên từ cell encode_queries() trong
    embedding_rxnorm.ipynb, chỉ đóng gói lại thành class để inject vào
    RxNormRetriever.
    """

    def __init__(self, index_config: dict[str, Any], device: str | None = None):
        if torch is None or AutoModel is None:
            raise RuntimeError("PyTorch and transformers are required for RxNorm dense retrieval")
        model_cfg = index_config["model"]

        self.model_id = resolve_project_path(model_cfg["model_id"])
        self.pooling = model_cfg["pooling"]
        self.max_length = int(model_cfg["max_length"])
        self.dimension = int(model_cfg["dimension"])

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

        if self.model.config.hidden_size != self.dimension:
            raise ValueError(
                f"Model hidden_size={self.model.config.hidden_size} "
                f"!= config dimension={self.dimension}"
            )

    @_inference_mode()
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        outputs: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            model_outputs = self.model(**inputs)
            hidden = model_outputs.last_hidden_state

            if self.pooling == "cls":
                embeddings = hidden[:, 0, :]
            elif self.pooling == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                raise ValueError(f"Pooling không hỗ trợ: {self.pooling}")

            embeddings = F.normalize(embeddings, p=2, dim=1)
            outputs.append(embeddings.cpu().numpy().astype(np.float32))

        return np.concatenate(outputs, axis=0)


class RxNormRetriever:
    def __init__(self, repository: RxNormRepository, encoder: SentenceEncoder):
        self.repository = repository
        self.encoder = encoder

    # ------------------------------------------------------------
    # Dense search theo tier
    # ------------------------------------------------------------

    def search_tier(self, query: str, tier: str, k: int) -> list[dict[str, Any]]:
        index = self.repository.indexes[tier]
        metadata = self.repository.metadata[tier]

        search_k = min(k, index.ntotal)
        vector = self.encoder.encode([query])
        scores, ids = index.search(vector, search_k)

        results: list[dict[str, Any]] = []

        for score, vector_id in zip(scores[0], ids[0]):
            if vector_id < 0:
                continue

            row = metadata[int(vector_id)]
            if row is None:
                continue

            results.append({**row, "tier": tier, "dense_score": float(score)})

        return results

    def search_tier_many(self, queries: list[str], tier: str, k: int) -> list[list[dict[str, Any]]]:
        """Search one tier for a query batch with a single encoder pass."""
        if not queries:
            return []
        index = self.repository.indexes[tier]
        metadata = self.repository.metadata[tier]
        search_k = min(k, index.ntotal)
        vectors = self.encoder.encode(queries)
        scores, ids = index.search(vectors, search_k)
        output = []
        for row_scores, row_ids in zip(scores, ids):
            row_results = []
            for score, vector_id in zip(row_scores, row_ids):
                if vector_id < 0:
                    continue
                row = metadata[int(vector_id)]
                if row is not None:
                    row_results.append({**row, "tier": tier, "dense_score": float(score)})
            output.append(row_results)
        return output


    def _recover_concatenated_components(self, parsed: ParsedDrugMention) -> None:
        """Recover missing separators using the repository term lexicon.

        Only exact support/brand terms are used. This prevents a concatenated
        multi-drug span from being linked to one constituent by fuzzy retrieval.
        """
        if not parsed.ingredient_core or len(parsed.ingredient_components) != 1:
            return
        original = parsed.ingredient_components[0]
        normalized_original = unicodedata.normalize("NFD", normalize_text(original))
        normalized_original = "".join(
            char for char in normalized_original
            if unicodedata.category(char) != "Mn"
        ).replace("đ", "d")
        compact = "".join(char for char in normalized_original if char.isalnum())
        if len(compact) < 8:
            return
        lexicon = self.repository.compact_component_terms
        recovered: list[str] | None = None
        for split in range(4, len(compact) - 3):
            left, right = compact[:split], compact[split:]
            if left in lexicon and right in lexicon:
                recovered = [lexicon[left], lexicon[right]]
                break
        if recovered is None:
            admin_suffixes = ("trongngay", "hangngay", "moingay", "nhung", "oral", "ngay")
            for suffix in admin_suffixes:
                if compact.endswith(suffix):
                    prefix = compact[:-len(suffix)]
                    if prefix in lexicon:
                        recovered = [lexicon[prefix]]
                        parsed.parse_warnings.append("attached_administration_suffix_removed")
                        if suffix == "oral":
                            parsed.route = parsed.route or "PO"
                            if "oral" not in parsed.dose_forms:
                                parsed.dose_forms.append("oral")
                        break
        if recovered is None:
            return
        parsed.ingredient_components = recovered
        parsed.ingredient_core = " / ".join(recovered)
        parsed.ingredient_aliases = list(dict.fromkeys([*recovered, *parsed.brand_hints]))
        if len(recovered) >= 2 and "multi_ingredient_mention" not in parsed.parse_warnings:
            parsed.parse_warnings.append("multi_ingredient_mention")
        parsed.query_variants = build_query_variants(parsed)

    def search_full_query(self, parsed: ParsedDrugMention) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        results += self.search_tier(parsed.normalized_text, "product", config.DEFAULT_PRODUCT_K)
        results += self.search_tier(parsed.normalized_text, "support", config.DEFAULT_SUPPORT_K)
        results += self.search_tier(
            parsed.normalized_text, "historical", config.DEFAULT_HISTORICAL_K
        )

        return results

    def search_core_query(self, parsed: ParsedDrugMention) -> list[dict[str, Any]]:
        if not parsed.ingredient_core:
            return []

        results: list[dict[str, Any]] = []
        results += self.search_tier(parsed.ingredient_core, "support", config.DEFAULT_SUPPORT_K)
        results += self.search_tier(parsed.ingredient_core, "product", config.DEFAULT_PRODUCT_K)

        return results

    # ------------------------------------------------------------
    # Exact-match injection (không để mất gold do dense miss)
    # ------------------------------------------------------------

    def inject_exact_term(self, parsed: ParsedDrugMention) -> list[dict[str, Any]]:
        matches = self.repository.exact_term_lookup.get(parsed.normalized_text, [])
        results: list[dict[str, Any]] = []

        for tier, vector_id in matches:
            row = self.repository.metadata[tier][vector_id]
            if row is None:
                continue

            results.append({**row, "tier": tier, "exact_term_match": True})

        return results

    def inject_exact_ingredient(self, parsed: ParsedDrugMention) -> list[dict[str, Any]]:
        components = parsed.ingredient_components or (
            [parsed.ingredient_core] if parsed.ingredient_core else []
        )
        if not components:
            return []
        component_sets = [
            set(self.repository.core_lookup.get(normalize_text(component), []))
            for component in components
        ]
        component_sets = [values for values in component_sets if values]
        if not component_sets:
            return []
        if len(components) >= 2 and len(component_sets) == len(components):
            # True combination candidates must contain every explicit component.
            rxcuis = sorted(set.intersection(*component_sets))
        else:
            rxcuis = sorted(set().union(*component_sets))
        return self._rows_for_rxcuis(rxcuis, exact_ingredient_match=True)

    def inject_exact_brand(self, parsed: ParsedDrugMention) -> list[dict[str, Any]]:
        rxcuis: list[str] = []
        for brand in parsed.brand_hints:
            rxcuis.extend(self.repository.brand_lookup.get(normalize_text(brand), []))
        return self._rows_for_rxcuis(rxcuis, exact_term_match=True)

    def _rows_for_rxcuis(self, rxcuis, **flags) -> list[dict[str, Any]]:
        results = []
        for rxcui in dict.fromkeys(str(value) for value in rxcuis):
            for tier, vector_id in self.repository.rows_by_rxcui.get(rxcui, []):
                row = self.repository.metadata[tier][vector_id]
                if row is not None:
                    results.append({**row, "tier": tier, "dense_score": 0.0, **flags})
        return results

    def inject_structured(self, parsed: ParsedDrugMention) -> list[dict[str, Any]]:
        components = parsed.ingredient_components or (
            [parsed.ingredient_core] if parsed.ingredient_core else []
        )
        if not components:
            return []
        rxcuis: list[str] = []
        for component in components:
            for strength in parsed.strengths:
                key = normalize_text(f"{component} {strength}")
                rxcuis.extend(self.repository.ingredient_strength_lookup.get(key, []))
            for form in parsed.dose_forms:
                key = normalize_text(f"{component} {form}")
                rxcuis.extend(self.repository.ingredient_form_lookup.get(key, []))
            for release in parsed.release_types:
                key = normalize_text(f"{component} {release}")
                rxcuis.extend(self.repository.ingredient_release_lookup.get(key, []))
        if len(components) >= 2:
            component_sets = [
                set(self.repository.core_lookup.get(normalize_text(component), []))
                for component in components
            ]
            if all(component_sets):
                rxcuis.extend(sorted(set.intersection(*component_sets)))
        return self._rows_for_rxcuis(rxcuis, exact_structured_match=True)

    # ------------------------------------------------------------
    # Gộp theo rxcui, giữ candidate priority tốt nhất cho mỗi tier gặp
    # ------------------------------------------------------------

    def aggregate_by_rxcui(
        self, raw_results: list[dict[str, Any]], source: str
    ) -> dict[str, RxNormCandidate]:
        candidates: dict[str, RxNormCandidate] = {}

        for row in raw_results:
            rxcui = row["rxcui"]
            structured = self.repository.get_structured_record(rxcui)

            if rxcui not in candidates:
                candidates[rxcui] = RxNormCandidate(
                    rxcui=rxcui,
                    tty=row["concept_tty"],
                    tier=row["tier"],
                    name=str(structured.get("name") or row["text"]),
                    active=row.get("active", True),
                    historical=row["tier"] == "historical",
                    candidate_priority=row.get("candidate_priority", 99),
                    current_rxcuis=row.get("current_rxcuis", []),
                    structured=structured,
                )

            candidate = candidates[rxcui]

            candidate.dense_score = max(candidate.dense_score, row.get("dense_score", 0.0))
            candidate.exact_term_match = candidate.exact_term_match or row.get(
                "exact_term_match", False
            )
            candidate.exact_ingredient_match = candidate.exact_ingredient_match or row.get(
                "exact_ingredient_match", False
            )
            candidate.features["exact_structured_match"] = bool(
                candidate.features.get("exact_structured_match")
                or row.get("exact_structured_match", False)
            )

            if row["text"] not in candidate.matched_terms:
                candidate.matched_terms.append(row["text"])

            if source not in candidate.retrieval_sources:
                candidate.retrieval_sources.append(source)

        return candidates

    def _lexical_score(self, parsed: ParsedDrugMention, candidate: RxNormCandidate) -> float:
        best = 0.0
        for term in candidate.matched_terms:
            normalized = normalize_text(term)
            best = max(
                best,
                fuzz.ratio(parsed.normalized_text, normalized) / 100.0,
                fuzz.token_set_ratio(parsed.normalized_text, normalized) / 100.0,
            )
        return best

    def retrieve(self, parsed: ParsedDrugMention) -> dict[str, RxNormCandidate]:
        return self.retrieve_many([parsed])[0]

    def retrieve_many(self, parsed_mentions: list[ParsedDrugMention]) -> list[dict[str, RxNormCandidate]]:
        """High-recall multi-variant retrieval with one unique-query encoding."""
        if not parsed_mentions:
            return []
        for item in parsed_mentions:
            self._recover_concatenated_components(item)
        variants_by_mention = [item.query_variants or [{
            "text": item.normalized_text, "source": "full_normalized",
        }] for item in parsed_mentions]
        unique_queries = list(dict.fromkeys(
            variant["text"] for variants in variants_by_mention for variant in variants
        ))
        query_index = {value: index for index, value in enumerate(unique_queries)}
        vectors = self.encoder.encode(unique_queries)
        tier_rows: dict[str, list[list[dict[str, Any]]]] = {}
        for tier, k in (
            ("product", config.DEFAULT_PRODUCT_K),
            ("support", config.DEFAULT_SUPPORT_K),
            ("historical", config.DEFAULT_HISTORICAL_K),
        ):
            index = self.repository.indexes[tier]
            scores, ids = index.search(vectors, min(k, index.ntotal))
            metadata = self.repository.metadata[tier]
            tier_rows[tier] = []
            for row_scores, row_ids in zip(scores, ids):
                values = []
                for score, vector_id in zip(row_scores, row_ids):
                    if vector_id < 0:
                        continue
                    row = metadata[int(vector_id)]
                    if row is not None:
                        values.append({**row, "tier": tier, "dense_score": float(score)})
                tier_rows[tier].append(values)

        outputs = []
        for index, parsed in enumerate(parsed_mentions):
            merged: dict[str, RxNormCandidate] = {}
            sources = []
            for variant in variants_by_mention[index]:
                query_row = query_index[variant["text"]]
                rows = (
                    tier_rows["product"][query_row]
                    + tier_rows["support"][query_row]
                    + tier_rows["historical"][query_row]
                )
                sources.append((f"query:{variant['source']}", rows))
            sources.extend((
                ("exact_term", self.inject_exact_term(parsed)),
                ("exact_ingredient", self.inject_exact_ingredient(parsed)),
                ("exact_brand", self.inject_exact_brand(parsed)),
                ("exact_structured", self.inject_structured(parsed)),
            ))
            for source_name, raw_results in sources:
                for rxcui, candidate in self.aggregate_by_rxcui(raw_results, source_name).items():
                    if rxcui not in merged:
                        merged[rxcui] = candidate
                        continue
                    existing = merged[rxcui]
                    existing.dense_score = max(existing.dense_score, candidate.dense_score)
                    existing.exact_term_match |= candidate.exact_term_match
                    existing.exact_ingredient_match |= candidate.exact_ingredient_match
                    existing.features["exact_structured_match"] = bool(
                        existing.features.get("exact_structured_match")
                        or candidate.features.get("exact_structured_match")
                    )
                    existing.matched_terms.extend(term for term in candidate.matched_terms if term not in existing.matched_terms)
                    existing.retrieval_sources.extend(src for src in candidate.retrieval_sources if src not in existing.retrieval_sources)
            for candidate in merged.values():
                candidate.lexical_score = self._lexical_score(parsed, candidate)
                candidate.query_variants = list(candidate.retrieval_sources)
            historical_current = []
            for candidate in list(merged.values()):
                if candidate.historical and candidate.current_rxcuis:
                    historical_current.extend(candidate.current_rxcuis)
            if historical_current:
                for rxcui, candidate in self.aggregate_by_rxcui(
                    self._rows_for_rxcuis(historical_current),
                    "historical_current_mapping",
                ).items():
                    if rxcui not in merged:
                        merged[rxcui] = candidate
                    elif "historical_current_mapping" not in merged[rxcui].retrieval_sources:
                        merged[rxcui].retrieval_sources.append("historical_current_mapping")
            outputs.append(merged)
        return outputs