"""Candidate generation: parsed mention -> dict[rxcui, RxNormCandidate].

Nhiệm vụ duy nhất: không để mất gold (recall cao). KHÔNG quyết định
candidate nào đúng nhất — việc đó thuộc về reranker.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rapidfuzz import fuzz
from transformers import AutoModel, AutoTokenizer

from . import config
from .repository import RxNormRepository
from .schemas import ParsedDrugMention, RxNormCandidate
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

    @torch.inference_mode()
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
        if not parsed.ingredient_core:
            return []

        rxcuis = self.repository.core_lookup.get(parsed.ingredient_core, [])
        if not rxcuis:
            return []

        results: list[dict[str, Any]] = []

        for tier, rows in self.repository.metadata.items():
            for row in rows:
                if row is not None and row["rxcui"] in rxcuis:
                    results.append(
                        {**row, "tier": tier, "dense_score": 0.0, "exact_ingredient_match": True}
                    )

        return results

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
                    name=row["text"],
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

            if row["text"] not in candidate.matched_terms:
                candidate.matched_terms.append(row["text"])

            if source not in candidate.retrieval_sources:
                candidate.retrieval_sources.append(source)

        return candidates

    def _lexical_score(self, parsed: ParsedDrugMention, candidate: RxNormCandidate) -> float:
        best = 0.0
        for term in candidate.matched_terms:
            best = max(best, fuzz.partial_ratio(parsed.normalized_text, term) / 100.0)
        return best

    def retrieve(self, parsed: ParsedDrugMention) -> dict[str, RxNormCandidate]:
        merged: dict[str, RxNormCandidate] = {}

        sources = (
            ("full_query", self.search_full_query(parsed)),
            ("core_query", self.search_core_query(parsed)),
            ("exact_term", self.inject_exact_term(parsed)),
            ("exact_ingredient", self.inject_exact_ingredient(parsed)),
        )

        for source_name, raw_results in sources:
            batch = self.aggregate_by_rxcui(raw_results, source_name)

            for rxcui, candidate in batch.items():
                if rxcui not in merged:
                    merged[rxcui] = candidate
                    continue

                existing = merged[rxcui]
                existing.dense_score = max(existing.dense_score, candidate.dense_score)
                existing.exact_term_match = existing.exact_term_match or candidate.exact_term_match
                existing.exact_ingredient_match = (
                    existing.exact_ingredient_match or candidate.exact_ingredient_match
                )

                for term in candidate.matched_terms:
                    if term not in existing.matched_terms:
                        existing.matched_terms.append(term)

                for src in candidate.retrieval_sources:
                    if src not in existing.retrieval_sources:
                        existing.retrieval_sources.append(src)

        for candidate in merged.values():
            candidate.lexical_score = self._lexical_score(parsed, candidate)

        return merged
