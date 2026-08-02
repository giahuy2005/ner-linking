"""Điều phối pipeline: mention -> parsed -> candidates -> ranked.

Không đặt regex, load JSONL, hay công thức score trực tiếp ở đây.
"""

from __future__ import annotations

from typing import Any

from .parser import parse_drug_mention
from .reranker import RxNormRuleReranker
from .repository import RxNormRepository
from .retriever import RxNormRetriever, SentenceEncoder


class RxNormLinker:
    def __init__(self, index_dir: str, clean_path: str | None = None, device: str | None = None):
        self.repository = RxNormRepository(index_dir=index_dir, clean_path=clean_path)
        self.encoder = SentenceEncoder(self.repository.config, device=device)
        self.retriever = RxNormRetriever(self.repository, self.encoder)
        self.reranker = RxNormRuleReranker()

    def link(self, mention: str, top_k: int = 10) -> dict[str, Any]:
        parsed = parse_drug_mention(mention)

        candidates = self.retriever.retrieve(parsed)
        ranked = self.reranker.rerank(parsed, list(candidates.values()))

        return {
            "mention": mention,
            "parsed": parsed,
            "candidates": ranked[:top_k],
        }

    def link_many(self, mentions: list[str], top_k: int = 10) -> list[list]:
        """Parse, encode, retrieve, and rerank a mention batch."""
        parsed_mentions = [parse_drug_mention(mention) for mention in mentions]
        retrieved = self.retriever.retrieve_many(parsed_mentions)
        return [
            self.reranker.rerank(parsed, list(candidates.values()))[:top_k]
            for parsed, candidates in zip(parsed_mentions, retrieved)
        ]

    def predict(self, mention: str) -> list:
        """Conservative final RxNorm result; ambiguous retrieval abstains."""
        result = self.link(mention, top_k=10)
        candidates = result["candidates"]
        if not candidates:
            return []
        top = candidates[0]
        features = top.features or {}
        no_conflict = all(
            features.get(name) not in {"mismatch", "order_dose_mismatch"}
            for name in ("strength_relation", "form_relation", "release_relation")
        )
        if no_conflict and (
            top.exact_term_match
            or features.get("ingredient_relation") == "exact" and top.final_score >= 0.60
        ):
            return [top]
        return []
