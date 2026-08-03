"""Điều phối pipeline: mention -> parsed -> candidates -> ranked.

Không đặt regex, load JSONL, hay công thức score trực tiếp ở đây.
"""

from __future__ import annotations

from typing import Any

from . import config
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
        return self.rank_many(mentions, top_k=top_k)

    def retrieve_many(self, mentions: list[str]):
        parsed_mentions = [parse_drug_mention(mention) for mention in mentions]
        retrieved = self.retriever.retrieve_many(parsed_mentions)
        return parsed_mentions, retrieved

    def rank_many(self, mentions: list[str], top_k: int = 50) -> list[list]:
        parsed_mentions, retrieved = self.retrieve_many(mentions)
        return [
            self.reranker.rerank(parsed, list(candidates.values()))[:top_k]
            for parsed, candidates in zip(parsed_mentions, retrieved)
        ]

    def predict_many(self, mentions: list[str]) -> list[list]:
        outputs = []
        for candidates in self.rank_many(mentions, top_k=10):
            if not candidates:
                outputs.append([])
                continue
            top = candidates[0]
            unique_exact = top.support_level == "exact" and not any(
                item.support_level == "exact" for item in candidates[1:]
            )
            if top.support_level in {"exact", "strong"} and not top.rejection_reasons and (
                unique_exact or (top.top1_margin or 0.0) >= config.DETERMINISTIC_MIN_MARGIN
            ):
                outputs.append([top])
            else:
                outputs.append([])
        return outputs

    def predict(self, mention: str) -> list:
        """Conservative final RxNorm result; ambiguous retrieval abstains."""
        return self.predict_many([mention])[0]
