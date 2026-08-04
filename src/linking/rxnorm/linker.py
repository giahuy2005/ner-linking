"""Orchestrate RxNorm parse -> retrieve -> rerank -> conservative prediction."""

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
        self.last_errors: list[dict[str, Any]] = []

    def link(self, mention: str, top_k: int = 10) -> dict[str, Any]:
        parsed = parse_drug_mention(mention)
        candidates = self.retriever.retrieve(parsed)
        ranked = self.reranker.rerank(parsed, list(candidates.values()))
        return {
            "mention": mention,
            "parsed": parsed,
            "candidates": ranked[:top_k],
        }

    def retrieve_many(self, mentions: list[str]):
        parsed_mentions = [parse_drug_mention(mention) for mention in mentions]
        retrieved = self.retriever.retrieve_many(parsed_mentions)
        return parsed_mentions, retrieved

    def rank_many(self, mentions: list[str], top_k: int = 50) -> list[list]:
        """Batch retrieval with per-mention failure isolation."""
        self.last_errors = []
        if not mentions:
            return []
        try:
            parsed_mentions, retrieved = self.retrieve_many(mentions)
            return [
                self.reranker.rerank(parsed, list(candidates.values()))[:top_k]
                for parsed, candidates in zip(parsed_mentions, retrieved)
            ]
        except Exception as batch_exc:
            outputs: list[list] = []
            for index, mention in enumerate(mentions):
                try:
                    outputs.append(self.link(mention, top_k=top_k)["candidates"])
                except Exception as row_exc:
                    self.last_errors.append({
                        "index": index,
                        "mention": mention,
                        "batch_error": str(batch_exc),
                        "row_error": str(row_exc),
                    })
                    outputs.append([])
            return outputs

    def link_many(self, mentions: list[str], top_k: int = 10) -> list[list]:
        return self.rank_many(mentions, top_k=top_k)

    @staticmethod
    def _predict_one(candidates: list) -> list:
        supported = [
            item for item in candidates
            if item.support_level in {"exact", "strong", "medium"}
            and not item.rejection_reasons
        ]
        if not supported:
            return []
        top = supported[0]
        exact_count = sum(item.support_level == "exact" for item in supported)
        if top.support_level == "exact" and exact_count == 1:
            return [top]
        if top.support_level == "strong" and (
            len(supported) == 1
            or float(top.top1_margin or 0.0) >= config.DETERMINISTIC_MIN_MARGIN
        ):
            return [top]
        return []

    def predict_many(self, mentions: list[str]) -> list[list]:
        return [self._predict_one(items) for items in self.rank_many(mentions, top_k=50)]

    def predict(self, mention: str) -> list:
        return self.predict_many([mention])[0]