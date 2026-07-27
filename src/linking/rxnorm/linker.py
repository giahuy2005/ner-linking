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
