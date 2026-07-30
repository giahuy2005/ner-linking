"""Kiểu dữ liệu dùng chung trong package inference.

Chỉ định nghĩa shape, không chứa logic xử lý.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NerEntity:
    """1 entity NER + assertion, TRƯỚC khi qua linking (chưa có candidates).

    position là char offset [start, end) trên raw text. Predictor notebook mới
    không mutate input trước tokenization để bảo toàn offset tuyệt đối.
    """

    text: str
    type: str
    assertions: list[str] = field(default_factory=list)
    position: tuple[int, int] = (0, 0)
    score: float = 1.0  # confidence proxy từ emission của nhãn CRF đã decode
    flag: str | None = None  # lý do repair_gate nghi ngờ (vd "suspect_truncated_diagnosis"), None = không nghi ngờ
    # Sidecar nội bộ từ Qwen 1.5B cho grouped handoff 7B; không xuất BTC JSON.
    review_hints: list[dict[str, Any]] = field(default_factory=list)

    def to_btc_dict(self, candidates: list[str] | None = None) -> dict[str, Any]:
        """Convert sang đúng format JSON BTC yêu cầu (thiếu 'candidates' thì để [])."""
        return {
            "text": self.text,
            "type": self.type,
            "candidates": candidates if candidates is not None else [],
            "assertions": list(self.assertions),
            "position": [self.position[0], self.position[1]],
        }


@dataclass
class SectionResult:
    section_no: int
    title: str
    entities: list[NerEntity] = field(default_factory=list)
