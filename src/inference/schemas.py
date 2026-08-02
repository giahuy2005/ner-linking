"""Kiểu dữ liệu và các enum dùng chung trong package inference.

Module này chỉ định nghĩa shape/constant; không tự sửa semantic entity.
Validation fail-fast trước output nằm trong ``io.validate_record_output``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


VALID_ENTITY_TYPES = frozenset({
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
})

VALID_ASSERTIONS = frozenset({
    "isNegated",
    "isFamily",
    "isHistorical",
})

# Chỉ ba loại này được phép mang assertion.
ASSERTION_ENTITY_TYPES = frozenset({
    "TRIỆU_CHỨNG",
    "CHẨN_ĐOÁN",
    "THUỐC",
})

# Chỉ hai loại này được phép có candidate ontology khác rỗng.
LINKING_TYPES = frozenset({
    "CHẨN_ĐOÁN",
    "THUỐC",
})

MAX_CANDIDATES_BY_TYPE = {
    "THUỐC": 1,
    "CHẨN_ĐOÁN": 2,
}


@dataclass
class NerEntity:
    """Một entity NER + assertion trước khi qua linking.

    ``position`` là char offset ``[start, end)`` trên raw text. Các field
    ``flag`` chỉ dùng nội bộ, không xuất ra JSON BTC.
    """

    text: str
    type: str
    assertions: list[str] = field(default_factory=list)
    position: tuple[int, int] = (0, 0)
    score: float = 1.0
    flag: str | None = None

    def to_btc_dict(self, candidates: list[str] | None = None) -> dict[str, Any]:
        """Chuyển sang đủ năm field BTC, không tự sửa text/span/assertion.

        Với entity không phải THUỐC/CHẨN_ĐOÁN, ``candidates`` luôn là ``[]``
        dù caller truyền nhầm code. Validator output vẫn là tầng fail-fast cuối.
        """
        candidate_values = (
            list(candidates or [])
            if self.type in LINKING_TYPES
            else []
        )
        return {
            "text": self.text,
            "type": self.type,
            "candidates": candidate_values,
            "assertions": list(self.assertions),
            "position": [self.position[0], self.position[1]],
        }


@dataclass
class SectionResult:
    section_no: int
    title: str
    entities: list[NerEntity] = field(default_factory=list)
