"""Load dữ liệu RxNorm (FAISS index, metadata, clean records).

Không làm retrieval, không chấm điểm. Khi lỗi count mismatch / vector_id
lệch / config sai / thiếu file -> chỉ sửa ở đây.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss

from . import config
from .parser import normalize_text


def _entity_name(entity: Any) -> str | None:
    """clean record lưu ingredient/dose_form/brand dạng {'rxcui','tty','name'}
    hoặc đôi khi string trần — hàm này lấy tên ra thống nhất."""

    if isinstance(entity, dict):
        return entity.get("name")
    if isinstance(entity, str):
        return entity
    return None


def _add_unique(bucket: list[str], value: str | None) -> None:
    if not value:
        return

    value = str(value).strip()
    if value and value not in bucket:
        bucket.append(value)


class RxNormRepository:
    def __init__(self, index_dir: str | Path, clean_path: str | Path | None = None):
        self.index_dir = Path(index_dir)
        self.clean_path = Path(clean_path) if clean_path else self.index_dir / "rxnorm_clean.jsonl"

        self.config = self._load_config()
        self._validate_config()

        self.indexes: dict[str, faiss.Index] = self._load_indexes()
        self.metadata: dict[str, list[dict[str, Any] | None]] = self._load_metadata()
        self.clean_by_rxcui: dict[str, dict[str, Any]] = self._load_clean_lookup()

        self.exact_term_lookup: dict[str, list[tuple[str, int]]] = {}
        self.core_lookup: dict[str, list[str]] = {}
        self._build_exact_lookups()

    # ----------------------------------------------------------------
    # Config
    # ----------------------------------------------------------------

    def _load_config(self) -> dict[str, Any]:
        config_path = self.index_dir / "rxnorm_index_config.json"

        if not config_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy {config_path}")

        return json.loads(config_path.read_text(encoding="utf-8"))

    def _validate_config(self) -> None:
        for key in ("model", "indexes"):
            if key not in self.config:
                raise ValueError(f"rxnorm_index_config.json thiếu key '{key}'")

        for tier in config.VALID_TIERS:
            if tier not in self.config["indexes"]:
                raise ValueError(f"rxnorm_index_config.json thiếu tier '{tier}'")

            info = self.config["indexes"][tier]
            for field_name in ("index_file", "metadata_file", "embedding_file"):
                if field_name not in info:
                    raise ValueError(f"Tier '{tier}' thiếu '{field_name}'")

        for field_name in ("model_id", "pooling", "dimension", "max_length"):
            if field_name not in self.config["model"]:
                raise ValueError(f"config['model'] thiếu '{field_name}'")

    # ----------------------------------------------------------------
    # Index & metadata
    # ----------------------------------------------------------------

    def _load_indexes(self) -> dict[str, faiss.Index]:
        indexes: dict[str, faiss.Index] = {}

        for tier in config.VALID_TIERS:
            info = self.config["indexes"][tier]
            index_path = self.index_dir / info["index_file"]

            if not index_path.is_file():
                raise FileNotFoundError(f"Thiếu index file cho tier '{tier}': {index_path}")

            indexes[tier] = faiss.read_index(str(index_path))

        return indexes

    def _load_metadata(self) -> dict[str, list[dict[str, Any] | None]]:
        metadata: dict[str, list[dict[str, Any] | None]] = {}

        for tier in config.VALID_TIERS:
            info = self.config["indexes"][tier]
            metadata_path = self.index_dir / info["metadata_file"]

            if not metadata_path.is_file():
                raise FileNotFoundError(f"Thiếu metadata file cho tier '{tier}': {metadata_path}")

            rows = self._load_compact_metadata(metadata_path)

            if self.indexes[tier].ntotal != len(rows):
                raise ValueError(
                    f"Count mismatch tier '{tier}': "
                    f"FAISS ntotal={self.indexes[tier].ntotal} != metadata={len(rows)}"
                )

            metadata[tier] = rows

        return metadata

    @staticmethod
    def _load_compact_metadata(path: Path) -> list[dict[str, Any] | None]:
        rows: list[dict[str, Any] | None] = []

        with path.open(encoding="utf-8") as file:
            for expected_id, line in enumerate(file):
                row = json.loads(line)

                actual_id = row.get("vector_id")
                if actual_id != expected_id:
                    raise ValueError(
                        f"Metadata lệch tại dòng {expected_id}: vector_id={actual_id}"
                    )

                tty = row["concept_tty"]

                if tty not in config.ALLOWED_OUTPUT_TTYS:
                    rows.append(None)
                    continue

                rows.append(
                    {
                        "vector_id": expected_id,
                        "term_id": row["term_id"],
                        "rxcui": str(row["rxcui"]),
                        "text": row["text"],
                        "term_type": row["term_type"],
                        "source_tty": row["source_tty"],
                        "concept_tty": tty,
                        "active": bool(row["active"]),
                        "candidate_priority": row.get("candidate_priority", 99),
                        "current_rxcuis": row.get("current_rxcuis", []),
                    }
                )

        return rows

    # ----------------------------------------------------------------
    # Clean records (structured: ingredients/strengths/dose_forms/brands)
    # ----------------------------------------------------------------

    def _load_clean_lookup(self) -> dict[str, dict[str, Any]]:
        lookup: dict[str, dict[str, Any]] = {}

        if not self.clean_path.is_file():
            print(
                f"[repository.py] CẢNH BÁO: không tìm thấy {self.clean_path} — "
                f"clean_by_rxcui sẽ rỗng, reranker sẽ mất hết structured "
                f"strength/dose_form/release_type để so khớp. Truyền đúng "
                f"clean_path khi khởi tạo RxNormRepository/RxNormLinker."
            )
            return lookup

        with self.clean_path.open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                rxcui = str(row.get("rxcui"))

                if rxcui:
                    lookup[rxcui] = self._compact_clean_record(row)

        if not lookup:
            print(
                f"[repository.py] CẢNH BÁO: {self.clean_path} tồn tại nhưng "
                f"không đọc được rxcui nào — kiểm tra lại tên field 'rxcui' "
                f"trong file có khớp không."
            )

        return lookup

    @staticmethod
    def _compact_clean_record(row: dict[str, Any]) -> dict[str, Any]:
        """Dẹp phẳng 1 record rxnorm_clean.jsonl (schema thật: strength nằm
        trong clinical_components[].strength.display, dose_forms là list
        object {'rxcui','tty','name'}) thành dict phẳng để reranker.py dùng
        trực tiếp qua candidate.structured.get('strengths'/'dose_forms'/...).
        """

        ingredients: list[str] = []
        precise_ingredients: list[str] = []
        strengths: list[str] = []
        dose_forms: list[str] = []
        release_types: list[str] = []
        brands: list[str] = []

        for component in row.get("clinical_components") or []:
            _add_unique(ingredients, _entity_name(component.get("ingredient")))
            _add_unique(precise_ingredients, _entity_name(component.get("precise_ingredient")))

            strength = component.get("strength")
            if isinstance(strength, dict) and strength.get("display"):
                _add_unique(strengths, strength["display"])
            elif isinstance(strength, str):
                _add_unique(strengths, strength)

        for dose_form in row.get("dose_forms") or []:
            name = _entity_name(dose_form)
            if not name:
                continue

            _add_unique(dose_forms, name)

            normalized_name = normalize_text(name)
            for phrase in config.RELEASE_TYPE_TERMS:
                if phrase in normalized_name:
                    _add_unique(release_types, phrase)

        _add_unique(brands, _entity_name(row.get("brand")))

        return {
            "rxcui": str(row.get("rxcui")),
            "name": row.get("canonical_name"),
            "ingredients": ingredients,
            "precise_ingredients": precise_ingredients,
            "strengths": strengths,
            "dose_forms": dose_forms,
            "release_types": release_types,
            "brands": brands,
        }

    def get_structured_record(self, rxcui: str) -> dict[str, Any]:
        return self.clean_by_rxcui.get(str(rxcui), {})

    # ----------------------------------------------------------------
    # Exact-match lookup (term -> (tier, vector_id), core ingredient -> rxcui)
    # ----------------------------------------------------------------

    def _build_exact_lookups(self) -> None:
        for tier, rows in self.metadata.items():
            for row in rows:
                if row is None:
                    continue

                term_key = normalize_text(row["text"])
                self.exact_term_lookup.setdefault(term_key, []).append((tier, row["vector_id"]))

        for rxcui, record in self.clean_by_rxcui.items():
            for ingredient in record.get("ingredients", []):
                name = ingredient.get("name") if isinstance(ingredient, dict) else ingredient
                if not name:
                    continue

                key = normalize_text(str(name))
                self.core_lookup.setdefault(key, []).append(rxcui)