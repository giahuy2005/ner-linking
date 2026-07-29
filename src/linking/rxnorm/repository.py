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


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _normalize_path_string(path_str: str | Path) -> str:
    """Chuẩn hóa slash và gỡ absolute Windows path cũ.

    Ví dụ:
    -> models/sapbert hoặc /workspace/ner-linking/models/sapbert tùy resolve.
    """
    s = str(path_str).strip().replace("\\", "/")

    project_root_posix = _PROJECT_ROOT.as_posix()

    old_roots = [
        "/workspace/ner-linking",
    ]

    for old_root in old_roots:
        if s.startswith(old_root):
            suffix = s[len(old_root):].lstrip("/")
            return f"{project_root_posix}/{suffix}" if suffix else project_root_posix

    # Trường hợp path Windows khác nhưng vẫn có tên project trong path
    marker_candidates = [
        "/viettel_ai_ner/",
        "/ner-linking/",
    ]
    for marker in marker_candidates:
        if marker in s:
            suffix = s.split(marker, 1)[1]
            return f"{project_root_posix}/{suffix}" if suffix else project_root_posix

    return s


def _looks_like_windows_abs(path_str: str) -> bool:
    return len(path_str) >= 3 and path_str[1] == ":" and path_str[2] == "/"


def resolve_project_path(
    path_str: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> str:

    s = _normalize_path_string(path_str)

    if not s:
        raise ValueError("Path rỗng")

    # Linux absolute path
    if s.startswith("/"):
        return Path(s).as_posix()

    # Windows absolute path còn sót nhưng không map được project root
    if _looks_like_windows_abs(s):
        # Cố gắng bỏ phần drive, nhưng báo rõ nếu sau đó không tồn tại.
        # Thực tế nên tránh case này bằng cách lưu relative path trong JSON.
        p = Path(s)
        return p.as_posix()

    rel = Path(s)

    candidates: list[Path] = []

    # Path dạng project-relative
    if s.startswith(("models/", "data/", "src/", "configs/")):
        candidates.append(_PROJECT_ROOT / rel)
    else:
        # Path dạng file name nằm trong index_dir, ví dụ product_sapbert.index
        if base_dir is not None:
            candidates.append(Path(base_dir) / rel)

        # Fallback: relative theo project root
        candidates.append(_PROJECT_ROOT / rel)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve().as_posix()

    # Nếu chưa tồn tại thì trả candidate đầu tiên để error message dễ hiểu
    return candidates[0].as_posix()


def _entity_name(entity: Any) -> str | None:
    """clean record lưu ingredient/dose_form/brand dạng {'rxcui','tty','name'}
    hoặc đôi khi string trần — hàm này lấy tên ra thống nhất.
    """

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
        self.index_dir = Path(resolve_project_path(index_dir, base_dir=_PROJECT_ROOT))

        if clean_path is not None:
            self.clean_path = Path(resolve_project_path(clean_path, base_dir=_PROJECT_ROOT))
        else:
            default_clean_path = getattr(config, "DEFAULT_CLEAN_PATH", None)
            if default_clean_path is not None:
                self.clean_path = Path(resolve_project_path(default_clean_path, base_dir=_PROJECT_ROOT))
            else:
                self.clean_path = self.index_dir / "rxnorm_clean.jsonl"

        self.config = self._load_config()
        self._validate_config()
        self._resolve_paths()

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

    def _resolve_paths(self) -> None:
        """Resolve toàn bộ path trong rxnorm_index_config.json.

        Sau hàm này:
        - config["model"]["model_id"] là path POSIX nếu là local model.
        - index_file / metadata_file / embedding_file là path POSIX tuyệt đối.
        """

        model_id = self.config["model"]["model_id"]
        model_id_str = str(model_id).replace("\\", "/")

        # Nếu là local path thì resolve. Nếu là HF repo id như "cambridgeltl/..."
        # thì giữ nguyên.
        if (
            model_id_str.startswith(("models/", "data/", ".", "/", "~"))
            or "Viettel_AI" in model_id_str
            or "viettel_ai_ner" in model_id_str
            or "ner-linking" in model_id_str
            or _looks_like_windows_abs(model_id_str)
        ):
            self.config["model"]["model_id"] = resolve_project_path(
                model_id_str,
                base_dir=_PROJECT_ROOT,
            )

        for tier in config.VALID_TIERS:
            info = self.config["indexes"][tier]

            for field_name in ("index_file", "metadata_file", "embedding_file"):
                info[field_name] = resolve_project_path(
                    info[field_name],
                    base_dir=self.index_dir,
                )

    # ----------------------------------------------------------------
    # Index & metadata
    # ----------------------------------------------------------------

    def _load_indexes(self) -> dict[str, faiss.Index]:
        indexes: dict[str, faiss.Index] = {}

        for tier in config.VALID_TIERS:
            info = self.config["indexes"][tier]
            index_path = Path(info["index_file"])

            if not index_path.is_file():
                raise FileNotFoundError(f"Thiếu index file cho tier '{tier}': {index_path}")

            indexes[tier] = faiss.read_index(index_path.as_posix())

        return indexes

    def _load_metadata(self) -> dict[str, list[dict[str, Any] | None]]:
        metadata: dict[str, list[dict[str, Any] | None]] = {}

        for tier in config.VALID_TIERS:
            info = self.config["indexes"][tier]
            metadata_path = Path(info["metadata_file"])

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
        """Dẹp phẳng 1 record rxnorm_clean.jsonl.

        Schema thật:
        - strength nằm trong clinical_components[].strength.display
        - dose_forms là list object {'rxcui','tty','name'}

        Output phẳng để reranker.py dùng qua:
        candidate.structured.get('strengths'/'dose_forms'/...)
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