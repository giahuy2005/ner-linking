#!/usr/bin/env python3
"""Query the SapBERT/FAISS ICD-10 index and aggregate term hits by code."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

try:
    from ..sapbert_encoder import SapBertEncoder, clean_query_text
except ImportError:  # Support direct execution by file path.
    from viettel_ai_ner.src.linking.sapbert_encoder import SapBertEncoder, clean_query_text

try:
    from . import config
except ImportError:  # Support direct execution by file path.
    from viettel_ai_ner.src.linking.icd10 import config


def aggregate_term_results(
    term_results: Iterable[dict[str, Any]],
    *,
    top_k_codes: int,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """Keep the maximum-scoring matched term for each ICD code."""
    if top_k_codes <= 0:
        raise ValueError("top_k_codes must be positive")

    by_code: dict[str, dict[str, Any]] = {}
    for result in term_results:
        score = float(result["score"])
        if min_score is not None and score < min_score:
            continue
        code = result["code"]
        current = by_code.get(code)
        if current is None or score > current["score"]:
            by_code[code] = {
                "code": code,
                "score": score,
                "matched_term": result["text"],
                "language": result["language"],
                "term_type": result["term_type"],
                "term_id": result["term_id"],
            }

    ranked = sorted(by_code.values(), key=lambda item: (-item["score"], item["code"]))
    return ranked[:top_k_codes]


def _normalize_alias(text: str) -> str:
    value = unicodedata.normalize("NFC", clean_query_text(text)).casefold()
    return re.sub(r"\s+", " ", value).strip(" \t\r\n.,;:()[]{}")


def _build_metadata_alias_codes(metadata: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Build unambiguous Vietnamese exact aliases from the index itself.

    This replaces private-test surface rules for ordinary ICD labels.  A
    generated alias is kept only when every matching metadata row points to
    one code.  For preferred labels we also expose a version without a generic
    leading ``bệnh``/``hội chứng`` when that shortened form remains unique.
    """
    aliases: dict[str, set[str]] = {}
    for row in metadata:
        if str(row.get("language", "")).casefold() != "vi":
            continue
        normalized = _normalize_alias(str(row.get("text", "")))
        if not normalized:
            continue
        forms = {normalized}
        if row.get("term_type") == "preferred":
            shortened = re.sub(r"^(?:bệnh|hội chứng)\s+", "", normalized).strip()
            if len(shortened) >= 3:
                forms.add(shortened)
        for form in forms:
            aliases.setdefault(form, set()).add(str(row["code"]))
    return {
        alias: next(iter(codes))
        for alias, codes in aliases.items()
        if len(codes) == 1
    }


def _exact_alias_result(
    mention: str,
    metadata_alias_codes: dict[str, str] | None = None,
) -> list[dict[str, Any]] | None:
    normalized = _normalize_alias(mention)
    code = config.EXACT_ALIAS_CODES.get(normalized)
    source = "configured_alias"
    if code is None and metadata_alias_codes is not None:
        code = metadata_alias_codes.get(normalized)
        source = "metadata_exact_alias"
    if code is None:
        return None
    return [{
        "code": code,
        "score": 1.0,
        "matched_term": mention,
        "language": "vi",
        "term_type": source,
        "term_id": f"{source}:{normalized}",
    }]


def _expected_chapters(mention: str) -> tuple[str, ...] | None:
    normalized = _normalize_alias(mention)
    for phrases, chapters in config.CHAPTER_HINTS:
        if any(phrase in normalized for phrase in phrases):
            return tuple(chapters)
    return None


def _finalize_term_results(
    mention: str,
    term_results: list[dict[str, Any]],
    *,
    top_k_codes: int,
    min_score: float | None,
) -> list[dict[str, Any]]:
    ranked = aggregate_term_results(
        term_results,
        top_k_codes=max(top_k_codes, 10),
        min_score=min_score,
    )
    chapters = _expected_chapters(mention)
    if chapters is not None:
        ranked = [item for item in ranked if item["code"].startswith(chapters)]
    return ranked[:min(top_k_codes, config.MAX_FINAL_CODES)]


class Icd10Linker:
    """SapBERT candidate retriever with max-score aggregation per ICD code."""

    def __init__(
        self,
        index_dir: str | Path = config.DEFAULT_INDEX_DIR,
        *,
        device: str = config.DEFAULT_DEVICE,
        query_batch_size: int = config.DEFAULT_QUERY_BATCH_SIZE,
    ) -> None:
        if query_batch_size <= 0:
            raise ValueError("query_batch_size must be positive")
        self.index_dir = Path(index_dir).resolve()
        self.query_batch_size = query_batch_size
        config_path = self.index_dir / config.INDEX_CONFIG_FILENAME
        if not config_path.exists():
            raise FileNotFoundError(
                f"Index config not found: {config_path}. Run build_icd10_faiss_index.py first."
            )
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self._validate_config()

        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("FAISS is required. Install faiss-cpu (or faiss-gpu) first.") from exc

        artifacts = self.config["artifacts"]
        index_path = self._artifact_path(artifacts["index"])
        metadata_path = self._artifact_path(artifacts["metadata"])
        # The .npy file is not loaded during queries, but its presence confirms
        # that the complete requested artifact set was built.
        embeddings_path = self._artifact_path(artifacts["embeddings"])
        for path in (index_path, metadata_path, embeddings_path):
            if not path.exists():
                raise FileNotFoundError(f"Index artifact not found: {path}")

        self.index = faiss.read_index(str(index_path))
        self.metadata = self._load_metadata(metadata_path)
        self.metadata_alias_codes = _build_metadata_alias_codes(self.metadata)
        expected_count = int(self.config["embedding"]["count"])
        expected_dimension = int(self.config["model"]["dimension"])
        if self.index.ntotal != expected_count or len(self.metadata) != expected_count:
            raise ValueError(
                "Artifact count mismatch: "
                f"index={self.index.ntotal}, metadata={len(self.metadata)}, config={expected_count}"
            )
        if self.index.d != expected_dimension:
            raise ValueError(
                f"Artifact dimension mismatch: index={self.index.d}, config={expected_dimension}"
            )

        model_config = self.config["model"]
        self.encoder = SapBertEncoder(
            model_config["model_id"],
            revision=model_config.get("revision"),
            device=device,
            max_length=int(model_config["max_length"]),
            pooling=model_config["pooling"],
        )
        if self.encoder.dimension != self.index.d:
            raise ValueError(
                f"Query encoder dimension {self.encoder.dimension} != index dimension {self.index.d}"
            )

    def _artifact_path(self, filename: str) -> Path:
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"Unsafe artifact filename in config: {filename!r}")
        return self.index_dir / filename

    def _validate_config(self) -> None:
        if self.config.get("config_version") != 1:
            raise ValueError(f"Unsupported config_version: {self.config.get('config_version')}")
        if self.config.get("faiss", {}).get("index_type") != "IndexFlatIP":
            raise ValueError("Only IndexFlatIP configs are supported")
        if self.config.get("faiss", {}).get("metric") != "inner_product":
            raise ValueError("Expected inner_product FAISS metric")
        if self.config.get("embedding", {}).get("l2_normalized") is not True:
            raise ValueError("Expected L2-normalized index embeddings")
        required_artifacts = {"embeddings", "index", "metadata"}
        if not required_artifacts.issubset(self.config.get("artifacts", {})):
            raise ValueError("Config is missing one or more artifact filenames")

    @staticmethod
    def _load_metadata(path: Path) -> list[dict[str, Any]]:
        metadata: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid metadata JSON at {path}:{line_number}: {exc}") from exc
                vector_id = len(metadata)
                if row.get("vector_id") != vector_id:
                    raise ValueError(
                        f"metadata vector_id mismatch at line {line_number}: "
                        f"{row.get('vector_id')} != {vector_id}"
                    )
                for field in ("term_id", "code", "text", "language", "term_type"):
                    if not isinstance(row.get(field), str):
                        raise ValueError(f"Invalid metadata field {field} at line {line_number}")
                metadata.append(row)
        return metadata

    def _term_results_from_search(self, scores: Any, indices: Any) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for score, vector_id in zip(scores, indices):
            vector_id = int(vector_id)
            if vector_id < 0:
                continue
            metadata = self.metadata[vector_id]
            results.append({"score": float(score), **metadata})
        return results

    def search_terms(
        self, mention: str, *, top_k_terms: int = config.DEFAULT_TOP_K_TERMS
    ) -> list[dict[str, Any]]:
        """Return raw nearest surface forms before code aggregation."""
        if top_k_terms <= 0:
            raise ValueError("top_k_terms must be positive")
        mention = clean_query_text(mention)
        query = self.encoder.encode(
            [mention], batch_size=1, show_progress=False, normalize=True
        )
        count = min(top_k_terms, int(self.index.ntotal))
        scores, indices = self.index.search(query, count)
        return self._term_results_from_search(scores[0], indices[0])

    def link(
        self,
        mention: str,
        *,
        top_k_terms: int = config.DEFAULT_TOP_K_TERMS,
        top_k_codes: int = config.DEFAULT_TOP_K_CODES,
        min_score: float | None = config.DEFAULT_MIN_SCORE,
    ) -> list[dict[str, Any]]:
        """Link one NER mention and return ICD codes ranked by maximum term score."""
        exact = _exact_alias_result(mention, self.metadata_alias_codes)
        if exact is not None:
            return exact
        term_results = self.search_terms(mention, top_k_terms=top_k_terms)
        return _finalize_term_results(
            mention,
            term_results,
            top_k_codes=top_k_codes,
            min_score=min_score,
        )

    def link_many(
        self,
        mentions: Iterable[str],
        *,
        top_k_terms: int = config.DEFAULT_TOP_K_TERMS,
        top_k_codes: int = config.DEFAULT_TOP_K_CODES,
        min_score: float | None = config.DEFAULT_MIN_SCORE,
    ) -> list[list[dict[str, Any]]]:
        """Batch-link multiple NER mentions with a single encoder pass."""
        if top_k_terms <= 0 or top_k_codes <= 0:
            raise ValueError("top_k_terms and top_k_codes must be positive")
        cleaned = [clean_query_text(mention) for mention in mentions]
        if not cleaned:
            return []
        queries = self.encoder.encode(
            cleaned,
            batch_size=self.query_batch_size,
            show_progress=False,
            normalize=True,
        )
        count = min(top_k_terms, int(self.index.ntotal))
        scores, indices = self.index.search(queries, count)
        output = []
        for mention, row_scores, row_indices in zip(cleaned, scores, indices):
            exact = _exact_alias_result(mention, self.metadata_alias_codes)
            if exact is not None:
                output.append(exact)
                continue
            term_results = self._term_results_from_search(row_scores, row_indices)
            output.append(
                _finalize_term_results(
                    mention,
                    term_results,
                    top_k_codes=top_k_codes,
                    min_score=min_score,
                )
            )
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mention", help="NER mention to link")
    parser.add_argument("--index-dir", type=Path, default=config.DEFAULT_INDEX_DIR)
    parser.add_argument("--device", default=config.DEFAULT_DEVICE)
    parser.add_argument("--top-k-terms", type=int, default=config.DEFAULT_TOP_K_TERMS)
    parser.add_argument("--top-k-codes", type=int, default=config.DEFAULT_TOP_K_CODES)
    parser.add_argument("--min-score", type=float, default=config.DEFAULT_MIN_SCORE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    linker = Icd10Linker(args.index_dir, device=args.device)
    results = linker.link(
        args.mention,
        top_k_terms=args.top_k_terms,
        top_k_codes=args.top_k_codes,
        min_score=args.min_score,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
