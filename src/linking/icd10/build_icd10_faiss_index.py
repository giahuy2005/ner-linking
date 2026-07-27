#!/usr/bin/env python3
"""Build a normalized SapBERT/FAISS ICD-10 retrieval index.

Input:
    data/processed/icd10/icd10_embedding_terms.jsonl

Default output directory:
    models/icd10/
        icd10_embeddings.npy
        icd10_faiss.index
        icd10_metadata.jsonl
        icd10_index_config.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ..sapbert_encoder import DEFAULT_MODEL_ID, SapBertEncoder
except ImportError:  # Support direct execution by file path.
    from viettel_ai_ner.src.linking.sapbert_encoder import DEFAULT_MODEL_ID, SapBertEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "icd10" / "icd10_embedding_terms.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "icd10"

EMBEDDINGS_FILENAME = "icd10_embeddings.npy"
INDEX_FILENAME = "icd10_faiss.index"
METADATA_FILENAME = "icd10_metadata.jsonl"
CONFIG_FILENAME = "icd10_index_config.json"
REQUIRED_TERM_FIELDS = ("term_id", "code", "text", "language", "term_type")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_embedding_terms(path: Path) -> list[dict[str, str]]:
    terms: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            missing = [field for field in REQUIRED_TERM_FIELDS if field not in row]
            if missing:
                raise ValueError(f"Missing fields {missing} at {path}:{line_number}")

            term = {field: row[field] for field in REQUIRED_TERM_FIELDS}
            if not all(isinstance(term[field], str) for field in REQUIRED_TERM_FIELDS):
                raise ValueError(f"All term fields must be strings at {path}:{line_number}")
            if not term["text"].strip():
                raise ValueError(f"Empty text at {path}:{line_number}")
            if term["term_id"] in seen_ids:
                raise ValueError(f"Duplicate term_id at {path}:{line_number}: {term['term_id']}")
            if term["language"] not in {"en", "vi"}:
                raise ValueError(f"Unsupported language at {path}:{line_number}: {term['language']}")
            if term["term_type"] not in {"preferred", "inclusion", "alias"}:
                raise ValueError(f"Unsupported term_type at {path}:{line_number}: {term['term_type']}")
            seen_ids.add(term["term_id"])
            terms.append(term)

    if not terms:
        raise ValueError(f"No terms found in {path}")
    return terms


def write_metadata(path: Path, terms: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for vector_id, term in enumerate(terms):
            row = {"vector_id": vector_id, **term}
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def temp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def build_index(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS is required. Install faiss-cpu (or faiss-gpu) first.") from exc

    terms = load_embedding_terms(args.input)
    texts = [term["text"] for term in terms]
    print(f"Loaded {len(terms):,} terms from {args.input}")
    print(f"Loading SapBERT: {args.model}")

    encoder = SapBertEncoder(
        args.model,
        revision=args.revision,
        device=args.device,
        max_length=args.max_length,
        pooling=args.pooling,
    )
    embeddings = encoder.encode(
        texts,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
        normalize=True,
    )
    if embeddings.shape != (len(terms), encoder.dimension):
        raise ValueError(
            f"Embedding shape mismatch: expected {(len(terms), encoder.dimension)}, "
            f"got {embeddings.shape}"
        )
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("Embeddings are not L2-normalized")

    index = faiss.IndexFlatIP(encoder.dimension)
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    if index.ntotal != len(terms):
        raise ValueError(f"FAISS ntotal mismatch: {index.ntotal} != {len(terms)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = args.output_dir / EMBEDDINGS_FILENAME
    index_path = args.output_dir / INDEX_FILENAME
    metadata_path = args.output_dir / METADATA_FILENAME
    config_path = args.output_dir / CONFIG_FILENAME
    temporary = {
        "embeddings": temp_path(embeddings_path),
        "index": temp_path(index_path),
        "metadata": temp_path(metadata_path),
        "config": temp_path(config_path),
    }

    try:
        with temporary["embeddings"].open("wb") as handle:
            np.save(handle, embeddings, allow_pickle=False)
        faiss.write_index(index, str(temporary["index"]))
        write_metadata(temporary["metadata"], terms)

        config: dict[str, Any] = {
            "config_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "path": str(args.input.resolve()),
                "sha256": sha256_file(args.input),
            },
            "model": {
                "model_id": args.model,
                "revision": encoder.resolved_revision,
                "pooling": args.pooling,
                "max_length": args.max_length,
                "dimension": encoder.dimension,
            },
            "embedding": {
                "dtype": "float32",
                "l2_normalized": True,
                "count": len(terms),
            },
            "faiss": {
                "index_type": "IndexFlatIP",
                "metric": "inner_product",
                "count": int(index.ntotal),
                "dimension": int(index.d),
            },
            "artifacts": {
                "embeddings": EMBEDDINGS_FILENAME,
                "index": INDEX_FILENAME,
                "metadata": METADATA_FILENAME,
                "sha256": {
                    EMBEDDINGS_FILENAME: sha256_file(temporary["embeddings"]),
                    INDEX_FILENAME: sha256_file(temporary["index"]),
                    METADATA_FILENAME: sha256_file(temporary["metadata"]),
                },
            },
        }
        temporary["config"].write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        os.replace(temporary["embeddings"], embeddings_path)
        os.replace(temporary["index"], index_path)
        os.replace(temporary["metadata"], metadata_path)
        # Config is the commit marker and is replaced last.
        os.replace(temporary["config"], config_path)
    finally:
        for path in temporary.values():
            if path.exists():
                path.unlink()

    print(f"Embeddings: {embeddings_path} {embeddings.shape}")
    print(f"FAISS index: {index_path} (IndexFlatIP, ntotal={index.ntotal:,})")
    print(f"Metadata: {metadata_path}")
    print(f"Config: {config_path}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.max_length <= 0:
        raise SystemExit("--max-length must be positive")
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    build_index(args)


if __name__ == "__main__":
    main()
