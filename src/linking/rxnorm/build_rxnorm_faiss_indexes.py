#!/usr/bin/env python3
"""Build separate product, support, and historical SapBERT FAISS indexes."""

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
except ImportError:
    from viettel_ai_ner.src.linking.sapbert_encoder import DEFAULT_MODEL_ID, SapBertEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "rxnorm" / "rxnorm_embedding_terms.jsonl"
DEFAULT_CLEAN_INPUT = PROJECT_ROOT / "data" / "processed" / "rxnorm" / "rxnorm_clean.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "rxnorm"
VALID_TIERS = ("product", "support", "historical")
REQUIRED_FIELDS = {"term_id", "rxcui", "text", "term_type", "source_tty", "concept_tty", "index_tier", "output_eligible", "candidate_priority", "active"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_terms(path: Path, tiers: set[str]) -> dict[str, list[dict[str, Any]]]:
    result = {tier: [] for tier in tiers}
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}: {exc}") from exc
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(f"Missing {sorted(missing)} at {path}:{number}")
            tier = row["index_tier"]
            if tier not in VALID_TIERS:
                raise ValueError(f"Invalid index_tier={tier!r} at {path}:{number}")
            if row["term_id"] in seen:
                raise ValueError(f"Duplicate term_id={row['term_id']} at {path}:{number}")
            seen.add(row["term_id"])
            if tier in result:
                result[tier].append(row)
    for tier, rows in result.items():
        if not rows:
            raise ValueError(f"No terms found for requested tier {tier}")
    return result


def metadata_row(vector_id: int, row: dict[str, Any]) -> dict[str, Any]:
    result = {
        "vector_id": vector_id, "term_id": row["term_id"], "rxcui": row["rxcui"],
        "text": row["text"], "term_type": row["term_type"], "source_tty": row["source_tty"],
        "concept_tty": row["concept_tty"], "active": row["active"],
        "output_eligible": row["output_eligible"],
        "candidate_priority": row["candidate_priority"],
    }
    if "current_rxcuis" in row:
        result["current_rxcuis"] = row["current_rxcuis"]
    return result


def load_clean_by_rxcui(
    path: Path, rxcuis: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load structured concept metadata for reranking after FAISS retrieval."""
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}: {exc}") from exc
            rxcui = row.get("rxcui")
            if not rxcui:
                raise ValueError(f"Missing rxcui at {path}:{number}")
            if rxcuis is None or rxcui in rxcuis:
                result[rxcui] = row
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Install faiss-cpu (or faiss-gpu) before building indexes") from exc
    terms_by_tier = load_terms(args.input, args.tiers)
    encoder = SapBertEncoder(args.model, revision=args.revision, device=args.device, max_length=args.max_length, pooling=args.pooling)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_config: dict[str, Any] = {}
    for tier in VALID_TIERS:
        if tier not in terms_by_tier:
            continue
        rows = terms_by_tier[tier]
        print(f"Encoding {tier}: {len(rows):,} terms")
        vectors = encoder.encode([row["embedding_text"] for row in rows], batch_size=args.batch_size, show_progress=not args.no_progress, normalize=True)
        if vectors.shape != (len(rows), encoder.dimension):
            raise ValueError(f"Embedding shape mismatch for {tier}: {vectors.shape}")
        index = faiss.IndexFlatIP(encoder.dimension)
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        stem = f"{tier}_sapbert"
        destinations = (
            args.output_dir / f"{stem}_embeddings.npy",
            args.output_dir / f"{stem}.index",
            args.output_dir / f"{tier}_metadata.jsonl",
        )
        temporary = tuple(Path(str(path) + ".tmp") for path in destinations)
        try:
            with temporary[0].open("wb") as handle:
                np.save(handle, vectors, allow_pickle=False)
            faiss.write_index(index, str(temporary[1]))
            with temporary[2].open("w", encoding="utf-8", newline="\n") as handle:
                for vector_id, row in enumerate(rows):
                    handle.write(json.dumps(metadata_row(vector_id, row), ensure_ascii=False, separators=(",", ":")) + "\n")
            for source, destination in zip(temporary, destinations):
                os.replace(source, destination)
        finally:
            for path in temporary:
                if path.exists():
                    path.unlink()
        if index.ntotal != len(rows):
            raise ValueError(f"FAISS/metadata count mismatch for {tier}")
        artifact_config[tier] = {
            "count": len(rows), "index_file": destinations[1].name,
            "metadata_file": destinations[2].name, "embedding_file": destinations[0].name,
            "sha256": {path.name: sha256_file(path) for path in destinations},
            **({"score_penalty": args.historical_penalty} if tier == "historical" else {}),
        }
    config = {
        "schema_version": "1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(args.input.resolve()), "sha256": sha256_file(args.input)},
        "structured_lookup": {
            "path": str(args.clean_input.resolve()),
            "sha256": sha256_file(args.clean_input),
            "key": "rxcui",
        },
        "model": {"model_id": args.model, "revision": encoder.resolved_revision, "pooling": args.pooling, "max_length": args.max_length, "dimension": encoder.dimension},
        "embedding": {"dtype": "float32", "l2_normalized": True},
        "faiss": {"index_type": "IndexFlatIP", "metric": "inner_product"},
        "candidate_policy": {
            "priority_0": ["SCD", "SBD"],
            "priority_1": ["GPCK", "BPCK"],
            "priority_2": "active support TTYs (IN/PIN/MIN/BN/SCDC/SCDF/...)",
            "priority_3": "historical fallback",
            "rule": "prefer the most specific concept supported by mention evidence; do not blindly promote support hits",
        },
        "indexes": artifact_config,
    }
    config_path = args.output_dir / "rxnorm_index_config.json"
    temporary_config = Path(str(config_path) + ".tmp")
    temporary_config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_config, config_path)
    return config


def parse_tiers(value: str) -> set[str]:
    tiers = {item.strip() for item in value.split(",") if item.strip()}
    invalid = tiers - set(VALID_TIERS)
    if not tiers or invalid:
        raise argparse.ArgumentTypeError(f"tiers must be a subset of {VALID_TIERS}; invalid={sorted(invalid)}")
    return tiers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--clean-input", type=Path, default=DEFAULT_CLEAN_INPUT,
        help="Structured rxnorm_clean.jsonl used for post-retrieval metadata lookup.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tiers", type=parse_tiers, default=set(VALID_TIERS))
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    parser.add_argument("--historical-penalty", type=float, default=0.05)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_length <= 0:
        raise SystemExit("batch size and max length must be positive")
    build(args)


if __name__ == "__main__":
    main()
