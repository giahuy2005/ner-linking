"""Portable detailed-NER artifacts used to benchmark only the LLM path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .ner.evidence import NerDetailedResult

SCHEMA_VERSION = "detailed_ner_artifact_v1"


def _raw_hash(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def save_artifact(directory: str | Path, record_id: str, raw_text: str, detail: NerDetailedResult) -> Path:
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    path = root / f"{record_id}.detailed.json"
    temporary = path.with_suffix(".json.tmp")
    payload = {
        "schema_version": SCHEMA_VERSION, "record_id": record_id,
        "raw_sha256": _raw_hash(raw_text), "detail": detail.to_audit_dict(),
    }
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)
    return path


def load_artifacts(
    directory: str | Path, raw_texts_by_id: dict[str, str],
) -> dict[str, NerDetailedResult]:
    root = Path(directory)
    outputs = {}
    for record_id, raw_text in raw_texts_by_id.items():
        path = root / f"{record_id}.detailed.json"
        if not path.exists():
            raise FileNotFoundError(f"missing detailed-NER artifact: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"stale detailed-NER schema: {path}")
        if payload.get("record_id") != record_id or payload.get("raw_sha256") != _raw_hash(raw_text):
            raise ValueError(f"detailed-NER artifact/raw mismatch: {path}")
        detail = NerDetailedResult.from_audit_dict(payload["detail"])
        detail.validate_offsets(raw_text)
        outputs[record_id] = detail
    return outputs
