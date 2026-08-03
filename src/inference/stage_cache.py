"""Atomic, hash-validated per-record stage checkpoints for trusted local runs."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StageCache:
    def __init__(self, root: str | Path, stage: str, config: Any):
        self.directory = Path(root) / stage
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config_hash = stable_hash(config)

    def _path(self, record_id: str) -> Path:
        safe_id = hashlib.sha1(record_id.encode("utf-8")).hexdigest()
        return self.directory / f"{safe_id}.pkl"

    def load(self, record_id: str, raw_text: str):
        path = self._path(record_id)
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                row = pickle.load(handle)
            if (
                row.get("record_id") != record_id
                or row.get("raw_hash") != stable_hash(raw_text)
                or row.get("config_hash") != self.config_hash
            ):
                return None
            return row.get("payload")
        except (OSError, EOFError, pickle.PickleError, AttributeError, TypeError):
            return None

    def put(self, record_id: str, raw_text: str, payload) -> None:
        path = self._path(record_id)
        temporary = path.with_suffix(".pkl.tmp")
        row = {
            "record_id": record_id, "raw_hash": stable_hash(raw_text),
            "config_hash": self.config_hash, "payload": payload,
        }
        with temporary.open("wb") as handle:
            pickle.dump(row, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
        temporary.replace(path)
