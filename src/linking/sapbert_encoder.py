"""Shared SapBERT encoder used by index building and online linking."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Any
from pathlib import Path
import numpy as np


DEFAULT_MODEL_ID = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
SUPPORTED_POOLING = {"cls", "mean"}


def clean_query_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    value = " ".join(unicodedata.normalize("NFC", text).split())
    if not value:
        raise ValueError("text must not be empty")
    return value


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return contiguous float32 rows normalized to unit L2 length."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape={matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("Embedding matrix contains a zero or non-finite vector")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def resolve_device(torch_module: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class SapBertEncoder:
    """Thin, deterministic Hugging Face encoder for SapBERT surface forms."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        revision: str | None = None,
        device: str = "auto",
        max_length: int = 64,
        pooling: str = "cls",
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        if pooling not in SUPPORTED_POOLING:
            raise ValueError(
                f"pooling must be one of {sorted(SUPPORTED_POOLING)}"
            )

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "SapBERT requires torch and transformers. "
                "Install project requirements first."
            ) from exc

        self.torch = torch
        self.device = resolve_device(torch, device)
        self.max_length = max_length
        self.pooling = pooling

        model_path = Path(model_id).expanduser()
        is_local_model = model_path.exists()

        if is_local_model:
            model_source = str(model_path.resolve())
            load_kwargs = {
                "local_files_only": True,
            }

            self.requested_revision = None
            print(f"Loading SapBERT from local: {model_source}")

        else:
            model_source = model_id
            load_kwargs = {}

            if revision:
                load_kwargs["revision"] = revision

            self.requested_revision = revision
            print(f"Loading SapBERT from Hugging Face: {model_source}")

        self.model_id = model_source

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            **load_kwargs,
        )

        self.model = AutoModel.from_pretrained(
            model_source,
            **load_kwargs,
        )

        self.model.eval()
        self.model.to(self.device)

        hidden_size = getattr(
            self.model.config,
            "hidden_size",
            None,
        )

        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(
                "Loaded model does not expose a valid "
                "config.hidden_size"
            )

        self.dimension = hidden_size

        self.resolved_revision = (
            getattr(self.model.config, "_commit_hash", None)
            or self.requested_revision
        )

    def _pool(self, hidden_state: Any, attention_mask: Any) -> Any:
        if self.pooling == "cls":
            return hidden_state[:, 0, :]
        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        summed = (hidden_state * mask).sum(dim=1)
        return summed / mask.sum(dim=1).clamp(min=1.0)

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        cleaned = [clean_query_text(text) for text in texts]
        if not cleaned:
            return np.empty((0, self.dimension), dtype=np.float32)

        starts: Any = range(0, len(cleaned), batch_size)
        if show_progress:
            try:
                from tqdm.auto import tqdm

                starts = tqdm(starts, total=(len(cleaned) + batch_size - 1) // batch_size)
            except ImportError:
                pass

        batches: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in starts:
                tokens = self.tokenizer(
                    cleaned[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {name: tensor.to(self.device) for name, tensor in tokens.items()}
                output = self.model(**tokens)
                pooled = self._pool(output.last_hidden_state, tokens["attention_mask"])
                batches.append(pooled.detach().cpu().to(self.torch.float32).numpy())

        vectors = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
        return l2_normalize(vectors) if normalize else np.ascontiguousarray(vectors)
