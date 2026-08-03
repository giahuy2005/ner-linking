"""Configuration for the single Qwen3-8B editor/linking selector."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalModelConfig:
    model_id: str
    revision: str | None
    cache_dir: str | None  # None -> dùng cache mặc định của HF (theo HF_HOME/HF_HUB_CACHE nếu bạn set)
    load_in_4bit: bool
    local_files_only: bool
    max_new_tokens: int
    # Qwen3 hỗ trợ enable_thinking trong apply_chat_template, Qwen2.5-Instruct
    # thì KHÔNG có tham số này (truyền vào sẽ lỗi) -> phải phân biệt rõ.
    supports_thinking: bool = False
    enable_thinking: bool = False
    batch_size: int = 4
    temperature: float = 0.0
    max_context_length: int = 8192
    retry_rounds: int = 1
    dtype: str = "auto"
    attention_implementation: str | None = "sdpa"
    device_map_mode: str = "single_gpu"
    require_full_gpu: bool = False
    max_batch_tokens: int = 16384
    min_batch_size: int = 1
    dynamic_batching: bool = True


QWEN3_8B_EDITOR_CONFIG = LocalModelConfig(
    model_id="Qwen/Qwen3-8B",
    revision=None,
    cache_dir=None,
    load_in_4bit=False,
    local_files_only=False,
    max_new_tokens=1024,
    supports_thinking=True,
    enable_thinking=False,
    batch_size=12,
    temperature=0.0,
    max_context_length=2048,
    retry_rounds=1,
    dtype="bfloat16",
    attention_implementation="sdpa",
    device_map_mode="single_gpu",
    require_full_gpu=False,
    max_batch_tokens=16384,
)
