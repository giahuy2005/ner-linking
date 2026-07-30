"""Config cho model local dùng để sửa NER và chọn candidate linking.

RÀNG BUỘC VÒNG 1: model self-host tối đa 9B tham số — Qwen2.5-1.5B và
Qwen2.5-7B-Instruct đều nằm trong giới hạn khi chạy TÁCH BIỆT (không
load đồng thời 2 model lên cùng 1 GPU, xem backend.py: load() ->
dùng -> unload() trước khi load model kia).
"""

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


# --- Máy dev (có mạng): model_id = repo Hub, local_files_only=False.
# --- Máy chấm không mạng: đổi model_id="/models/<tên-thư-mục-local>",
#     local_files_only=True (đã tải sẵn bằng download_models.py hoặc
#     huggingface-cli download). KHÔNG cần sửa gì khác ở backend.py.

NER_FIXER_CONFIG = LocalModelConfig(
    # Giữ đúng small-model stage của notebook V11.
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    revision=None,
    cache_dir=None,
    load_in_4bit=False,
    local_files_only=False,
    max_new_tokens=240,
    supports_thinking=False,
    enable_thinking=False,
    batch_size=4,
    temperature=0.0,
    max_context_length=8192,
    retry_rounds=1,
)

NER_REVIEWER_7B_CONFIG = LocalModelConfig(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    revision=None,
    cache_dir=None,
    load_in_4bit=True,
    local_files_only=False,
    max_new_tokens=512,
    supports_thinking=False,
    enable_thinking=False,
    batch_size=4,
    temperature=0.0,
    max_context_length=8192,
    retry_rounds=1,
)

# Compatibility alias: cùng model 7B xử lý NER trước rồi linking rerank.
CANDIDATE_SELECTOR_CONFIG = NER_REVIEWER_7B_CONFIG
