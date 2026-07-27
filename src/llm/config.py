"""Config cho 2 model LLM local: sửa span NER + chọn candidate linking.

RÀNG BUỘC VÒNG 1: model self-host tối đa 9B tham số — Qwen3-1.7B và
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


# --- Máy dev (có mạng): model_id = repo Hub, local_files_only=False.
# --- Máy chấm không mạng: đổi model_id="/models/<tên-thư-mục-local>",
#     local_files_only=True (đã tải sẵn bằng download_models.py hoặc
#     huggingface-cli download). KHÔNG cần sửa gì khác ở backend.py.

NER_FIXER_CONFIG = LocalModelConfig(
    model_id="Qwen/Qwen3-1.7B",
    revision=None,
    cache_dir=None,
    load_in_4bit=True,
    local_files_only=False,
    max_new_tokens=256,
    supports_thinking=True,
    enable_thinking=False,  # tắt thinking: chỉ cần chọn boundary/type, không cần suy luận dài
)

CANDIDATE_SELECTOR_CONFIG = LocalModelConfig(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    revision=None,
    cache_dir=None,
    load_in_4bit=True,
    local_files_only=False,
    max_new_tokens=256,
    supports_thinking=False,
    enable_thinking=False,
)