"""Constants cho pipeline linking ICD-10 — nguồn tham số duy nhất.

icd10_linker.py và evaluate.py đều import từ đây, không tự định nghĩa
lại default path/top_k/device ở chỗ khác nữa.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../viettel_ai_ner

DEFAULT_INDEX_DIR = _PROJECT_ROOT / "models" / "icd10"
INDEX_CONFIG_FILENAME = "icd10_index_config.json"

DEFAULT_DEVICE = "auto"
DEFAULT_QUERY_BATCH_SIZE = 32

DEFAULT_TOP_K_TERMS = 10
DEFAULT_TOP_K_CODES = 2
DEFAULT_MIN_SCORE: float | None = None