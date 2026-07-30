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
DEFAULT_MIN_SCORE: float | None = 0.55
MAX_FINAL_CODES = 2

# High-precision Vietnamese aliases are resolved before FAISS.  These entries
# intentionally return one code only and prevent semantically unrelated dense
# neighbors from being appended.
EXACT_ALIAS_CODES = {
    "ăng huyết áp": "I10",
    "tăng huyết áp": "I10",
    "tăng huyết áp nguyên phát": "I10",
    "ổ loét trong bao tử": "K25",
    "loét dạ dày": "K25",
    "viêm bao tử": "K29",
    "viêm dạ dày": "K29",
    "bệnh đa xơ cứng": "G35",
    "đa xơ cứng": "G35",
    "bại não": "G80",
    "hội chứng parkinson": "G20",
    "bệnh parkinson": "G20",
    "thiếu men g6pd": "D55.0",
    "thiếu hụt men g6pd": "D55.0",
    "bệnh kawasaki": "M30.3",
    "trào ngược dạ dày thực quản": "K21",
    "viêm dạ dày ruột do virus": "A08.4",
}

# Conservative chapter constraints for frequent Vietnamese concepts.  If none
# of the retrieved codes belong to the expected chapter, return no candidate.
CHAPTER_HINTS = (
    (("tăng huyết áp", "ăng huyết áp", "hạ huyết áp", "mạch vành", "nhồi máu cơ tim"), ("I",)),
    (("ổ loét", "bao tử", "viêm dạ dày", "trào ngược dạ dày"), ("K",)),
    (("đa xơ cứng", "bại não", "parkinson"), ("G",)),
)
