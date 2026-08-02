"""Single source of truth for ICD-10 retrieval configuration."""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INDEX_DIR = _PROJECT_ROOT / "models" / "icd10"
INDEX_CONFIG_FILENAME = "icd10_index_config.json"

DEFAULT_DEVICE = "auto"
DEFAULT_QUERY_BATCH_SIZE = 32
DEFAULT_TOP_K_TERMS = 50
DEFAULT_TOP_K_CODES = 10
DEFAULT_MIN_SCORE: float | None = 0.55
MAX_FINAL_CODES = 2

# The former long hand-written surface table mixed general concepts with
# observed long mentions. Exact aliases are now derived from index metadata and
# retained only when every metadata occurrence resolves to one code.
EXACT_ALIAS_CODES: dict[str, str] = {}

# General hints are soft ranking evidence only; they never remove candidates.
CHAPTER_HINTS = (
    (("tăng huyết áp", "hạ huyết áp", "mạch vành", "nhồi máu cơ tim"), ("I",)),
    (("ổ loét", "bao tử", "viêm dạ dày", "trào ngược dạ dày"), ("K",)),
    (("đa xơ cứng", "bại não", "parkinson"), ("G",)),
    (("viêm cơ tim", "tràn dịch màng tim", "ngoại tâm thu", "viêm tim"), ("I",)),
    (("trầm cảm", "loạn thần", "nghiện rượu"), ("F",)),
    (("sốt siêu vi", "sốt phát ban", "nhiễm virus"), ("A", "B")),
)
