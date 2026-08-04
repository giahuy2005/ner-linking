"""Single source of truth for ICD-10 retrieval and selection evidence."""

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

# General, reusable lexical normalizations. These are concept-language bridges,
# not record-specific full-surface rules.
SEMANTIC_PHRASE_VARIANTS: tuple[tuple[str, str], ...] = (
    ("tan huyết", "tan máu"),
    ("tan máu", "tan huyết"),
    ("nhiễm khuẩn", "nhiễm trùng"),
    ("nhiễm trùng", "nhiễm khuẩn"),
    ("trí tuệ", "tâm thần"),
    ("tâm thần", "trí tuệ"),
    ("thiếu men", "thiếu enzyme"),
    ("thiếu enzyme", "thiếu men"),
)

# Tokens that express ICD catalogue wording rather than a clinically
# discriminating qualifier. They may be ignored for lexical coverage, but are
# still retained in labels shown to the selector.
NON_SEMANTIC_TOKENS = frozenset({
    "benh", "hoi", "chung", "do", "cua", "va", "hoac", "kem", "he",
    "vi", "tri", "typ", "type", "nos", "disease", "syndrome", "disorder",
})

# Evidence calibration. These thresholds are deliberately conservative: exact
# and strong evidence may bypass Qwen, medium evidence is only a whitelist
# candidate for ambiguity resolution, and weak evidence never becomes output.
SUPPORT_EXACT_MIN_SCORE = 0.72
SUPPORT_STRONG_MIN_SCORE = 0.76
SUPPORT_MEDIUM_MIN_SCORE = 0.64
DETERMINISTIC_STRONG_MARGIN = 0.055
LEXICAL_RETRIEVAL_LIMIT = 96
SELECTOR_SHORTLIST_SIZE = 10

# General hints are soft ranking evidence only; they never remove candidates.
CHAPTER_HINTS = (
    (("tăng huyết áp", "hạ huyết áp", "mạch vành", "nhồi máu cơ tim"), ("I",)),
    (("ổ loét", "bao tử", "viêm dạ dày", "trào ngược dạ dày"), ("K",)),
    (("đa xơ cứng", "bại não", "parkinson"), ("G",)),
    (("viêm cơ tim", "tràn dịch màng tim", "ngoại tâm thu", "viêm tim"), ("I",)),
    (("trầm cảm", "loạn thần", "nghiện rượu"), ("F",)),
    (("sốt siêu vi", "sốt phát ban", "nhiễm virus"), ("A", "B")),
)