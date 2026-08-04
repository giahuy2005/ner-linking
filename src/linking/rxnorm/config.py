"""Constants và policy cho pipeline linking RxNorm.

Không đặt logic ở đây. Muốn tune weight / thêm alias / thêm dose form
thì chỉ sửa file này, không đụng tới parser/retriever/reranker.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Đường dẫn mặc định tới model/index đã build (models/rxnorm/...)
# Sửa 2 dòng này khi đổi vị trí output của build_rxnorm_faiss_indexes.py
# --------------------------------------------------------------------------

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../viettel_ai_ner

DEFAULT_INDEX_DIR = _PROJECT_ROOT / "models" / "rxnorm"
DEFAULT_CLEAN_PATH = _PROJECT_ROOT / "data" / "processed" / "rxnorm" / "rxnorm_clean.jsonl"

# --------------------------------------------------------------------------
# Tier & TTY
# --------------------------------------------------------------------------

VALID_TIERS = (
    "product",
    "support",
    "historical",
)

PRODUCT_TTYS = {
    "SCD",
    "SBD",
    "GPCK",
    "BPCK",
}

SUPPORT_TTYS = {
    "SCDC",
    "SCDF",
    "IN",
    "PIN",
    "MIN",
    "BN",
}

ALLOWED_OUTPUT_TTYS = PRODUCT_TTYS | SUPPORT_TTYS

# --------------------------------------------------------------------------
# Retrieval K mặc định (candidate generation, ưu tiên recall)
# --------------------------------------------------------------------------

DEFAULT_PRODUCT_K = 300
DEFAULT_SUPPORT_K = 300
DEFAULT_HISTORICAL_K = 200

# --------------------------------------------------------------------------
# Trọng số score cuối (dense + lexical + rule bonus)
# --------------------------------------------------------------------------

DENSE_WEIGHT = 0.28
LEXICAL_WEIGHT = 0.12
INGREDIENT_EXACT_BONUS = 0.30
DETERMINISTIC_MIN_SCORE = 0.58
DETERMINISTIC_MIN_MARGIN = 0.035
AMBIGUITY_MARGIN = 0.08

# --------------------------------------------------------------------------
# Chuẩn hoá text: alias thuốc, cụm từ Việt -> Anh, viết tắt
# --------------------------------------------------------------------------

DRUG_ALIAS_MAP = {
    "paracetamol": "acetaminophen",
    "natri clorid": "sodium chloride",
    "kali clorid": "potassium chloride",
    "adrenalin": "epinephrine",
    "acid acetylsalicylic": "aspirin",
    "senna": "sennosides",
    "trimetazidin": "trimetazidine",
    "levafloxacin": "levofloxacin",
    "nhôm hydroxid": "aluminum hydroxide",
    "nhom hydroxid": "aluminum hydroxide",
    "magie hydroxid": "magnesium hydroxide",
    "simethicon": "simethicone",
    "alverin": "alverine",
    "cotrimoxazol": "sulfamethoxazole / trimethoprim",
    "cotrimoxazole": "sulfamethoxazole / trimethoprim",
    "co-trimoxazole": "sulfamethoxazole / trimethoprim",
    "doxycyclin": "doxycycline",
}

# value là cụm dose-form chuẩn (không phải release-type đơn thuần)
PHRASE_MAP = {
    "viên nén phóng thích kéo dài": "extended release oral tablet",
    "viên nén giải phóng kéo dài": "extended release oral tablet",
    "viên nén phóng thích chậm": "delayed release oral tablet",
    "viên bao tan trong ruột": "delayed release oral tablet",
    "viên nén": "oral tablet",
    "viên nang": "oral capsule",
    "viên con nhộng": "oral capsule",
    "viên nhai": "chewable tablet",
    "viên ngậm dưới lưỡi": "sublingual tablet",
    "dung dịch uống": "oral solution",
    "hỗn dịch uống": "oral suspension",
    "kem bôi": "topical cream",
    "thuốc mỡ": "topical ointment",
    "dung dịch nhỏ mắt": "ophthalmic solution",
    "thuốc tiêm": "injection",
    "dung dịch tiêm": "injection",
    "dung dịch truyền": "injection",
    "truyền tĩnh mạch": "injection",
    "tiêm tĩnh mạch": "injection",
    "phóng thích kéo dài": "extended release",
    "giải phóng kéo dài": "extended release",
    "tác dụng kéo dài": "extended release",
    "phóng thích chậm": "delayed release",
    "bao tan trong ruột": "delayed release",
}

ABBREVIATION_MAP = {
    " xr ": " extended release ",
    " xl ": " extended release ",
    " er ": " extended release ",
    " sr ": " extended release ",
    " cr ": " controlled release ",
}

# Chỉ dùng khi build câu query cho dense embedding (normalize_text).
# KHÔNG dùng trước bước parse route/frequency/quantity, vì các pattern
# này (po, prn, q6h, daily, ...) chính là input cho parser.
NOISE_PATTERNS = [
    r"\bpo\b",
    r"\bprn\b",
    r"\bbid\b",
    r"\btid\b",
    r"\bqid\b",
    r"\bqhs\b",
    r"\bqam\b",
    r"\bq\d+h\b",
    r"\bdaily\b",
    r"\bonce daily\b",
    r"\btwice daily\b",
    r"\buống\b",
    r"\bngày\s+\d+\s+lần\b",
    r"\bmỗi ngày\b",
    r"\bsau ăn\b",
    r"\btrước ăn\b",
    r"\bsáng\b",
    r"\btối\b",
    r"\bkhi đau\b",
]

# --------------------------------------------------------------------------
# Đơn vị strength/quantity
# --------------------------------------------------------------------------

UNIT_MAP = {
    "mg": "MG",
    "g": "G",
    "mcg": "MCG",
    "ml": "ML",
    "l": "L",
    "meq": "MEQ",
    "iu": "IU",
    "unit": "UNIT",
    "units": "UNIT",
    "unt": "UNIT",
    "gm": "G",
    "gram": "G",
    "grams": "G",
}

# Đơn vị coi là "quantity" (thể tích/khối lượng dispense) khi đứng một
# mình, không có tỉ lệ nồng độ (không có dấu "/"). Có mặt số + đơn vị
# strength thật (MG/MCG/...) thì luôn là strength.
QUANTITY_ONLY_UNITS = {"ML", "L"}
STRENGTH_UNITS = {"MG", "G", "MCG", "MEQ", "IU", "UNIT"}

# --------------------------------------------------------------------------
# Dose form & release type (đã ở dạng English, sau khi qua PHRASE_MAP)
# --------------------------------------------------------------------------

DOSE_FORM_TERMS = (
    "extended release oral tablet",
    "delayed release oral tablet",
    "oral tablet",
    "oral capsule",
    "chewable tablet",
    "sublingual tablet",
    "oral solution",
    "oral suspension",
    "topical cream",
    "topical ointment",
    "ophthalmic solution",
    "injection",
    "injectable solution",
)

RELEASE_TYPE_TERMS = (
    "extended release",
    "delayed release",
    "controlled release",
)

# --------------------------------------------------------------------------
# Route & frequency (đọc trên text CHƯA strip noise)
# --------------------------------------------------------------------------

ROUTE_PHRASE_MAP = {
    "truyền tĩnh mạch": "IV",
    "tiêm tĩnh mạch": "IV",
    "đường tĩnh mạch": "IV",
    "tiêm bắp": "IM",
    "tiêm dưới da": "SC",
    "nhỏ mắt": "OPHTH",
    "bôi ngoài da": "TOP",
    "ngậm dưới lưỡi": "SL",
    "đường uống": "PO",
}

ROUTE_MAP = {
    "po": "PO",
    "iv": "IV",
    "im": "IM",
    "sc": "SC",
    "sq": "SC",
    "top": "TOP",
    "ophth": "OPHTH",
    "sl": "SL",
}

# Thứ tự ưu tiên khi match (dài -> ngắn) để tránh "daily" bị q\d+h nuốt nhầm.
FREQUENCY_MAP = {
    "qid": "QID",
    "tid": "TID",
    "bid": "BID",
    "qhs": "QHS",
    "qam": "QAM",
    "once daily": "QD",
    "twice daily": "BID",
    "daily": "QD",
}

FREQUENCY_INTERVAL_HOURS = {
    "QD": 24,
    "BID": 12,
    "TID": 8,
    "QID": 6,
    "QHS": 24,
    "QAM": 24,
}

# --------------------------------------------------------------------------
# TTY bonus theo mức độ cụ thể (specificity) của mention — dùng ở reranker
# --------------------------------------------------------------------------

TTY_BONUS_TABLE = {
    "full_product": {  # mention có cả strength + dose form -> ưu tiên clinical drug đầy đủ
        "SCD": 0.130,
        "SBD": 0.090,
        "SCDC": -0.060,
        "SCDF": -0.060,
        "IN": -0.080,
        "PIN": -0.070,
        "MIN": -0.070,
    },
    "ingredient_strength": {
        # mention có strength nhưng KHÔNG nói dose form -> BTC vẫn chọn SCD
        # (thuốc lâm sàng với dose form mặc định), KHÔNG chọn SCDC (chỉ mới
        # hoạt chất + liều, chưa phải sản phẩm hoàn chỉnh). Đây là chỗ bảng
        # cũ bị đảo ngược, gây sai gần hết các ca có strength trong 11 gold.
        "SCD": 0.140,
        "SBD": 0.090,
        "SCDC": -0.080,
        "SCDF": -0.080,
        "IN": -0.050,
        "PIN": -0.040,
        "MIN": -0.040,
    },
    "ingredient_form": {
        # mention có dose form nhưng KHÔNG có strength -> BTC không tự đoán
        # liều, chọn IN (hoạt chất trần) thay vì SCDF/SCD cụ thể.
        "IN": 0.100,
        "PIN": 0.080,
        "MIN": 0.080,
        "SCDF": -0.040,
        "SCD": -0.050,
        "SBD": -0.050,
    },
    "ingredient_only": {
        "IN": 0.070,
        "PIN": 0.065,
        "MIN": 0.060,
        "BN": 0.055,
        "SCD": -0.050,
        "SBD": -0.055,
        "SCDC": -0.030,
        "SCDF": -0.030,
        "GPCK": -0.060,
        "BPCK": -0.060,
    },
}


# Generic non-linkable medication-class or administration mentions. These are
# semantic categories rather than RxNorm drug concepts and should abstain.
GENERIC_DRUG_CLASS_PATTERNS = (
    r"^(?:thuoc\s+)?khang\s+sinh$",
    r"^(?:thuoc\s+)?uc\s+che\s+mien\s+dich$",
    r"^(?:thuoc\s+)?loi\s+tieu$",
    r"^(?:thuoc\s+)?chong\s+non$",
    r"^(?:thuoc\s+)?chong\s+tram\s+cam$",
    r"^(?:thuoc\s+)?khang\s+histamin(?:\s+h1)?$",
    r"^corticoid(?:\s+lieu.*)?$",
    r"^nsaids?$",
    r"^vacc?in(?:e)?(?:\s+song)?$",
    r"^intravenous\s+fluids?$",
)

# Route-compatible dose-form keywords used as a hard safety gate only when the
# route is explicit in the mention.
ROUTE_FORM_KEYWORDS = {
    "PO": ("oral", "tablet", "capsule", "solution", "suspension", "sublingual"),
    "IV": ("injection", "injectable", "intravenous"),
    "IM": ("injection", "injectable", "intramuscular"),
    "SC": ("injection", "injectable", "subcutaneous"),
    "OPHTH": ("ophthalmic",),
    "TOP": ("topical", "cream", "ointment", "lotion", "foam", "gel"),
    "SL": ("sublingual",),
}

ROUTE_CONFLICT_KEYWORDS = {
    "IV": ("oral", "tablet", "capsule", "topical", "ophthalmic"),
    "IM": ("oral", "tablet", "capsule", "topical", "ophthalmic"),
    "SC": ("oral", "tablet", "capsule", "topical", "ophthalmic"),
    "PO": ("injectable", "injection", "ophthalmic", "topical"),
    "OPHTH": ("oral", "tablet", "capsule", "injection", "topical"),
    "TOP": ("oral", "tablet", "capsule", "injection", "ophthalmic"),
    "SL": ("injection", "ophthalmic", "topical"),
}

COMBINATION_CONNECTOR_RE = r"(?:\b(?:va|và|and)\b|\+|\s/\s)"