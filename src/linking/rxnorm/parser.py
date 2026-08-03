"""Parser: raw mention -> ParsedDrugMention.

Lưu ý: notebook gốc (embedding_rxnorm.ipynb) gọi parse_strengths() và
parse_dose_forms() nhưng không định nghĩa chúng trong các cell còn lại
(chắc nằm trong build_rxnorm_faiss_indexes.py không upload). File này
viết lại hai hàm đó từ đầu, dựa theo cách chúng được dùng trong notebook
(normalize_text(), PHRASE_MAP, UNIT_MAP, ví dụ print_rxnorm_results).
Nên chạy lại evaluate.py để tune regex theo dữ liệu thật, đặc biệt các
case biên như "guaifenesin ml po q6h:prn" (thiếu số liều trong gold).
"""

from __future__ import annotations

import re
import unicodedata

from . import config
from .schemas import ParsedDrugMention

_STRENGTH_RANGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*"
    r"(mg|mcg|g|ml|l|meq|iu|units?)\b"
)

_STRENGTH_RATIO_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|l|meq|iu|units?)"
    r"\s*/\s*(\d+(?:\.\d+)?)?\s*(mg|mcg|g|ml|l|meq|iu|units?)\b"
)

_AMOUNT_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|l|meq|iu|units?)\b"
)


def normalize_text(text: str) -> str:
    """Chuẩn hoá đầy đủ, dùng cho dense embedding query (đã strip noise)."""

    padded = _semantic_normalize(text)

    for pattern in config.NOISE_PATTERNS:
        padded = re.sub(pattern, " ", padded)

    padded = re.sub(r"[^a-z0-9À-ỹ./]+", " ", padded)
    padded = re.sub(r"\s+", " ", padded).strip()

    return padded


def _semantic_normalize(text: str) -> str:
    """Chuẩn hoá nhưng GIỮ route/frequency — dùng để parser đọc trước
    khi normalize_text() xoá chúng đi."""

    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()

    text = (
        text.replace("µg", "mcg")
        .replace("μg", "mcg")
        .replace("ug", "mcg")
    )

    padded = f" {text} "

    for source, target in sorted(
        config.PHRASE_MAP.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        padded = padded.replace(source, target)

    for source, target in config.ABBREVIATION_MAP.items():
        padded = padded.replace(source, target)

    for source, target in config.DRUG_ALIAS_MAP.items():
        padded = re.sub(rf"\b{re.escape(source)}\b", target, padded)

    padded = re.sub(r"\s+", " ", padded).strip()
    padded = f" {padded} "

    return padded


def parse_strength_range(text: str) -> str | None:
    match = _STRENGTH_RANGE_RE.search(text)

    if not match:
        return None

    low, high, unit = match.groups()
    unit_norm = config.UNIT_MAP.get(unit.rstrip("s"), unit.upper())

    return f"{low}-{high} {unit_norm}"


def parse_strengths(text: str) -> list[str]:
    """Trả về list strength đã chuẩn hoá đơn vị (vd '10 MG', '0.4 MG/ML').

    Không tính các amount có đơn vị ML/L đứng một mình không có tỉ lệ
    nồng độ — cái đó là quantity (parse_quantity), không phải strength.
    """

    normalized = _semantic_normalize(text)
    strengths: list[str] = []

    range_match = parse_strength_range(normalized)
    if range_match:
        strengths.append(range_match)
        # xoá phần range khỏi text để không bị parse trùng bởi ratio/amount
        normalized = _STRENGTH_RANGE_RE.sub(" ", normalized)

    for match in _STRENGTH_RATIO_RE.finditer(normalized):
        num_val, num_unit, den_val, den_unit = match.groups()
        num_unit_norm = config.UNIT_MAP.get(num_unit.rstrip("s"), num_unit.upper())
        den_unit_norm = config.UNIT_MAP.get(den_unit.rstrip("s"), den_unit.upper())

        if den_val:
            display = f"{num_val} {num_unit_norm}/{den_val} {den_unit_norm}"
        else:
            display = f"{num_val} {num_unit_norm}/{den_unit_norm}"

        strengths.append(display)

    consumed_spans = [m.span() for m in _STRENGTH_RATIO_RE.finditer(normalized)]

    for match in _AMOUNT_UNIT_RE.finditer(normalized):
        if any(start <= match.start() < end for start, end in consumed_spans):
            continue

        value, unit = match.groups()
        unit_norm = config.UNIT_MAP.get(unit.rstrip("s"), unit.upper())

        if unit_norm not in config.STRENGTH_UNITS:
            continue  # ML/L một mình -> quantity, không phải strength

        strengths.append(f"{value} {unit_norm}")

    seen: set[str] = set()
    ordered: list[str] = []
    for item in strengths:
        if item not in seen:
            seen.add(item)
            ordered.append(item)

    return ordered


def parse_quantity(text: str) -> str | None:
    """Bắt thể tích/khối lượng dispense (ML/L đứng một mình, không tỉ lệ)."""

    normalized = _semantic_normalize(text)
    normalized = _STRENGTH_RATIO_RE.sub(" ", normalized)
    normalized = _STRENGTH_RANGE_RE.sub(" ", normalized)

    for match in _AMOUNT_UNIT_RE.finditer(normalized):
        value, unit = match.groups()
        unit_norm = config.UNIT_MAP.get(unit.rstrip("s"), unit.upper())

        if unit_norm in config.QUANTITY_ONLY_UNITS:
            return f"{value} {unit_norm}"

    return None


def parse_dose_forms(text: str) -> list[str]:
    normalized = _semantic_normalize(text)
    found: list[str] = []

    for phrase in sorted(config.DOSE_FORM_TERMS, key=len, reverse=True):
        if phrase in normalized and phrase not in found:
            found.append(phrase)

    return found


def parse_release_types(text: str) -> list[str]:
    normalized = _semantic_normalize(text)
    dose_forms = parse_dose_forms(text)
    found: list[str] = []

    for phrase in sorted(config.RELEASE_TYPE_TERMS, key=len, reverse=True):
        # nếu release type đã nằm trong 1 dose form ghép (vd "extended
        # release oral tablet") thì không tách ra thêm lần nữa
        if any(phrase in form and form != phrase for form in dose_forms):
            continue

        if phrase in normalized and phrase not in found:
            found.append(phrase)

    return found


def parse_route(text: str) -> str | None:
    normalized = _semantic_normalize(text)

    for token, route in config.ROUTE_MAP.items():
        if re.search(rf"\b{token}\b", normalized):
            return route

    return None


def parse_frequency(text: str) -> tuple[str | None, int | None]:
    normalized = _semantic_normalize(text)

    for phrase in sorted(config.FREQUENCY_MAP, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", normalized):
            code = config.FREQUENCY_MAP[phrase]
            return code, config.FREQUENCY_INTERVAL_HOURS.get(code)

    q_match = re.search(r"\bq(\d+)h\b", normalized)
    if q_match:
        hours = int(q_match.group(1))
        return f"Q{hours}H", hours

    return None, None


def extract_drug_core(text: str) -> str | None:
    """Lấy phần tên hoạt chất/thuốc, trước khi gặp số liều/dose form đầu tiên."""

    normalized = _semantic_normalize(text)
    # Route/frequency/PRN are administration instructions, never ingredients.
    for pattern in config.NOISE_PATTERNS:
        normalized = re.sub(pattern, " ", normalized)
    normalized = re.sub(r"\b(?:iv|im|sc|sq|top|ophth|sl|prn)\b", " ", normalized)
    # A bare liquid unit ("guaifenesin ml") is a form hint, not part of the
    # ingredient and not a measurable quantity without a number.
    normalized = re.sub(r"\b(?:ml|l)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    cut_positions = []

    for pattern in (_STRENGTH_RANGE_RE, _STRENGTH_RATIO_RE, _AMOUNT_UNIT_RE):
        match = pattern.search(normalized)
        if match:
            cut_positions.append(match.start())

    for phrase in config.DOSE_FORM_TERMS:
        idx = normalized.find(phrase)
        if idx != -1:
            cut_positions.append(idx)

    core = normalized if not cut_positions else normalized[: min(cut_positions)]
    core = core.strip(" \t\r\n:;,.-/")

    return core or None


def classify_strength_role(strengths: list[str]) -> str:
    if not strengths:
        return "missing"

    if len(strengths) > 1 or "-" in strengths[0]:
        return "range"

    if "/" in strengths[0]:
        return "ratio"

    return "single"


def parse_drug_mention(mention: str) -> ParsedDrugMention:
    strengths = parse_strengths(mention)
    dose_forms = parse_dose_forms(mention)

    core = extract_drug_core(mention)
    warnings = []
    semantic = _semantic_normalize(mention)
    if re.search(r"(?<!\d)\b(?:ml|l)\b", semantic) and not parse_quantity(mention):
        warnings.append("bare_liquid_unit_without_quantity")
    parsed = ParsedDrugMention(
        raw_text=mention,
        normalized_text=normalize_text(mention),
        ingredient_core=core,
        ingredient_aliases=[core] if core else [],
        strengths=strengths,
        strength_role=classify_strength_role(strengths),
        dose_forms=dose_forms,
        release_types=parse_release_types(mention),
        quantity=parse_quantity(mention),
        route=parse_route(mention),
        frequency=parse_frequency(mention)[0],
        interval_hours=parse_frequency(mention)[1],
        parse_warnings=warnings,
    )
    parsed.query_variants = build_query_variants(parsed)
    return parsed


def build_query_variants(parsed: ParsedDrugMention) -> list[dict[str, str]]:
    """Return deduplicated, auditable retrieval interpretations."""
    values: list[tuple[str, str]] = [("full_normalized", parsed.normalized_text)]
    core = parsed.ingredient_core or ""
    if core:
        values.append(("ingredient_core", core))
        for strength in parsed.strengths:
            values.append(("ingredient_strength", f"{core} {strength}"))
        for form in parsed.dose_forms:
            values.append(("ingredient_form", f"{core} {form}"))
        for release in parsed.release_types:
            values.append(("ingredient_release", f"{core} {release}"))
        for strength in parsed.strengths:
            for form in parsed.dose_forms:
                values.append(("ingredient_strength_form", f"{core} {strength} {form}"))
        if "bare_liquid_unit_without_quantity" in parsed.parse_warnings:
            values.extend([
                ("liquid_form_hint", f"{core} oral solution"),
                ("liquid_form_hint", f"{core} oral suspension"),
            ])
        if parsed.strength_role == "range" and parsed.strengths:
            match = re.match(r"([0-9.]+)-([0-9.]+)\s+(.+)", parsed.strengths[0])
            if match:
                low, high, unit = match.groups()
                values.extend([
                    ("range_lower_endpoint", f"{core} {low} {unit}"),
                    ("range_upper_endpoint", f"{core} {high} {unit}"),
                ])
    output, seen = [], set()
    for source, text in values:
        normalized = normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append({"text": normalized, "source": source})
    return output
