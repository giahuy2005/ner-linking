"""Parse a medication mention into structured RxNorm retrieval evidence.

The parser is deliberately conservative: it extracts explicit ingredient,
strength, form, release and route evidence but never invents a drug from a
class name or an administration fragment.
"""

from __future__ import annotations

import re
import unicodedata

from . import config
from .schemas import ParsedDrugMention

_DASH = r"[-–—−]"
_NUMBER = r"\d+(?:[\.,]\d+)?"
_UNIT = r"mg|mcg|g|gm|grams?|ml|l|meq|iu|unt|units?|unit"

_STRENGTH_RANGE_RE = re.compile(
    rf"({_NUMBER})\s*{_DASH}\s*({_NUMBER})\s*({_UNIT})\b", re.I
)
_STRENGTH_RATIO_RE = re.compile(
    rf"({_NUMBER})\s*({_UNIT})\s*/\s*({_NUMBER})?\s*({_UNIT})\b", re.I
)
_AMOUNT_UNIT_RE = re.compile(rf"({_NUMBER})\s*({_UNIT})\b", re.I)
_PERCENT_RE = re.compile(rf"({_NUMBER})\s*%")
_MULTIPLIER_RE = re.compile(
    r"\bx\s*\d+(?:[\.,]\d+)?\s*(?:vien|ong|goi|lo|chai|lan|lieu)?\b", re.I
)
_COORDINATOR_RE = re.compile(config.COMBINATION_CONNECTOR_RE, re.I)

_ADMIN_WORDS = {
    "thuoc", "lieu", "ngay", "tuan", "lan", "cach", "moi", "cham",
    "uong", "tiem", "truyen", "giot", "phut", "vien", "ong", "goi",
    "po", "iv", "im", "sc", "sq", "prn", "bid", "tid", "qid", "qhs",
    "qam", "daily", "oral", "injection", "x", "giot", "phut", "vien", "ong", "goi", "chai",
}


def _decimal(value: str) -> str:
    return value.replace(",", ".")


def _fold_ascii(text: str) -> str:
    value = unicodedata.normalize("NFD", text.casefold()).replace("đ", "d")
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", value).strip()


def _semantic_normalize(text: str) -> str:
    """Normalize semantics while preserving route/frequency evidence."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = (
        text.replace("µg", "mcg")
        .replace("μg", "mcg")
        .replace("ug", "mcg")
    )
    padded = f" {text} "
    for source, target in sorted(config.PHRASE_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        padded = padded.replace(source, target)
    for source, target in config.ABBREVIATION_MAP.items():
        padded = padded.replace(source, target)
    for source, target in config.DRUG_ALIAS_MAP.items():
        padded = re.sub(rf"(?<!\w){re.escape(source)}(?!\w)", target, padded)
    return f" {re.sub(r'\s+', ' ', padded).strip()} "


def normalize_text(text: str) -> str:
    padded = _semantic_normalize(text)
    for pattern in config.NOISE_PATTERNS:
        padded = re.sub(pattern, " ", padded)
    padded = re.sub(r"\b(?:iv|im|sc|sq|top|ophth|sl|prn)\b", " ", padded)
    padded = re.sub(r"[^a-z0-9À-ỹ%./+]+", " ", padded)
    return re.sub(r"\s+", " ", padded).strip()


def _unit_norm(unit: str) -> str:
    return config.UNIT_MAP.get(unit.casefold().rstrip("s"), unit.upper())


def parse_strength_range(text: str) -> str | None:
    match = _STRENGTH_RANGE_RE.search(_semantic_normalize(text))
    if not match:
        return None
    low, high, unit = match.groups()
    return f"{_decimal(low)}-{_decimal(high)} {_unit_norm(unit)}"


def parse_strengths(text: str) -> list[str]:
    normalized = _semantic_normalize(text)
    strengths: list[str] = []

    range_match = parse_strength_range(normalized)
    if range_match:
        strengths.append(range_match)
        normalized = _STRENGTH_RANGE_RE.sub(" ", normalized)

    for match in _STRENGTH_RATIO_RE.finditer(normalized):
        num_val, num_unit, den_val, den_unit = match.groups()
        numerator = f"{_decimal(num_val)} {_unit_norm(num_unit)}"
        denominator = (
            f"{_decimal(den_val)} {_unit_norm(den_unit)}"
            if den_val else _unit_norm(den_unit)
        )
        strengths.append(f"{numerator}/{denominator}")

    ratio_spans = [match.span() for match in _STRENGTH_RATIO_RE.finditer(normalized)]
    for match in _AMOUNT_UNIT_RE.finditer(normalized):
        if any(start <= match.start() < end for start, end in ratio_spans):
            continue
        value, unit = match.groups()
        unit_norm = _unit_norm(unit)
        if unit_norm in config.STRENGTH_UNITS:
            strengths.append(f"{_decimal(value)} {unit_norm}")

    # A percentage is retained explicitly and converted generically to a w/v
    # mass concentration for solution candidates (e.g. 5% -> 50 MG/ML).
    for match in _PERCENT_RE.finditer(normalized):
        percent = float(_decimal(match.group(1)))
        strengths.append(f"{_decimal(match.group(1))} PERCENT")
        strengths.append(f"{percent * 10:g} MG/ML")

    return list(dict.fromkeys(strengths))


def parse_quantity(text: str) -> str | None:
    normalized = _STRENGTH_RATIO_RE.sub(" ", _semantic_normalize(text))
    normalized = _STRENGTH_RANGE_RE.sub(" ", normalized)
    for match in _AMOUNT_UNIT_RE.finditer(normalized):
        value, unit = match.groups()
        unit_norm = _unit_norm(unit)
        if unit_norm in config.QUANTITY_ONLY_UNITS:
            return f"{_decimal(value)} {unit_norm}"
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
        if any(phrase in form and form != phrase for form in dose_forms):
            continue
        if phrase in normalized and phrase not in found:
            found.append(phrase)
    return found


def parse_route(text: str) -> str | None:
    folded = _fold_ascii(text)
    for phrase, route in sorted(config.ROUTE_PHRASE_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        if _fold_ascii(phrase) in folded:
            return route
    normalized = _semantic_normalize(text)
    for token, route in config.ROUTE_MAP.items():
        if re.search(rf"\b{re.escape(token)}\b", normalized):
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


def _strip_structured_parts(text: str) -> str:
    value = _semantic_normalize(text)
    for pattern in config.NOISE_PATTERNS:
        value = re.sub(pattern, " ", value)
    value = re.sub(r"\b(?:iv|im|sc|sq|top|ophth|sl|prn)\b", " ", value)
    value = _STRENGTH_RANGE_RE.sub(" ", value)
    value = _STRENGTH_RATIO_RE.sub(" ", value)
    value = _AMOUNT_UNIT_RE.sub(" ", value)
    value = _PERCENT_RE.sub(" ", value)
    value = _MULTIPLIER_RE.sub(" ", value)
    for phrase in sorted(config.DOSE_FORM_TERMS, key=len, reverse=True):
        value = value.replace(phrase, " ")
    for phrase in sorted(config.RELEASE_TYPE_TERMS, key=len, reverse=True):
        value = value.replace(phrase, " ")
    value = re.sub(r"\b(?:ml|l|giot|giọt|phut|phút|vien|viên|ong|ống|goi|gói|chai|x)\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n:;,.-/()")
    return value


def _component_is_meaningful(component: str) -> bool:
    tokens = re.findall(r"[a-z0-9]+", _fold_ascii(component))
    meaningful = [token for token in tokens if token not in _ADMIN_WORDS and not token.isdigit()]
    return bool(meaningful)


def extract_drug_components(text: str) -> tuple[list[str], list[str]]:
    """Return explicit ingredient components and optional brand hints."""
    raw_semantic = _semantic_normalize(text).strip()
    brand_hints: list[str] = []
    ingredient_source = raw_semantic

    # Prefer an explicit generic name after a colon or inside parentheses.
    if ":" in ingredient_source:
        prefix, suffix = ingredient_source.split(":", 1)
        if _component_is_meaningful(prefix):
            brand_hints.append(_strip_structured_parts(prefix))
        if _component_is_meaningful(suffix):
            ingredient_source = suffix
    paren_match = re.search(r"\(([^)]{3,})\)?", ingredient_source)
    if paren_match and _component_is_meaningful(paren_match.group(1)):
        prefix = ingredient_source[:paren_match.start()]
        if _component_is_meaningful(prefix):
            brand_hints.append(_strip_structured_parts(prefix))
        ingredient_source = paren_match.group(1)

    cleaned = _strip_structured_parts(ingredient_source)
    cleaned = re.sub(r"^(?:(?:thuốc|thuoc)(?:i\s+|\s*))+", "", cleaned).strip()
    components = [part.strip(" ,;:/()") for part in _COORDINATOR_RE.split(cleaned)]
    components = [part for part in components if _component_is_meaningful(part)]
    if not components and _component_is_meaningful(cleaned):
        components = [cleaned]
    return list(dict.fromkeys(components)), [item for item in dict.fromkeys(brand_hints) if item]


def extract_drug_core(text: str) -> str | None:
    components, _brands = extract_drug_components(text)
    return " / ".join(components) if components else None


def classify_strength_role(strengths: list[str]) -> str:
    if not strengths:
        return "missing"
    if any("PERCENT" in value for value in strengths):
        return "percent"
    if any("-" in value for value in strengths):
        return "range"
    if any("/" in value for value in strengths):
        return "ratio"
    return "single" if len(strengths) == 1 else "range"


def _generic_class_warning(core: str | None) -> bool:
    if not core:
        return False
    folded = _fold_ascii(core)
    return any(re.fullmatch(pattern, folded, flags=re.I) for pattern in config.GENERIC_DRUG_CLASS_PATTERNS)


def parse_drug_mention(mention: str) -> ParsedDrugMention:
    strengths = parse_strengths(mention)
    dose_forms = parse_dose_forms(mention)
    components, brand_hints = extract_drug_components(mention)
    core = " / ".join(components) if components else None
    warnings: list[str] = []
    semantic = _semantic_normalize(mention)
    folded = _fold_ascii(mention)

    if re.search(r"(?<!\d)\b(?:ml|l)\b", semantic) and not parse_quantity(mention):
        warnings.append("bare_liquid_unit_without_quantity")
    if _generic_class_warning(core):
        warnings.append("generic_drug_class")
    if re.search(r"\b(?:khong|without|free)\b.{0,24}\b(?:caffeine|thuoc|drug)\b", folded):
        warnings.append("negated_or_excluded_ingredient")
    if not components:
        warnings.append("missing_ingredient")
    if len(components) >= 2:
        warnings.append("multi_ingredient_mention")

    frequency, interval = parse_frequency(mention)
    parsed = ParsedDrugMention(
        raw_text=mention,
        normalized_text=normalize_text(mention),
        ingredient_core=core,
        ingredient_components=components,
        brand_hints=brand_hints,
        ingredient_aliases=list(dict.fromkeys([*components, *(brand_hints or [])])),
        strengths=strengths,
        strength_role=classify_strength_role(strengths),
        dose_forms=dose_forms,
        release_types=parse_release_types(mention),
        quantity=parse_quantity(mention),
        route=parse_route(mention),
        frequency=frequency,
        interval_hours=interval,
        parse_warnings=warnings,
    )
    parsed.query_variants = build_query_variants(parsed)
    return parsed


def build_query_variants(parsed: ParsedDrugMention) -> list[dict[str, str]]:
    values: list[tuple[str, str]] = [("full_normalized", parsed.normalized_text)]
    components = parsed.ingredient_components or ([parsed.ingredient_core] if parsed.ingredient_core else [])
    if parsed.ingredient_core:
        values.append(("ingredient_core", parsed.ingredient_core))
    for component in components:
        values.append(("ingredient_component", component))
        for strength in parsed.strengths:
            values.append(("ingredient_strength", f"{component} {strength}"))
        for form in parsed.dose_forms:
            values.append(("ingredient_form", f"{component} {form}"))
        if parsed.route in {"IV", "IM", "SC"}:
            values.append(("ingredient_route_form", f"{component} injection"))
    if len(components) >= 2:
        joined = " / ".join(components)
        values.append(("combination_ingredients", joined))
        for strength in parsed.strengths:
            values.append(("combination_strength", f"{joined} {strength}"))
    for brand in parsed.brand_hints:
        values.append(("brand_hint", brand))
    if "bare_liquid_unit_without_quantity" in parsed.parse_warnings and parsed.ingredient_core:
        values.extend([
            ("liquid_form_hint", f"{parsed.ingredient_core} oral solution"),
            ("liquid_form_hint", f"{parsed.ingredient_core} oral suspension"),
        ])
    if parsed.strength_role == "range" and parsed.strengths:
        match = re.match(r"([0-9.]+)-([0-9.]+)\s+(.+)", parsed.strengths[0])
        if match and parsed.ingredient_core:
            low, high, unit = match.groups()
            values.extend([
                ("range_lower_endpoint", f"{parsed.ingredient_core} {low} {unit}"),
                ("range_upper_endpoint", f"{parsed.ingredient_core} {high} {unit}"),
            ])

    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for source, text in values:
        normalized = normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append({"text": normalized, "source": source})
    return output