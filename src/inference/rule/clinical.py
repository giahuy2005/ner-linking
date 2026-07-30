"""Surface-agnostic validation between NER and the 7B reviewer.

Only deterministic structure is handled here: offset validation, mechanical
boundary repair, assertion scope, overlap resolution and medication-list
layout. Clinical mentions are not recovered, deleted, or retyped from a
private-test vocabulary.
"""

from __future__ import annotations

import re

from ..schemas import NerEntity

ALLOWED_TYPES = {
    "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC",
    "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM",
}
ALLOWED_ASSERTIONS = {"isHistorical", "isNegated", "isFamily"}

_DEVICE_PREFIXES = (
    "stent", "catheter", "picc", "foley", "ống dẫn mật", "ống dẫn lưu",
)
_GENERIC_NON_ENTITIES = {
    "kết quả", "xét nghiệm", "thuốc", "mẫu", "dấu hiệu", "triệu chứng",
}
_MEASUREMENT_ONLY_RE = re.compile(
    r"^\d+(?:[.,]\d+)?\s*(?:kg|fr|tuần|week|weeks|w)$", re.I
)
_DRUG_HEADER_RE = re.compile(r"danh\s+sách\s+thuốc\s+trước\s+nhập\s+viện", re.I)
_NUMBERED_ITEM_RE = re.compile(r"(?:^|\s)(\d+)\.\s+", re.M)


def _exact(raw_text: str, entity: NerEntity) -> bool:
    start, end = entity.position
    return (
        entity.type in ALLOWED_TYPES
        and 0 <= start < end <= len(raw_text)
        and raw_text[start:end] == entity.text
    )


def _copy(entity: NerEntity, *, start: int | None = None, end: int | None = None,
          text: str | None = None, entity_type: str | None = None,
          assertions: list[str] | None = None, flag: str | None = None) -> NerEntity:
    return NerEntity(
        text=entity.text if text is None else text,
        type=entity.type if entity_type is None else entity_type,
        assertions=list(entity.assertions if assertions is None else assertions),
        position=(entity.position[0] if start is None else start,
                  entity.position[1] if end is None else end),
        score=entity.score,
        flag=flag,
        review_hints=list(entity.review_hints),
    )


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split()).strip(" \t\r\n.,;:()[]{}")


def _is_hard_negative(entity: NerEntity) -> str | None:
    normalized = _normalize(entity.text)
    if normalized in _GENERIC_NON_ENTITIES:
        return "generic_non_entity"
    if normalized.isdigit():
        return "isolated_number"
    if _MEASUREMENT_ONLY_RE.fullmatch(normalized):
        return "isolated_measurement"
    if entity.type in {"THUỐC", "CHẨN_ĐOÁN"} and normalized.startswith(_DEVICE_PREFIXES):
        return "device_or_procedure"
    return None


def is_linkable_entity(entity: NerEntity) -> bool:
    """Defense-in-depth gate used immediately before RxNorm/ICD retrieval."""
    return entity.type in {"THUỐC", "CHẨN_ĐOÁN"} and _is_hard_negative(entity) is None


def _numbered_section_heading(raw_text: str, position: int) -> str:
    headings = list(re.finditer(r"(?im)^\s*\d+\.\s+([^\n]+)$", raw_text[:position]))
    return _normalize(headings[-1].group(1)) if headings else ""


def _repair_assertion_scope(raw_text: str, entity: NerEntity) -> NerEntity:
    assertions = [item for item in entity.assertions if item in ALLOWED_ASSERTIONS]
    start, _end = entity.position
    heading = _numbered_section_heading(raw_text, start)
    local_start = max(raw_text.rfind("\n", 0, start), raw_text.rfind(".", 0, start)) + 1
    local_prefix = _normalize(raw_text[local_start:start])
    negated_history = "phủ nhận tiền sử" in local_prefix
    exception_scope = "ngoại trừ" in local_prefix

    if exception_scope and "isNegated" in assertions:
        assertions = [item for item in assertions if item != "isNegated"]
    elif negated_history:
        # "phủ nhận tiền sử X" carries both assertion dimensions.  An
        # exception later in the same clause is handled above and is positive.
        if "isNegated" not in assertions:
            assertions.append("isNegated")
        if "isHistorical" not in assertions:
            assertions.append("isHistorical")
    elif re.search(r"(?:^|\s)(?:không|chưa|phủ nhận)(?:\s+\S+){0,12}\s*$", local_prefix) \
            and entity.type not in {"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"} \
            and "isNegated" not in assertions:
        assertions.append("isNegated")

    current_cue = re.search(
        r"(?i)\b(?:hiện\s+đang|nay\s+.*?đang|hiện\s+tại|đang\s+có)\b",
        raw_text[max(0, start - 100):start],
    )
    current_section = any(marker in heading for marker in (
        "hiện tại", "đánh giá tại bệnh viện", "tình trạng hiện tại",
    ))
    if (current_cue or current_section) and not negated_history:
        assertions = [item for item in assertions if item != "isHistorical"]
    elif heading == "tiền sử bệnh" and entity.type not in {
        "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM",
    } and "isHistorical" not in assertions:
        assertions.append("isHistorical")

    if entity.type in {"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}:
        assertions = []
    return _copy(entity, assertions=assertions, flag=entity.flag)


def _repair_boundary(raw_text: str, entity: NerEntity) -> NerEntity:
    """Apply only repairs whose replacement is an exact nearby substring."""
    text = entity.text
    start, end = entity.position
    candidates: list[tuple[int, int]] = []

    # Leading/trailing patient markers and punctuation leaked by BIO.
    trimmed = re.sub(r"^(?:bn|bệnh\s+nhân)\s+", "", text, flags=re.I)
    trimmed = re.sub(r"\s+(?:bn|ở|khi)$", "", trimmed, flags=re.I)
    trimmed = re.sub(r"\s*[\(\[\{]+\s*$", "", trimmed).strip()
    trimmed = trimmed.splitlines()[0].rstrip() if "\n" in trimmed else trimmed
    if trimmed and trimmed != text:
        local = text.find(trimmed)
        if local >= 0:
            candidates.append((start + local, start + local + len(trimmed)))

    # Prefer the second copy for duplicated prefixes.  It yields an exact span.
    tokens = list(re.finditer(r"\S+", text))
    if len(tokens) >= 2 and tokens[0].group(0).casefold() == tokens[1].group(0).casefold():
        candidates.append((start + tokens[1].start(), end))
    words = text.split()
    for width in range(1, min(4, len(words) // 2) + 1):
        if [w.casefold() for w in words[:width]] == [w.casefold() for w in words[width:2 * width]]:
            needle = " ".join(words[width:])
            local = text.find(needle)
            candidates.append((start + local, start + local + len(needle)))

    # Fused trailing connector: "ổn địnhkhi" -> "ổn định".
    fused = re.sub(r"(?i)(?<=\w)(?:khi|ở)$", "", text).rstrip()
    if fused and fused != text:
        candidates.append((start, start + len(fused)))

    valid = [(s, e) for s, e in candidates if 0 <= s < e <= len(raw_text)]
    if not valid:
        return entity
    # Most conservative repair: discard the fewest characters.
    new_start, new_end = max(valid, key=lambda span: span[1] - span[0])
    new_text = raw_text[new_start:new_end]
    return _copy(entity, start=new_start, end=new_end, text=new_text, flag=None)


def _resolve_overlaps(entities: list[NerEntity]) -> list[NerEntity]:
    """Weighted interval selection with deterministic notebook priorities."""
    ranked = sorted(
        entities,
        key=lambda e: (
            -float(e.score),
            -(e.position[1] - e.position[0]),
            e.position[0],
            e.type,
        ),
    )
    kept: list[NerEntity] = []
    for entity in ranked:
        exact_duplicate = any(
            entity.position == other.position and entity.type == other.type
            for other in kept
        )
        overlap = any(
            entity.position[0] < other.position[1]
            and entity.position[1] > other.position[0]
            for other in kept
        )
        if not exact_duplicate and not overlap:
            kept.append(entity)
    return sorted(kept, key=lambda e: (e.position[0], e.position[1], e.type))


def deterministic_cleanup(raw_text: str, entities: list[NerEntity]) -> tuple[list[NerEntity], list[dict]]:
    logs: list[dict] = []
    cleaned: list[NerEntity] = []
    for entity in entities:
        if not _exact(raw_text, entity):
            logs.append({"status": "drop", "reason": "invalid_exact_span", "text": entity.text})
            continue
        drop_reason = _is_hard_negative(entity)
        if drop_reason:
            logs.append({"status": "drop", "reason": drop_reason, "text": entity.text})
            continue
        repaired = _repair_boundary(raw_text, entity)
        if repaired.position != entity.position:
            logs.append({"status": "repair", "reason": "deterministic_boundary", "before": entity.text,
                         "after": repaired.text, "position": list(repaired.position)})
        repaired = _repair_assertion_scope(raw_text, repaired)
        cleaned.append(repaired)
    # Re-check repaired candidates before overlap resolution.
    cleaned = [entity for entity in cleaned if _exact(raw_text, entity)
               and _is_hard_negative(entity) is None]
    return _resolve_overlaps(cleaned), logs


def _medication_list_entities(raw_text: str) -> list[NerEntity]:
    """Recover medication spans from a numbered pre-admission list.

    Text after ``điều trị`` is deliberately left to NER/LLMs: it may be a
    symptom or a diagnosis and must not be classified from a memorized phrase.
    """
    header = _DRUG_HEADER_RE.search(raw_text)
    if not header:
        return []
    markers = list(_NUMBERED_ITEM_RE.finditer(raw_text, header.end()))
    if not markers:
        return []
    result: list[NerEntity] = []
    for index, marker in enumerate(markers):
        item_start = marker.end()
        item_end = markers[index + 1].start() if index + 1 < len(markers) else len(raw_text)
        item = raw_text[item_start:item_end].strip()
        item_end = item_start + len(item)
        if not item:
            continue
        indication = re.search(r"\s+điều\s+trị\s+", item, flags=re.I)
        drug_text = item[:indication.start()].strip() if indication else item
        drug_end = item_start + len(drug_text)
        result.append(NerEntity(drug_text, "THUỐC", ["isHistorical"],
                                (item_start, drug_end), score=1.0))
    return result


def apply_clinical_rules(raw_text: str, entities: list[NerEntity]) -> tuple[list[NerEntity], list[dict]]:
    """Cleanup model candidates and inject only high-precision rule recovery."""
    cleaned, logs = deterministic_cleanup(raw_text, entities)
    med_entities = _medication_list_entities(raw_text)
    if med_entities:
        # Structured parsing is authoritative only for each drug span. Keep
        # NER mentions in the indication text that follows it.
        cleaned = [
            entity for entity in cleaned
            if not any(
                entity.position[0] < drug.position[1]
                and entity.position[1] > drug.position[0]
                for drug in med_entities
            )
        ]
        cleaned.extend(med_entities)
        cleaned.sort(key=lambda e: (e.position[0], e.position[1]))
        logs.append({"status": "recover", "reason": "pre_admission_medication_list",
                     "count": len(med_entities)})
    return cleaned, logs
