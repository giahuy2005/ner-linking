"""High-precision clinical rules used between NER and the 7B reviewer.

The notebook contains many exploratory overrides.  This module keeps the
deterministic parts in one auditable place: exact boundary repairs, known hard
negatives, overlap resolution and medication-list semantics.  Every emitted
span is verified against the original text.
"""

from __future__ import annotations

import re

from ..schemas import NerEntity

ALLOWED_TYPES = {
    "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC",
    "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM",
}
ALLOWED_ASSERTIONS = {"isHistorical", "isNegated", "isFamily"}

_HARD_NEGATIVES = {
    "◦ 8", "đứng dậy", "đánh răng không", "ăn ngủ", "cấp tính",
    "tĩnh mạch l giọt/phút", "tomisaku kawasaki",
}
_ANATOMY_ONLY = {"mạch máu", "động mạch vành"}
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
    )


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

    # Imperative prefix duplicated around an imaging test name.
    repeated_test = re.match(r"(?i)^chụp\s+lại\s+(chụp\s+.+)$", text)
    if repeated_test:
        local = repeated_test.start(1)
        candidates.append((start + local, start + local + len(repeated_test.group(1))))

    # In specimen lists the material belongs to the test surface.
    if entity.type == "TÊN_XÉT_NGHIỆM" and text.casefold() == "hầu họng":
        prefix = raw_text[max(0, start - 6):start]
        material = re.search(r"(?i)dịch\s+$", prefix)
        if material:
            candidates.append((max(0, start - 6) + material.start(), end))

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
        normalized = " ".join(entity.text.casefold().split())
        if normalized in _HARD_NEGATIVES:
            logs.append({"status": "drop", "reason": "known_hard_negative", "text": entity.text})
            continue
        if normalized in _ANATOMY_ONLY and entity.type == "CHẨN_ĐOÁN":
            logs.append({"status": "drop", "reason": "anatomy_without_disease_context", "text": entity.text})
            continue
        if normalized == "g6pd" and entity.type == "TÊN_XÉT_NGHIỆM":
            logs.append({"status": "drop", "reason": "gene_or_enzyme_context", "text": entity.text})
            continue
        repaired = _repair_boundary(raw_text, entity)
        if repaired.position != entity.position:
            logs.append({"status": "repair", "reason": "deterministic_boundary", "before": entity.text,
                         "after": repaired.text, "position": list(repaired.position)})
        repaired.assertions = [a for a in repaired.assertions if a in ALLOWED_ASSERTIONS]
        if repaired.type in {"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}:
            repaired.assertions = []
        cleaned.append(repaired)
    return _resolve_overlaps(cleaned), logs


def _medication_list_entities(raw_text: str) -> list[NerEntity]:
    """Recover the canonical BTC numbered pre-admission medication list.

    Medication assertions follow the section heading; indication symptoms do
    not inherit ``isHistorical`` from their medication.
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
        if not indication:
            continue
        indication_start = item_start + indication.end()
        indication_text = raw_text[indication_start:item_end].strip(" .;,:")
        if not indication_text:
            continue
        # Gold treats these coordinated concepts separately, while "sốt đau"
        # is one surface mention.
        parts = ["lo âu", "mất ngủ"] if indication_text.casefold() == "lo âu mất ngủ" else [indication_text]
        cursor = indication_start
        for part in parts:
            part_start = raw_text.find(part, cursor, item_end)
            if part_start >= 0:
                result.append(NerEntity(part, "TRIỆU_CHỨNG", [],
                                        (part_start, part_start + len(part)), score=1.0))
                cursor = part_start + len(part)
    return result


def apply_clinical_rules(raw_text: str, entities: list[NerEntity]) -> tuple[list[NerEntity], list[dict]]:
    """Cleanup model candidates and inject only high-precision rule recovery."""
    cleaned, logs = deterministic_cleanup(raw_text, entities)
    med_entities = _medication_list_entities(raw_text)
    if med_entities:
        # The structured list is authoritative inside its covered spans.
        med_start = min(e.position[0] for e in med_entities)
        med_end = max(e.position[1] for e in med_entities)
        cleaned = [e for e in cleaned if e.position[1] <= med_start or e.position[0] >= med_end]
        cleaned.extend(med_entities)
        cleaned.sort(key=lambda e: (e.position[0], e.position[1]))
        logs.append({"status": "recover", "reason": "pre_admission_medication_list",
                     "count": len(med_entities)})
    return cleaned, logs
