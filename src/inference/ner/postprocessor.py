"""Offset-safe BIO helpers synchronized with the final notebook predictor."""

from __future__ import annotations

import re

ASSERTION_PRIORITY = ["isNegated", "isFamily", "isHistorical"]


def get_assertion_threshold(threshold, label: str) -> float:
    """Support one global threshold or notebook-style per-label thresholds."""
    if isinstance(threshold, dict):
        return float(threshold.get(label, 0.5))
    return float(threshold)


def clean_text_for_inference(text: str) -> str:
    """Do not mutate input; every returned position remains a raw-text offset."""
    return text


def get_label(id2label: dict, idx: int) -> str:
    if idx in id2label:
        return id2label[idx]
    return id2label[str(idx)]


def collapse_assertions(assertions: list[str]) -> list[str]:
    """Ép 1 assertion/entity theo priority — dùng nếu format submit yêu cầu
    giá trị đơn (BTC ví dụ chỉ có 1 assertion/entity)."""
    assertions = assertions or []
    for a in ASSERTION_PRIORITY:
        if a in assertions:
            return [a]
    return []


def repair_bio_tags(tags: list[str]) -> list[str]:
    """Nếu model sinh I-X mà trước đó không phải B/I-X thì đổi thành B-X
    (CRF transition matrix vẫn có thể sinh sequence bất hợp lệ hiếm khi)."""
    fixed = []
    prev_type = None

    for tag in tags:
        if tag == "O":
            fixed.append(tag)
            prev_type = None
            continue
        if tag.startswith("B-"):
            ent_type = tag[2:]
            fixed.append(tag)
            prev_type = ent_type
            continue
        if tag.startswith("I-"):
            ent_type = tag[2:]
            fixed.append(tag if prev_type == ent_type else "B-" + ent_type)
            prev_type = ent_type
            continue
        fixed.append("O")
        prev_type = None

    return fixed


def extract_entities_from_word_tags(tags: list[str], line_ids: list[int] | None = None):
    """Không cho merge entity qua ranh giới newline/bullet line."""
    tags = repair_bio_tags(tags)

    entities = []
    i = 0
    n = len(tags)

    while i < n:
        tag = tags[i]
        if not tag.startswith("B-"):
            i += 1
            continue

        ent_type = tag[2:]
        start = i
        start_line = line_ids[i] if line_ids is not None else None

        j = i + 1
        while j < n and tags[j] == f"I-{ent_type}":
            if line_ids is not None and line_ids[j] != start_line:
                break
            j += 1

        entities.append({"word_start": start, "word_end": j, "type": ent_type})
        i = j

    return entities


def drop_nested_entities(results: list[dict]) -> list[dict]:
    """Loại entity mà span nằm hoàn toàn trong 1 entity KHÁC cùng type đã
    giữ lại (xử lý overlap boundary lệch giữa 2 chunk, khác dedup exact-key
    ở merge_chunk_results — cái đó chỉ bắt trùng offset y hệt)."""
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)

    kept = []
    for ents in by_type.values():
        ents = sorted(ents, key=lambda r: (r["char_start"], -(r["char_end"] - r["char_start"])))
        local_kept: list[dict] = []
        for r in ents:
            if local_kept and r["char_start"] >= local_kept[-1]["char_start"] and r["char_end"] <= local_kept[-1]["char_end"]:
                continue
            local_kept.append(r)
        kept.extend(local_kept)
    return kept


def merge_chunk_results(all_results: list[dict]) -> list[dict]:
    """Dedup entity trùng offset+type xuất hiện ở nhiều chunk overlap
    (giữ bản score cao hơn, hoặc nhiều assertion hơn nếu score bằng nhau),
    rồi loại entity lồng nhau và sort theo vị trí."""
    best: dict[tuple[int, int, str], dict] = {}

    for r in all_results:
        key = (r["char_start"], r["char_end"], r["type"])
        if key not in best:
            best[key] = r
            continue
        old = best[key]
        if r["score"] > old["score"] + 1e-6:
            best[key] = r
        elif abs(r["score"] - old["score"]) < 1e-6 and len(r["assertions"]) > len(old["assertions"]):
            best[key] = r

    results = list(best.values())
    results = drop_nested_entities(results)
    results.sort(key=lambda x: (x["char_start"], x["char_end"]))
    return results
