"""Làm sạch text đầu vào + convert word-tag BIO -> entity + dedup/merge
kết quả giữa các chunk overlap.

LƯU Ý: bản `clean_text_for_inference` dưới đây là bản ĐẦY ĐỦ (từ cell 7
của predict_ner_crf_final.ipynb). Notebook gốc có 1 bản rút gọn định
nghĩa SAU đó (cell 15) vô tình ghi đè bản này — nghĩa là lúc chạy
predict() thật sự trong notebook, các rule xử lý case bẩn (VS98.3,
dấu câu dính chữ, v.v.) KHÔNG được áp dụng. Ở đây dùng lại bản đầy đủ
vì rõ ràng đó mới là bản có chủ đích (xem markdown mô tả ở đầu notebook).
"""

from __future__ import annotations

import re

ASSERTION_PRIORITY = ["isNegated", "isFamily", "isHistorical"]


def clean_text_for_inference(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Thêm space sau dấu câu khi bị dính chữ, không phá số thập phân
    text = re.sub(r'(?<!\d)\.(?=[^\s\d])', '. ', text)
    text = re.sub(r'(?<!\d),(?=[^\s\d])', ', ', text)
    text = re.sub(r'([:;])(?=\S)', r'\1 ', text)

    # Sửa lại decimal lỡ bị tách ở bước trên
    text = re.sub(r'(?<=\d)\.\s+(?=\d)', '.', text)
    text = re.sub(r'(?<=\d),\s+(?=\d)', ',', text)

    # Tách chữ-số dính liền kiểu chỉ số sinh tồn viết tắt
    # ("VS98.3 12987 56 18 99RA" -> "VS 98.3 12987 56 18 99 RA")
    # Chỉ trigger khi >=2 chữ HOA liên tiếp cạnh số, để không đụng B12, T2, O2.
    text = re.sub(r'([A-ZĐ]{2,})(?=\d)', r'\1 ', text)
    text = re.sub(r'(?<=\d)([A-ZĐ]{2,})', r' \1', text)

    # Tách 1 số ranh giới mất khoảng trắng làm BIO word-level bất khả thi.
    text = re.sub(
        r'(?i)\b(bị|đang|không|chưa|có)(?=(?:chảy|đau|sốt|ho|khó|phù|nôn|tiểu|vàng|ban)\b)',
        r'\1 ',
        text,
    )
    text = re.sub(r'(?i)(cảm\s+giác)(?=(?:khó|đau|nóng|tê|rợn)\b)', r'\1 ', text)
    text = re.sub(r'(?i)\b(CT|MRI|ECG)(?=(?:chưa|không|ghi|cho|phát)\b)', r'\1 ', text)
    text = re.sub(
        r'(?i)(?<=\d)(?=(?:bạch\s*cầu|hồng\s*cầu|tiểu\s*cầu|kali|natri|clo)\b)',
        ' ',
        text,
    )

    # Xóa bullet chỉ ở đầu dòng, không đụng gạch trong từ (x-quang, 88-92)
    text = re.sub(r'(?m)^\s*[-•]\s+', '', text)

    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)

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