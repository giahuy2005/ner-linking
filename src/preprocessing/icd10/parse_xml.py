#!/usr/bin/env python3
"""
Parse file ICD-10 dạng ClaML (icd102019en.xml) -> icd10_clean.jsonl

Mỗi dòng output là 1 record cho 1 mã bệnh (Class kind="category"):
{
  "code": "A06.6",
  "preferred_en": "Amoebic brain abscess",
  "preferred_en_raw": "Amoebic brain abscess (G07)",     # chỉ có nếu khác preferred_en
  "inclusions_en_raw": ["Amoebic abscess of brain (and liver)(and lung)"],
  "aliases_en": [
      "Amoebic abscess of brain",
      "Amoebic abscess of brain and liver",
      "Amoebic abscess of brain and lung",
      "Amoebic abscess of brain, liver and lung"
  ],
  "references": ["G07"],
  "exclusions_en": [],
  "parent": "A06",
  "children": []
}

Chỉ giữ Class kind="category". Bỏ qua chapter/block/Modifier/ModifierClass.

2 vấn đề được xử lý riêng so với bản v1:

1. <Reference class="in brackets">CODE</Reference> (dùng để trỏ chéo sang
   1 mã khác, hay gặp ở preferred/exclusion, VD "Amoebic brain
   abscess<Reference>G07</Reference>") KHÔNG được coi là 1 phần tên bệnh.
   -> preferred_en: bản sạch, không có mã tham chiếu.
   -> preferred_en_raw: bản gốc đầy đủ (chỉ lưu nếu khác bản sạch).
   -> references: list các mã code được trỏ tới (dùng để mở rộng
      candidate khi cần, không dùng để embedding).

2. Pattern "(and X)(and Y)" trong inclusion (VD "Amoebic abscess of
   brain (and liver)(and lung)") là dạng rút gọn của 3 cụm:
   "...of brain", "...of brain and liver", "...of brain and lung",
   "...of brain, liver and lung". Giữ nguyên bản gốc trong
   inclusions_en_raw, đồng thời sinh ra các biến thể "sạch" cho
   aliases_en để dùng làm câu embedding riêng lẻ (không để 1 câu dài
   dạng "(and X)(and Y)" lấn át tên chính khi đưa vào SapBERT).

Không dịch / không tạo alias tiếng Việt ở bước này -- việc đó nên tách
thành 1 script riêng (dịch qua LLM/MT) đọc từ file clean.jsonl này,
vì dịch 11k+ record cần rate-limit / batch / caching riêng, gộp chung
vào đây sẽ làm bước parse (chạy trong vài giây) bị phụ thuộc mạng.
"""

import json
import os
import re
from lxml import etree
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

    
INPUT_XML = BASE_DIR / "data/raw/icd10/icd102019en.xml"
OUTPUT_JSONL = BASE_DIR / "data/processed/icd10/icd10_clean.jsonl"


RUBRIC_PREFERRED = {"preferred"}
RUBRIC_PREFERRED_FALLBACK = {"preferred", "preferredLong"}
RUBRIC_INCLUSION = {"inclusion"}
RUBRIC_EXCLUSION = {"exclusion"}

AND_GROUP_RE = re.compile(r"\(and ([^()]+)\)")

# --- Lọc alias hỏng ---
#
# Có 2 kiểu noise không nên đưa vào aliases_en (dùng cho embedding),
# nhưng vẫn phải giữ nguyên trong inclusions_en_raw:
#
# 1) Reference nằm GIỮA câu (không phải dạng ngoặc cuối câu như
#    "Amoebic brain abscess (G07)"), khi tách mã ra thì câu đứt gãy.
#    VD: "conditions classifiable to <Ref>I05.0</Ref> and
#    <Ref>I05.2-I05.9</Ref>, whether specified as rheumatic or not"
#    -> sau khi tách mã: "conditions classifiable to and , whether
#    specified as rheumatic or not" (câu vô nghĩa, mất chủ thể).
#    Nhận diện bằng dấu vết còn sót: "to and", "and ," (khoảng trắng
#    thừa/giới từ nối trực tiếp với dấu phẩy).
#
# 2) Câu hướng dẫn mã hóa (coding-hint) lẫn trong inclusion, không phải
#    tên đồng nghĩa của bệnh. VD: "haematuria: with morphological
#    lesion specified in .0-.8 before N00.-". Đây là text thường,
#    KHÔNG dùng thẻ <Reference> nên bước tách mã ở trên không xử lý
#    được. Nhận diện qua pattern "specified in .N" (chỉ mục subcode
#    dạng ".0-.8") hoặc "before N00.-" kiểu tham chiếu chuỗi mã.
BAD_ALIAS_PATTERNS = [
    re.compile(r"\bto\s+and\b"),        # "classifiable to and" - đứt câu do gộp 2 reference liền nhau
    re.compile(r"\band\s*,"),           # "and ," - artifact khi reference bị xóa giữa câu
    re.compile(r"specified in \.\d"),   # "specified in .0-.8 ..." - coding-hint, không phải tên bệnh
]

# Giữ thêm blocklist tường minh làm lưới an toàn cho đúng 4 câu đã biết,
# phòng trường hợp regex ở trên vô tình không khớp do khác biệt nhỏ.
BAD_ALIASES_EXACT = {
    "conditions classifiable to and , whether specified as rheumatic or not",
    "haematuria with morphological lesion specified in .0-.8 before N00.-",
    "nephropathy NOS and renal disease NOS with morphological lesion specified in .0-.8 before N00.-",
    "proteinuria (isolated)(orthostatic)(persistent) with morphological lesion specified in .0-.8 before N00.-",
}


def is_bad_alias(alias):
    if alias in BAD_ALIASES_EXACT:
        return True
    return any(pat.search(alias) for pat in BAD_ALIAS_PATTERNS)


def normalize_ws(text):
    return re.sub(r"\s+", " ", text).strip()


def has_fragment_descendant(label_el):
    return label_el.find(".//Fragment") is not None


def parse_label(label_el):
    """
    Trả về (clean_text, raw_text, references) cho 1 <Label>.

    - clean_text: text hiển thị, KHÔNG bao gồm nội dung <Reference>.
    - raw_text: text đầy đủ như trong XML, riêng <Reference>CODE</Reference>
      được bọc thành " (CODE)" cho dễ đọc.
    - references: list mã code lấy từ các thẻ <Reference>, DUYỆT ĐỆ QUY
      toàn bộ cây con (Reference hay nằm lồng bên trong <Fragment>,
      không phải con trực tiếp của <Label> -- VD:
      <Fragment>arthritis<Reference>M01.3</Reference></Fragment> -- nên
      không thể chỉ duyệt 1 cấp).

    Xử lý riêng <Fragment type="list">: ClaML dùng cặp Fragment kiểu
    "Salmonella:" + "arthritis" để ghép thành enum "Salmonella:
    arthritis / meningitis / ...". Với alias dùng để nhúng câu, ta bỏ
    dấu ':' để có câu tự nhiên "Salmonella arthritis" thay vì
    "Salmonella: arthritis".
    """
    references = []

    def build(el):
        clean_parts = []
        raw_parts = []
        if el.text:
            clean_parts.append(el.text)
            raw_parts.append(el.text)
        for child in el:
            tag = etree.QName(child).localname
            if tag == "Reference":
                code = normalize_ws("".join(child.itertext()))
                if code:
                    references.append(code)
                raw_parts.append(f" ({code})")
                # KHÔNG thêm vào clean_parts.
            else:
                c_clean, c_raw = build(child)
                clean_parts.append(c_clean)
                raw_parts.append(c_raw)
            if child.tail:
                clean_parts.append(child.tail)
                raw_parts.append(child.tail)
        return "".join(clean_parts), "".join(raw_parts)

    clean_text, raw_text = build(label_el)

    if has_fragment_descendant(label_el):
        # "Salmonella:  arthritis" -> "Salmonella arthritis"
        clean_text = re.sub(r":\s+", " ", clean_text)
        raw_text = re.sub(r":\s+", " ", raw_text)

    return normalize_ws(clean_text), normalize_ws(raw_text), references


def expand_and_groups(text):
    """
    "Amoebic abscess of brain (and liver)(and lung)"
    -> base = "Amoebic abscess of brain"
    -> additions = ["liver", "lung"]
    """
    additions = [normalize_ws(a) for a in AND_GROUP_RE.findall(text)]
    base = AND_GROUP_RE.sub("", text)
    base = normalize_ws(base)
    return base, additions


def build_aliases(base, additions):
    """
    base="Amoebic abscess of brain", additions=["liver","lung"] ->
    [
      "Amoebic abscess of brain",
      "Amoebic abscess of brain and liver",
      "Amoebic abscess of brain and lung",
      "Amoebic abscess of brain, liver and lung",
    ]
    Nếu không có additions thì chỉ trả về [base].
    """
    aliases = [base] if base else []
    for add in additions:
        aliases.append(normalize_ws(f"{base} and {add}"))
    if len(additions) >= 2:
        combined = f"{base}, " + ", ".join(additions[:-1]) + f" and {additions[-1]}"
        aliases.append(normalize_ws(combined))

    seen = set()
    out = []
    for a in aliases:
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return out


def get_rubric_label_els(class_el, kind_set, lang="en"):
    """Lấy các <Label xml:lang="en"> trực tiếp trong Rubric kind thuộc kind_set."""
    results = []
    for rubric in class_el.findall("Rubric"):
        if rubric.get("kind") not in kind_set:
            continue
        for label in rubric.findall("Label"):
            xml_lang = label.get("{http://www.w3.org/XML/1998/namespace}lang")
            if xml_lang is not None and xml_lang != lang:
                continue
            results.append(label)
    return results


def main():
    print(f"Đang parse {INPUT_XML} ...")
    parser = etree.XMLParser(dtd_validation=False, load_dtd=False, resolve_entities=False)
    tree = etree.parse(INPUT_XML, parser)
    root = tree.getroot()

    # Pass 1: build children map (code cha -> list code con) dựa trên
    # SuperClass thực tế của từng Class category (đáng tin hơn SubClass
    # khai ở class cha, vì 1 số class cha liệt kê SubClass không khớp
    # 100% với category con thực tế trong file).
    children_map = {}
    category_classes = []
    for class_el in root.findall("Class"):
        if class_el.get("kind") != "category":
            continue
        category_classes.append(class_el)
        code = class_el.get("code")
        super_el = class_el.find("SuperClass")
        if super_el is not None:
            children_map.setdefault(super_el.get("code"), []).append(code)

    print(f"Tổng số Class kind=category: {len(category_classes)}")

    records = []
    n_no_preferred = 0
    n_with_reference = 0
    n_with_and_group = 0

    for class_el in category_classes:
        code = class_el.get("code")

        # --- preferred ---
        pref_labels = get_rubric_label_els(class_el, RUBRIC_PREFERRED)
        if not pref_labels:
            pref_labels = get_rubric_label_els(class_el, RUBRIC_PREFERRED_FALLBACK)

        preferred_en = ""
        preferred_en_raw = None
        all_references = []

        if pref_labels:
            clean, raw, refs = parse_label(pref_labels[0])
            # Phòng trường hợp hiếm: preferred cũng có pattern "(and X)"
            base, additions = expand_and_groups(clean)
            preferred_en = base if base else clean
            if raw != preferred_en:
                preferred_en_raw = raw
            all_references.extend(refs)
        if not preferred_en:
            n_no_preferred += 1
        if preferred_en_raw:
            n_with_reference += 1

        # --- inclusion ---
        inclusions_en_raw = []
        aliases_en = []
        incl_labels = get_rubric_label_els(class_el, RUBRIC_INCLUSION)
        for label_el in incl_labels:
            clean, raw, refs = parse_label(label_el)
            inclusions_en_raw.append(raw)
            all_references.extend(refs)
            base, additions = expand_and_groups(clean)
            if additions:
                n_with_and_group += 1
            for alias in build_aliases(base, additions):
                if alias not in aliases_en and not is_bad_alias(alias):
                    aliases_en.append(alias)

        # Nếu preferred có and-group, cũng đẩy các biến thể vào aliases_en
        if pref_labels:
            pref_base, pref_additions = expand_and_groups(
                parse_label(pref_labels[0])[0]
            )
            if pref_additions:
                for alias in build_aliases(pref_base, pref_additions):
                    if alias not in aliases_en and alias != preferred_en and not is_bad_alias(alias):
                        aliases_en.append(alias)

        # --- exclusion (giữ nguyên dạng raw gộp mã tham chiếu, không dùng
        # để embedding positive nên không cần tách and-group / reference) ---
        exclusions_en = []
        for label_el in get_rubric_label_els(class_el, RUBRIC_EXCLUSION):
            _, raw, _ = parse_label(label_el)
            exclusions_en.append(raw)

        # dedup references, giữ thứ tự
        seen_ref = set()
        references = []
        for r in all_references:
            if r not in seen_ref:
                seen_ref.add(r)
                references.append(r)

        super_el = class_el.find("SuperClass")
        parent = super_el.get("code") if super_el is not None else None
        children = children_map.get(code, [])

        record = {
            "code": code,
            "preferred_en": preferred_en,
        }
        if preferred_en_raw:
            record["preferred_en_raw"] = preferred_en_raw
        record["inclusions_en_raw"] = inclusions_en_raw
        record["aliases_en"] = aliases_en
        record["references"] = references
        record["exclusions_en"] = exclusions_en
        record["parent"] = parent
        record["children"] = children

        records.append(record)

    print(f"Số record không có preferred_en: {n_no_preferred}")
    print(f"Số record preferred_en có mã tham chiếu (Reference): {n_with_reference}")
    print(f"Số lượt inclusion có pattern (and X): {n_with_and_group}")

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Đã ghi {len(records)} record vào {OUTPUT_JSONL}")

    print("\n--- Mẫu kiểm tra ---")
    for rec in records:
        if rec["code"] in ("K21.0", "A06.6"):
            print(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()