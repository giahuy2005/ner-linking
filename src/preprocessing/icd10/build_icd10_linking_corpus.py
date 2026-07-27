#!/usr/bin/env python3
"""Build the clean, flattened ICD-10 term corpus used for embedding/FAISS.

The merged ontology remains unchanged.  This script writes one term per JSONL
line and writes translation collisions/cross-code ambiguities to separate audit
files.  It intentionally does not reconstruct Vietnamese aliases from
``inclusion_notes_vi`` because that would destroy the existing EN-VI alignment.

Usage:
    python src/preprocessing/icd10/build_icd10_linking_corpus.py
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "icd10" / "icd10_merged.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "icd10" / "icd10_embedding_terms.jsonl"
DEFAULT_COLLISIONS = (
    PROJECT_ROOT / "data" / "processed" / "icd10" / "icd10_translation_collisions.jsonl"
)
DEFAULT_AMBIGUITIES = (
    PROJECT_ROOT / "data" / "processed" / "icd10" / "icd10_cross_code_ambiguities.jsonl"
)
DEFAULT_EXCLUDED_NOTES = (
    PROJECT_ROOT / "data" / "processed" / "icd10" / "icd10_excluded_coding_notes.jsonl"
)
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "icd10" / "icd10_embedding_report.json"

WHITESPACE_RE = re.compile(r"\s+")
# The Vietnamese spreadsheet also contains 52 literal ``?`` replacement
# characters exactly where catalogue markers were corrupted (at the beginning
# of Z80-Z91 labels, plus S11.9).  No English/alias term contains a question
# mark, so all three characters are source artefacts in this dataset.
SOURCE_ARTIFACT_RE = re.compile(r"[?†*]")
PRIVATE_CHAR_MAP = {
    "\uf061": "alpha",
    "\uf062": "beta",
}
RESERVED_U_CODE_RE = re.compile(r"^U(?:1[3-9]|[2-4][0-9])(?:\.|$)", re.IGNORECASE)

# References seen in ICD labels include G30.-, C80† and ordinary codes such as
# G02.0*.  Only a trailing parenthesised code is removed; meaningful medical
# parentheticals are retained.
CODE_REFERENCE = r"[A-Z]\d{2}(?:(?:\.[0-9A-Z]+)|(?:\.\-))?[†*]?"
REFERENCE_SUFFIX_RE = re.compile(
    rf"\s*\(\s*{CODE_REFERENCE}(?:\s*[,;]\s*{CODE_REFERENCE})*\s*\)\s*$",
    re.IGNORECASE,
)

NOTE_PREFIXES = tuple(
    unicodedata.normalize("NFC", value).casefold()
    for value in (
        "bất kỳ tình trạng nào trong",
        "bất kỳ bệnh nào trong",
        "các tình trạng liệt kê trong",
        "các bệnh được liệt kê trong",
        "any condition in",
        "conditions listed in",
    )
)

# Scope/instruction sentences that survived the earlier source QC.  They are
# valid ontology notes but poor natural-language retrieval terms.  Filtering is
# performed after IDs are assigned so the audit file can point to the exact
# term that would otherwise have been embedded.
EMBEDDING_SCOPE_NOTE_PREFIXES = tuple(
    unicodedata.normalize("NFC", value).casefold()
    for value in (
        "any condition listed",
        "any condition classified",
        "conditions listed",
        "diseases listed",
        "examples of the use of this category",
        "combination of disorders specified",
        "combination of conditions listed",
        "infestation classifiable to",
        "bất kỳ tình trạng nào liệt kê",
        "bất kỳ bệnh nào liệt kê",
        "các tình trạng liệt kê",
        "các bệnh được liệt kê",
        "ví dụ sử dụng mã này",
        "kết hợp các rối loạn được mô tả",
        "kết hợp các tình trạng liệt kê",
    )
)

# Corrections are deliberately explicit and auditable.  The first six repair
# malformed brackets in the official Vietnamese catalogue.  H65.9 is the
# requested canonical wording for the target corpus.
PREFERRED_VI_CORRECTIONS = {
    "A87.1": "Viêm màng não do Adenovirus",
    "E63.0": "Thiếu acid béo cần thiết [EFA]",
    "G03.2": "Viêm màng não tái diễn lành tính [Mollaret]",
    "G12.0": "Teo cơ do tủy trẻ em, loại I [Werdnig - Hofman]",
    "G73.1": "Hội chứng Lambert-Eaton",
    "H65.9": "Viêm tai giữa không mủ, không xác định",
    "M09.1": "Viêm khớp trẻ em trong bệnh Crohn (viêm ruột đoạn)",
    "Y51.4": "Chất chủ vận thụ thể alpha-adrenergic, không xếp loại ở nơi khác",
    "Y51.5": "Chất chủ vận thụ thể beta-adrenergic, không xếp loại ở nơi khác",
    "Y51.6": "Chất đối kháng thụ thể alpha-adrenergic, không xếp loại ở nơi khác",
    "Y51.7": "Chất đối kháng thụ thể beta-adrenergic, không xếp loại ở nơi khác",
}

TERM_FIELDS = ("term_id", "code", "text", "language", "term_type")


def clean_text(text: str) -> str:
    """Normalize text without changing its medical meaning."""
    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def normalized_key(text: str) -> str:
    """Comparison key used only for deduplication and audit grouping."""
    return clean_text(text).casefold()


def clean_embedding_text(text: str) -> str:
    """Clean a term and remove a terminal ICD cross-reference, if present."""
    value = clean_text(text)
    while True:
        cleaned = REFERENCE_SUFFIX_RE.sub("", value).strip()
        if cleaned == value:
            # Dagger/asterisk are ICD coding markers, not part of the natural
            # language surface form passed to the embedding model.
            cleaned = "".join(PRIVATE_CHAR_MAP.get(char, char) for char in cleaned)
            return clean_text(SOURCE_ARTIFACT_RE.sub("", cleaned))
        value = cleaned


def is_coding_note(text: str) -> bool:
    return normalized_key(text).startswith(NOTE_PREFIXES)


def is_embedding_coding_note(term: dict[str, str]) -> bool:
    return (
        term.get("term_type") == "inclusion"
        and normalized_key(term["text"]).startswith(EMBEDDING_SCOPE_NOTE_PREFIXES)
    )


def filter_embedding_coding_notes(
    terms: Iterable[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    kept: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for term in terms:
        if is_embedding_coding_note(term):
            excluded.append(
                {
                    "term_id": term["term_id"],
                    "reason": "coding_scope_note",
                    "text": term["text"],
                }
            )
        else:
            kept.append(term)
    return kept, excluded


def is_reserved_u_code(code: str) -> bool:
    return bool(RESERVED_U_CODE_RE.match(code))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON lỗi tại {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Record tại {path}:{line_number} không phải JSON object")
            rows.append(row)
    return rows


def validate_source(records: Iterable[dict[str, Any]]) -> None:
    seen_codes: set[str] = set()
    for row_number, record in enumerate(records, 1):
        code = record.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"Record {row_number} thiếu code hợp lệ")
        if code in seen_codes:
            raise ValueError(f"Code trùng trong input: {code}")
        seen_codes.add(code)

        for field in ("preferred_en", "preferred_vi"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"{code} thiếu {field}")
        for field in ("aliases_en", "aliases_vi"):
            value = record.get(field, [])
            if value is not None and not isinstance(value, list):
                raise ValueError(f"{code}.{field} phải là list")


def source_cleaning_stats(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    stats: Counter[str] = Counter()
    for record in records:
        values = [record["preferred_en"], record["preferred_vi"]]
        values.extend(record.get("aliases_en") or [])
        values.extend(record.get("aliases_vi") or [])
        for value in values:
            if not isinstance(value, str):
                continue
            normalized = unicodedata.normalize("NFC", value)
            if normalized != value:
                stats["source_non_nfc_strings_cleaned"] += 1
            if clean_text(value) != normalized.strip():
                stats["source_whitespace_strings_cleaned"] += 1
            if SOURCE_ARTIFACT_RE.search(value):
                stats["source_artifact_strings_cleaned"] += 1
            if any(char in PRIVATE_CHAR_MAP for char in value):
                stats["source_private_use_strings_cleaned"] += 1
    return dict(stats)


def preferred_text(record: dict[str, Any], language: str) -> str:
    if language == "vi" and record["code"] in PREFERRED_VI_CORRECTIONS:
        return PREFERRED_VI_CORRECTIONS[record["code"]]
    return clean_embedding_text(record[f"preferred_{language}"])


def add_term(
    terms: list[dict[str, str]],
    seen_within_code: set[tuple[str, str]],
    counters: dict[tuple[str, str], int],
    *,
    code: str,
    text: str,
    language: str,
    term_type: str,
) -> bool:
    text = clean_embedding_text(text)
    if not text or is_coding_note(text):
        return False

    dedup_key = (language, normalized_key(text))
    if dedup_key in seen_within_code:
        return False
    seen_within_code.add(dedup_key)

    counter_key = (language, term_type)
    index = counters[counter_key]
    counters[counter_key] += 1
    terms.append(
        {
            "term_id": f"{code}|{language}|{term_type}|{index}",
            "code": code,
            "text": text,
            "language": language,
            "term_type": term_type,
        }
    )
    return True


def find_translation_collisions(record: dict[str, Any]) -> list[dict[str, Any]]:
    aliases_en = record.get("aliases_en") or []
    aliases_vi = record.get("aliases_vi") or []
    mapped: dict[str, dict[str, Any]] = {}

    # Only the aligned portion is eligible; manual VI aliases appended later
    # have no English counterpart and are handled as term_type=alias.
    for text_en, text_vi in zip(aliases_en, aliases_vi):
        if not isinstance(text_en, str) or not isinstance(text_vi, str):
            continue
        clean_en = clean_embedding_text(text_en)
        clean_vi = clean_embedding_text(text_vi)
        if not clean_en or not clean_vi or is_coding_note(clean_en) or is_coding_note(clean_vi):
            continue
        key_vi = normalized_key(clean_vi)
        bucket = mapped.setdefault(key_vi, {"text_vi": clean_vi, "texts_en": {}})
        bucket["texts_en"].setdefault(normalized_key(clean_en), clean_en)

    collisions = []
    for bucket in mapped.values():
        texts_en = list(bucket["texts_en"].values())
        if len(texts_en) > 1:
            collisions.append(
                {
                    "type": "translation_collision",
                    "code": record["code"],
                    "text_vi": bucket["text_vi"],
                    "texts_en": texts_en,
                }
            )
    return collisions


def build_terms(
    records: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, int]]:
    terms: list[dict[str, str]] = []
    collisions: list[dict[str, Any]] = []
    stats: defaultdict[str, int] = defaultdict(int)

    for record in records:
        code = clean_text(record["code"])
        if is_reserved_u_code(code):
            stats["reserved_u_records_excluded"] += 1
            continue

        collisions.extend(find_translation_collisions(record))
        seen_within_code: set[tuple[str, str]] = set()
        counters: defaultdict[tuple[str, str], int] = defaultdict(int)

        for language in ("en", "vi"):
            text = preferred_text(record, language)
            if not add_term(
                terms,
                seen_within_code,
                counters,
                code=code,
                text=text,
                language=language,
                term_type="preferred",
            ):
                raise ValueError(f"Preferred term không hợp lệ sau cleaning: {code}/{language}")

        aliases_en = record.get("aliases_en") or []
        aliases_vi = record.get("aliases_vi") or []
        preferred_keys = {
            "en": normalized_key(preferred_text(record, "en")),
            "vi": normalized_key(preferred_text(record, "vi")),
        }

        for index, alias in enumerate(aliases_en):
            if not isinstance(alias, str):
                stats["invalid_aliases_excluded"] += 1
                continue
            clean_alias = clean_embedding_text(alias)
            if is_coding_note(clean_alias):
                stats["coding_notes_excluded"] += 1
                continue
            if normalized_key(clean_alias) == preferred_keys["en"]:
                stats["aliases_equal_preferred_excluded"] += 1
                continue
            if not add_term(
                terms,
                seen_within_code,
                counters,
                code=code,
                text=clean_alias,
                language="en",
                term_type="inclusion",
            ):
                stats["within_code_duplicates_excluded"] += 1

        for index, alias in enumerate(aliases_vi):
            if not isinstance(alias, str):
                stats["invalid_aliases_excluded"] += 1
                continue
            clean_alias = clean_embedding_text(alias)
            if is_coding_note(clean_alias):
                stats["coding_notes_excluded"] += 1
                continue
            if normalized_key(clean_alias) == preferred_keys["vi"]:
                stats["aliases_equal_preferred_excluded"] += 1
                continue
            term_type = "inclusion" if index < len(aliases_en) else "alias"
            if not add_term(
                terms,
                seen_within_code,
                counters,
                code=code,
                text=clean_alias,
                language="vi",
                term_type=term_type,
            ):
                stats["within_code_duplicates_excluded"] += 1

    stats["terms_written"] = len(terms)
    stats["translation_collision_groups"] = len(collisions)
    return terms, collisions, dict(stats)


def find_cross_code_ambiguities(terms: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for term in terms:
        key = (term["language"], normalized_key(term["text"]))
        bucket = groups.setdefault(
            key,
            {
                "normalized_text": key[1],
                "language": term["language"],
                "codes": set(),
            },
        )
        bucket["codes"].add(term["code"])

    ambiguities = []
    for bucket in groups.values():
        if len(bucket["codes"]) > 1:
            ambiguities.append(
                {
                    "normalized_text": bucket["normalized_text"],
                    "language": bucket["language"],
                    "codes": sorted(bucket["codes"]),
                }
            )
    return ambiguities


def validate_terms(terms: Iterable[dict[str, str]]) -> None:
    seen_ids: set[str] = set()
    seen_code_language_text: set[tuple[str, str, str]] = set()
    for row_number, term in enumerate(terms, 1):
        if tuple(term) != TERM_FIELDS:
            raise ValueError(f"Sai schema term tại dòng {row_number}: {tuple(term)}")
        if term["term_id"] in seen_ids:
            raise ValueError(f"term_id trùng: {term['term_id']}")
        seen_ids.add(term["term_id"])
        if term["text"] != clean_text(term["text"]):
            raise ValueError(f"Text chưa NFC/whitespace-clean: {term['term_id']}")
        key = (term["code"], term["language"], normalized_key(term["text"]))
        if key in seen_code_language_text:
            raise ValueError(f"Term trùng trong cùng code/language: {term['term_id']}")
        seen_code_language_text.add(key)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--collisions", type=Path, default=DEFAULT_COLLISIONS)
    parser.add_argument("--ambiguities", type=Path, default=DEFAULT_AMBIGUITIES)
    parser.add_argument("--excluded-notes", type=Path, default=DEFAULT_EXCLUDED_NOTES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    validate_source(records)

    terms, collisions, stats = build_terms(records)
    terms, excluded_notes = filter_embedding_coding_notes(terms)
    validate_terms(terms)
    ambiguities = find_cross_code_ambiguities(terms)

    write_jsonl(args.output, terms)
    write_jsonl(args.collisions, collisions)
    write_jsonl(args.ambiguities, ambiguities)
    write_jsonl(args.excluded_notes, excluded_notes)

    stats.update(
        {
            "source_records": len(records),
            "terms_written": len(terms),
            "coding_scope_notes_excluded": len(excluded_notes),
            "cross_code_ambiguity_groups": len(ambiguities),
            "terms_en": sum(term["language"] == "en" for term in terms),
            "terms_vi": sum(term["language"] == "vi" for term in terms),
            "preferred_terms": sum(term["term_type"] == "preferred" for term in terms),
            "inclusion_terms": sum(term["term_type"] == "inclusion" for term in terms),
            "manual_alias_terms": sum(term["term_type"] == "alias" for term in terms),
            "output": str(args.output),
        }
    )
    stats.update(source_cleaning_stats(records))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Input records: {len(records):,}")
    print(f"Embedding terms: {len(terms):,}")
    print(f"Coding scope notes excluded: {len(excluded_notes):,}")
    print(f"Translation collisions: {len(collisions):,}")
    print(f"Cross-code ambiguities: {len(ambiguities):,}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
