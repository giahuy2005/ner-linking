"""Audit or repair truncated drug route/frequency spans in synthetic JSONL files."""

import argparse
from collections import Counter
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_gen.gen_reject import (
    assign_spans_sequential,
    autofix_drug_regimen_span,
    sort_entities_by_text_order,
)


def remove_entities_nested_in_expanded_drugs(entities, input_text, expanded_texts, logs):
    """Remove only overlaps introduced by this repair, leaving unrelated overlaps untouched."""
    spans = assign_spans_sequential(entities, input_text)
    if spans is None:
        return entities
    remaining = Counter(expanded_texts)
    expanded_indexes = []
    for index, (_start, _end, entity) in enumerate(spans):
        text = entity["text"]
        if entity["type"] == "THUỐC" and remaining[text] > 0:
            expanded_indexes.append(index)
            remaining[text] -= 1

    drop = set()
    for drug_index in expanded_indexes:
        drug_start, drug_end, drug = spans[drug_index]
        for index, (start, end, entity) in enumerate(spans):
            if index == drug_index:
                continue
            if drug_start <= start and end <= drug_end:
                drop.add(index)
                logs.append(
                    f"[fix-drug-span-nested] giữ '{drug['text']}', loại '{entity['text']}'"
                )
    return [entity for index, entity in enumerate(entities) if index not in drop]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand THUỐC entities through contiguous route/frequency text, following "
            "the official BTC medication-span examples."
        )
    )
    parser.add_argument("inputs", nargs="+", help="Input JSONL file(s).")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Atomically replace each input file. Without this flag, only audit.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=10,
        help="Maximum before/after examples per file. Default: 10.",
    )
    return parser.parse_args()


def audit_or_fix(path: Path, in_place: bool, show: int) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    temp_path = path.with_name(f"{path.name}.drug-span.tmp")
    stats = {
        "samples": 0,
        "changed_samples": 0,
        "changed_entities": 0,
        "removed_nested_entities": 0,
    }
    examples = []

    output = temp_path.open("w", encoding="utf-8", newline="\n") if in_place else None
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, 1):
                if not raw_line.strip():
                    if output is not None:
                        output.write(raw_line)
                    continue

                stats["samples"] += 1
                record = json.loads(raw_line)
                entities = record.get("entities", [])
                before = [entity.get("text") for entity in entities]
                logs = []
                fixed = autofix_drug_regimen_span(entities, record["input_text"], logs)
                expanded = [entity.get("text") for entity in fixed]

                changes = [
                    (old, new)
                    for old, new in zip(before, expanded)
                    if old != new
                ]
                if changes:
                    fixed = sort_entities_by_text_order(fixed, record["input_text"], logs)
                    before_nested_count = len(fixed)
                    fixed = remove_entities_nested_in_expanded_drugs(
                        fixed,
                        record["input_text"],
                        [new for _old, new in changes],
                        logs,
                    )
                    stats["removed_nested_entities"] += before_nested_count - len(fixed)
                    stats["changed_samples"] += 1
                    stats["changed_entities"] += len(changes)
                    record["entities"] = fixed
                    for old, new in changes:
                        if new not in record["input_text"]:
                            raise ValueError(
                                f"{path}:{line_number}: expanded span is not a substring: {new!r}"
                            )
                        if len(examples) < show:
                            examples.append((line_number, old, new))

                if output is not None:
                    if changes:
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    else:
                        output.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
    except Exception:
        if output is not None:
            output.close()
        temp_path.unlink(missing_ok=True)
        raise
    else:
        if output is not None:
            output.close()
            os.replace(temp_path, path)

    print(
        f"{path}: samples={stats['samples']}, changed_samples={stats['changed_samples']}, "
        f"changed_drug_entities={stats['changed_entities']}, "
        f"removed_nested_entities={stats['removed_nested_entities']}, "
        f"mode={'fixed' if in_place else 'audit'}"
    )
    for line_number, old, new in examples:
        print(f"  line {line_number}: {old!r} -> {new!r}")
    return stats


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    for value in args.inputs:
        audit_or_fix(Path(value), args.in_place, args.show)


if __name__ == "__main__":
    main()
