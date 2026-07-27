"""Audit or fix result entities that incorrectly contain multiple lab pairs."""

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_GEN_SRC = REPO_ROOT / "src" / "data_gen"
if str(DATA_GEN_SRC) not in sys.path:
    sys.path.insert(0, str(DATA_GEN_SRC))

from gen_reject import autofix_split_merged_lab, sort_entities_by_text_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split merged multi-pair lab results such as WBC:12,5; NEUT%:78,2."
    )
    parser.add_argument("inputs", nargs="+", help="JSONL file(s) to audit or fix.")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Atomically replace input files. Without this flag, only audit.",
    )
    return parser.parse_args()


def process_file(path: Path, in_place: bool) -> tuple[int, list[int]]:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    changed_records = []
    changed_entities = 0
    output = temp_path.open("w", encoding="utf-8", newline="\n") if in_place else None

    try:
        with path.open(encoding="utf-8") as source:
            for line_no, raw_line in enumerate(source, start=1):
                record = json.loads(raw_line)
                fixed_entities = []
                record_changed = False

                for entity in record.get("entities", []):
                    logs = []
                    split = autofix_split_merged_lab([entity], logs)
                    if any("autofix-split-multi-lab" in log for log in logs):
                        fixed_entities.extend(split)
                        changed_entities += 1
                        record_changed = True
                    else:
                        fixed_entities.append(entity)

                if record_changed:
                    changed_records.append(line_no)
                    record["entities"] = sort_entities_by_text_order(
                        fixed_entities, record["input_text"], []
                    )

                if output is not None:
                    if record_changed:
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    else:
                        output.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
    finally:
        if output is not None:
            output.close()

    if in_place:
        os.replace(temp_path, path)

    return changed_entities, changed_records


def main() -> None:
    args = parse_args()
    total = 0
    for value in args.inputs:
        path = Path(value)
        changed_entities, records = process_file(path, args.in_place)
        total += changed_entities
        action = "fixed" if args.in_place else "found"
        print(f"{path}: {action} {changed_entities} merged entities; records={records}")
    print(f"Total: {total}")


if __name__ == "__main__":
    main()
