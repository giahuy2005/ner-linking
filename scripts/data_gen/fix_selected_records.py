"""Run the current QC/autofix pipeline on selected one-based JSONL line numbers."""

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_GEN_SRC = REPO_ROOT / "src" / "data_gen"
if str(DATA_GEN_SRC) not in sys.path:
    sys.path.insert(0, str(DATA_GEN_SRC))

from gen_reject import process_record


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("lines", nargs="+", type=int, help="One-based line numbers")
    args = parser.parse_args()

    selected = set(args.lines)
    raw_lines = args.input.read_text(encoding="utf-8").splitlines()
    invalid = sorted(line for line in selected if line < 1 or line > len(raw_lines))
    if invalid:
        raise ValueError(f"Line numbers outside 1..{len(raw_lines)}: {invalid}")

    changed = []
    for line_no in sorted(selected):
        record = json.loads(raw_lines[line_no - 1])
        result = process_record(record)
        if result[0] != "keep":
            raise RuntimeError(f"Line {line_no} rejected: {result[1]}")
        _, cleaned, logs = result
        raw_lines[line_no - 1] = json.dumps(cleaned, ensure_ascii=False)
        changed.append((line_no, logs))

    temp_path = args.input.with_suffix(args.input.suffix + ".tmp")
    temp_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    os.replace(temp_path, args.input)

    for line_no, logs in changed:
        print(f"line {line_no}: {len(logs)} change log(s)")
        for log in logs:
            print(f"  {log}")


if __name__ == "__main__":
    main()
