"""Command line entrypoint for converting JSONL NER data to BIO JSONL."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JSONL samples with input_text/entities to BIO token-level JSONL."
    )
    parser.add_argument(
        "--input",
        default="data/synthetic/train_1.jsonl",
        help="Input JSONL path. Default: data/synthetic/train_1.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/processed/train_bio.jsonl",
        help="Output JSONL path. Default: data/processed/train_bio.jsonl",
    )
    parser.add_argument(
        "--vncorenlp-jar",
        default="vncorenlp/VnCoreNLP-1.1.1.jar",
        help="Path to VnCoreNLP jar. Default: vncorenlp/VnCoreNLP-1.1.1.jar",
    )
    parser.add_argument(
        "--max-heap-size",
        default="-Xmx2g",
        help="JVM heap size for VnCoreNLP. Default: -Xmx2g",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-sample conversion logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.preprocessing.json_to_bio import JSONToBioConverter

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with JSONToBioConverter(
        jar_path=args.vncorenlp_jar,
        max_heap_size=args.max_heap_size,
    ) as converter:
        stats = converter.convert_file(
            str(input_path),
            str(output_path),
            verbose=not args.quiet,
        )

    print(f"Output written to: {output_path}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
