"""Command line entrypoint for generating synthetic Vietnamese medical NER data."""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_gen.generate_data import DEFAULT_PROFILE, N_SAMPLES, run_generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Vietnamese medical NER JSONL data."
    )
    parser.add_argument(
        "--profile",
        choices=("baseline", "quota_v2", "mixed_v3", "mixed_v4", "mixed_v5"),
        default=DEFAULT_PROFILE,
        help=("baseline keeps the base prompt; quota_v2 uses the old soft focus; "
              "mixed_v3 mixes baseline/V3; mixed_v4 mixes long QA/theory; "
              "mixed_v5 generates contrastive, sparse, boundary and dirty-text data."),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=N_SAMPLES,
        help=f"Number of samples to generate. Default: {N_SAMPLES}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Default depends on --profile.",
    )
    parser.add_argument(
        "--reject-output",
        default=None,
        help="Rejected-sample JSONL path. Default depends on --profile.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    stats = run_generation(
        profile=args.profile,
        n_samples=args.samples,
        output=args.output,
        reject_output=args.reject_output,
    )
    if stats is not None:
        print(f"Output written to: {stats['output']}")
        print(f"Reject log written to: {stats['reject_output']}")


if __name__ == "__main__":
    main()
