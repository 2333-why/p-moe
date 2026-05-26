"""Collect Stage 1-4 run summaries into Stage 5 result tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.result_utils import collect_result_records, write_csv, write_markdown_summary


def parse_args() -> argparse.Namespace:
    """Parse result collection arguments."""
    parser = argparse.ArgumentParser(description="Collect experiment summary files into tables.")
    parser.add_argument("--outputs_dir", default="outputs", help="Directory containing run_* outputs.")
    parser.add_argument("--csv_path", default=None, help="Output CSV path.")
    parser.add_argument("--md_path", default=None, help="Output Markdown summary path.")
    return parser.parse_args()


def main() -> int:
    """Collect summaries and write outputs/results_table.csv and outputs/results_summary.md."""
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    csv_path = Path(args.csv_path) if args.csv_path else outputs_dir / "results_table.csv"
    md_path = Path(args.md_path) if args.md_path else outputs_dir / "results_summary.md"

    records = collect_result_records(outputs_dir)
    write_csv(records, csv_path)
    write_markdown_summary(records, md_path)

    print(f"Collected {len(records)} summary file(s).")
    print(f"CSV: {csv_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
