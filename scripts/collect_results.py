"""Collect p-MoE experiment summaries into table and markdown reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.result_utils import collect_result_records, write_results_markdown, write_results_table


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for offline result collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs_dir", default="outputs", help="Directory containing run outputs.")
    parser.add_argument(
        "--csv_path",
        default="outputs/results_table.csv",
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--md_path",
        default="outputs/results_summary.md",
        help="Destination markdown summary path.",
    )
    return parser.parse_args()


def main() -> None:
    """Collect result JSON files and write aggregate artifacts."""
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    records = collect_result_records(outputs_dir)
    write_results_table(records, Path(args.csv_path))
    write_results_markdown(records, Path(args.md_path))


if __name__ == "__main__":
    main()
