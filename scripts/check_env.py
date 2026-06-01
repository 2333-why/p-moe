"""Check local environment readiness for p-MoE Stage 1.

This script performs diagnostics only. It does not download assets, load
DeepSeek, evaluate, train, or initialize WandB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gpu_utils import cuda_summary, disk_summary, dtype_support_summary, hf_environment_summary
from src.logging_utils import safe_env_status, write_json


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Inspect p-MoE runtime environment without loading models.")
    parser.add_argument("--output", default="outputs/env_summary.json", help="Path to write JSON summary.")
    parser.add_argument(
        "--disk_path",
        action="append",
        default=None,
        help="Path to include in disk checks. May be passed multiple times.",
    )
    parser.add_argument("--print_json", action="store_true", help="Print the token-safe summary as JSON.")
    return parser


def main() -> None:
    """Run environment diagnostics."""

    args = build_parser().parse_args()
    disk_paths = args.disk_path or ["models", "outputs", "."]
    summary = {
        "hf_environment": hf_environment_summary(),
        "secret_status": safe_env_status(["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "WANDB_API_KEY"]),
        "dtype_support": dtype_support_summary(),
        "cuda": cuda_summary(),
        "disk": disk_summary(disk_paths),
    }
    path = write_json(args.output, summary)
    if args.print_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Wrote environment summary to {path}")


if __name__ == "__main__":
    main()
