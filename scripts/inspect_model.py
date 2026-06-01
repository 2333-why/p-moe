"""Inspect model module names for MoE/router/gate/expert planning.

This is a Stage 1/6 inspection helper. It loads only when executed and writes a
plain-text module listing plus a JSON summary. It does not patch the model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gpu_utils import cuda_summary
from src.logging_utils import ensure_parent, write_json
from src.model_utils import infer_offload_status, load_causal_lm, model_parameter_summary


SEARCH_TERMS = ("moe", "mixtral", "expert", "experts", "gate", "router", "mlp")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Inspect model modules and find MoE/router/expert candidates.")
    parser.add_argument("--model_name", default="models/deepseek-moe-16b-base", help="Local path or HF model id.")
    parser.add_argument("--revision", default=None, help="Optional model revision.")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16", help="Model dtype.")
    parser.add_argument("--device_map", default="auto", help="Transformers/Accelerate device_map, e.g. auto/cpu.")
    parser.add_argument("--local_files_only", action="store_true", help="Do not access the network.")
    parser.add_argument("--offload_folder", default=None, help="Optional accelerate disk offload directory.")
    parser.add_argument("--modules_path", default="outputs/model_modules.txt", help="Text output path for module list.")
    parser.add_argument("--summary_path", default="outputs/model_inspection_summary.json", help="Summary JSON path.")
    return parser


def module_rows(model) -> list[dict[str, str]]:
    """Return module name/type rows from a loaded model."""

    rows = []
    for name, module in model.named_modules():
        rows.append({"name": name, "type": type(module).__name__})
    return rows


def find_candidate_modules(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Filter module rows for likely MoE/router/gate/expert modules."""

    candidates = []
    for row in rows:
        haystack = f"{row['name']} {row['type']}".lower()
        if any(term in haystack for term in SEARCH_TERMS):
            candidates.append(row)
    return candidates


def write_module_text(path: str, rows: list[dict[str, str]], candidates: list[dict[str, str]]) -> Path:
    """Write a human-readable module report."""

    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8") as handle:
        handle.write("# Candidate MoE/router/gate/expert modules\n")
        for row in candidates:
            handle.write(f"{row['name']}\t{row['type']}\n")
        handle.write("\n# All modules\n")
        for row in rows:
            handle.write(f"{row['name']}\t{row['type']}\n")
    return resolved


def main() -> None:
    """Load the model and write module inspection reports."""

    args = build_parser().parse_args()
    model = load_causal_lm(
        args.model_name,
        dtype=args.dtype,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
        offload_folder=args.offload_folder,
        revision=args.revision,
    )
    rows = module_rows(model)
    candidates = find_candidate_modules(rows)
    modules_path = write_module_text(args.modules_path, rows, candidates)
    summary = {
        "model_name": args.model_name,
        "revision": args.revision,
        "local_files_only": args.local_files_only,
        "module_count": len(rows),
        "candidate_count": len(candidates),
        "candidate_modules": candidates,
        "modules_path": str(modules_path),
        "offload_status": infer_offload_status(model),
        "parameter_summary": model_parameter_summary(model),
        "cuda": cuda_summary(),
    }
    summary_path = write_json(args.summary_path, summary)
    print(f"Wrote module list to {modules_path}")
    print(f"Wrote inspection summary to {summary_path}")


if __name__ == "__main__":
    main()
