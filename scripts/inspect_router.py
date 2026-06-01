"""Inspect model modules for router-like components and save a report.

The script loads a model only when executed directly by the user. It never
patches the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.router_patch import list_candidate_router_modules


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Inspect router-like modules in a causal LM.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/router_inspection"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--torch_dtype", type=str, default="auto")
    parser.add_argument(
        "--keywords",
        type=str,
        default="router,gate,moe,expert",
        help="Comma-separated module name/class keywords to include.",
    )
    return parser.parse_args()


def load_model_for_inspection(args: argparse.Namespace) -> Any:
    """Load a model for structural inspection using user-provided options."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("Install transformers to inspect router modules.") from exc
    return AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a router inspection report without applying patches."""
    model = load_model_for_inspection(args)
    keywords = [item.strip() for item in args.keywords.split(",") if item.strip()]
    targets = list_candidate_router_modules(model, name_keywords=keywords)
    return {
        "model_name": args.model_name,
        "local_files_only": bool(args.local_files_only),
        "keywords": keywords,
        "num_candidates": len(targets),
        "candidates": [asdict(target) for target in targets],
        "patch_applied": False,
        "note": "Inspection only. Review this report before designing any model-specific patch.",
    }


def write_report(report: Dict[str, Any], output_dir: Path) -> None:
    """Write JSON and text inspection reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "router_inspection.json"
    txt_path = output_dir / "router_inspection.txt"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    txt_path.write_text(_format_text_report(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")


def _format_text_report(report: Dict[str, Any]) -> str:
    """Return a readable text report."""
    lines: List[str] = [
        "Router inspection report",
        f"model_name: {report['model_name']}",
        f"local_files_only: {report['local_files_only']}",
        f"num_candidates: {report['num_candidates']}",
        "patch_applied: false",
        "",
    ]
    for index, candidate in enumerate(report["candidates"], start=1):
        lines.extend(
            [
                f"[{index}] {candidate['name']}",
                f"  class: {candidate['class_name']}",
                f"  module_path: {candidate['module_path']}",
                f"  num_experts: {candidate['num_experts']}",
                f"  top_k: {candidate['top_k']}",
            ]
        )
    lines.append("")
    lines.append(str(report["note"]))
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    report = build_report(args)
    write_report(report, args.output_dir)


if __name__ == "__main__":
    main()
