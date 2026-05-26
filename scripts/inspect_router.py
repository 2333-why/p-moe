"""Inspect router-like modules before Stage 4 p-bit patching.

This script loads a model only when the user runs it manually. It does not
modify model weights and does not apply the p-bit router patch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_utils import load_causal_lm
from src.router_patch import ROUTER_KEYWORDS, build_router_patch_plan, list_router_like_modules


DEFAULT_CONFIG_PATH = Path("configs") / "pbit_router_config.yaml"
DEFAULT_MODEL_NAME = "deepseek-ai/deepseek-moe-16b-base"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Inspect router-like modules in a causal LM.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default=None)
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--offload_folder", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="Router discovery keywords. Defaults to config or gate/router/moe/expert.",
    )
    return parser.parse_args()


def redact_secrets(text: str) -> str:
    """Redact known secret environment variable values from text."""
    redacted = text
    for name in ("HF_TOKEN", "WANDB_API_KEY"):
        value = os.environ.get(name)
        if value and len(value) >= 4:
            redacted = redacted.replace(value, f"<{name}:redacted>")
    return redacted


def load_config(path: str) -> dict[str, Any]:
    """Load YAML config if present."""
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """Resolve config dtype name to torch dtype."""
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    try:
        return mapping[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype '{dtype_name}'.") from exc


def create_run_dir(base_dir: str) -> Path:
    """Create a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = Path(base_dir) / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def inspect_router(config: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path]:
    """Load model, list router candidates, and save JSON/TXT reports."""
    model_name = args.model_name or config.get("model_name", DEFAULT_MODEL_NAME)
    dtype_name = args.dtype or config.get("dtype", "bf16")
    device_map = args.device_map or config.get("device_map", "auto")
    output_dir = args.output_dir or config.get("output_dir", "outputs")
    trust_remote_code = bool(config.get("trust_remote_code", True))
    low_cpu_mem_usage = bool(config.get("low_cpu_mem_usage", True))
    offload_folder = args.offload_folder or config.get("offload_folder")
    keywords = tuple(args.keywords or config.get("router_keywords", ROUTER_KEYWORDS))

    run_dir = create_run_dir(output_dir)
    json_path = run_dir / "router_inspection.json"
    txt_path = run_dir / "router_inspection.txt"

    print(f"Loading model for router inspection: {model_name}")
    print("This may download large weights if they are not already cached.")
    print(f"local_files_only={args.local_files_only}")

    model = load_causal_lm(
        model_name=model_name,
        dtype_name=dtype_name,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage,
        offload_folder=offload_folder,
        local_files_only=bool(args.local_files_only),
    )
    candidates = list_router_like_modules(model, keywords=keywords)
    candidate_rows = [candidate.to_dict() for candidate in candidates]
    patch_plan = build_router_patch_plan(model)

    result = {
        "model_name": model_name,
        "model_class": model.__class__.__name__,
        "dtype": dtype_name,
        "device_map": device_map,
        "local_files_only": bool(args.local_files_only),
        "router_keywords": list(keywords),
        "candidate_count": len(candidate_rows),
        "router_candidates": candidate_rows,
        "patch_plan": patch_plan,
        "note": (
            "Review candidate names before calling apply_pbit_patch. "
            "This script does not modify the model."
        ),
    }

    lines = [
        "Router Inspection",
        "=================",
        f"Model name: {model_name}",
        f"Model class: {model.__class__.__name__}",
        f"Keywords: {', '.join(keywords)}",
        f"Candidate count: {len(candidate_rows)}",
        "",
        "Router candidates:",
    ]
    if candidate_rows:
        lines.extend(f"- {row['name']}: {row['class_name']}" for row in candidate_rows)
    else:
        lines.append("- <none found>")
    lines.extend(
        [
            "",
            "Patch note:",
            "Review these names manually before passing one exact name to apply_pbit_patch.",
        ]
    )

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved router inspection JSON to: {json_path}")
    print(f"Saved router inspection text to: {txt_path}")
    return json_path, txt_path


def main() -> None:
    """Entry point for manual router inspection."""
    args = parse_args()
    config = load_config(args.config)
    try:
        inspect_router(config, args)
    except Exception as exc:
        output_dir = args.output_dir or config.get("output_dir", "outputs")
        run_dir = create_run_dir(output_dir)
        txt_path = run_dir / "router_inspection.txt"
        json_path = run_dir / "router_inspection.json"
        message = redact_secrets(str(exc))
        failure = {
            "success": False,
            "error_type": exc.__class__.__name__,
            "error_message": message,
            "suggestion": (
                "Check local_files_only/cache settings, Hugging Face access, "
                "GPU memory, and disk space. This script loads the model only "
                "for manual inspection."
            ),
        }
        json_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        txt_path.write_text(
            "Router Inspection Failed\n"
            "========================\n"
            f"Error type: {exc.__class__.__name__}\n"
            f"Error message: {message}\n",
            encoding="utf-8",
        )
        print(f"Router inspection failed: {message}")
        print(f"Saved failure reports to: {json_path} and {txt_path}")
        raise


if __name__ == "__main__":
    main()
