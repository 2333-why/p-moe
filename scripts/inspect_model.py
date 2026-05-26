"""Inspect DeepSeek-MoE model structure for Stage 1.

This script intentionally loads the configured model only when it is run
manually. It does not modify model weights or implement training/evaluation.
"""

from __future__ import annotations

import argparse
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


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "generation_config.yaml"
DEFAULT_MODEL_NAME = "deepseek-ai/deepseek-moe-16b-base"
MODULE_KEYWORDS = ("moe", "gate", "router", "expert")


def redact_secrets(text: str) -> str:
    """Redact known secret environment variable values from report text."""
    redacted = text
    for name in ("HF_TOKEN", "WANDB_API_KEY"):
        value = os.environ.get(name)
        if value and len(value) >= 4:
            redacted = redacted.replace(value, f"<{name}:redacted>")
    return redacted


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model inspection."""
    parser = argparse.ArgumentParser(
        description="Load a causal LM and inspect MoE-related module names."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config. Defaults to configs/generation_config.yaml.",
    )
    parser.add_argument("--model_name", default=None, help="Hugging Face model name.")
    parser.add_argument(
        "--dtype",
        default=None,
        choices=("bf16", "fp16", "fp32"),
        help="Torch dtype for loading the model.",
    )
    parser.add_argument(
        "--device_map",
        default=None,
        help='Device map passed to from_pretrained, for example "auto".',
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Base output directory for run_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--offload_folder",
        default=None,
        help="Optional folder used by accelerate for CPU/disk offload.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        default=None,
        help="Load model only from local files or Hugging Face cache.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config if present, otherwise return an empty config."""
    path = Path(config_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return loaded


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """Resolve bf16, fp16, or fp32 to a torch dtype."""
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    try:
        return dtype_map[dtype_name.lower()]
    except KeyError as exc:
        supported = ", ".join(dtype_map)
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Use one of: {supported}.") from exc


def create_run_dir(base_dir: str) -> Path:
    """Create outputs/run_YYYYMMDD_HHMMSS and return its path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = Path(base_dir) / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return total, trainable


def get_top_level_modules(model: torch.nn.Module) -> list[str]:
    """Return top-level child module names and class names."""
    return [
        f"{name}: {module.__class__.__name__}"
        for name, module in model.named_children()
    ]


def find_matching_modules(model: torch.nn.Module) -> list[str]:
    """Return module names containing moe/gate/router/expert."""
    matches: list[str] = []
    for name, module in model.named_modules():
        lowered = name.lower()
        if any(keyword in lowered for keyword in MODULE_KEYWORDS):
            display_name = name if name else "<root>"
            matches.append(f"{display_name}: {module.__class__.__name__}")
    return matches


def format_count(value: int) -> str:
    """Format a parameter count with separators."""
    return f"{value:,}"


def build_report(model: torch.nn.Module, model_name: str) -> str:
    """Build the full text report for model structure inspection."""
    total_params, trainable_params = count_parameters(model)
    top_level_modules = get_top_level_modules(model)
    matching_modules = find_matching_modules(model)

    lines = [
        "DeepSeek MoE Model Structure Inspection",
        "=" * 43,
        f"Model name: {model_name}",
        f"Model class: {model.__class__.__name__}",
        f"Total parameters: {format_count(total_params)}",
        f"Trainable parameters: {format_count(trainable_params)}",
        "",
        "Top-level modules:",
    ]

    if top_level_modules:
        lines.extend(f"- {module_name}" for module_name in top_level_modules)
    else:
        lines.append("- <none>")

    lines.extend(
        [
            "",
            "Modules containing moe/gate/router/expert:",
        ]
    )

    if matching_modules:
        lines.extend(f"- {module_name}" for module_name in matching_modules)
    else:
        lines.append("- <none found>")

    lines.extend(
        [
            "",
            "# TODO Stage 2:",
            "# Add WikiText-2 perplexity evaluation.",
            "",
            "# TODO Stage 3:",
            "# Add LoRA / QLoRA continued pretraining.",
            "",
            "# TODO Stage 4:",
            "# Inspect and modify MoE router for hard-forward p-bit-backward routing.",
        ]
    )
    return "\n".join(lines) + "\n"


def inspect_model(config: dict[str, Any], args: argparse.Namespace) -> tuple[str, Path]:
    """Load the configured model and write the model_modules.txt report."""
    model_name = args.model_name or config.get("model_name", DEFAULT_MODEL_NAME)
    dtype_name = args.dtype or config.get("dtype", "bf16")
    device_map = args.device_map or config.get("device_map", "auto")
    output_dir = args.output_dir or config.get("output_dir", "outputs")
    trust_remote_code = bool(config.get("trust_remote_code", True))
    low_cpu_mem_usage = bool(config.get("low_cpu_mem_usage", True))
    offload_folder = args.offload_folder or config.get("offload_folder")
    local_files_only = (
        bool(args.local_files_only)
        if args.local_files_only is not None
        else bool(config.get("local_files_only", False))
    )

    run_dir = create_run_dir(output_dir)
    output_path = run_dir / "model_modules.txt"

    print(f"Loading model for inspection: {model_name}")
    print(f"dtype={dtype_name}, device_map={device_map}")
    if local_files_only:
        print("local_files_only=True: using local files or Hugging Face cache only.")
    else:
        print("This may download large weights if they are not already cached.")

    model = load_causal_lm(
        model_name=model_name,
        dtype_name=dtype_name,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage,
        offload_folder=offload_folder,
        local_files_only=local_files_only,
    )
    report = build_report(model, model_name)

    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved model structure report to: {output_path}")
    return report, output_path


def main() -> None:
    """Entry point for manual model structure inspection."""
    args = parse_args()
    config = load_config(args.config)

    try:
        inspect_model(config, args)
    except Exception as exc:
        output_dir = args.output_dir or config.get("output_dir", "outputs")
        run_dir = create_run_dir(output_dir)
        output_path = run_dir / "model_modules.txt"
        message = (
            "DeepSeek MoE Model Structure Inspection Failed\n"
            "=============================================\n"
            f"Error type: {exc.__class__.__name__}\n"
            f"Error message: {redact_secrets(str(exc))}\n\n"
            "Suggestions:\n"
            "1. Check network access and Hugging Face model permissions.\n"
            "2. Verify HF_ENDPOINT and Hugging Face cache settings if using a mirror.\n"
            "3. Ensure sufficient GPU memory, CPU memory, and disk space.\n"
            "4. Try device_map=\"auto\" or provide an offload folder for inspection only.\n"
        )
        output_path.write_text(message, encoding="utf-8")
        print(message)
        print(f"Saved failure report to: {output_path}")
        raise


if __name__ == "__main__":
    main()
