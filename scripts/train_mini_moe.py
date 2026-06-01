"""CLI entry point for Stage 5 Mini-MoE training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mini_moe_train import load_yaml_config, train_mini_moe


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Mini-MoE training."""

    parser = argparse.ArgumentParser(description="Train a controllable MiniGPT/Mini-MoE model.")
    parser.add_argument("--config", default="configs/mini_moe_config.yaml", help="YAML config path.")
    parser.add_argument("--output_dir", default=None, help="Override output directory.")
    parser.add_argument("--routing_method", default=None, help="Override routing method.")
    parser.add_argument("--block_size", type=int, default=None, help="Override model block size.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override train batch size.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override maximum train steps.")
    parser.add_argument("--eval_steps", type=int, default=None, help="Override eval interval.")
    parser.add_argument("--save_steps", type=int, default=None, help="Override checkpoint interval.")
    parser.add_argument("--device", default=None, help="Device string, for example cuda or cpu.")
    return parser.parse_args()


def main() -> None:
    """Load config, apply simple overrides, and start Mini-MoE training."""

    args = parse_args()
    config = load_yaml_config(args.config)
    for key in ["output_dir", "batch_size", "max_steps", "eval_steps", "save_steps", "device"]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.routing_method is not None:
        config.setdefault("model", {})["routing_method"] = args.routing_method
    if args.block_size is not None:
        config.setdefault("model", {})["block_size"] = args.block_size
    train_mini_moe(config)


if __name__ == "__main__":
    main()
