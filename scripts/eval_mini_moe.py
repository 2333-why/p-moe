"""CLI entry point for Stage 5 Mini-MoE evaluation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mini_moe_train import (
    build_dataloader,
    evaluate,
    load_model_for_eval,
    load_wikitext_tokens,
    load_yaml_config,
    save_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Mini-MoE evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate a trained MiniGPT/Mini-MoE checkpoint.")
    parser.add_argument("--config", default="configs/mini_moe_config.yaml", help="YAML config path.")
    parser.add_argument("--checkpoint", required=True, help="Path to model.pt checkpoint.")
    parser.add_argument("--output_dir", default=None, help="Directory for eval_summary.json.")
    parser.add_argument("--split", default=None, help="Dataset split to evaluate.")
    parser.add_argument("--batch_size", type=int, default=None, help="Evaluation batch size.")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional batch limit.")
    parser.add_argument("--device", default=None, help="Device string, for example cuda or cpu.")
    return parser.parse_args()


def main() -> None:
    """Load a checkpoint, evaluate WikiText PPL, and save eval_summary.json."""

    args = parse_args()
    config = load_yaml_config(args.config)
    device = torch.device(args.device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model_for_eval(args.config, args.checkpoint, device)
    model_config = config.get("model", {})
    split = args.split or config.get("eval_split", "validation")
    tokens = load_wikitext_tokens(
        dataset_name=config.get("dataset_name", "wikitext"),
        dataset_config_name=config.get("dataset_config_name", "wikitext-2-raw-v1"),
        split=split,
        tokenizer_name=config.get("tokenizer_name", "gpt2"),
        max_chars=config.get("max_eval_chars"),
    )
    loader = build_dataloader(
        tokens,
        block_size=int(model_config.get("block_size", 256)),
        batch_size=args.batch_size or int(config.get("eval_batch_size", config.get("batch_size", 4))),
        shuffle=False,
    )
    metrics = evaluate(model, loader, device=device, max_batches=args.max_batches)
    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/mini_moe_run"))
    summary = {
        "stage": "stage5_mini_moe_eval",
        "checkpoint": args.checkpoint,
        "split": split,
        "metrics": metrics,
    }
    save_json(output_dir / "eval_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
