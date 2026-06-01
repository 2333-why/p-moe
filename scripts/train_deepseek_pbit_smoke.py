"""Run a small DeepSeek-MoE continued-pretraining smoke experiment.

This script is intentionally conservative. It only loads and trains a model
when executed directly by the user, supports offline caches, and writes all
artifacts under ``output_dir``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.data_utils import extract_nonempty_texts, load_wikitext_dataset
from src.deepseek_pbit_patch import (
    DeepSeekPBitPatchConfig,
    apply_deepseek_pbit_patch,
    collect_deepseek_pbit_metrics,
)
from src.gpu_utils import cuda_summary
from src.logging_utils import write_json
from src.model_utils import infer_offload_status, load_causal_lm, load_tokenizer
from src.train_utils import seed_everything, write_yaml_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for DeepSeek smoke training."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_deepseek_pbit_smoke.yaml")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--local_files_only", action="store_true", default=None)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--offload_folder", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_config_name", default=None)
    parser.add_argument("--dataset_split", default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--max_train_tokens", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--freeze_non_router", action="store_true", default=None)
    parser.add_argument("--use_pbit_patch", action="store_true", default=None)
    parser.add_argument("--no_pbit_patch", action="store_true", help="Disable p-bit patch for baseline smoke runs.")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Load config, run smoke training, and save summary."""

    args = parse_args()
    config = _load_config(args)
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml_config(output_dir / "train_config_used.yaml", config)

    try:
        summary = run_smoke_training(config)
    except Exception as exc:
        summary = {
            "stage": "stage6_deepseek_pbit_smoke",
            "success": False,
            "failure": True,
            "error_message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
            "output_dir": str(output_dir),
        }
        write_json(output_dir / "train_summary.json", summary)
        raise

    write_json(output_dir / "train_summary.json", summary)
    print(f"Wrote train summary to {output_dir / 'train_summary.json'}")


def run_smoke_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a short DeepSeek-MoE training loop for baseline or p-bit routing."""

    import torch

    seed_everything(int(config.get("seed", 42)))
    start_time = time.time()

    tokenizer = load_tokenizer(
        str(config["model_name"]),
        local_files_only=bool(config.get("local_files_only", True)),
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )
    model = load_causal_lm(
        str(config["model_name"]),
        dtype=str(config.get("dtype", "bf16")),
        device_map=config.get("device_map", "auto"),
        local_files_only=bool(config.get("local_files_only", True)),
        offload_folder=config.get("offload_folder"),
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )

    patch_report: Dict[str, Any] = {"enabled": False}
    if bool(config.get("use_pbit_patch", False)):
        patch_report = apply_deepseek_pbit_patch(
            model,
            DeepSeekPBitPatchConfig(
                enabled=True,
                alpha=float(config.get("alpha", 0.1)),
                beta=float(config.get("beta", 0.1)),
                temperature=float(config.get("temperature", 1.0)),
                min_temperature=float(config.get("min_temperature", 0.2)),
                load_ema_decay=float(config.get("load_ema_decay", 0.99)),
                use_load_bias=bool(config.get("use_load_bias", True)),
                use_competition=bool(config.get("use_competition", True)),
                patch_train_only=bool(config.get("patch_train_only", True)),
            ),
        )

    if bool(config.get("gradient_checkpointing", False)) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if bool(config.get("freeze_non_router", True)):
        _freeze_non_router_parameters(model)

    model.train()
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters remain. Check freeze_non_router/router naming.")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config.get("learning_rate", 1.0e-5)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    input_ids = _load_training_tokens(tokenizer, config)
    batches = _iter_lm_batches(input_ids, int(config.get("block_size", 512)))
    max_steps = int(config.get("max_steps", 20))
    grad_accum = int(config.get("gradient_accumulation_steps", 8))
    if max_steps <= 0:
        raise ValueError("max_steps must be positive for smoke training.")
    if grad_accum <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")

    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(max_steps):
        batch = next(batches)
        outputs = model(input_ids=batch, labels=batch)
        loss = outputs.loss / grad_accum
        loss.backward()
        losses.append(float(outputs.loss.detach().float().item()))
        if (step + 1) % grad_accum == 0 or step + 1 == max_steps:
            if config.get("max_grad_norm") is not None:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, float(config["max_grad_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if (step + 1) % int(config.get("logging_steps", 5)) == 0:
            print(f"step {step + 1}/{max_steps} loss={losses[-1]:.6f}")

    router_metrics = collect_deepseek_pbit_metrics(model)
    elapsed = time.time() - start_time
    final_loss = losses[-1] if losses else None
    return {
        "stage": "stage6_deepseek_pbit_smoke",
        "method": "deepseek_pbit" if config.get("use_pbit_patch", False) else "deepseek_baseline_router",
        "success": True,
        "failure": False,
        "model_name": config.get("model_name"),
        "output_dir": config.get("output_dir"),
        "max_steps": max_steps,
        "block_size": int(config.get("block_size", 512)),
        "gradient_accumulation_steps": grad_accum,
        "freeze_non_router": bool(config.get("freeze_non_router", True)),
        "use_pbit_patch": bool(config.get("use_pbit_patch", False)),
        "pbit_patch": patch_report,
        "train_loss": final_loss,
        "train_ppl_estimate": math.exp(final_loss) if isinstance(final_loss, float) and final_loss < 20 else None,
        "losses": losses,
        "router_metrics": router_metrics,
        "load_variance": router_metrics.get("load_variance"),
        "dead_expert_ratio": router_metrics.get("dead_expert_ratio"),
        "expert_entropy": router_metrics.get("expert_entropy"),
        "router_grad_norm": router_metrics.get("router_grad_norm"),
        "offload_status": infer_offload_status(model),
        "cuda": cuda_summary(),
        "elapsed_sec": elapsed,
    }


def _load_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge YAML config and explicit CLI overrides."""

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a mapping.")

    for key, value in vars(args).items():
        if key in {"config", "no_pbit_patch"} or value is None:
            continue
        config[key] = value
    if args.no_pbit_patch:
        config["use_pbit_patch"] = False
    return config


def _freeze_non_router_parameters(model: Any) -> None:
    """Freeze all parameters except DeepSeek MoEGate weights."""

    for name, param in model.named_parameters():
        param.requires_grad = name.endswith(".mlp.gate.weight") or ".mlp.gate.weight" in name


def _load_training_tokens(tokenizer: Any, config: Dict[str, Any]):
    """Load cached text data and tokenize it into a flat tensor."""

    import torch

    dataset = load_wikitext_dataset(
        dataset_name=str(config.get("dataset_name", "wikitext")),
        dataset_config_name=config.get("dataset_config_name", "wikitext-2-raw-v1"),
        split=str(config.get("dataset_split", "train")),
        local_files_only=bool(config.get("local_files_only", True)),
    )
    texts = extract_nonempty_texts(dataset, text_column=str(config.get("text_column", "text")))
    text = "\n\n".join(texts)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].view(-1)
    max_tokens = config.get("max_train_tokens")
    if max_tokens:
        input_ids = input_ids[: int(max_tokens)]
    if input_ids.numel() <= int(config.get("block_size", 512)):
        raise ValueError("Not enough training tokens for the requested block_size.")
    return input_ids.to(_first_input_device(config))


def _iter_lm_batches(input_ids: Any, block_size: int):
    """Yield deterministic one-sample causal LM batches over a flat token tensor."""

    position = 0
    while True:
        if position + block_size + 1 > input_ids.numel():
            position = 0
        batch = input_ids[position : position + block_size].unsqueeze(0)
        position += block_size
        yield batch


def _first_input_device(config: Dict[str, Any]) -> str:
    """Return the device used for input IDs with accelerate-dispatched models."""

    device = config.get("input_device")
    if device:
        return str(device)
    return "cuda:0"


if __name__ == "__main__":
    main()
