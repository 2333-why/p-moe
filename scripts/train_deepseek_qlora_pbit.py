"""DeepSeek-MoE QLoRA continued pretraining with optional p-bit router patch.

This is the main DeepSeek-side training entry point for checking what happens
when model adaptation parameters participate in training. It loads and trains
only when executed directly by the user.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deepseek_pbit_patch import (
    DeepSeekPBitPatchConfig,
    apply_deepseek_gate_metric_patch,
    apply_deepseek_pbit_patch,
    collect_deepseek_pbit_metrics,
)
from src.lora_utils import (
    apply_lora,
    build_qlora_quantization_config,
    prepare_model_for_lora_training,
    print_trainable_parameters,
)
from src.train_utils import (
    build_training_arguments,
    load_text_dataset,
    maybe_init_wandb,
    merge_config,
    read_yaml_config,
    save_training_artifacts,
    seed_everything,
    str_to_bool,
    tokenize_and_group_texts,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_deepseek_qlora_pbit.yaml")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--local_files_only", type=str_to_bool, nargs="?", const=True, default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_config_name", default=None)
    parser.add_argument("--dataset_split", default=None)
    parser.add_argument("--text_column", default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_target_modules", default=None)
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--use_pbit_patch", action="store_true", default=None)
    parser.add_argument("--no_pbit_patch", action="store_true")
    parser.add_argument("--train_router", type=str_to_bool, nargs="?", const=True, default=None)
    parser.add_argument("--record_router_metrics", type=str_to_bool, nargs="?", const=True, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--forward_preserve", type=str_to_bool, nargs="?", const=True, default=None)
    parser.add_argument("--no_forward_preserve", action="store_true")
    parser.add_argument("--use_trust_region", type=str_to_bool, nargs="?", const=True, default=None)
    parser.add_argument("--no_trust_region", action="store_true")
    parser.add_argument("--trust_region_gamma", type=float, default=None)
    parser.add_argument("--load_margin", type=float, default=None)
    parser.add_argument("--top_m", type=int, default=None)
    parser.add_argument("--load_bias_forward", type=str_to_bool, nargs="?", const=True, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_wandb", type=str_to_bool, nargs="?", const=True, default=None)
    return parser


def default_config() -> Dict[str, Any]:
    """Return conservative QLoRA+p-bit defaults."""

    return {
        "model_name": "deepseek-ai/deepseek-moe-16b-base",
        "output_dir": "outputs/deepseek_qlora_pbit",
        "local_files_only": True,
        "trust_remote_code": True,
        "dataset_name": "wikitext",
        "dataset_config_name": "wikitext-2-raw-v1",
        "dataset_split": "train",
        "text_column": "text",
        "block_size": 1024,
        "preprocessing_num_workers": None,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "max_steps": 1000,
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1.0e-4,
        "weight_decay": 0.0,
        "warmup_steps": 50,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "logging_steps": 10,
        "save_steps": 500,
        "save_total_limit": 2,
        "dataloader_num_workers": 0,
        "overwrite_output_dir": False,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "lora_bias": "none",
        "task_type": "CAUSAL_LM",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bf16",
        "device_map": "auto",
        "prepare_model_for_kbit_training": True,
        "use_wandb": False,
        "wandb_project": "p-moe",
        "run_name": "deepseek_qlora_pbit",
        "seed": 42,
        "use_pbit_patch": True,
        "train_router": True,
        "record_router_metrics": True,
        "alpha": 0.0,
        "beta": 0.001,
        "temperature": 2.0,
        "min_temperature": 0.2,
        "load_ema_decay": 0.99,
        "use_load_bias": True,
        "use_competition": True,
        "forward_preserve": True,
        "use_trust_region": True,
        "trust_region_gamma": 0.03,
        "load_margin": 0.005,
        "top_m": None,
        "load_bias_forward": False,
        "patch_train_only": True,
    }


def normalize_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge defaults, YAML config, and CLI overrides."""

    config = default_config()
    config.update(read_yaml_config(args.config))
    config = merge_config(args, config)
    if args.no_pbit_patch:
        config["use_pbit_patch"] = False
    if args.no_forward_preserve:
        config["forward_preserve"] = False
    if args.no_trust_region:
        config["use_trust_region"] = False
    return config


def main() -> None:
    """Parse config and run DeepSeek QLoRA+p-bit training."""

    config = normalize_config(build_parser().parse_args())
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_training(config)
    except Exception as exc:
        summary = {
            "stage": "stage6_deepseek_qlora_pbit",
            "method": _method_name(config),
            "success": False,
            "failure": True,
            "model_name": config.get("model_name"),
            "output_dir": config.get("output_dir"),
            "error_message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
        write_json(output_dir / "train_summary.json", summary)
        raise


def run_training(config: Dict[str, Any]) -> None:
    """Load 4-bit DeepSeek, attach LoRA, optionally patch router, and train."""

    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer

    seed_everything(int(config["seed"]))
    maybe_init_wandb(config)

    tokenizer = AutoTokenizer.from_pretrained(
        str(config["model_name"]),
        local_files_only=bool(config.get("local_files_only", True)),
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(config["model_name"]),
        local_files_only=bool(config.get("local_files_only", True)),
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        quantization_config=build_qlora_quantization_config(config),
        device_map=config.get("device_map", "auto"),
    )
    model = prepare_model_for_lora_training(model, config)
    model = apply_lora(model, config)
    if bool(config.get("train_router", True)):
        _unfreeze_deepseek_router_gates(model)

    patch_report: Dict[str, Any] = {"enabled": False}
    if bool(config.get("use_pbit_patch", False)):
        patch_report = apply_deepseek_pbit_patch(model, _pbit_config_from_mapping(config))
    elif bool(config.get("record_router_metrics", True)):
        patch_report = apply_deepseek_gate_metric_patch(model)

    print_trainable_parameters(model)
    dataset = load_text_dataset(config)
    train_dataset = tokenize_and_group_texts(dataset, tokenizer, config)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=build_training_arguments(config),
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    train_result = trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.save_model(str(config["output_dir"]))

    metrics = dict(getattr(train_result, "metrics", {}) or {})
    train_loss = metrics.get("train_loss")
    router_metrics = collect_deepseek_pbit_metrics(model)
    summary = {
        "stage": "stage6_deepseek_qlora_pbit",
        "method": _method_name(config),
        "success": True,
        "failure": False,
        "model_name": config.get("model_name"),
        "output_dir": config.get("output_dir"),
        "local_files_only": bool(config.get("local_files_only", True)),
        "use_pbit_patch": bool(config.get("use_pbit_patch", False)),
        "train_router": bool(config.get("train_router", True)),
        "max_steps": int(config.get("max_steps", -1)),
        "block_size": int(config.get("block_size", 1024)),
        "lora_r": int(config.get("lora_r", 16)),
        "lora_target_modules": config.get("lora_target_modules"),
        "train_loss": train_loss,
        "train_ppl_estimate": math.exp(train_loss) if isinstance(train_loss, (int, float)) and train_loss < 20 else None,
        "metrics": metrics,
        "router_metrics": router_metrics,
        "load_variance": router_metrics.get("load_variance"),
        "dead_expert_ratio": router_metrics.get("dead_expert_ratio"),
        "expert_entropy": router_metrics.get("expert_entropy"),
        "router_grad_norm": router_metrics.get("router_grad_norm"),
        "pbit_patch": patch_report,
        "model_config": {
            "routing_method": _method_name(config),
            "alpha": config.get("alpha"),
            "beta": config.get("beta"),
            "temperature": config.get("temperature"),
            "forward_preserve": config.get("forward_preserve"),
            "use_trust_region": config.get("use_trust_region"),
            "trust_region_gamma": config.get("trust_region_gamma"),
            "load_margin": config.get("load_margin"),
            "top_m": config.get("top_m"),
            "load_bias_forward": config.get("load_bias_forward"),
        },
    }
    save_training_artifacts(config, summary)


def _pbit_config_from_mapping(config: Dict[str, Any]) -> DeepSeekPBitPatchConfig:
    """Build p-bit patch config from script config."""

    top_m = config.get("top_m")
    if top_m in ("", "none", "None"):
        top_m = None
    return DeepSeekPBitPatchConfig(
        enabled=True,
        alpha=float(config.get("alpha", 0.0)),
        beta=float(config.get("beta", 0.01)),
        temperature=float(config.get("temperature", 1.0)),
        min_temperature=float(config.get("min_temperature", 0.2)),
        load_ema_decay=float(config.get("load_ema_decay", 0.99)),
        use_load_bias=bool(config.get("use_load_bias", True)),
        use_competition=bool(config.get("use_competition", True)),
        forward_preserve=bool(config.get("forward_preserve", True)),
        use_trust_region=bool(config.get("use_trust_region", True)),
        trust_region_gamma=float(config.get("trust_region_gamma", 0.03)),
        load_margin=float(config.get("load_margin", 0.005)),
        top_m=None if top_m is None else int(top_m),
        load_bias_forward=bool(config.get("load_bias_forward", False)),
        patch_train_only=bool(config.get("patch_train_only", True)),
    )


def _unfreeze_deepseek_router_gates(model: Any) -> None:
    """Make DeepSeek MoE gate weights trainable in addition to LoRA adapters."""

    for name, param in model.named_parameters():
        if name.endswith(".mlp.gate.weight") or ".mlp.gate.weight" in name:
            param.requires_grad = True


def _method_name(config: Dict[str, Any]) -> str:
    """Return a stable method name for result aggregation."""

    if config.get("use_pbit_patch", False):
        return "deepseek_qlora_pbit"
    return "deepseek_qlora_baseline"


if __name__ == "__main__":
    main()
