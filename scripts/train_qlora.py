"""Stage 3 QLoRA continued pretraining entrypoint.

The script supports 4-bit bitsandbytes loading plus PEFT LoRA adapters. It is
safe at import time and performs no model or dataset work until main() runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lora_utils import (
    apply_lora,
    build_qlora_quantization_config,
    prepare_model_for_lora_training,
    print_trainable_parameters,
    router_lora_todo,
)
from src.train_utils import (
    build_training_arguments,
    compute_train_summary,
    load_text_dataset,
    maybe_init_wandb,
    merge_config,
    read_yaml_config,
    save_training_artifacts,
    seed_everything,
    str_to_bool,
    tokenize_and_group_texts,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for QLoRA continued pretraining."""
    parser = argparse.ArgumentParser(description="QLoRA continued pretraining scaffold for p-MoE Stage 3.")
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument("--model_name", default=None, help="Model name or local model directory.")
    parser.add_argument("--output_dir", default=None, help="Directory for adapter checkpoints and summaries.")
    parser.add_argument("--local_files_only", type=str_to_bool, nargs="?", const=True, default=None)
    parser.add_argument("--bf16", type=str_to_bool, nargs="?", const=True, default=None)
    parser.add_argument("--gradient_checkpointing", type=str_to_bool, nargs="?", const=True, default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_config_name", default=None)
    parser.add_argument("--dataset_split", default=None)
    parser.add_argument("--text_column", default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_target_modules", default=None)
    parser.add_argument("--lora_bias", default=None)
    parser.add_argument("--bnb_4bit_quant_type", default=None, choices=["nf4", "fp4"])
    parser.add_argument("--bnb_4bit_use_double_quant", type=str_to_bool, nargs="?", const=True, default=None)
    parser.add_argument("--bnb_4bit_compute_dtype", default=None, choices=["bf16", "fp16"])
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--use_wandb", type=str_to_bool, nargs="?", const=True, default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def normalize_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge YAML and CLI configuration and fill safe QLoRA defaults."""
    config = merge_config(args, read_yaml_config(args.config))
    defaults = {
        "model_name": "models/deepseek-moe-16b-base",
        "output_dir": "outputs/stage3_qlora",
        "local_files_only": True,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "dataset_name": "wikitext",
        "dataset_config_name": "wikitext-2-raw-v1",
        "dataset_split": "train",
        "text_column": "text",
        "block_size": 1024,
        "max_steps": -1,
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "weight_decay": 0.0,
        "warmup_steps": 100,
        "save_steps": 500,
        "save_total_limit": 2,
        "logging_steps": 10,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": None,
        "lora_bias": "none",
        "task_type": "CAUSAL_LM",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bf16",
        "device_map": "auto",
        "use_wandb": False,
        "wandb_project": "p-moe",
        "run_name": "stage3_qlora",
        "seed": 42,
        "optim": "paged_adamw_8bit",
        "lr_scheduler_type": "cosine",
        "dataloader_num_workers": 0,
        "preprocessing_num_workers": None,
        "overwrite_output_dir": False,
        "prepare_model_for_kbit_training": True,
        "router_lora_todo": router_lora_todo(),
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    return config


def run_training(config: Dict[str, Any]) -> None:
    """Load a 4-bit model, attach LoRA adapters, and run Trainer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer

    seed_everything(int(config["seed"]))
    maybe_init_wandb(config)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        local_files_only=bool(config["local_files_only"]),
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = build_qlora_quantization_config(config)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        local_files_only=bool(config["local_files_only"]),
        quantization_config=quantization_config,
        device_map=config.get("device_map", "auto"),
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )
    model = prepare_model_for_lora_training(model, config)
    model = apply_lora(model, config)
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
    summary = compute_train_summary(trainer, config, train_result)
    save_training_artifacts(config, summary)


def main() -> None:
    """Parse arguments and launch QLoRA training."""
    parser = build_parser()
    config = normalize_config(parser.parse_args())
    Path(str(config["output_dir"])).mkdir(parents=True, exist_ok=True)
    run_training(config)


if __name__ == "__main__":
    main()
