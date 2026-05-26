"""Shared Stage 3 Trainer utilities for LoRA and QLoRA runs."""

from __future__ import annotations

import inspect
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import DownloadConfig, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.logging_utils import create_run_dir, save_json, save_text
from src.lora_utils import resolve_torch_dtype


def load_yaml_config(path: str) -> dict[str, Any]:
    """
    Load a YAML config file.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml_config(config: dict[str, Any], path: str) -> None:
    """
    Save a YAML config file.
    """
    save_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), path)


def apply_cli_overrides(config: dict[str, Any], args: Any) -> dict[str, Any]:
    """
    Apply common Stage 3 command-line overrides to a config dictionary.
    """
    merged = dict(config)
    for key in (
        "model_name",
        "dataset_name",
        "dataset_config_name",
        "train_split",
        "eval_split",
        "dtype",
        "device_map",
        "output_dir",
        "block_size",
    ):
        value = getattr(args, key, None)
        if value is not None:
            merged[key] = value
    if getattr(args, "local_files_only", False):
        merged["local_files_only"] = True
    if getattr(args, "max_seq_length", None) is not None:
        merged["block_size"] = int(args.max_seq_length)
    if getattr(args, "max_steps", None) is not None:
        merged.setdefault("training", {})["max_steps"] = int(args.max_steps)
    if getattr(args, "num_train_epochs", None) is not None:
        merged.setdefault("training", {})["num_train_epochs"] = float(args.num_train_epochs)
    if getattr(args, "gradient_accumulation_steps", None) is not None:
        merged.setdefault("training", {})["gradient_accumulation_steps"] = int(
            args.gradient_accumulation_steps
        )
    if getattr(args, "save_steps", None) is not None:
        merged.setdefault("training", {})["save_steps"] = int(args.save_steps)
    if getattr(args, "learning_rate", None) is not None:
        merged.setdefault("training", {})["learning_rate"] = float(args.learning_rate)
    return merged


def secret_safe_env_summary() -> dict[str, Any]:
    """
    Return environment metadata without exposing raw secret values.
    """
    return {
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "not set"),
        "hf_home": os.environ.get("HF_HOME", "not set"),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE", "not set"),
        "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE", "not set"),
        "hf_token_is_set": bool(os.environ.get("HF_TOKEN")),
        "wandb_api_key_is_set": bool(os.environ.get("WANDB_API_KEY")),
    }


def create_training_run_dir(output_dir: str) -> Path:
    """
    Create a timestamped training run directory under the configured output directory.
    """
    return Path(create_run_dir(output_dir))


def load_tokenizer_for_training(config: dict[str, Any]):
    """
    Load a tokenizer and ensure a pad token is available for batching.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        local_files_only=bool(config.get("local_files_only", False)),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm_for_training(
    config: dict[str, Any],
    quantization_config: Any | None = None,
):
    """
    Load a causal LM for Stage 3 training when the user explicitly runs a train script.
    """
    kwargs: dict[str, Any] = {
        "torch_dtype": resolve_torch_dtype(config.get("dtype", "bf16")),
        "device_map": config.get("device_map", "auto"),
        "trust_remote_code": bool(config.get("trust_remote_code", True)),
        "low_cpu_mem_usage": bool(config.get("low_cpu_mem_usage", True)),
        "local_files_only": bool(config.get("local_files_only", False)),
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    return AutoModelForCausalLM.from_pretrained(config["model_name"], **kwargs)


def load_and_tokenize_lm_dataset(config: dict[str, Any], tokenizer):
    """
    Load WikiText-style text data and tokenize it into fixed-length causal LM blocks.
    """
    dataset_kwargs = {
        "path": config["dataset_name"],
        "name": config.get("dataset_config_name"),
    }
    dataset_kwargs = {key: value for key, value in dataset_kwargs.items() if value}
    if bool(config.get("local_files_only", False)):
        dataset_kwargs["download_config"] = DownloadConfig(local_files_only=True)
    raw = load_dataset(**dataset_kwargs)
    train_split = str(config.get("train_split", "train"))
    eval_split = str(config.get("eval_split", "validation"))
    if train_split not in raw:
        raise ValueError(f"Train split '{train_split}' not found in dataset. Available: {list(raw)}")
    if eval_split not in raw:
        raise ValueError(f"Eval split '{eval_split}' not found in dataset. Available: {list(raw)}")

    text_column = "text" if "text" in raw[train_split].column_names else raw[train_split].column_names[0]
    block_size = int(config.get("block_size", config.get("max_seq_length", 1024)))
    workers = int(config.get("preprocessing_num_workers", 1))

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        texts = [text for text in batch[text_column] if text and not text.isspace()]
        return tokenizer(texts, add_special_tokens=True)

    tokenized = raw.map(
        tokenize_batch,
        batched=True,
        remove_columns=raw[train_split].column_names,
        num_proc=workers,
        desc="Tokenizing text",
    )

    def group_texts(examples: dict[str, list[list[int]]]) -> dict[str, Any]:
        concatenated = {key: sum(examples[key], []) for key in examples}
        total_length = len(concatenated["input_ids"])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            key: [
                values[index : index + block_size]
                for index in range(0, total_length, block_size)
            ]
            for key, values in concatenated.items()
        }
        result["labels"] = list(result["input_ids"])
        return result

    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        num_proc=workers,
        desc=f"Grouping into {block_size}-token blocks",
    )
    return lm_dataset[train_split], lm_dataset[eval_split]


def build_training_arguments(config: dict[str, Any], run_dir: Path, use_wandb: bool) -> TrainingArguments:
    """
    Build TrainingArguments while keeping WandB disabled unless explicitly requested.
    """
    training = dict(config.get("training", {}))
    report_to = ["wandb"] if use_wandb else []
    args: dict[str, Any] = {
        "output_dir": str(run_dir),
        "run_name": config.get("run_name", run_dir.name),
        "overwrite_output_dir": False,
        "num_train_epochs": training.get("num_train_epochs", 1),
        "max_steps": training.get("max_steps", -1),
        "per_device_train_batch_size": training.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size": training.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps", 16),
        "learning_rate": training.get("learning_rate", 1e-4),
        "weight_decay": training.get("weight_decay", 0.0),
        "warmup_ratio": training.get("warmup_ratio", 0.03),
        "lr_scheduler_type": training.get("lr_scheduler_type", "cosine"),
        "logging_steps": training.get("logging_steps", 5),
        "eval_steps": training.get("eval_steps", 50),
        "save_steps": training.get("save_steps", 50),
        "save_total_limit": training.get("save_total_limit", 2),
        "gradient_checkpointing": training.get("gradient_checkpointing", True),
        "bf16": training.get("bf16", config.get("dtype") == "bf16"),
        "fp16": training.get("fp16", config.get("dtype") == "fp16"),
        "optim": training.get("optim", "adamw_torch"),
        "report_to": report_to,
        "remove_unused_columns": False,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        args["eval_strategy"] = "steps"
    else:
        args["evaluation_strategy"] = "steps"
    return TrainingArguments(**args)


def maybe_enable_wandb(use_wandb: bool) -> None:
    """
    Validate optional WandB availability without reading or printing raw keys.
    """
    if not use_wandb:
        os.environ.setdefault("WANDB_DISABLED", "true")
        return
    import importlib.util

    if importlib.util.find_spec("wandb") is None:
        raise RuntimeError("WandB requested with --use_wandb, but wandb is not installed.")
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY: not set. WandB may require prior login or environment setup.")


def run_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: dict[str, Any],
    run_dir: Path,
    use_wandb: bool,
) -> dict[str, Any]:
    """
    Run Hugging Face Trainer and save adapter/checkpoint artifacts.
    """
    training_args = build_training_arguments(config, run_dir, use_wandb)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    train_output = trainer.train()
    trainer.save_model(str(run_dir / "final_adapter"))
    tokenizer.save_pretrained(str(run_dir / "final_adapter"))
    metrics = dict(train_output.metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    state_path = run_dir / "trainer_state.json"
    if state_path.exists():
        shutil.copy2(state_path, run_dir / "trainer_state.final.json")
    return metrics


def save_train_summary(
    run_dir: Path,
    config: dict[str, Any],
    success: bool,
    metrics: dict[str, Any] | None = None,
    error: BaseException | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Save a secret-safe train_summary.json file.
    """
    summary: dict[str, Any] = {
        "model_name": config.get("model_name"),
        "dataset_name": config.get("dataset_name"),
        "dataset_config_name": config.get("dataset_config_name"),
        "success": success,
        "metrics": metrics or {},
        **secret_safe_env_summary(),
    }
    if extra:
        summary.update(extra)
    if error is not None:
        summary["error_type"] = type(error).__name__
        summary["error_message"] = redact_known_secrets(str(error))
    save_json(summary, str(run_dir / "train_summary.json"))


def redact_known_secrets(text: str) -> str:
    """
    Redact known secret environment values from a loggable string.
    """
    redacted = text
    for name in ("HF_TOKEN", "WANDB_API_KEY"):
        value = os.environ.get(name)
        if value and len(value) >= 4:
            redacted = redacted.replace(value, f"<{name}:redacted>")
    return redacted
