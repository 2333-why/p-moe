"""Shared utilities for continued-pretraining scripts.

The helpers in this module are intentionally safe at import time: they do not
load models, datasets, initialize WandB, or allocate CUDA tensors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml


def str_to_bool(value: Any) -> bool:
    """Parse common CLI/config boolean spellings."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def read_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """Read a YAML config file, returning an empty dict when no path is given."""
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def write_yaml_config(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write a YAML mapping with stable key order disabled for readability."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write a JSON file without leaking environment secret values."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


def merge_config(cli_values: argparse.Namespace, file_values: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge YAML and CLI values, with explicit CLI values taking precedence."""
    merged = dict(file_values)
    for key, value in vars(cli_values).items():
        if key == "config":
            continue
        if value is not None:
            merged[key] = value
    return merged


def require_package(import_name: str, install_hint: str):
    """Import an optional dependency or raise a clear installation error."""
    try:
        return __import__(import_name)
    except ImportError as exc:
        raise ImportError(
            f"Missing optional dependency '{import_name}'. Install it with: {install_hint}"
        ) from exc


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when available."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sanitize_env_flags() -> Dict[str, bool]:
    """Return secret-safe environment status flags."""
    return {
        "hf_token_set": bool(os.environ.get("HF_TOKEN")),
        "wandb_api_key_set": bool(os.environ.get("WANDB_API_KEY")),
        "hf_endpoint_set": bool(os.environ.get("HF_ENDPOINT")),
        "hf_home_set": bool(os.environ.get("HF_HOME")),
        "hf_hub_cache_set": bool(os.environ.get("HF_HUB_CACHE")),
        "hf_datasets_cache_set": bool(os.environ.get("HF_DATASETS_CACHE")),
    }


def resolve_torch_dtype(dtype_name: str):
    """Resolve a torch dtype name lazily."""
    import torch

    normalized = str(dtype_name).lower()
    if normalized in {"auto", "none"}:
        return "auto"
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp16":
        return torch.float16
    if normalized == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype '{dtype_name}'. Use auto, bf16, fp16, or fp32.")


def build_training_arguments(config: Mapping[str, Any]):
    """Create Hugging Face TrainingArguments from a normalized config."""
    from transformers import TrainingArguments

    report_to = ["wandb"] if config.get("use_wandb", False) else []
    return TrainingArguments(
        output_dir=str(config["output_dir"]),
        overwrite_output_dir=bool(config.get("overwrite_output_dir", False)),
        max_steps=int(config.get("max_steps", -1)),
        num_train_epochs=float(config.get("num_train_epochs", 1.0)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        learning_rate=float(config.get("learning_rate", 2e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        warmup_steps=int(config.get("warmup_steps", 0)),
        logging_steps=int(config.get("logging_steps", 10)),
        save_steps=int(config.get("save_steps", 500)),
        save_total_limit=int(config.get("save_total_limit", 2)),
        bf16=bool(config.get("bf16", False)),
        fp16=bool(config.get("fp16", False)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
        optim=str(config.get("optim", "adamw_torch")),
        lr_scheduler_type=str(config.get("lr_scheduler_type", "cosine")),
        report_to=report_to,
        run_name=config.get("run_name"),
        dataloader_num_workers=int(config.get("dataloader_num_workers", 0)),
        remove_unused_columns=False,
    )


def load_text_dataset(config: Mapping[str, Any]):
    """Load a Hugging Face text dataset lazily from script entrypoints."""
    datasets = require_package("datasets", "pip install datasets")
    dataset_name = config.get("dataset_name")
    if not dataset_name:
        raise ValueError("dataset_name is required for continued pretraining.")
    dataset_config_name = config.get("dataset_config_name")
    args = [str(dataset_name)]
    if dataset_config_name not in (None, "", "null", "None"):
        args.append(str(dataset_config_name))
    return datasets.load_dataset(
        *args,
        split=str(config.get("dataset_split", "train")),
        data_files=config.get("data_files"),
        streaming=bool(config.get("streaming", False)),
    )


def tokenize_and_group_texts(dataset, tokenizer, config: Mapping[str, Any]):
    """Tokenize text rows and group tokens into fixed-length LM blocks."""
    text_column = str(config.get("text_column", "text"))
    block_size = int(config.get("block_size", 1024))
    preprocessing_num_workers = config.get("preprocessing_num_workers")

    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    def tokenize_function(examples):
        texts = [text for text in examples[text_column] if text is not None and str(text).strip()]
        return tokenizer(texts)

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=list(dataset.column_names),
        num_proc=preprocessing_num_workers,
        desc="Tokenizing dataset",
    )

    def group_texts(examples):
        concatenated = {key: sum(examples[key], []) for key in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            key: [tokens[i : i + block_size] for i in range(0, total_length, block_size)]
            for key, tokens in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenized.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_workers,
        desc=f"Grouping texts into {block_size}-token blocks",
    )


def maybe_init_wandb(config: Mapping[str, Any]) -> None:
    """Initialize WandB only when explicitly requested."""
    if not config.get("use_wandb", False):
        return
    wandb = require_package("wandb", "pip install wandb")
    wandb.init(
        project=config.get("wandb_project", "p-moe"),
        name=config.get("run_name"),
        config={key: value for key, value in config.items() if "token" not in key.lower() and "key" not in key.lower()},
    )


def compute_train_summary(trainer, config: Mapping[str, Any], train_result: Any) -> Dict[str, Any]:
    """Build a compact training summary suitable for result collection."""
    metrics = dict(getattr(train_result, "metrics", {}) or {})
    train_loss = metrics.get("train_loss")
    perplexity = math.exp(train_loss) if isinstance(train_loss, (int, float)) and train_loss < 20 else None
    return {
        "stage": "stage3_lora_qlora",
        "run_name": config.get("run_name"),
        "model_name": config.get("model_name"),
        "dataset_name": config.get("dataset_name"),
        "dataset_config_name": config.get("dataset_config_name"),
        "dataset_split": config.get("dataset_split"),
        "output_dir": config.get("output_dir"),
        "local_files_only": bool(config.get("local_files_only", False)),
        "use_wandb": bool(config.get("use_wandb", False)),
        "bf16": bool(config.get("bf16", False)),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", False)),
        "max_steps": int(config.get("max_steps", -1)),
        "num_train_epochs": float(config.get("num_train_epochs", 1.0)),
        "train_loss": train_loss,
        "train_ppl_estimate": perplexity,
        "metrics": metrics,
        "env": sanitize_env_flags(),
    }


def save_training_artifacts(config: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    """Save the normalized config and train summary under output_dir."""
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml_config(output_dir / "train_config_used.yaml", config)
    write_json(output_dir / "train_summary.json", summary)
