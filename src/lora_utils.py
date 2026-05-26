"""LoRA and QLoRA helpers for Stage 3 continued pretraining."""

from __future__ import annotations

import importlib.util
from typing import Any

import torch


def require_package(package_name: str, install_hint: str) -> None:
    """
    Raise a clear error when an optional Stage 3 dependency is unavailable.
    """
    if importlib.util.find_spec(package_name) is None:
        raise RuntimeError(
            f"Optional dependency '{package_name}' is required for this command. "
            f"Install it explicitly first, for example: {install_hint}"
        )


def resolve_torch_dtype(dtype_name: str):
    """
    Resolve bf16, fp16, and fp32 dtype names to torch dtypes.
    """
    normalized = str(dtype_name).lower().strip()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Expected bf16, fp16, or fp32.")
    return mapping[normalized]


def build_lora_config(lora_config: dict[str, Any]):
    """
    Build a PEFT LoraConfig from a plain dictionary.
    """
    require_package("peft", "pip install peft")
    from peft import LoraConfig

    target_modules = lora_config.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]
    return LoraConfig(
        r=int(lora_config.get("r", 8)),
        lora_alpha=int(lora_config.get("alpha", 16)),
        lora_dropout=float(lora_config.get("dropout", 0.05)),
        bias=str(lora_config.get("bias", "none")),
        task_type=str(lora_config.get("task_type", "CAUSAL_LM")),
        target_modules=list(target_modules),
    )


def apply_lora(model, lora_config: dict[str, Any]):
    """
    Attach trainable LoRA adapters to a model.
    """
    require_package("peft", "pip install peft")
    from peft import get_peft_model

    peft_config = build_lora_config(lora_config)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def build_qlora_quantization_config(qlora_config: dict[str, Any]):
    """
    Build a Transformers BitsAndBytesConfig for 4-bit QLoRA loading.
    """
    require_package("bitsandbytes", "pip install bitsandbytes")
    from transformers import BitsAndBytesConfig

    compute_dtype = resolve_torch_dtype(qlora_config.get("bnb_4bit_compute_dtype", "bf16"))
    return BitsAndBytesConfig(
        load_in_4bit=bool(qlora_config.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(qlora_config.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(qlora_config.get("bnb_4bit_use_double_quant", True)),
    )


def prepare_model_for_qlora(model, gradient_checkpointing: bool = True):
    """
    Prepare a quantized causal LM for k-bit PEFT training.
    """
    require_package("peft", "pip install peft")
    from peft import prepare_model_for_kbit_training

    return prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(gradient_checkpointing),
    )


def trainable_parameter_summary(model) -> dict[str, int | float]:
    """
    Return trainable and total parameter counts for a model.
    """
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    ratio = trainable / total if total else 0.0
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": ratio,
    }
