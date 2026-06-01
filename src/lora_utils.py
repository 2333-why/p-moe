"""LoRA and QLoRA helpers for Stage 3 continued pretraining.

All optional ML dependencies are imported inside functions so this module stays
safe to import in lightweight environments.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .train_utils import require_package


def parse_lora_target_modules(value: Any):
    """Normalize target module config into the PEFT format."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not text or text.lower() in {"auto", "none"}:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def build_lora_config(config: Mapping[str, Any]):
    """Create a PEFT LoraConfig from a script config mapping."""
    peft = require_package("peft", "pip install peft")
    target_modules = parse_lora_target_modules(config.get("lora_target_modules"))
    return peft.LoraConfig(
        r=int(config.get("lora_r", 16)),
        lora_alpha=int(config.get("lora_alpha", 32)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias=str(config.get("lora_bias", "none")),
        task_type=str(config.get("task_type", "CAUSAL_LM")),
        target_modules=target_modules,
    )


def apply_lora(model, config: Mapping[str, Any]):
    """Attach LoRA adapters to a causal LM model."""
    peft = require_package("peft", "pip install peft")
    lora_config = build_lora_config(config)
    return peft.get_peft_model(model, lora_config)


def prepare_model_for_lora_training(model, config: Mapping[str, Any]):
    """Apply gradient checkpointing and PEFT input-gradient preparation."""
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    peft = require_package("peft", "pip install peft")
    if hasattr(peft, "prepare_model_for_kbit_training") and config.get("prepare_model_for_kbit_training", False):
        model = peft.prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
        )
    return model


def build_qlora_quantization_config(config: Mapping[str, Any]):
    """Create a Transformers BitsAndBytesConfig for 4-bit QLoRA loading."""
    require_package("bitsandbytes", "pip install bitsandbytes")
    from transformers import BitsAndBytesConfig
    import torch

    compute_dtype = torch.bfloat16 if config.get("bnb_4bit_compute_dtype", "bf16") == "bf16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(config.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(config.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def print_trainable_parameters(model) -> None:
    """Print trainable parameter counts without exposing secrets."""
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
        return
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    percent = 100.0 * trainable / total if total else 0.0
    print(f"trainable params: {trainable} || all params: {total} || trainable%: {percent:.4f}")


def router_lora_todo() -> str:
    """Return the explicit router-specific LoRA follow-up note for reports."""
    return "TODO: add router-specific LoRA target selection and p-bit ablation hooks after router inspection."
