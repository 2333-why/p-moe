"""Model and tokenizer loading helpers for Stage 1 inference."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


OOM_SUGGESTION = """Suggestions:
1. Use multiple GPUs with device_map="auto".
2. Add max_memory config.
3. Use CPU offload only for loading tests.
4. Try a smaller model first.
5. Consider quantized loading in a later stage."""

DOWNLOAD_SUGGESTION = """Suggestions:
1. Check the network connection.
2. Check Hugging Face access.
3. Check whether HF_ENDPOINT should be set.
4. Check disk space.
5. Check whether huggingface-cli login is required."""

DISK_SUGGESTION = (
    "DeepSeek-MoE-16B weights are large. Please ensure sufficient disk space "
    "for Hugging Face cache."
)


def resolve_dtype(dtype_name: str):
    """
    Support bf16, fp16, and fp32.
    bf16 -> torch.bfloat16
    fp16 -> torch.float16
    fp32 -> torch.float32
    """
    normalized = dtype_name.lower().strip()
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in dtype_map:
        valid = ", ".join(sorted(dtype_map))
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Expected one of: {valid}")
    return dtype_map[normalized]


def load_tokenizer(
    model_name: str,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
):
    """
    Load tokenizer.
    If pad_token_id is missing, try to set it to eos_token_id.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_fast=True,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load tokenizer for '{model_name}'.\n{classify_exception(exc)}"
        ) from exc

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_name: str,
    dtype_name: str = "bf16",
    device_map: str | dict[str, Any] | None = "auto",
    trust_remote_code: bool = True,
    low_cpu_mem_usage: bool = True,
    max_memory: dict[str, Any] | None = None,
    offload_folder: str | None = None,
    local_files_only: bool = False,
):
    """
    Load AutoModelForCausalLM and return the model.
    """
    resolved_dtype = resolve_dtype(dtype_name)
    kwargs: dict[str, Any] = {
        "torch_dtype": resolved_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "local_files_only": local_files_only,
    }
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    if offload_folder:
        kwargs["offload_folder"] = offload_folder

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(f"CUDA out of memory while loading model.\n{OOM_SUGGESTION}") from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model '{model_name}'.\n{classify_exception(exc)}"
        ) from exc

    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map is not None:
        print("hf_device_map:")
        print(hf_device_map)
        has_cpu, has_disk = detect_offload(hf_device_map)
        if has_cpu:
            print("Warning: hf_device_map contains CPU offload. Inference may be slow.")
        if has_disk:
            print("Warning: hf_device_map contains disk offload. Memory is likely insufficient.")
    else:
        print("hf_device_map is not available on the loaded model.")
    return model


def detect_offload(hf_device_map: dict[str, Any] | None) -> tuple[bool, bool]:
    """
    Return whether a Hugging Face device map contains CPU or disk offload.
    """
    if not hf_device_map:
        return False, False
    values = [str(value).lower() for value in hf_device_map.values()]
    return any(value == "cpu" for value in values), any(value == "disk" for value in values)


def classify_exception(exc: BaseException) -> str:
    """
    Return an actionable, secret-safe suggestion for common model loading failures.
    """
    message = str(exc)
    lowered = message.lower()
    if "cuda out of memory" in lowered:
        return OOM_SUGGESTION
    if any(term in lowered for term in ("no space left", "disk quota", "errno 28")):
        return DISK_SUGGESTION
    if any(
        term in lowered
        for term in (
            "connection",
            "timed out",
            "timeout",
            "http",
            "401",
            "403",
            "404",
            "repository not found",
            "gated",
            "huggingface",
            "couldn't connect",
        )
    ):
        return DOWNLOAD_SUGGESTION
    return (
        "Check the traceback above, installed package versions, available GPU memory, "
        "Hugging Face access, and free disk space."
    )
