"""Model and tokenizer loading helpers for DeepSeek/MoE scaffolding.

No model or tokenizer is loaded at import time. Callers opt into loading via
explicit functions and may force offline behavior with ``local_files_only``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def parse_torch_dtype(dtype_name: str | None):
    """Map bf16/fp16/fp32/auto strings to torch dtype values."""

    if dtype_name is None or dtype_name == "auto":
        return "auto"
    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency diagnostic
        raise RuntimeError(f"PyTorch is required to parse dtype {dtype_name!r}: {exc}") from exc

    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError("Unsupported dtype. Expected one of: auto, bf16, fp16, fp32.")


def local_model_exists(model_name: str) -> bool:
    """Return true when model_name points to an existing local path."""

    return Path(model_name).expanduser().exists()


def looks_like_local_model_path(model_name: str) -> bool:
    """Return true when a model reference is intended to be a filesystem path."""

    path = Path(model_name).expanduser()
    return (
        path.is_absolute()
        or model_name.startswith(("./", "../", "~/"))
        or model_name == "models"
        or model_name.startswith("models/")
        or model_name.startswith("models\\")
    )


def looks_like_local_model_path(model_name: str) -> bool:
    """Return true when a model reference is intended to be a filesystem path."""

    path = Path(model_name).expanduser()
    return (
        path.is_absolute()
        or model_name.startswith(("./", "../", "~/"))
        or model_name == "models"
        or model_name.startswith("models/")
        or model_name.startswith("models\\")
    )


def build_model_load_kwargs(
    *,
    dtype: str = "bf16",
    device_map: str | None = "auto",
    local_files_only: bool = False,
    offload_folder: str | None = None,
    trust_remote_code: bool = True,
    revision: str | None = None,
    low_cpu_mem_usage: bool = True,
) -> dict[str, Any]:
    """Build kwargs for ``AutoModelForCausalLM.from_pretrained``."""

    kwargs: dict[str, Any] = {
        "torch_dtype": parse_torch_dtype(dtype),
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    if device_map:
        kwargs["device_map"] = device_map
    if revision:
        kwargs["revision"] = revision
    if offload_folder:
        kwargs["offload_folder"] = offload_folder
        kwargs["offload_state_dict"] = True
        Path(offload_folder).mkdir(parents=True, exist_ok=True)
    return kwargs


def explain_load_error(exc: BaseException, *, model_name: str, local_files_only: bool) -> RuntimeError:
    """Convert common loading failures into clear actionable errors."""

    text = str(exc)
    lower = text.lower()
    prefix = f"Failed to load model/tokenizer {model_name!r}."
    if "cuda out of memory" in lower or "outofmemoryerror" in lower:
        return RuntimeError(
            f"{prefix} CUDA OOM. Try --device_map auto, --offload_folder outputs/offload, "
            "--dtype fp16/bf16, a smaller model, or more GPU memory."
        )
    if local_files_only and ("couldn't find" in lower or "not found" in lower or "no such file" in lower):
        return RuntimeError(
            f"{prefix} Local files were requested but required files are missing. "
            "Run scripts/download_assets.py first or pass the correct local model path."
        )
    if "no space left on device" in lower or "disk quota" in lower:
        return RuntimeError(f"{prefix} Disk space is insufficient. Check HF_HOME/cache and offload folders.")
    if "connection" in lower or "timed out" in lower or "offline" in lower:
        return RuntimeError(
            f"{prefix} Network/cache access failed. For offline runs, pass --local_files_only "
            "and point --model_name to a downloaded local directory."
        )
    if "trust_remote_code" in lower:
        return RuntimeError(f"{prefix} This model may require --trust_remote_code.")
    return RuntimeError(f"{prefix} Original error: {type(exc).__name__}: {exc}")


def load_tokenizer(
    model_name: str,
    *,
    local_files_only: bool = False,
    revision: str | None = None,
    trust_remote_code: bool = True,
    use_fast: bool | None = None,
):
    """Load a tokenizer with explicit offline and revision controls."""

    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - dependency diagnostic
        raise RuntimeError("transformers is required to load tokenizers. Install requirements.txt.") from exc

    kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if revision:
        kwargs["revision"] = revision
    if use_fast is not None:
        kwargs["use_fast"] = use_fast
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    except Exception as exc:
        raise explain_load_error(exc, model_name=model_name, local_files_only=local_files_only) from exc
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_name: str,
    *,
    dtype: str = "bf16",
    dtype_name: str | None = None,
    device_map: str | None = "auto",
    local_files_only: bool = False,
    offload_folder: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = True,
    low_cpu_mem_usage: bool = True,
):
    """Load a causal language model with clear DeepSeek-friendly defaults."""

    try:
        from transformers import AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover - dependency diagnostic
        raise RuntimeError("transformers is required to load causal LMs. Install requirements.txt.") from exc

    if (
        local_files_only
        and looks_like_local_model_path(model_name)
        and not local_model_exists(model_name)
    ):
        raise RuntimeError(
            f"Local model path {model_name!r} does not exist while --local_files_only is set. "
            "Download assets first or pass a valid local directory."
        )
    kwargs = build_model_load_kwargs(
        dtype=dtype_name or dtype,
        device_map=device_map,
        local_files_only=local_files_only,
        offload_folder=offload_folder,
        trust_remote_code=trust_remote_code,
        revision=revision,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as exc:
        raise explain_load_error(exc, model_name=model_name, local_files_only=local_files_only) from exc


def infer_offload_status(model: Any) -> dict[str, Any]:
    """Infer whether accelerate dispatched any modules to CPU or disk."""

    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return {"has_device_map": False, "cpu_modules": 0, "disk_modules": 0, "device_map": None}
    values = [str(value) for value in device_map.values()]
    return {
        "has_device_map": True,
        "cpu_modules": sum(value == "cpu" for value in values),
        "disk_modules": sum(value == "disk" for value in values),
        "device_map": device_map,
    }


def detect_offload(device_map: dict[str, Any]) -> tuple[bool, bool]:
    """Return whether a serialized device map uses CPU or disk offload."""

    values = [str(value).lower() for value in device_map.values()]
    return any(value == "cpu" for value in values), any(value == "disk" for value in values)


def classify_exception(exc: BaseException) -> str:
    """Return a concise, secret-safe suggestion for common runtime failures."""

    text = str(exc).lower()
    if "out of memory" in text or "cuda" in text and "memory" in text:
        return "CUDA memory error. Try smaller block/stride settings, offload, or a larger GPU."
    if "local" in text and ("not found" in text or "missing" in text):
        return "Local files are missing. Download assets first or pass a valid local path."
    if "dataset" in text or "cache" in text:
        return "Dataset cache/load error. Verify dataset config, split, and HF cache settings."
    if "transformers" in text or "datasets" in text:
        return "Missing dependency. Install requirements.txt in the active environment."
    return f"{type(exc).__name__}: review the saved traceback and command arguments."


def model_parameter_summary(model: Any) -> dict[str, Any]:
    """Return a compact parameter count and dtype/device summary."""

    total = 0
    trainable = 0
    dtype_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}
    for param in model.parameters():
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
        dtype_counts[str(param.dtype)] = dtype_counts.get(str(param.dtype), 0) + count
        device_counts[str(param.device)] = device_counts.get(str(param.device), 0) + count
    return {
        "parameters": total,
        "trainable_parameters": trainable,
        "dtype_counts": dtype_counts,
        "device_counts": device_counts,
    }


def huggingface_token_kwargs() -> dict[str, Any]:
    """Return a token kwarg without exposing token values in logs."""

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return {"token": token} if token else {}
