"""Environment, GPU, dtype, and disk inspection helpers.

These functions only inspect local state. They do not download models, load
models, initialize training, or start external services.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


def torch_import_status() -> tuple[bool, str]:
    """Return whether PyTorch imports and a short version/error string."""

    try:
        import torch

        return True, str(torch.__version__)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return False, f"{type(exc).__name__}: {exc}"


def cuda_summary() -> dict[str, Any]:
    """Return CUDA availability and per-device memory details."""

    ok, torch_status = torch_import_status()
    summary: dict[str, Any] = {
        "torch_available": ok,
        "torch_status": torch_status,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }
    if not ok:
        return summary

    import torch

    summary["cuda_available"] = bool(torch.cuda.is_available())
    if not torch.cuda.is_available():
        return summary

    summary["device_count"] = int(torch.cuda.device_count())
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        free_bytes = None
        total_bytes = int(props.total_memory)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        except Exception:
            free_bytes = None
        devices.append(
            {
                "index": index,
                "name": props.name,
                "capability": f"{props.major}.{props.minor}",
                "total_gb": round(total_bytes / 1024**3, 2),
                "free_gb": None if free_bytes is None else round(int(free_bytes) / 1024**3, 2),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    summary["devices"] = devices
    return summary


def get_cuda_summary() -> dict[str, Any]:
    """Return a Stage 2 compatible CUDA summary."""

    summary = cuda_summary()
    devices = summary.get("devices", [])
    return {
        **summary,
        "gpu_count": summary.get("device_count", 0),
        "cuda_version": _cuda_version(),
        "bf16_supported": any(bool(device.get("bf16_supported")) for device in devices),
    }


def _cuda_version() -> str | None:
    """Return the PyTorch CUDA build version when available."""

    ok, _ = torch_import_status()
    if not ok:
        return None
    import torch

    return getattr(torch.version, "cuda", None)


def get_gpu_info() -> list[dict[str, Any]]:
    """Return per-GPU memory information for evaluation summaries."""

    return list(cuda_summary().get("devices", []))


def reset_peak_memory_stats() -> None:
    """Reset CUDA peak memory stats when CUDA is available."""

    ok, _ = torch_import_status()
    if not ok:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def print_gpu_memory(label: str) -> None:
    """Print compact GPU memory information for a named checkpoint."""

    print(f"GPU memory [{label}]:")
    info = get_gpu_info()
    if not info:
        print("  no CUDA devices")
        return
    for device in info:
        print(
            f"  cuda:{device.get('index')}: "
            f"free={device.get('free_gb')} GB total={device.get('total_gb')} GB"
        )


def dtype_support_summary() -> dict[str, Any]:
    """Return BF16/FP16/FP32 support information for the active environment."""

    ok, torch_status = torch_import_status()
    summary: dict[str, Any] = {
        "torch_available": ok,
        "torch_status": torch_status,
        "fp32": True,
        "fp16_cuda": False,
        "bf16_cuda": False,
    }
    if not ok:
        return summary

    import torch

    summary["fp16_cuda"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            summary["bf16_cuda"] = bool(torch.cuda.is_bf16_supported())
        except Exception:
            summary["bf16_cuda"] = False
    return summary


def disk_summary(paths: list[str | os.PathLike[str]]) -> dict[str, dict[str, Any]]:
    """Return disk usage for the filesystem containing each requested path."""

    results: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        probe = path if path.exists() else path.parent
        if not str(probe):
            probe = Path(".")
        try:
            usage = shutil.disk_usage(probe)
            results[str(path)] = {
                "exists": path.exists(),
                "total_gb": round(usage.total / 1024**3, 2),
                "used_gb": round(usage.used / 1024**3, 2),
                "free_gb": round(usage.free / 1024**3, 2),
            }
        except Exception as exc:
            results[str(path)] = {"exists": path.exists(), "error": f"{type(exc).__name__}: {exc}"}
    return results


def hf_environment_summary() -> dict[str, Any]:
    """Return token-safe Hugging Face environment settings."""

    keys = ["HF_ENDPOINT", "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE"]
    summary = {key: os.environ.get(key, "") for key in keys}
    summary["HF_TOKEN"] = "set" if os.environ.get("HF_TOKEN") else "not set"
    summary["HUGGINGFACE_HUB_TOKEN"] = "set" if os.environ.get("HUGGINGFACE_HUB_TOKEN") else "not set"
    return summary


def raise_if_cuda_required(require_cuda: bool) -> None:
    """Raise a clear error when CUDA is required but unavailable."""

    if not require_cuda:
        return
    ok, torch_status = torch_import_status()
    if not ok:
        raise RuntimeError(f"PyTorch is unavailable: {torch_status}")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device_map cpu or run on a CUDA GPU node.")
