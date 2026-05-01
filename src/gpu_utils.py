"""CUDA and GPU memory helpers for Stage 1 inference probing."""

from __future__ import annotations

from typing import Any


GB = 1024**3


def _load_torch() -> Any | None:
    """Import torch lazily so environment checks can run without PyTorch installed."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def _bytes_to_gb(value: int | float) -> float:
    """Convert bytes to GB using a binary GiB divisor."""
    return float(value) / GB


def get_gpu_info() -> list[dict]:
    """
    Return information for every GPU.

    Each item includes GPU index, name, total memory, allocated memory,
    reserved memory, and peak allocated memory. Memory values are in GB.
    """
    torch = _load_torch()
    if torch is None or not torch.cuda.is_available():
        return []

    gpu_info: list[dict] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpu_info.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_gb": _bytes_to_gb(props.total_memory),
                "allocated_memory_gb": _bytes_to_gb(torch.cuda.memory_allocated(index)),
                "reserved_memory_gb": _bytes_to_gb(torch.cuda.memory_reserved(index)),
                "peak_allocated_memory_gb": _bytes_to_gb(
                    torch.cuda.max_memory_allocated(index)
                ),
            }
        )
    return gpu_info


def print_gpu_memory(stage: str) -> None:
    """
    Print GPU memory usage for a named stage.

    Example stages: before_loading, after_loading, after_generation.
    """
    print(f"== GPU memory: {stage} ==")
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("No CUDA GPUs available.")
        return

    for gpu in gpu_info:
        print(f"GPU {gpu['index']}: {gpu['name']}")
        print(f"  total_memory_gb:          {gpu['total_memory_gb']:.2f}")
        print(f"  allocated_memory_gb:      {gpu['allocated_memory_gb']:.2f}")
        print(f"  reserved_memory_gb:       {gpu['reserved_memory_gb']:.2f}")
        print(f"  peak_allocated_memory_gb: {gpu['peak_allocated_memory_gb']:.2f}")


def reset_peak_memory_stats() -> None:
    """Reset PyTorch CUDA peak memory statistics for all available GPUs."""
    torch = _load_torch()
    if torch is None or not torch.cuda.is_available():
        return

    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)


def get_cuda_summary() -> dict:
    """
    Return CUDA availability, GPU count, CUDA version, BF16 support, and GPU names.

    The function is intentionally defensive so it can be used on CPU-only
    machines or incomplete Python environments.
    """
    torch = _load_torch()
    if torch is None:
        return {
            "torch_installed": False,
            "cuda_available": False,
            "cuda_version": "not available",
            "gpu_count": 0,
            "gpu_names": [],
            "bf16_supported": False,
        }

    cuda_available = bool(torch.cuda.is_available())
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu_names = [torch.cuda.get_device_name(index) for index in range(gpu_count)]

    bf16_supported = False
    if cuda_available and hasattr(torch.cuda, "is_bf16_supported"):
        try:
            bf16_supported = bool(torch.cuda.is_bf16_supported())
        except RuntimeError:
            bf16_supported = False

    return {
        "torch_installed": True,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda or "not available",
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "bf16_supported": bf16_supported,
    }
