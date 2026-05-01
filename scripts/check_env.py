"""Print the Stage 1 environment and GPU readiness summary."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gpu_utils import get_cuda_summary, get_gpu_info  # noqa: E402


def _import_version(module_name: str) -> str:
    """Return an installed package version or a readable unavailable marker."""
    try:
        module = __import__(module_name)
    except ImportError:
        return "not installed"
    return getattr(module, "__version__", "unknown")


def _env_value(name: str) -> str:
    """Return an environment variable value, or 'not set' when absent."""
    value = os.environ.get(name)
    return value if value else "not set"


def _secret_status(name: str) -> str:
    """Return only whether a secret environment variable is set."""
    return "set" if os.environ.get(name) else "not set"


def _format_gb(num_bytes: int) -> str:
    """Format bytes as GiB with two decimal places."""
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def _print_disk_usage(label: str, path: str | Path) -> None:
    """Print disk usage for a path when it exists or can be resolved."""
    try:
        resolved = Path(path).expanduser()
        if not resolved.exists():
            print(f"{label}: {resolved} does not exist")
            return
        usage = shutil.disk_usage(resolved)
    except OSError as exc:
        print(f"{label}: unable to read disk usage ({exc})")
        return

    print(f"{label}: {resolved}")
    print(f"  total: {_format_gb(usage.total)}")
    print(f"  used:  {_format_gb(usage.used)}")
    print(f"  free:  {_format_gb(usage.free)}")


def _print_gpu_info() -> None:
    """Print per-GPU memory information in GB."""
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("GPU details: none")
        return

    print("GPU details:")
    for gpu in gpu_info:
        print(f"  GPU {gpu['index']}: {gpu['name']}")
        print(f"    total_memory_gb:          {gpu['total_memory_gb']:.2f}")
        print(f"    allocated_memory_gb:      {gpu['allocated_memory_gb']:.2f}")
        print(f"    reserved_memory_gb:       {gpu['reserved_memory_gb']:.2f}")
        print(f"    peak_allocated_memory_gb: {gpu['peak_allocated_memory_gb']:.2f}")


def main() -> None:
    """Run the lightweight Stage 1 environment check."""
    cuda_summary = get_cuda_summary()

    print("== Python and package versions ==")
    print(f"Python: {sys.version.replace(os.linesep, ' ')}")
    print(f"PyTorch: {_import_version('torch')}")
    print(f"Transformers: {_import_version('transformers')}")
    print()

    print("== CUDA summary ==")
    print(f"CUDA available: {cuda_summary['cuda_available']}")
    print(f"CUDA version: {cuda_summary['cuda_version']}")
    print(f"GPU count: {cuda_summary['gpu_count']}")
    print(f"BF16 supported: {cuda_summary['bf16_supported']}")
    if not cuda_summary["cuda_available"]:
        print("CUDA is not available. DeepSeek-MoE-16B inference is not recommended on CPU.")
    print()

    print("== GPU memory ==")
    _print_gpu_info()
    print()

    print("== Paths and environment ==")
    print(f"Current working directory: {Path.cwd()}")
    print(f"HF_ENDPOINT: {_env_value('HF_ENDPOINT')}")
    print(f"HF_HOME: {_env_value('HF_HOME')}")
    print(f"HF_HUB_CACHE: {_env_value('HF_HUB_CACHE')}")
    print(f"HF_DATASETS_CACHE: {_env_value('HF_DATASETS_CACHE')}")
    print(f"HF_TOKEN: {_secret_status('HF_TOKEN')}")
    print(f"WANDB_API_KEY: {_secret_status('WANDB_API_KEY')}")
    print()

    print("== Disk space ==")
    _print_disk_usage("Current directory", Path.cwd())

    hf_cache_paths = [
        ("HF_HOME", os.environ.get("HF_HOME")),
        ("HF_HUB_CACHE", os.environ.get("HF_HUB_CACHE")),
        ("HF_DATASETS_CACHE", os.environ.get("HF_DATASETS_CACHE")),
    ]
    seen_paths: set[Path] = set()
    for label, value in hf_cache_paths:
        if not value:
            continue
        resolved = Path(value).expanduser()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        _print_disk_usage(label, resolved)


if __name__ == "__main__":
    main()
