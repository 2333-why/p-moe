"""Run Stage 1 DeepSeek-MoE loading and text generation probe."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation_utils import generate_text
from src.logging_utils import create_run_dir, save_json, save_text
from src.model_utils import classify_exception, detect_offload, load_causal_lm, load_tokenizer

try:
    from src.gpu_utils import get_cuda_summary, get_gpu_info, print_gpu_memory, reset_peak_memory_stats
except ModuleNotFoundError:
    def get_gpu_info() -> list[dict[str, Any]]:
        """Fallback GPU memory snapshot until Agent B provides src.gpu_utils."""
        if not torch.cuda.is_available():
            return []
        gpu_info = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpu_info.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / 1024**3,
                    "allocated_memory_gb": torch.cuda.memory_allocated(index) / 1024**3,
                    "reserved_memory_gb": torch.cuda.memory_reserved(index) / 1024**3,
                    "peak_allocated_memory_gb": torch.cuda.max_memory_allocated(index) / 1024**3,
                }
            )
        return gpu_info

    def print_gpu_memory(stage: str) -> None:
        """Fallback GPU memory printer until Agent B provides src.gpu_utils."""
        print(f"GPU memory at {stage}:")
        info = get_gpu_info()
        if not info:
            print("  CUDA is not available. DeepSeek-MoE-16B inference is not recommended on CPU.")
            return
        for gpu in info:
            print(
                "  GPU {index} {name}: total={total_memory_gb:.2f} GB, "
                "allocated={allocated_memory_gb:.2f} GB, reserved={reserved_memory_gb:.2f} GB, "
                "peak={peak_allocated_memory_gb:.2f} GB".format(**gpu)
            )

    def reset_peak_memory_stats() -> None:
        """Fallback peak memory reset until Agent B provides src.gpu_utils."""
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(index)

    def get_cuda_summary() -> dict[str, Any]:
        """Fallback CUDA summary until Agent B provides src.gpu_utils."""
        return {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        }


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "generation_config.yaml"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line overrides for important generation settings.
    """
    parser = argparse.ArgumentParser(description="Stage 1 DeepSeek-MoE generation probe")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--offload_folder", default=None)
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        default=None,
        help="Load tokenizer/model only from local files or Hugging Face cache.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load YAML config from disk, returning an empty config if it does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        print(f"Config file not found: {path}. Using built-in defaults.")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """
    Apply non-null CLI overrides to a config dictionary.
    """
    merged = dict(default_config())
    merged.update(config)
    for key in (
        "model_name",
        "dtype",
        "device_map",
        "prompt",
        "max_new_tokens",
        "output_dir",
        "offload_folder",
        "local_files_only",
    ):
        value = getattr(args, key)
        if value is not None:
            merged[key] = value
    return merged


def default_config() -> dict[str, Any]:
    """
    Return safe defaults matching configs/generation_config.yaml.
    """
    return {
        "model_name": "deepseek-ai/deepseek-moe-16b-base",
        "dtype": "bf16",
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "prompt": "The history of artificial intelligence can be traced back to",
        "max_new_tokens": 120,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "output_dir": "outputs",
        "local_files_only": False,
    }


def env_summary() -> dict[str, Any]:
    """
    Return secret-safe Hugging Face and CUDA environment information.
    """
    return {
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "not set"),
        "hf_home": os.environ.get("HF_HOME", "not set"),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE", "not set"),
        "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE", "not set"),
        "hf_token_is_set": bool(os.environ.get("HF_TOKEN")),
        "wandb_api_key_is_set": bool(os.environ.get("WANDB_API_KEY")),
        "cuda_summary": get_cuda_summary(),
    }


def redact_secrets(text: str) -> str:
    """
    Redact known secret environment variable values from loggable text.
    """
    redacted = text
    for name in ("HF_TOKEN", "WANDB_API_KEY"):
        value = os.environ.get(name)
        if value and len(value) >= 4:
            redacted = redacted.replace(value, f"<{name}:redacted>")
    return redacted


def print_environment_summary(summary: dict[str, Any]) -> None:
    """
    Print a compact, secret-safe environment summary.
    """
    print("Environment summary:")
    for key in ("hf_endpoint", "hf_home", "hf_hub_cache", "hf_datasets_cache"):
        print(f"  {key}: {summary[key]}")
    print(f"  HF_TOKEN: {'set' if summary['hf_token_is_set'] else 'not set'}")
    print(f"  WANDB_API_KEY: {'set' if summary['wandb_api_key_is_set'] else 'not set'}")
    cuda_summary = summary["cuda_summary"]
    print(f"  CUDA available: {cuda_summary.get('cuda_available')}")
    print(f"  CUDA version: {cuda_summary.get('cuda_version')}")
    print(f"  GPU count: {cuda_summary.get('gpu_count')}")
    print(f"  BF16 supported: {cuda_summary.get('bf16_supported')}")
    if not cuda_summary.get("cuda_available"):
        print("CUDA is not available. DeepSeek-MoE-16B inference is not recommended on CPU.")


def serialize_device_map(device_map: Any) -> dict[str, str]:
    """
    Convert model.hf_device_map into a JSON-serializable dictionary.
    """
    if not isinstance(device_map, dict):
        return {}
    return {str(key): str(value) for key, value in device_map.items()}


def write_failure_summary(
    run_dir: Path,
    config: dict[str, Any],
    exc: BaseException,
    gpu_info_before_loading: list[dict[str, Any]] | None = None,
    gpu_info_after_loading: list[dict[str, Any]] | None = None,
    load_success: bool = False,
) -> None:
    """
    Save a failure summary with actionable suggestions.
    """
    error_message = redact_secrets(str(exc))
    suggestion = classify_exception(exc)
    failure = {
        "model_name": config.get("model_name"),
        "dtype": config.get("dtype"),
        "device_map": config.get("device_map"),
        "load_success": load_success,
        "generation_success": False,
        "error_type": type(exc).__name__,
        "error_message": error_message,
        "suggestion": suggestion,
        "gpu_info_before_loading": gpu_info_before_loading or [],
        "gpu_info_after_loading": gpu_info_after_loading or [],
        **env_summary(),
    }
    save_json(failure, str(run_dir / "summary.json"))
    print("Failure summary saved to", run_dir / "summary.json")
    print("Error:", error_message)
    print(suggestion)


def main() -> int:
    """
    Execute model loading, text generation, and result logging.
    """
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    run_dir = Path(create_run_dir(config.get("output_dir", "outputs")))
    print(f"Run directory: {run_dir}")
    save_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), str(run_dir / "config_used.yaml"))

    env = env_summary()
    print_environment_summary(env)
    reset_peak_memory_stats()

    gpu_info_before_loading = get_gpu_info()
    print_gpu_memory("before_loading")
    gpu_info_after_loading: list[dict[str, Any]] = []
    model_loaded = False

    try:
        print("Loading tokenizer...")
        tokenizer = load_tokenizer(
            config["model_name"],
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            local_files_only=bool(config.get("local_files_only", False)),
        )

        print("Loading model...")
        model = load_causal_lm(
            model_name=config["model_name"],
            dtype_name=config.get("dtype", "bf16"),
            device_map=config.get("device_map", "auto"),
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            low_cpu_mem_usage=bool(config.get("low_cpu_mem_usage", True)),
            max_memory=config.get("max_memory"),
            offload_folder=config.get("offload_folder"),
            local_files_only=bool(config.get("local_files_only", False)),
        )
        model_loaded = True
        gpu_info_after_loading = get_gpu_info()
        print_gpu_memory("after_loading")

        hf_device_map = serialize_device_map(getattr(model, "hf_device_map", None))
        save_json(hf_device_map, str(run_dir / "device_map.json"))
        has_cpu_offload, has_disk_offload = detect_offload(hf_device_map)

        print("Generating text...")
        generation = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=str(config["prompt"]),
            max_new_tokens=int(config.get("max_new_tokens", 120)),
            do_sample=bool(config.get("do_sample", True)),
            temperature=float(config.get("temperature", 0.7)),
            top_p=float(config.get("top_p", 0.9)),
            repetition_penalty=float(config.get("repetition_penalty", 1.05)),
        )
        gpu_info_after_generation = get_gpu_info()
        print_gpu_memory("after_generation")

        save_text(generation["generated_text"], str(run_dir / "generated.txt"))
        summary = {
            "model_name": config["model_name"],
            "dtype": config.get("dtype", "bf16"),
            "device_map": config.get("device_map", "auto"),
            "load_success": True,
            "generation_success": True,
            **generation,
            "gpu_info_before_loading": gpu_info_before_loading,
            "gpu_info_after_loading": gpu_info_after_loading,
            "gpu_info_after_generation": gpu_info_after_generation,
            "hf_device_map": hf_device_map,
            "has_cpu_offload": has_cpu_offload,
            "has_disk_offload": has_disk_offload,
            **env,
        }
        save_json(summary, str(run_dir / "summary.json"))

        print("\nGenerated text:")
        print(generation["generated_text"])
        print("\nKey metrics:")
        print(f"  New tokens: {generation['new_tokens']}")
        print(f"  Generation time: {generation['generation_time_sec']:.3f} sec")
        print(f"  Tokens/sec: {generation['tokens_per_sec']:.3f}")
        print(f"  CPU offload: {has_cpu_offload}")
        print(f"  Disk offload: {has_disk_offload}")
        print(f"Summary saved to {run_dir / 'summary.json'}")
        return 0
    except Exception as exc:
        print(redact_secrets(traceback.format_exc()), file=sys.stderr)
        write_failure_summary(
            run_dir=run_dir,
            config=config,
            exc=exc,
            gpu_info_before_loading=gpu_info_before_loading,
            gpu_info_after_loading=gpu_info_after_loading,
            load_success=model_loaded,
        )
        return 1


# TODO Stage 2:
# Add WikiText-2 perplexity evaluation.
#
# TODO Stage 3:
# Add LoRA / QLoRA continued pretraining.
#
# TODO Stage 4:
# Inspect and modify MoE router for hard-forward p-bit-backward routing.


if __name__ == "__main__":
    raise SystemExit(main())
