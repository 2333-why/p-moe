"""Run Stage 2 WikiText perplexity evaluation for a causal LM."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import concatenate_texts, extract_nonempty_texts, load_wikitext_dataset
from src.eval_utils import compute_sliding_window_ppl
from src.gpu_utils import get_cuda_summary, get_gpu_info, print_gpu_memory, reset_peak_memory_stats
from src.logging_utils import create_run_dir, save_json, save_text
from src.model_utils import classify_exception, detect_offload, load_causal_lm, load_tokenizer


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "eval_wikitext_config.yaml"


def parse_args() -> argparse.Namespace:
    """
    Parse Stage 2 evaluation config path and CLI overrides.
    """
    parser = argparse.ArgumentParser(description="Stage 2 WikiText perplexity evaluation")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_config_name", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--text_column", default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None, help="Deprecated alias for --block_size")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max_eval_tokens", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--offload_folder", default=None)
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        default=None,
        help="Load model and dataset only from local files or Hugging Face caches.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress display.",
    )
    return parser.parse_args()


def default_config() -> dict[str, Any]:
    """
    Return defaults matching configs/eval_wikitext_config.yaml.
    """
    return {
        "model_name": "models/deepseek-moe-16b-base",
        "dtype": "bf16",
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        "offload_folder": None,
        "dataset_name": "wikitext",
        "dataset_config_name": "wikitext-2-raw-v1",
        "split": "test",
        "text_column": "text",
        "block_size": 2048,
        "stride": 1024,
        "max_eval_tokens": None,
        "output_dir": "outputs",
    }


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load a YAML config from disk, returning an empty dict when missing.
    """
    path = Path(config_path)
    if not path.exists():
        print(f"Config file not found: {path}. Using built-in defaults.")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """
    Merge defaults, YAML config, and non-null CLI overrides.
    """
    merged = default_config()
    merged.update(config)
    for key in (
        "model_name",
        "dtype",
        "device_map",
        "dataset_name",
        "dataset_config_name",
        "split",
        "text_column",
        "block_size",
        "stride",
        "max_eval_tokens",
        "output_dir",
        "offload_folder",
        "local_files_only",
    ):
        value = getattr(args, key)
        if value is not None:
            merged[key] = value
    if args.max_length is not None:
        merged["block_size"] = int(args.max_length)
    return merged


def env_summary() -> dict[str, Any]:
    """
    Return secret-safe environment fields relevant to evaluation.
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
    Remove known secret values from text before printing or saving.
    """
    redacted = text
    for name in ("HF_TOKEN", "WANDB_API_KEY"):
        value = os.environ.get(name)
        if value and len(value) >= 4:
            redacted = redacted.replace(value, f"<{name}:redacted>")
    return redacted


def serialize_device_map(device_map: Any) -> dict[str, str]:
    """
    Convert a Hugging Face device map into a JSON-safe dictionary.
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
    Save eval_summary.json for failed evaluation runs.
    """
    error_message = redact_secrets(str(exc))
    if "Dataset loading failed" in error_message:
        suggestion = error_message
    else:
        suggestion = classify_exception(exc)
    summary = {
        "stage": "stage_2_wikitext_ppl",
        "model_name": config.get("model_name"),
        "dtype": config.get("dtype"),
        "device_map": config.get("device_map"),
        "dataset_name": config.get("dataset_name"),
        "dataset_config_name": config.get("dataset_config_name"),
        "split": config.get("split"),
        "load_success": load_success,
        "eval_success": False,
        "error_type": type(exc).__name__,
        "error_message": error_message,
        "suggestion": suggestion,
        "gpu_info_before_loading": gpu_info_before_loading or [],
        "gpu_info_after_loading": gpu_info_after_loading or [],
        **env_summary(),
    }
    save_json(summary, str(run_dir / "eval_summary.json"))
    print("Failure summary saved to", run_dir / "eval_summary.json")
    print("Error:", error_message)
    print(suggestion)


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
        print("CUDA is not available. DeepSeek-MoE-16B evaluation is not recommended on CPU.")


def main() -> int:
    """
    Execute WikiText loading, model loading, sliding-window PPL, and result saving.
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
        print("Loading WikiText dataset...")
        dataset = load_wikitext_dataset(
            dataset_name=str(config["dataset_name"]),
            dataset_config_name=config.get("dataset_config_name"),
            split=str(config["split"]),
            local_files_only=bool(config.get("local_files_only", False)),
        )
        texts = extract_nonempty_texts(dataset, text_column=str(config.get("text_column", "text")))
        eval_text = concatenate_texts(texts)

        print("Loading tokenizer...")
        tokenizer = load_tokenizer(
            str(config["model_name"]),
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            local_files_only=bool(config.get("local_files_only", False)),
        )

        print("Loading model...")
        model = load_causal_lm(
            model_name=str(config["model_name"]),
            dtype_name=str(config.get("dtype", "bf16")),
            device_map=config.get("device_map", "auto"),
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            low_cpu_mem_usage=bool(config.get("low_cpu_mem_usage", True)),
            offload_folder=config.get("offload_folder"),
            local_files_only=bool(config.get("local_files_only", False)),
        )
        model_loaded = True
        gpu_info_after_loading = get_gpu_info()
        print_gpu_memory("after_loading")

        hf_device_map = serialize_device_map(getattr(model, "hf_device_map", None))
        save_json(hf_device_map, str(run_dir / "device_map.json"))
        has_cpu_offload, has_disk_offload = detect_offload(hf_device_map)
        gpu_info_before_eval = get_gpu_info()

        print("Computing sliding-window perplexity...")
        eval_result = compute_sliding_window_ppl(
            model=model,
            tokenizer=tokenizer,
            text=eval_text,
            block_size=int(config.get("block_size", 2048)),
            stride=int(config.get("stride", 1024)),
            max_eval_tokens=config.get("max_eval_tokens"),
            show_progress=not args.no_progress,
        )
        gpu_info_after_eval = get_gpu_info()
        print_gpu_memory("after_eval")

        summary = {
            "stage": "stage_2_wikitext_ppl",
            "model_name": config["model_name"],
            "dtype": config.get("dtype", "bf16"),
            "device_map": config.get("device_map", "auto"),
            "dataset_name": config["dataset_name"],
            "dataset_config_name": config.get("dataset_config_name"),
            "split": config["split"],
            "text_column": config.get("text_column", "text"),
            "load_success": True,
            "eval_success": True,
            **eval_result,
            "num_text_rows": len(texts),
            "gpu_info_before_loading": gpu_info_before_loading,
            "gpu_info_after_loading": gpu_info_after_loading,
            "gpu_info_before_eval": gpu_info_before_eval,
            "gpu_info_after_eval": gpu_info_after_eval,
            "hf_device_map": hf_device_map,
            "has_cpu_offload": has_cpu_offload,
            "has_disk_offload": has_disk_offload,
            **env,
        }
        save_json(summary, str(run_dir / "eval_summary.json"))

        print("\nEvaluation metrics:")
        print(f"  Perplexity: {eval_result['ppl']:.4f}")
        print(f"  Mean NLL: {eval_result['mean_nll']:.6f}")
        print(f"  Target tokens: {eval_result['target_tokens']}")
        print(f"  Eval time: {eval_result['eval_time_sec']:.3f} sec")
        print(f"  Tokens/sec: {eval_result['tokens_per_sec']:.3f}")
        print(f"  CPU offload: {has_cpu_offload}")
        print(f"  Disk offload: {has_disk_offload}")
        print(f"Summary saved to {run_dir / 'eval_summary.json'}")
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


if __name__ == "__main__":
    raise SystemExit(main())
