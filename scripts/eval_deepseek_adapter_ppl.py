"""Evaluate a DeepSeek base model plus PEFT adapter on WikiText PPL."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import concatenate_texts, extract_nonempty_texts, load_wikitext_dataset
from src.eval_utils import compute_sliding_window_ppl
from src.gpu_utils import get_gpu_info
from src.logging_utils import create_run_dir, save_json, save_text
from src.lora_utils import build_qlora_quantization_config
from src.model_utils import detect_offload, load_tokenizer
from src.train_utils import read_yaml_config, require_package, str_to_bool


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--base_model", default="deepseek-ai/deepseek-moe-16b-base")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--local_files_only", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--dataset_name", default="wikitext")
    parser.add_argument("--dataset_config_name", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max_eval_tokens", type=int, default=None)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--load_in_4bit", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--bnb_4bit_quant_type", default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--bnb_4bit_use_double_quant", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("--bnb_4bit_compute_dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--no_progress", action="store_true")
    return parser


def main() -> int:
    """Load base model and adapter, then compute PPL."""

    args = build_parser().parse_args()
    config = _merge_config(args)
    run_dir = Path(create_run_dir(str(config["output_dir"])))
    save_text(yaml.safe_dump(config, sort_keys=False), str(run_dir / "eval_config_used.yaml"))

    try:
        summary = run_eval(config)
        summary["run_dir"] = str(run_dir)
        save_json(summary, str(run_dir / "eval_summary.json"))
        print(f"Adapter PPL: {summary['ppl']:.6f}")
        print(f"Summary saved to {run_dir / 'eval_summary.json'}")
        return 0
    except Exception as exc:
        summary = {
            "stage": "stage6_deepseek_adapter_ppl",
            "success": False,
            "failure": True,
            "base_model": config.get("base_model"),
            "adapter_path": config.get("adapter_path"),
            "error_message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
        save_json(summary, str(run_dir / "eval_summary.json"))
        raise


def run_eval(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run adapter perplexity evaluation."""

    from transformers import AutoModelForCausalLM

    peft = require_package("peft", "pip install peft")
    tokenizer = load_tokenizer(
        str(config["base_model"]),
        local_files_only=bool(config.get("local_files_only", True)),
        trust_remote_code=True,
    )

    model_kwargs: Dict[str, Any] = {
        "local_files_only": bool(config.get("local_files_only", True)),
        "trust_remote_code": True,
        "device_map": config.get("device_map", "auto"),
    }
    if bool(config.get("load_in_4bit", True)):
        model_kwargs["quantization_config"] = build_qlora_quantization_config(config)

    base_model = AutoModelForCausalLM.from_pretrained(str(config["base_model"]), **model_kwargs)
    model = peft.PeftModel.from_pretrained(
        base_model,
        str(config["adapter_path"]),
        is_trainable=False,
        local_files_only=bool(config.get("local_files_only", True)),
    )

    dataset = load_wikitext_dataset(
        dataset_name=str(config["dataset_name"]),
        dataset_config_name=config.get("dataset_config_name"),
        split=str(config["split"]),
        local_files_only=bool(config.get("local_files_only", True)),
    )
    text = concatenate_texts(extract_nonempty_texts(dataset, text_column=str(config.get("text_column", "text"))))
    before = get_gpu_info()
    ppl = compute_sliding_window_ppl(
        model,
        tokenizer,
        text,
        block_size=int(config["block_size"]),
        stride=int(config["stride"]),
        max_eval_tokens=config.get("max_eval_tokens"),
        show_progress=not bool(config.get("no_progress", False)),
    )
    after = get_gpu_info()
    device_map = getattr(model, "hf_device_map", getattr(base_model, "hf_device_map", {}))
    has_cpu_offload, has_disk_offload = detect_offload(device_map if isinstance(device_map, dict) else {})
    return {
        "stage": "stage6_deepseek_adapter_ppl",
        "method": "deepseek_adapter_ppl",
        "success": True,
        "failure": False,
        "model_name": config.get("base_model"),
        "adapter_path": config.get("adapter_path"),
        "dataset_name": config.get("dataset_name"),
        "dataset_config_name": config.get("dataset_config_name"),
        "split": config.get("split"),
        "gpu_info_before_eval": before,
        "gpu_info_after_eval": after,
        "has_cpu_offload": has_cpu_offload,
        "has_disk_offload": has_disk_offload,
        **ppl,
    }


def _merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge optional YAML config and CLI arguments."""

    config = read_yaml_config(args.config)
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            config[key] = value
    return config


if __name__ == "__main__":
    raise SystemExit(main())
