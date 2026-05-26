"""Stage 3 QLoRA continued pretraining entry point.

This script intentionally does no work at import time. It loads the model and
starts Trainer only when the user explicitly runs it.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lora_utils import (
    apply_lora,
    build_qlora_quantization_config,
    prepare_model_for_qlora,
    require_package,
    trainable_parameter_summary,
)
from src.train_utils import (
    apply_cli_overrides,
    create_training_run_dir,
    load_and_tokenize_lm_dataset,
    load_causal_lm_for_training,
    load_tokenizer_for_training,
    load_yaml_config,
    maybe_enable_wandb,
    redact_known_secrets,
    run_trainer,
    save_train_summary,
    save_yaml_config,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_qlora_config.yaml"


def parse_args() -> argparse.Namespace:
    """
    Parse Stage 3 QLoRA training command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Stage 3 QLoRA continued pretraining framework")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to YAML config")
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_config_name", default=None)
    parser.add_argument("--train_split", default=None)
    parser.add_argument("--eval_split", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None, help="Deprecated alias for --block_size")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB Trainer reporting.")
    return parser.parse_args()


def main() -> int:
    """
    Load config, quantize the base model, attach LoRA adapters, and run Trainer.
    """
    args = parse_args()
    config = apply_cli_overrides(load_yaml_config(args.config), args)
    run_dir = create_training_run_dir(str(config.get("output_dir", "outputs")))
    save_yaml_config(config, str(run_dir / "train_config_used.yaml"))

    try:
        require_package("peft", "pip install peft")
        require_package("bitsandbytes", "pip install bitsandbytes")
        maybe_enable_wandb(args.use_wandb)

        print(f"Run directory: {run_dir}")
        print("Loading tokenizer...")
        tokenizer = load_tokenizer_for_training(config)
        print("Building 4-bit quantization config...")
        quantization_config = build_qlora_quantization_config(config.get("qlora", {}))
        print("Loading model for QLoRA training...")
        model = load_causal_lm_for_training(config, quantization_config=quantization_config)
        model.config.use_cache = False

        print("Preparing quantized model for PEFT training...")
        model = prepare_model_for_qlora(
            model,
            gradient_checkpointing=bool(config.get("training", {}).get("gradient_checkpointing", True)),
        )
        print("Applying LoRA adapters...")
        model = apply_lora(model, config.get("lora", {}))
        parameter_summary = trainable_parameter_summary(model)

        print("Loading and tokenizing dataset...")
        train_dataset, eval_dataset = load_and_tokenize_lm_dataset(config, tokenizer)

        print("Starting Trainer. This is the first point where training begins.")
        metrics = run_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            config=config,
            run_dir=run_dir,
            use_wandb=args.use_wandb,
        )
        save_train_summary(
            run_dir,
            config,
            success=True,
            metrics=metrics,
            extra={
                "qlora": config.get("qlora", {}),
                "lora": config.get("lora", {}),
                **parameter_summary,
            },
        )
        print(f"Training summary saved to {run_dir / 'train_summary.json'}")
        return 0
    except Exception as exc:
        print(redact_known_secrets(traceback.format_exc()), file=sys.stderr)
        save_train_summary(run_dir, config, success=False, error=exc)
        print(f"Failure summary saved to {run_dir / 'train_summary.json'}")
        return 1


# TODO Stage 4:
# Add router-specific LoRA target discovery after MoE router modules are inspected.
#
# TODO Stage 5:
# Add p-bit backward-router ablation switches after the surrogate objective is defined.


if __name__ == "__main__":
    raise SystemExit(main())
