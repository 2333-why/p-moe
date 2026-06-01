"""Run a single Stage 1 generation probe.

This script loads the requested model only when executed directly. Use
--local_files_only with a local model path for offline GPU-server runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation_utils import generate_text
from src.gpu_utils import cuda_summary, disk_summary
from src.logging_utils import write_json
from src.model_utils import infer_offload_status, load_causal_lm, load_tokenizer, model_parameter_summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Load a causal LM and generate text for Stage 1 validation.")
    parser.add_argument("--model_name", default="models/deepseek-moe-16b-base", help="Local path or HF model id.")
    parser.add_argument("--revision", default=None, help="Optional model revision.")
    parser.add_argument("--prompt", default="The key idea of sparse mixture-of-experts routing is", help="Prompt text.")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16", help="Model dtype.")
    parser.add_argument("--device_map", default="auto", help="Transformers/Accelerate device_map, e.g. auto/cpu.")
    parser.add_argument("--device", default=None, help="Optional explicit input tensor device.")
    parser.add_argument("--local_files_only", action="store_true", help="Do not access the network.")
    parser.add_argument("--offload_folder", default=None, help="Optional accelerate disk offload directory.")
    parser.add_argument("--max_new_tokens", type=int, default=120, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling p.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling cutoff.")
    parser.add_argument("--no_sample", action="store_true", help="Use greedy/beam-free deterministic generation.")
    parser.add_argument("--summary_path", default="outputs/generation_summary.json", help="Summary JSON path.")
    return parser


def main() -> None:
    """Load model/tokenizer and generate a short sample."""

    args = build_parser().parse_args()
    tokenizer = load_tokenizer(args.model_name, local_files_only=args.local_files_only, revision=args.revision)
    model = load_causal_lm(
        args.model_name,
        dtype=args.dtype,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
        offload_folder=args.offload_folder,
        revision=args.revision,
    )
    model.eval()
    generation = generate_text(
        model,
        tokenizer,
        args.prompt,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=not args.no_sample,
    )
    summary = {
        "model_name": args.model_name,
        "revision": args.revision,
        "local_files_only": args.local_files_only,
        "offload_folder": args.offload_folder,
        "offload_status": infer_offload_status(model),
        "parameter_summary": model_parameter_summary(model),
        "cuda": cuda_summary(),
        "disk": disk_summary(["models", "outputs", args.offload_folder or "outputs"]),
        "generation": generation,
    }
    path = write_json(args.summary_path, summary)
    print(generation["text"])
    print(f"Wrote generation summary to {path}")


if __name__ == "__main__":
    main()
