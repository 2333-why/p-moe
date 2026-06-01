"""Small random-score checks for p-bit router primitives.

This script is intentionally self-contained and does not load models, datasets,
or external assets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pbit_router import PBitBackwardRouter, PBitRouterConfig, sample_top_k_mask, top_k_mask
from src.router_metrics import router_metric_summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Random-score p-bit router smoke checks.")
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_sampling", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def run_checks(args: argparse.Namespace) -> Dict[str, object]:
    """Run shape and gradient checks on random router scores."""
    torch.manual_seed(args.seed)
    scores = torch.randn(args.batch_size, args.seq_len, args.num_experts, requires_grad=True)
    config = PBitRouterConfig(
        num_experts=args.num_experts,
        top_k=args.top_k,
        alpha=args.alpha,
        beta=args.beta,
        temperature=args.temperature,
        use_sampling=args.use_sampling,
    )
    router = PBitBackwardRouter(config)
    gates, aux = router(scores, return_aux=True)
    loss = (gates * scores).sum()
    loss.backward()
    hard = aux["hard_mask"]
    sampled = sample_top_k_mask(aux["surrogate_prob"].detach(), args.top_k)
    topk = top_k_mask(scores.detach(), args.top_k)
    return {
        "scores_shape": list(scores.shape),
        "gates_shape": list(gates.shape),
        "hard_selected_per_token_min": float(hard.sum(dim=-1).min().item()),
        "hard_selected_per_token_max": float(hard.sum(dim=-1).max().item()),
        "topk_selected_per_token_min": float(topk.sum(dim=-1).min().item()),
        "sampled_selected_per_token_min": float(sampled.sum(dim=-1).min().item()),
        "score_grad_norm": float(scores.grad.norm().item()) if scores.grad is not None else 0.0,
        "metrics": router_metric_summary(hard),
    }


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    summary = run_checks(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
