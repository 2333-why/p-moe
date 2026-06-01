"""Metrics for sparse MoE router diagnostics."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch import Tensor, nn


def expert_load_distribution(mask: Tensor, eps: float = 1.0e-8) -> Tensor:
    """Return normalized expert activation frequency over all routed tokens."""
    flat = _as_2d(mask).float()
    counts = flat.sum(dim=0)
    total = counts.sum().clamp_min(eps)
    return counts / total


def load_variance(mask: Tensor, eps: float = 1.0e-8) -> Tensor:
    """Return variance of normalized expert load."""
    load = expert_load_distribution(mask, eps=eps)
    return torch.var(load, unbiased=False)


def expert_entropy(mask: Tensor, eps: float = 1.0e-8, normalized: bool = True) -> Tensor:
    """Return entropy of expert load distribution."""
    load = expert_load_distribution(mask, eps=eps).clamp_min(eps)
    entropy = -(load * load.log()).sum()
    if normalized and load.numel() > 1:
        entropy = entropy / torch.log(torch.tensor(float(load.numel()), device=load.device))
    return entropy


def dead_expert_ratio(mask: Tensor, threshold: float = 0.0, eps: float = 1.0e-8) -> Tensor:
    """Return fraction of experts whose normalized load is at or below threshold."""
    load = expert_load_distribution(mask, eps=eps)
    return (load <= threshold).float().mean()


def routing_jaccard_stability(previous_mask: Tensor, current_mask: Tensor, eps: float = 1.0e-8) -> Tensor:
    """Return average Jaccard similarity between two routing masks."""
    if previous_mask.shape != current_mask.shape:
        raise ValueError("previous_mask and current_mask must have the same shape.")
    prev = _as_2d(previous_mask).bool()
    curr = _as_2d(current_mask).bool()
    intersection = (prev & curr).sum(dim=-1).float()
    union = (prev | curr).sum(dim=-1).float().clamp_min(eps)
    return (intersection / union).mean()


def gradient_norm(
    module_or_parameters: nn.Module | Iterable[nn.Parameter],
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> float:
    """Return gradient norm for a module or iterable of parameters."""
    parameters = (
        module_or_parameters.parameters()
        if isinstance(module_or_parameters, nn.Module)
        else module_or_parameters
    )
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    device = grads[0].device
    norms = torch.stack([torch.linalg.vector_norm(g.to(device), ord=norm_type) for g in grads])
    total = torch.linalg.vector_norm(norms, ord=norm_type)
    if error_if_nonfinite and not torch.isfinite(total):
        raise RuntimeError("Non-finite gradient norm.")
    return float(total.item())


def router_metric_summary(mask: Tensor, previous_mask: Optional[Tensor] = None) -> Dict[str, float]:
    """Return common router metrics as Python floats."""
    summary = {
        "load_variance": float(load_variance(mask).item()),
        "expert_entropy": float(expert_entropy(mask).item()),
        "dead_expert_ratio": float(dead_expert_ratio(mask).item()),
    }
    if previous_mask is not None:
        summary["routing_jaccard_stability"] = float(
            routing_jaccard_stability(previous_mask, mask).item()
        )
    return summary


def _as_2d(mask: Tensor) -> Tensor:
    """Flatten all leading dimensions while preserving expert dimension."""
    if mask.ndim < 1:
        raise ValueError("mask must include an expert dimension.")
    return mask.reshape(-1, mask.shape[-1])
