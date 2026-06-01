"""Runtime p-bit patch for DeepSeek-MoE gate modules.

This module does not modify Hugging Face cache files. It replaces in-memory
``MoEGate.forward`` methods after a DeepSeek-MoE model has been loaded by an
explicit training or evaluation script.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from types import MethodType
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.router_metrics import router_metric_summary


@dataclass
class DeepSeekPBitPatchConfig:
    """Configuration for runtime patching DeepSeek-MoE gate modules."""

    enabled: bool = True
    alpha: float = 0.1
    beta: float = 0.1
    temperature: float = 1.0
    min_temperature: float = 0.2
    load_ema_decay: float = 0.99
    use_load_bias: bool = True
    use_competition: bool = True
    normalize_topk_prob: bool | None = None
    patch_train_only: bool = True
    eps: float = 1.0e-20


def is_deepseek_moe_gate(module: nn.Module) -> bool:
    """Return whether a module looks like DeepSeek's ``MoEGate``."""

    return (
        module.__class__.__name__ == "MoEGate"
        and hasattr(module, "weight")
        and hasattr(module, "top_k")
        and hasattr(module, "n_routed_experts")
    )


def list_deepseek_moe_gates(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """List DeepSeek-MoE gate modules by name."""

    return [(name, module) for name, module in model.named_modules() if is_deepseek_moe_gate(module)]


def apply_deepseek_pbit_patch(
    model: nn.Module,
    config: DeepSeekPBitPatchConfig,
    *,
    module_names: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """Patch selected DeepSeek ``MoEGate`` modules in memory.

    The patch preserves hard top-k expert dispatch. It only changes the gate
    weights returned by ``MoEGate.forward`` so their backward path uses a p-bit
    straight-through surrogate with optional load-aware bias and competition.
    """

    if not config.enabled:
        return {"enabled": False, "patched_modules": [], "num_patched": 0}

    allowed = set(module_names) if module_names is not None else None
    patched: list[str] = []
    for name, module in list_deepseek_moe_gates(model):
        if allowed is not None and name not in allowed:
            continue
        _patch_gate_module(module, name, config)
        patched.append(name)

    if not patched:
        raise RuntimeError(
            "No DeepSeek MoEGate modules were patched. Run scripts/inspect_router.py first "
            "and verify that model.layers.*.mlp.gate modules exist."
        )
    return {
        "enabled": True,
        "config": asdict(config),
        "patched_modules": patched,
        "num_patched": len(patched),
    }


def collect_deepseek_pbit_metrics(model: nn.Module) -> Dict[str, Any]:
    """Collect aggregate p-bit gate metrics from patched DeepSeek modules."""

    modules: List[Dict[str, Any]] = []
    variances: list[float] = []
    entropies: list[float] = []
    dead_ratios: list[float] = []
    router_grad_norms: list[float] = []

    for name, module in list_deepseek_moe_gates(model):
        if not getattr(module, "_pmoe_pbit_patched", False):
            continue
        load = getattr(module, "_pmoe_load_ema", None)
        hard_mask = getattr(module, "_pmoe_last_hard_mask", None)
        if isinstance(hard_mask, Tensor):
            metric_values = router_metric_summary(hard_mask.detach())
        elif isinstance(load, Tensor):
            metric_values = _metrics_from_load(load.detach())
        else:
            metric_values = {}

        grad_norm = _parameter_grad_norm([module.weight])
        router_grad_norms.append(grad_norm)
        if "load_variance" in metric_values:
            variances.append(float(metric_values["load_variance"]))
        if "expert_entropy" in metric_values:
            entropies.append(float(metric_values["expert_entropy"]))
        if "dead_expert_ratio" in metric_values:
            dead_ratios.append(float(metric_values["dead_expert_ratio"]))
        modules.append(
            {
                "name": name,
                "load_updates": int(getattr(module, "_pmoe_load_updates", torch.tensor(0)).item()),
                "router_grad_norm": grad_norm,
                **metric_values,
            }
        )

    return {
        "patched_gate_count": len(modules),
        "load_variance": _mean_or_none(variances),
        "expert_entropy": _mean_or_none(entropies),
        "dead_expert_ratio": _mean_or_none(dead_ratios),
        "router_grad_norm": _mean_or_none(router_grad_norms),
        "modules": modules,
    }


def _patch_gate_module(module: nn.Module, name: str, config: DeepSeekPBitPatchConfig) -> None:
    """Attach p-bit state and replace one gate module forward method."""

    if getattr(module, "_pmoe_pbit_patched", False):
        return
    num_experts = int(module.n_routed_experts)
    initial_load = torch.full((num_experts,), 1.0 / num_experts)
    module.register_buffer("_pmoe_load_ema", initial_load)
    module.register_buffer("_pmoe_load_updates", torch.zeros((), dtype=torch.long))
    module._pmoe_original_forward = module.forward
    module._pmoe_pbit_config = config
    module._pmoe_pbit_name = name
    module._pmoe_pbit_patched = True
    module.forward = MethodType(_pbit_gate_forward, module)


def _pbit_gate_forward(self: nn.Module, hidden_states: Tensor):
    """DeepSeek ``MoEGate.forward`` replacement with p-bit ST gate weights."""

    config: DeepSeekPBitPatchConfig = self._pmoe_pbit_config
    if config.patch_train_only and not self.training:
        return self._pmoe_original_forward(hidden_states)

    bsz, seq_len, hidden_dim = hidden_states.shape
    flat_hidden = hidden_states.view(-1, hidden_dim)
    logits = F.linear(flat_hidden, self.weight, None)
    if self.scoring_func == "softmax":
        scores = logits.softmax(dim=-1)
    else:
        raise NotImplementedError(f"Unsupported scoring function for p-bit MoE gate: {self.scoring_func}")

    bias = _load_bias(self, scores, config)
    hard_scores = scores + bias
    _, topk_idx = torch.topk(hard_scores, k=self.top_k, dim=-1, sorted=False)
    hard_mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)

    local_field = logits
    if config.use_competition and config.alpha != 0.0:
        active_others = hard_mask.sum(dim=-1, keepdim=True) - hard_mask
        local_field = local_field - config.alpha * active_others
    local_field = local_field + bias

    temperature = max(float(config.temperature), float(config.min_temperature))
    q = torch.sigmoid(local_field / temperature)
    st_mask = hard_mask.detach() - q.detach() + q

    selected_scores = scores.gather(dim=-1, index=topk_idx)
    selected_st = st_mask.gather(dim=-1, index=topk_idx)
    topk_weight = selected_scores.detach() - selected_st.detach() + selected_st

    normalize = self.norm_topk_prob if config.normalize_topk_prob is None else config.normalize_topk_prob
    if self.top_k > 1 and normalize:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + config.eps
        topk_weight = topk_weight / denominator

    _update_load_state(self, hard_mask, config)
    self._pmoe_last_hard_mask = hard_mask.detach()
    self._pmoe_last_topk_idx = topk_idx.detach()

    aux_loss = _deepseek_aux_loss(
        self,
        scores=scores,
        topk_idx=topk_idx,
        bsz=bsz,
        seq_len=seq_len,
    )
    return topk_idx, topk_weight, aux_loss


def _load_bias(module: nn.Module, scores: Tensor, config: DeepSeekPBitPatchConfig) -> Tensor:
    """Return broadcasted load-aware bias for one patched gate."""

    if not config.use_load_bias or config.beta == 0.0:
        return torch.zeros_like(scores)
    load = module._pmoe_load_ema.to(device=scores.device, dtype=scores.dtype)
    target = 1.0 / int(module.n_routed_experts)
    bias = config.beta * (target - load)
    return bias.view(1, -1).expand_as(scores)


@torch.no_grad()
def _update_load_state(module: nn.Module, hard_mask: Tensor, config: DeepSeekPBitPatchConfig) -> None:
    """Update per-gate EMA load from the hard forward mask."""

    denom = hard_mask.sum(dim=-1, keepdim=True).clamp_min(float(config.eps))
    batch_load = (hard_mask / denom).mean(dim=0)
    load = module._pmoe_load_ema.to(device=hard_mask.device, dtype=hard_mask.dtype)
    load.mul_(float(config.load_ema_decay)).add_(batch_load, alpha=1.0 - float(config.load_ema_decay))
    module._pmoe_load_ema.copy_(load.to(device=module._pmoe_load_ema.device, dtype=module._pmoe_load_ema.dtype))
    module._pmoe_load_updates.add_(1)


def _deepseek_aux_loss(
    module: nn.Module,
    *,
    scores: Tensor,
    topk_idx: Tensor,
    bsz: int,
    seq_len: int,
) -> Tensor | None:
    """Reproduce DeepSeek ``MoEGate`` auxiliary loss for patched gates."""

    if not module.training or float(module.alpha) <= 0.0:
        return None
    aux_topk = int(module.top_k)
    topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
    if module.seq_aux:
        scores_for_seq_aux = scores.view(bsz, seq_len, -1)
        ce = torch.zeros(bsz, module.n_routed_experts, device=scores.device)
        ones = torch.ones(bsz, seq_len * aux_topk, device=scores.device)
        ce.scatter_add_(1, topk_idx_for_aux_loss, ones).div_(seq_len * aux_topk / module.n_routed_experts)
        return (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * module.alpha

    mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=module.n_routed_experts)
    ce = mask_ce.float().mean(0)
    pi = scores.mean(0)
    fi = ce * module.n_routed_experts
    return (pi * fi).sum() * module.alpha


def _metrics_from_load(load: Tensor) -> Dict[str, float]:
    """Compute metrics from an already-normalized load vector."""

    eps = 1.0e-8
    load = load.float()
    load = load / load.sum().clamp_min(eps)
    entropy = -(load.clamp_min(eps) * load.clamp_min(eps).log()).sum()
    if load.numel() > 1:
        entropy = entropy / math.log(float(load.numel()))
    return {
        "load_variance": float(torch.var(load, unbiased=False).item()),
        "expert_entropy": float(entropy.item()),
        "dead_expert_ratio": float((load <= 0.0).float().mean().item()),
    }


def _parameter_grad_norm(parameters: Iterable[nn.Parameter]) -> float:
    """Return L2 gradient norm for a small parameter iterable."""

    grads = [param.grad.detach() for param in parameters if param.grad is not None]
    if not grads:
        return 0.0
    device = grads[0].device
    norms = torch.stack([torch.linalg.vector_norm(grad.to(device), ord=2) for grad in grads])
    return float(torch.linalg.vector_norm(norms, ord=2).item())


def _mean_or_none(values: list[float]) -> float | None:
    """Return the arithmetic mean for non-empty lists."""

    if not values:
        return None
    return float(sum(values) / len(values))
