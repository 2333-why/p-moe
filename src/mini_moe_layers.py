"""Mini-MoE layers and routing utilities for controllable experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiniMoERouterConfig:
    """Configuration for Mini-MoE router behavior."""

    hidden_size: int = 256
    num_experts: int = 8
    top_k: int = 2
    routing_method: str = "pbit"
    noisy_std: float = 1.0
    temperature: float = 1.0
    min_temperature: float = 0.2
    alpha: float = 0.1
    beta: float = 0.1
    load_ema_decay: float = 0.99
    dead_expert_threshold: float = 0.01
    warmup_steps: int = 0
    warmup_mode: str = "none"
    warmup_top_k: int = 4
    eps: float = 1.0e-8


def top_k_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Return a binary mask selecting the top-k scores along the last axis."""

    if k <= 0:
        raise ValueError("top_k must be positive")
    k = min(k, scores.shape[-1])
    indices = torch.topk(scores, k=k, dim=-1).indices
    mask = torch.zeros_like(scores)
    return mask.scatter(-1, indices, 1.0)


def straight_through_mask(hard_mask: torch.Tensor, soft_mask: torch.Tensor) -> torch.Tensor:
    """Use hard values in the forward pass and soft values in the backward pass."""

    return hard_mask.detach() - soft_mask.detach() + soft_mask


def load_statistics(
    mask: torch.Tensor, eps: float = 1.0e-8, dead_expert_threshold: float = 0.01
) -> Dict[str, torch.Tensor]:
    """Compute load variance, dead expert ratio, and entropy for a routing mask."""

    load = mask.detach().float().mean(dim=0)
    total = load.sum().clamp_min(eps)
    prob = load / total
    entropy = -(prob * (prob + eps).log()).sum()
    max_entropy = math.log(mask.shape[-1]) if mask.shape[-1] > 1 else 1.0
    return {
        "expert_load": load,
        "expert_load_variance": load.var(unbiased=False),
        "dead_expert_ratio": (load <= dead_expert_threshold).float().mean(),
        "expert_entropy": entropy / max(max_entropy, eps),
    }


class ExpertMLP(nn.Module):
    """Feed-forward expert used inside the Mini-MoE block."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the expert MLP to token states."""

        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class MiniMoERouter(nn.Module):
    """Router supporting dense, top-k, noisy, Gumbel, ST, and p-bit variants."""

    def __init__(self, config: MiniMoERouterConfig) -> None:
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        initial_load = torch.full((config.num_experts,), 1.0 / config.num_experts)
        self.register_buffer("load_ema", initial_load)

    def temperature_at(self, step: Optional[int]) -> float:
        """Return the warmup-aware router temperature for a global step."""

        if step is None or self.config.warmup_steps <= 0:
            return max(self.config.temperature, self.config.min_temperature)
        progress = min(max(step / float(self.config.warmup_steps), 0.0), 1.0)
        temp = self.config.temperature + progress * (
            self.config.min_temperature - self.config.temperature
        )
        return max(temp, self.config.min_temperature)

    def effective_top_k(self, step: Optional[int]) -> int:
        """Return the top-k value after applying wider-top-k warmup."""

        if (
            self.config.warmup_mode == "wider_top_k"
            and step is not None
            and step < self.config.warmup_steps
        ):
            return min(self.config.warmup_top_k, self.config.num_experts)
        return min(self.config.top_k, self.config.num_experts)

    def update_load(self, hard_mask: torch.Tensor) -> None:
        """Update the expert-load EMA from a detached routing mask."""

        if not self.training:
            return
        batch_load = hard_mask.detach().float().mean(dim=0)
        self.load_ema.mul_(self.config.load_ema_decay).add_(
            batch_load, alpha=1.0 - self.config.load_ema_decay
        )

    def load_bias(self, enabled: bool) -> torch.Tensor:
        """Return load-aware bias that favors under-used experts."""

        if not enabled:
            return torch.zeros_like(self.load_ema)
        target = 1.0 / float(self.config.num_experts)
        return self.config.beta * (target - self.load_ema)

    def _gumbel_softmax(
        self, scores: torch.Tensor, temperature: float, step: Optional[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a Concrete/Gumbel-Softmax mask and its hard top-k view."""

        gumbel = -torch.empty_like(scores).exponential_().log()
        soft = F.softmax((scores + gumbel) / temperature, dim=-1)
        hard = top_k_mask(soft, self.effective_top_k(step))
        return hard, soft

    def _pbit_mask(
        self,
        scores: torch.Tensor,
        temperature: float,
        use_load_bias: bool,
        use_competition: bool,
        step: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute hard-forward p-bit-backward masks with optional load and competition."""

        k = self.effective_top_k(step)
        biased_scores = scores + self.load_bias(use_load_bias).to(scores.device)
        hard = top_k_mask(biased_scores, k)
        competition = 0.0
        if use_competition:
            competition = self.config.alpha * (hard.sum(dim=-1, keepdim=True) - hard)
        local_field = scores + self.load_bias(use_load_bias).to(scores.device) - competition
        q = torch.sigmoid(local_field / max(temperature, self.config.eps))
        return hard, q

    def forward(
        self, x: torch.Tensor, step: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return routing weights and metrics for flattened token states."""

        scores = self.router(x)
        method = self.config.routing_method
        temperature = self.temperature_at(step)
        k = self.effective_top_k(step)

        if method == "dense":
            hard = torch.ones_like(scores)
            weights = hard / float(self.config.num_experts)
        elif method == "topk":
            hard = top_k_mask(scores, k)
            weights = hard / hard.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        elif method == "noisy_topk":
            noisy_scores = scores + torch.randn_like(scores) * self.config.noisy_std
            hard = top_k_mask(noisy_scores, k)
            weights = hard / hard.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        elif method == "gumbel":
            hard, soft = self._gumbel_softmax(scores, temperature, step)
            weights = soft
        elif method == "topk_st":
            hard = top_k_mask(scores, k)
            soft = F.softmax(scores / max(temperature, self.config.eps), dim=-1)
            weights = straight_through_mask(hard, soft)
        elif method in {"pbit", "pbit_load", "pbit_competition", "pbit_load_competition"}:
            hard, soft = self._pbit_mask(
                scores=scores,
                temperature=temperature,
                use_load_bias=method in {"pbit_load", "pbit_load_competition"},
                use_competition=method in {"pbit_competition", "pbit_load_competition"},
                step=step,
            )
            if self.config.warmup_mode == "soft" and step is not None and step < self.config.warmup_steps:
                weights = soft
            else:
                weights = straight_through_mask(hard, soft)
        else:
            raise ValueError(f"Unsupported routing_method: {method}")

        self.update_load(hard)
        stats = load_statistics(hard, self.config.eps, self.config.dead_expert_threshold)
        stats.update(
            {
                "router_scores_mean": scores.detach().mean(),
                "router_scores_std": scores.detach().std(unbiased=False),
                "router_temperature": torch.tensor(temperature, device=scores.device),
            }
        )
        return weights, stats


class MiniMoEMLP(nn.Module):
    """Mixture-of-experts MLP that combines expert outputs by router weights."""

    def __init__(
        self,
        router_config: MiniMoERouterConfig,
        intermediate_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.router = MiniMoERouter(router_config)
        self.experts = nn.ModuleList(
            [
                ExpertMLP(router_config.hidden_size, intermediate_size, dropout)
                for _ in range(router_config.num_experts)
            ]
        )

    def forward(
        self, x: torch.Tensor, step: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply all experts and weight their token outputs by router decisions."""

        original_shape = x.shape
        flat_x = x.reshape(-1, original_shape[-1])
        weights, stats = self.router(flat_x, step=step)
        expert_outputs = torch.stack([expert(flat_x) for expert in self.experts], dim=1)
        mixed = torch.einsum("te,teh->th", weights, expert_outputs)
        return mixed.reshape(original_shape), stats


class CausalSelfAttention(nn.Module):
    """Small causal self-attention layer for MiniGPT experiments."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float, block_size: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run masked multi-head self-attention."""

        batch, seq_len, hidden = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        attn = self.dropout(F.softmax(attn, dim=-1))
        y = attn @ value
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        return self.proj(self.dropout(y))


class MiniTransformerBlock(nn.Module):
    """Transformer block with attention and a dense or MoE feed-forward path."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float,
        block_size: int,
        router_config: MiniMoERouterConfig,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads, dropout, block_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.mlp = MiniMoEMLP(router_config, intermediate_size, dropout)

    def forward(
        self, x: torch.Tensor, step: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply one residual transformer block and return routing metrics."""

        x = x + self.attn(self.ln_1(x))
        mlp_out, stats = self.mlp(self.ln_2(x), step=step)
        x = x + mlp_out
        return x, stats
