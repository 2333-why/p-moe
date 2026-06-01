"""Load-aware p-bit competitive routing primitives.

This module only computes routing masks and gates. It does not call experts or
patch any model implementation directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class PBitRouterConfig:
    """Configuration for load-aware p-bit straight-through routing."""

    num_experts: int
    top_k: int = 2
    alpha: float = 0.0
    beta: float = 0.0
    temperature: float = 1.0
    min_temperature: float = 0.1
    load_ema_decay: float = 0.99
    use_load_bias: bool = True
    use_competition: bool = True
    use_sampling: bool = False
    eps: float = 1.0e-8
    temperature_decay: float = 1.0
    normalize_gates: bool = True

    def __post_init__(self) -> None:
        """Validate p-bit router settings."""
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError("top_k must be in [1, num_experts].")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if self.min_temperature <= 0.0:
            raise ValueError("min_temperature must be positive.")
        if not 0.0 <= self.load_ema_decay < 1.0:
            raise ValueError("load_ema_decay must be in [0, 1).")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")
        if self.temperature_decay <= 0.0:
            raise ValueError("temperature_decay must be positive.")


class PBitLoadState(nn.Module):
    """Tracks exponential moving average expert load."""

    def __init__(self, num_experts: int, ema_decay: float = 0.99, eps: float = 1.0e-8) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1).")
        self.num_experts = num_experts
        self.ema_decay = ema_decay
        self.eps = eps
        initial = torch.full((num_experts,), 1.0 / num_experts)
        self.register_buffer("load_ema", initial)
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, mask: Tensor) -> Tensor:
        """Update EMA load from a hard routing mask and return current load."""
        flat = _flatten_expert_dim(mask, self.num_experts).float()
        denom = flat.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        per_token = flat / denom
        batch_load = per_token.mean(dim=0)
        self.load_ema.mul_(self.ema_decay).add_(batch_load, alpha=1.0 - self.ema_decay)
        self.updates.add_(1)
        return self.load_ema.detach().clone()

    @torch.no_grad()
    def reset(self, uniform: bool = True) -> None:
        """Reset load state to uniform load or zeros."""
        value = 1.0 / self.num_experts if uniform else 0.0
        self.load_ema.fill_(value)
        self.updates.zero_()

    def bias(self, beta: float) -> Tensor:
        """Return load-aware bias beta * (1 / N - L_i)."""
        target = 1.0 / self.num_experts
        return beta * (target - self.load_ema)


class TemperatureScheduler:
    """Simple multiplicative temperature scheduler with a lower bound."""

    def __init__(self, temperature: float, min_temperature: float, decay: float = 1.0) -> None:
        if temperature <= 0.0 or min_temperature <= 0.0:
            raise ValueError("temperature and min_temperature must be positive.")
        if decay <= 0.0:
            raise ValueError("decay must be positive.")
        self.initial_temperature = float(temperature)
        self.temperature = float(temperature)
        self.min_temperature = float(min_temperature)
        self.decay = float(decay)
        self.step_count = 0

    def step(self) -> float:
        """Advance one scheduler step and return the new temperature."""
        self.step_count += 1
        self.temperature = max(self.min_temperature, self.temperature * self.decay)
        return self.temperature

    def reset(self) -> None:
        """Restore the initial temperature."""
        self.temperature = self.initial_temperature
        self.step_count = 0

    def state_dict(self) -> Dict[str, float]:
        """Return scheduler state as JSON-serializable values."""
        return {
            "initial_temperature": self.initial_temperature,
            "temperature": self.temperature,
            "min_temperature": self.min_temperature,
            "decay": self.decay,
            "step_count": float(self.step_count),
        }

    def load_state_dict(self, state: Dict[str, float]) -> None:
        """Load scheduler state from a dictionary."""
        self.initial_temperature = float(state["initial_temperature"])
        self.temperature = float(state["temperature"])
        self.min_temperature = float(state["min_temperature"])
        self.decay = float(state["decay"])
        self.step_count = int(state["step_count"])


def top_k_mask(scores: Tensor, top_k: int) -> Tensor:
    """Return a hard top-k binary mask over the last dimension."""
    if scores.ndim == 0:
        raise ValueError("scores must have an expert dimension.")
    num_experts = scores.shape[-1]
    if top_k <= 0 or top_k > num_experts:
        raise ValueError("top_k must be in [1, num_experts].")
    indices = torch.topk(scores, k=top_k, dim=-1).indices
    return torch.zeros_like(scores).scatter_(-1, indices, 1.0)


def sample_top_k_mask(probabilities: Tensor, top_k: int, eps: float = 1.0e-8) -> Tensor:
    """Sample k experts without replacement from per-expert probabilities."""
    if probabilities.ndim == 0:
        raise ValueError("probabilities must have an expert dimension.")
    num_experts = probabilities.shape[-1]
    if top_k <= 0 or top_k > num_experts:
        raise ValueError("top_k must be in [1, num_experts].")
    flat = probabilities.reshape(-1, num_experts).clamp_min(eps)
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(eps)
    sampled = torch.multinomial(flat, num_samples=top_k, replacement=False)
    mask = torch.zeros_like(flat).scatter_(-1, sampled, 1.0)
    return mask.reshape_as(probabilities)


def straight_through_mask(hard_mask: Tensor, surrogate_prob: Tensor) -> Tensor:
    """Use hard forward values and surrogate p-bit gradients."""
    if hard_mask.shape != surrogate_prob.shape:
        raise ValueError("hard_mask and surrogate_prob must have the same shape.")
    return hard_mask.detach() - surrogate_prob.detach() + surrogate_prob


class PBitBackwardRouter(nn.Module):
    """Compute p-bit straight-through masks for sparse MoE routing."""

    def __init__(self, config: PBitRouterConfig) -> None:
        super().__init__()
        self.config = config
        self.load_state = PBitLoadState(
            num_experts=config.num_experts,
            ema_decay=config.load_ema_decay,
            eps=config.eps,
        )
        self.temperature_scheduler = TemperatureScheduler(
            temperature=config.temperature,
            min_temperature=config.min_temperature,
            decay=config.temperature_decay,
        )

    @property
    def temperature(self) -> float:
        """Current p-bit temperature."""
        return self.temperature_scheduler.temperature

    def step_temperature(self) -> float:
        """Advance temperature schedule and return current value."""
        return self.temperature_scheduler.step()

    def load_bias(self, scores: Tensor) -> Tensor:
        """Return broadcastable load-aware bias for scores."""
        if not self.config.use_load_bias:
            return torch.zeros_like(scores)
        bias = self.load_state.bias(self.config.beta).to(device=scores.device, dtype=scores.dtype)
        return bias.view(*([1] * (scores.ndim - 1)), self.config.num_experts).expand_as(scores)

    def local_field(self, scores: Tensor, z: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Compute p-bit local field I_i and load bias."""
        self._validate_scores(scores)
        bias = self.load_bias(scores)
        if z is None:
            z = top_k_mask(scores + bias, self.config.top_k)
        competition = torch.zeros_like(scores)
        if self.config.use_competition and self.config.alpha != 0.0:
            z = z.to(dtype=scores.dtype, device=scores.device)
            active_others = z.sum(dim=-1, keepdim=True) - z
            competition = self.config.alpha * active_others
        return scores - competition + bias, bias

    def probabilities(self, scores: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Return p-bit activation probabilities q_i = sigmoid(I_i / T)."""
        field, _ = self.local_field(scores, z=z)
        temperature = max(self.temperature, self.config.min_temperature)
        return torch.sigmoid(field / temperature)

    def forward(
        self,
        scores: Tensor,
        update_load: Optional[bool] = None,
        return_aux: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        """Return straight-through routing mask, optionally with diagnostics."""
        self._validate_scores(scores)
        update_load = self.training if update_load is None else update_load
        bias = self.load_bias(scores)
        hard_scores = scores + bias
        initial_hard = top_k_mask(hard_scores, self.config.top_k)
        q = self.probabilities(scores, z=initial_hard)
        hard_mask = (
            sample_top_k_mask(q, self.config.top_k, eps=self.config.eps)
            if self.config.use_sampling
            else top_k_mask(hard_scores, self.config.top_k)
        )
        if update_load:
            self.load_state.update(hard_mask)
        mask = straight_through_mask(hard_mask, q)
        gates = self._normalize(mask) if self.config.normalize_gates else mask
        if not return_aux:
            return gates
        aux = {
            "hard_mask": hard_mask.detach(),
            "surrogate_prob": q,
            "load_bias": bias.detach(),
            "load_ema": self.load_state.load_ema.detach().clone(),
            "temperature": torch.tensor(self.temperature, device=scores.device, dtype=scores.dtype),
        }
        return gates, aux

    def _normalize(self, mask: Tensor) -> Tensor:
        """Normalize selected gate mass per token while preserving ST gradients."""
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
        return mask / denom

    def _validate_scores(self, scores: Tensor) -> None:
        """Validate score shape against configured expert count."""
        if scores.ndim < 1:
            raise ValueError("scores must include an expert dimension.")
        if scores.shape[-1] != self.config.num_experts:
            raise ValueError(
                f"Expected last dimension {self.config.num_experts}, got {scores.shape[-1]}."
            )


def _flatten_expert_dim(values: Tensor, num_experts: int) -> Tensor:
    """Flatten all leading dimensions while keeping the expert dimension."""
    if values.ndim < 1 or values.shape[-1] != num_experts:
        raise ValueError(f"Expected last dimension to be {num_experts}.")
    return values.reshape(-1, num_experts)
