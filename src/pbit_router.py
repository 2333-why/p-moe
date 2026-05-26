"""P-bit backward surrogate routing scaffold.

This module implements only tensor-level router-mask logic. It does not modify
DeepSeek experts or assume a concrete DeepSeek router class.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(slots=True)
class PBitRouterConfig:
    """Configuration for hard-forward, p-bit-backward surrogate masks."""

    num_experts: int | None = None
    top_k: int | None = None
    alpha: float = 1.0
    temperature: float = 1.0
    use_load_bias: bool = False
    load_ema_decay: float = 0.99
    eps: float = 1e-8
    logits_index: int = 0

    def validate(self, inferred_num_experts: int | None = None) -> None:
        """Validate config values against an optional inferred expert count."""
        num_experts = self.num_experts or inferred_num_experts
        if num_experts is not None and num_experts < 1:
            raise ValueError("num_experts must be >= 1.")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1 when set.")
        if self.top_k is not None and num_experts is not None and self.top_k > num_experts:
            raise ValueError(f"top_k={self.top_k} exceeds num_experts={num_experts}.")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0.")
        if not 0 <= self.load_ema_decay < 1:
            raise ValueError("load_ema_decay must be in [0, 1).")
        if self.eps <= 0:
            raise ValueError("eps must be > 0.")
        if self.logits_index < 0:
            raise ValueError("logits_index must be >= 0.")

    def resolved_top_k(self, inferred_num_experts: int) -> int:
        """Return the configured top-k, defaulting to 2 or fewer experts."""
        self.validate(inferred_num_experts)
        return int(self.top_k if self.top_k is not None else min(2, inferred_num_experts))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe dictionary."""
        return asdict(self)


class PBitSurrogateMask(nn.Module):
    """Return hard top-k masks with p-bit surrogate gradients.

    For router scores ``scores``:

    ``m_hard = top_k_mask(scores)``

    ``q_pbit = sigmoid((scores - alpha * competition + bias) / temperature)``

    ``m = m_hard.detach() - q_pbit.detach() + q_pbit``

    Forward values are the hard top-k mask. Backward gradients flow through
    ``q_pbit``. This affects only the mask/gate surrogate, not expert forward
    computation.
    """

    def __init__(self, config: PBitRouterConfig | None = None) -> None:
        super().__init__()
        self.config = config or PBitRouterConfig()
        self.register_buffer("load_ema", torch.empty(0), persistent=False)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Build a hard-forward, p-bit-backward mask from router scores."""
        if scores.ndim < 1:
            raise ValueError("scores must have at least one dimension.")
        num_experts = int(scores.shape[-1])
        self.config.validate(num_experts)
        top_k = self.config.resolved_top_k(num_experts)

        topk_indices = torch.topk(scores, k=top_k, dim=-1).indices
        m_hard = torch.zeros_like(scores).scatter_(-1, topk_indices, 1.0)

        competition = self._competition(scores)
        bias = self._load_bias(scores, m_hard)
        q_pbit = torch.sigmoid(
            (scores - float(self.config.alpha) * competition + bias)
            / float(self.config.temperature)
        )
        return m_hard.detach() - q_pbit.detach() + q_pbit

    def _competition(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute a simple differentiable competition term per expert."""
        num_experts = scores.shape[-1]
        if num_experts <= 1:
            return torch.zeros_like(scores)
        other_sum = scores.sum(dim=-1, keepdim=True) - scores
        return other_sum / max(num_experts - 1, 1)

    def _load_bias(self, scores: torch.Tensor, hard_mask: torch.Tensor) -> torch.Tensor:
        """Return optional load-balancing bias without exposing trainable state."""
        if not self.config.use_load_bias:
            return torch.zeros_like(scores)

        reduce_dims = tuple(range(hard_mask.ndim - 1))
        current_load = hard_mask.detach().mean(dim=reduce_dims)
        if self.load_ema.numel() != current_load.numel() or self.load_ema.device != current_load.device:
            self.load_ema = current_load.clone()
        else:
            decay = float(self.config.load_ema_decay)
            self.load_ema.mul_(decay).add_(current_load, alpha=1.0 - decay)

        target = 1.0 / current_load.numel()
        bias = target - self.load_ema
        return bias.to(dtype=scores.dtype, device=scores.device).view(
            *([1] * (scores.ndim - 1)),
            scores.shape[-1],
        )


class PBitBackwardRouter(nn.Module):
    """Wrap a router and replace its logits output with a p-bit mask scaffold."""

    def __init__(
        self,
        base_router: nn.Module,
        config: PBitRouterConfig | None = None,
        router_name: str | None = None,
    ) -> None:
        super().__init__()
        self.base_router = base_router
        self.config = config or PBitRouterConfig()
        self.router_name = router_name
        self.mask = PBitSurrogateMask(self.config)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the base router and replace its logits tensor with a p-bit mask."""
        router_output = self.base_router(*args, **kwargs)
        if torch.is_tensor(router_output):
            return self.mask(router_output)
        if isinstance(router_output, tuple):
            values = list(router_output)
            index = self._checked_logits_index(values)
            values[index] = self.mask(values[index])
            return tuple(values)
        if isinstance(router_output, list):
            values = list(router_output)
            index = self._checked_logits_index(values)
            values[index] = self.mask(values[index])
            return values
        raise TypeError(
            "PBitBackwardRouter only supports Tensor, tuple, or list router outputs. "
            f"Inspect router {self.router_name or '<unnamed>'} before patching."
        )

    def _checked_logits_index(self, values: list[Any]) -> int:
        """Return a valid logits index for tuple/list router outputs."""
        index = self.config.logits_index
        if index >= len(values):
            raise IndexError(f"logits_index={index} is out of range for router output.")
        if not torch.is_tensor(values[index]):
            raise TypeError(f"Router output at logits_index={index} is not a tensor.")
        return index
