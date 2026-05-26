"""Unit checks for p-bit router scaffold using random tensors only."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pbit_router import PBitBackwardRouter, PBitRouterConfig, PBitSurrogateMask
from src.router_patch import apply_pbit_patch, build_router_patch_plan, list_router_like_modules


class ToyRouter(nn.Module):
    """Small deterministic router used for local unit tests."""

    def __init__(self, in_features: int, num_experts: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features, num_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return toy router logits."""
        return self.proj(hidden_states)


class ToyBlock(nn.Module):
    """Small module with a router-like child name."""

    def __init__(self) -> None:
        super().__init__()
        self.router = ToyRouter(4, 6)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the router child."""
        return self.router(hidden_states)


def assert_hard_forward_and_grad() -> None:
    """Verify hard top-k forward values and nonzero surrogate gradients."""
    torch.manual_seed(7)
    logits = torch.randn(3, 6, requires_grad=True)
    config = PBitRouterConfig(num_experts=6, top_k=2, alpha=1.0, temperature=0.75)
    mask = PBitSurrogateMask(config)(logits)

    expected = torch.zeros_like(logits).scatter_(-1, logits.topk(k=2, dim=-1).indices, 1.0)
    if not torch.equal(mask.detach(), expected):
        raise AssertionError("Forward mask is not the exact hard top-k mask.")

    weights = torch.linspace(-1.0, 1.0, steps=logits.numel(), dtype=logits.dtype).reshape_as(logits)
    loss = (mask * weights).sum()
    loss.backward()
    if logits.grad is None:
        raise AssertionError("Expected logits gradients, got None.")
    if not torch.isfinite(logits.grad).all():
        raise AssertionError("Expected finite logits gradients.")
    if logits.grad.abs().sum().item() <= 0:
        raise AssertionError("Expected nonzero surrogate gradients.")


def assert_router_wrapper() -> None:
    """Verify wrapper output shape and hard forward mask."""
    torch.manual_seed(11)
    base_router = ToyRouter(4, 5)
    wrapped = PBitBackwardRouter(base_router, PBitRouterConfig(num_experts=5, top_k=1))
    hidden = torch.randn(2, 4)
    output = wrapped(hidden)
    if output.shape != (2, 5):
        raise AssertionError(f"Unexpected wrapped output shape: {tuple(output.shape)}")
    if not torch.allclose(output.detach().sum(dim=-1), torch.ones(2)):
        raise AssertionError("top_k=1 hard mask should sum to 1 per row.")


def assert_patch_helpers() -> None:
    """Verify candidate discovery and explicit patch application."""
    model = ToyBlock()
    candidates = list_router_like_modules(model)
    names = [candidate.name for candidate in candidates]
    if "router" not in names:
        raise AssertionError(f"Expected to discover router candidate, got {names}")
    plan = build_router_patch_plan(model)
    if plan["candidate_count"] < 1:
        raise AssertionError("Expected at least one router-like module in patch plan.")

    apply_pbit_patch(model, "router", PBitRouterConfig(num_experts=6, top_k=2))
    if not isinstance(model.router, PBitBackwardRouter):
        raise AssertionError("Router child was not replaced by PBitBackwardRouter.")


def main() -> None:
    """Run all local unit checks."""
    assert_hard_forward_and_grad()
    assert_router_wrapper()
    assert_patch_helpers()
    print("P-bit router unit tests passed.")


if __name__ == "__main__":
    main()
