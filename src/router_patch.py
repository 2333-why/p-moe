"""Safe p-bit router patch planning scaffold.

These helpers never guess a DeepSeek router replacement automatically. Inspect
the model first, review the candidate module names, then patch an explicit
module name only when the router contract is understood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from torch import nn

from src.pbit_router import PBitBackwardRouter, PBitRouterConfig


ROUTER_KEYWORDS = ("gate", "router", "moe", "expert")


@dataclass(frozen=True, slots=True)
class RouterLikeModule:
    """Summary of a module whose name or class looks router-related."""

    name: str
    class_name: str
    module: nn.Module

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe module summary."""
        return {"name": self.name, "class_name": self.class_name}


def list_router_like_modules(
    model: nn.Module,
    keywords: Iterable[str] = ROUTER_KEYWORDS,
) -> list[RouterLikeModule]:
    """List modules whose name or class contains gate/router/moe/expert."""
    lowered_keywords = tuple(keyword.lower() for keyword in keywords)
    if not lowered_keywords:
        raise ValueError("At least one router keyword is required.")
    matches: list[RouterLikeModule] = []
    for name, module in model.named_modules():
        if not name:
            continue
        haystack = f"{name} {module.__class__.__name__}".lower()
        if any(keyword in haystack for keyword in lowered_keywords):
            matches.append(RouterLikeModule(name, module.__class__.__name__, module))
    return matches


def build_router_patch_plan(model: nn.Module) -> dict[str, object]:
    """Build a review-only patch plan from router-like module candidates."""
    candidates = list_router_like_modules(model)
    return {
        "candidate_count": len(candidates),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "warning": (
            "This is an inspection scaffold. Do not patch until the true "
            "DeepSeek-MoE router contract has been confirmed."
        ),
    }


def apply_pbit_patch(
    model: nn.Module,
    module_name: str,
    config: PBitRouterConfig | None = None,
) -> nn.Module:
    """Patch one explicit module name with ``PBitBackwardRouter``.

    The function intentionally refuses empty names and unknown modules. If the
    correct router name is unclear, run ``scripts/inspect_router.py`` first.
    """
    if not module_name:
        raise ValueError("module_name is required. Run inspect_router.py first.")
    named_modules = dict(model.named_modules())
    if module_name not in named_modules:
        raise KeyError(
            f"Module '{module_name}' was not found. Run inspect_router.py and "
            "choose one exact router-like module name."
        )

    if "." in module_name:
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = named_modules.get(parent_name)
    else:
        parent_name, child_name = "<root>", module_name
        parent = model
    if parent is None:
        raise KeyError(f"Parent module '{parent_name}' was not found.")

    child = getattr(parent, child_name, None)
    if child is None:
        raise AttributeError(f"Parent module '{parent_name}' has no child '{child_name}'.")
    if not isinstance(child, nn.Module):
        raise TypeError(f"Attribute '{module_name}' is not an nn.Module.")
    if isinstance(child, PBitBackwardRouter):
        raise TypeError(f"Module '{module_name}' is already wrapped with PBitBackwardRouter.")

    setattr(parent, child_name, PBitBackwardRouter(child, config=config, router_name=module_name))
    return model


# Backward-compatible aliases for earlier scaffold names.
list_router_candidates = list_router_like_modules
apply_pbit_router_patch = lambda model, router_names, config=None: _apply_many(model, router_names, config)


def _apply_many(
    model: nn.Module,
    router_names: Iterable[str],
    config: PBitRouterConfig | None = None,
) -> list[str]:
    """Patch multiple explicit router names and return the patched names."""
    patched: list[str] = []
    for name in router_names:
        apply_pbit_patch(model, name, config=config)
        patched.append(name)
    if not patched:
        raise ValueError("No router names were provided. Run inspect_router.py first.")
    return patched
