"""Safe scaffold for p-bit router patch planning.

This module intentionally does not mutate DeepSeek or any other model unless an
explicit unsupported apply call is made. It is a planning surface for inspection
results and future model-specific patch code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

from .pbit_router import PBitRouterConfig


@dataclass
class RouterPatchTarget:
    """Description of a candidate router module found during inspection."""

    name: str
    class_name: str
    module_path: str
    num_experts: Optional[int] = None
    top_k: Optional[int] = None
    notes: str = ""


@dataclass
class RouterPatchPlan:
    """Serializable patch plan for future manual review."""

    model_name: str
    targets: List[RouterPatchTarget]
    pbit_config: Dict[str, Any]
    scaffold_only: bool = True
    warning: str = "Patch application is intentionally disabled until a model-specific adapter is reviewed."

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable patch plan."""
        return {
            "model_name": self.model_name,
            "targets": [asdict(target) for target in self.targets],
            "pbit_config": self.pbit_config,
            "scaffold_only": self.scaffold_only,
            "warning": self.warning,
        }


def list_candidate_router_modules(model: Any, name_keywords: Optional[Iterable[str]] = None) -> List[RouterPatchTarget]:
    """List modules whose names or class names look router-related."""
    keywords = tuple(k.lower() for k in (name_keywords or ("router", "gate", "moe", "expert")))
    targets: List[RouterPatchTarget] = []
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        haystack = f"{name} {class_name}".lower()
        if any(keyword in haystack for keyword in keywords):
            targets.append(
                RouterPatchTarget(
                    name=name,
                    class_name=class_name,
                    module_path=f"{module.__class__.__module__}.{class_name}",
                    num_experts=_maybe_int_attr(module, ("num_experts", "n_routed_experts", "n_experts")),
                    top_k=_maybe_int_attr(module, ("top_k", "num_experts_per_tok", "k")),
                )
            )
    return targets


def build_patch_plan(
    model: Any,
    model_name: str,
    config: PBitRouterConfig,
    name_keywords: Optional[Iterable[str]] = None,
) -> RouterPatchPlan:
    """Build a scaffold-only p-bit router patch plan from an inspected model."""
    targets = list_candidate_router_modules(model, name_keywords=name_keywords)
    return RouterPatchPlan(model_name=model_name, targets=targets, pbit_config=asdict(config))


def apply_patch_plan(model: Any, plan: RouterPatchPlan, explicit: bool = False) -> Any:
    """Refuse to patch by default; future adapters must opt in explicitly."""
    if not explicit:
        raise RuntimeError(
            "Router patching is scaffold-only. Re-run after implementing and reviewing a "
            "model-specific adapter, then call with explicit=True."
        )
    raise NotImplementedError(
        "No model-specific p-bit router patch adapter is implemented. Inspect router modules first "
        "and add an adapter for the exact target architecture."
    )


def _maybe_int_attr(module: Any, names: Iterable[str]) -> Optional[int]:
    """Return the first integer-like module attribute from names."""
    for name in names:
        value = getattr(module, name, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None
