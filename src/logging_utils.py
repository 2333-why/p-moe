"""Small logging and JSON helpers for p-MoE scripts.

The helpers in this module are intentionally dependency-light and safe at
import time. They never print raw secret values.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping


SECRET_ENV_NAMES = {
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "WANDB_API_KEY",
    "BARK_KEY",
    "OPENAI_API_KEY",
}


def project_root() -> Path:
    """Return the project root from HROOT or the current working directory."""

    return Path(os.environ.get("HROOT", ".")).expanduser().resolve()


def resolve_output_path(path: str | os.PathLike[str]) -> Path:
    """Resolve an output path relative to the project root when needed."""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def ensure_parent(path: str | os.PathLike[str]) -> Path:
    """Create the parent directory for a file path and return the path."""

    resolved = resolve_output_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def safe_env_status(names: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Return set/not set status for environment variables without values."""

    return {name: "set" if os.environ.get(name) else "not set" for name in names}


def redact_secrets(data: Any) -> Any:
    """Recursively redact likely secret values from dictionaries and lists."""

    if isinstance(data, Mapping):
        redacted: dict[str, Any] = {}
        for key, value in data.items():
            key_text = str(key).upper()
            if key_text in SECRET_ENV_NAMES or "TOKEN" in key_text or "KEY" in key_text or "PASSWORD" in key_text:
                redacted[str(key)] = "set" if value else "not set"
            else:
                redacted[str(key)] = redact_secrets(value)
        return redacted
    if isinstance(data, list):
        return [redact_secrets(item) for item in data]
    if isinstance(data, tuple):
        return [redact_secrets(item) for item in data]
    return data


def write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    """Write a redacted JSON object with stable formatting."""

    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(redact_secrets(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return resolved


def save_json(payload: Mapping[str, Any], path: str | os.PathLike[str]) -> Path:
    """Write a redacted JSON object using the Stage 2 call convention."""

    return write_json(path, payload)


def save_text(text: str, path: str | os.PathLike[str]) -> Path:
    """Write text to an output path relative to the project root."""

    resolved = ensure_parent(path)
    resolved.write_text(text, encoding="utf-8")
    return resolved


def create_run_dir(output_dir: str | os.PathLike[str], prefix: str = "run") -> Path:
    """Create and return a timestamped run directory under output_dir."""

    root = resolve_output_path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    suffix = 0
    candidate = run_dir
    while candidate.exists():
        suffix += 1
        candidate = root / f"{run_dir.name}_{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def print_kv(title: str, values: Mapping[str, Any]) -> None:
    """Print a compact key-value section for CLI diagnostics."""

    print(title)
    for key, value in values.items():
        print(f"  {key}: {value}")
