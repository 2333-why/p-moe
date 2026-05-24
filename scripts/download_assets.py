"""Download Stage 1 model assets without running inference."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "download_config.yaml"


def parse_args() -> argparse.Namespace:
    """Parse download-stage command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download DeepSeek-MoE assets to Hugging Face cache or a local directory."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to download YAML config")
    parser.add_argument("--model_name", default=None, help="Hugging Face model id")
    parser.add_argument("--local_dir", default=None, help="Local output directory for --mode local")
    parser.add_argument("--mode", choices=["cache", "local"], default=None, help="Download mode")
    parser.add_argument(
        "--resume_download",
        action="store_true",
        default=None,
        help="Resume partially downloaded files when supported by huggingface_hub",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML config from disk."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Download config must be a YAML mapping: {config_path}")
    return loaded


def default_config() -> dict[str, Any]:
    """Return default download settings."""
    return {
        "model_name": "deepseek-ai/deepseek-moe-16b-base",
        "mode": "cache",
        "local_dir": "models/deepseek-moe-16b-base",
        "resume_download": True,
    }


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply non-null CLI overrides to download config."""
    merged = default_config()
    merged.update(config)
    for key in ("model_name", "local_dir", "mode", "resume_download"):
        value = getattr(args, key)
        if value is not None:
            merged[key] = value
    return merged


def env_status() -> dict[str, str]:
    """Return Hugging Face environment status without exposing secrets."""
    return {
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", "not set"),
        "HF_HOME": os.environ.get("HF_HOME", "not set"),
        "HF_TOKEN": "set" if os.environ.get("HF_TOKEN") else "not set",
    }


def print_env_status() -> None:
    """Print Hugging Face download environment without raw token values."""
    print("Hugging Face environment:")
    for name, value in env_status().items():
        print(f"  {name}: {value}")


def download_assets(config: dict[str, Any]) -> str:
    """Download model assets and return the resolved snapshot path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for downloads. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    model_name = str(config["model_name"])
    mode = str(config.get("mode", "cache"))
    resume_download = bool(config.get("resume_download", True))

    kwargs: dict[str, Any] = {
        "repo_id": model_name,
        "resume_download": resume_download,
    }

    if mode == "local":
        local_dir = Path(str(config.get("local_dir") or "models/deepseek-moe-16b-base"))
        local_dir.mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = str(local_dir)
        kwargs["local_dir_use_symlinks"] = False
        print(f"Downloading {model_name} to local directory: {local_dir}")
    elif mode == "cache":
        print(f"Downloading {model_name} to Hugging Face cache.")
    else:
        raise ValueError("mode must be either 'cache' or 'local'")

    snapshot_path = snapshot_download(**kwargs)
    return str(snapshot_path)


def main() -> int:
    """Run the download-only stage."""
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)

    print_env_status()
    print(f"Download mode: {config['mode']}")
    print(f"Model name: {config['model_name']}")
    if config["mode"] == "local":
        print(f"Local directory: {config['local_dir']}")

    try:
        snapshot_path = download_assets(config)
    except Exception as exc:
        print(f"Download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Suggestions: check network access, HF_ENDPOINT, Hugging Face permissions, "
            "HF_HOME/cache disk space, and whether HF_TOKEN is set when needed.",
            file=sys.stderr,
        )
        return 1

    print(f"Download complete: {snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
