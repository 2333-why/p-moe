"""Small file logging helpers for Stage 1 inference runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def create_run_dir(base_dir: str = "outputs") -> str:
    """
    Create outputs/run_YYYYMMDD_HHMMSS and return its path.
    """
    base_path = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_path / f"run_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = base_path / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return str(run_dir)


def save_json(obj: dict[str, Any], path: str) -> None:
    """
    Save JSON with ensure_ascii=False and indent=2.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")


def save_text(text: str, path: str) -> None:
    """
    Save plain text.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
