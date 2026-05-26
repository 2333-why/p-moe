"""Utilities for collecting experiment summaries into reviewable tables."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


RESULT_FIELDS = [
    "run_dir",
    "summary_file",
    "stage",
    "timestamp",
    "model_name",
    "dtype",
    "dataset",
    "split",
    "status",
    "load_success",
    "generation_success",
    "eval_success",
    "train_success",
    "inspect_success",
    "perplexity",
    "eval_loss",
    "train_loss",
    "loss",
    "new_tokens",
    "tokens_per_sec",
    "has_cpu_offload",
    "has_disk_offload",
    "peak_gpu_memory_gb",
    "error_type",
    "error_message",
    "summary_path",
]


SUMMARY_PATTERNS = (
    "summary.json",
    "eval_summary.json",
    "train_summary.json",
    "training_summary.json",
    "qlora_summary.json",
    "router_summary.json",
    "router_inspect_summary.json",
)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from a path."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def iter_summary_paths(outputs_dir: Path) -> list[Path]:
    """Return known summary files from outputs/run_* directories."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for run_dir in sorted(outputs_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        for pattern in SUMMARY_PATTERNS:
            for path in sorted(run_dir.glob(pattern)):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(path)
        for path in sorted(run_dir.glob("*summary*.json")):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(path)
    return paths


def first_value(data: dict[str, Any], keys: tuple[str, ...], default: Any = "") -> Any:
    """Return the first present, non-None value from a dictionary."""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def infer_stage(path: Path, data: dict[str, Any]) -> str:
    """Infer experiment stage from summary keys and file names."""
    explicit = first_value(data, ("stage", "experiment_stage"), "")
    if explicit:
        return str(explicit)

    name = path.name.lower()
    text = " ".join(str(key).lower() for key in data.keys())
    if "router" in name or "router" in text:
        return "stage4_router_inspect"
    if "train" in name or "qlora" in name or "train_" in text:
        return "stage3_qlora"
    if "eval" in name or "perplexity" in text or "ppl" in text:
        return "stage2_eval"
    if "generation_success" in data or "generated_text" in data:
        return "stage1_generation"
    return "unknown"


def infer_status(data: dict[str, Any]) -> str:
    """Infer a compact success/failure status from known summary booleans."""
    for key in ("success", "generation_success", "eval_success", "train_success", "inspect_success"):
        value = data.get(key)
        if value is True:
            return "success"
        if value is False:
            return "failed"
    if data.get("error_type") or data.get("error_message"):
        return "failed"
    return "unknown"


def run_timestamp(run_dir: Path) -> str:
    """Extract run timestamp from run_YYYYMMDD_HHMMSS style directory names."""
    match = re.match(r"run_(\d{8}_\d{6}(?:_\d+)?)", run_dir.name)
    if match:
        return match.group(1)
    return ""


def max_peak_gpu_memory(data: dict[str, Any]) -> str:
    """Return the largest recorded peak GPU memory value in GB, if available."""
    candidates: list[float] = []
    for key, value in data.items():
        if not key.startswith("gpu_info"):
            continue
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                peak = first_value(
                    item,
                    (
                        "peak_allocated_memory_gb",
                        "peak_memory_gb",
                        "max_memory_allocated_gb",
                        "allocated_memory_gb",
                    ),
                    None,
                )
                if isinstance(peak, (int, float)):
                    candidates.append(float(peak))
    if not candidates:
        value = first_value(data, ("peak_gpu_memory_gb", "max_gpu_memory_gb"), None)
        if isinstance(value, (int, float)):
            candidates.append(float(value))
    return f"{max(candidates):.3f}" if candidates else ""


def normalize_record(path: Path, data: dict[str, Any], outputs_dir: Path) -> dict[str, Any]:
    """Convert one summary JSON into a flat CSV/Markdown row."""
    run_dir = path.parent
    try:
        run_dir_text = str(run_dir.relative_to(outputs_dir.parent))
    except ValueError:
        run_dir_text = str(run_dir)

    dataset = first_value(data, ("dataset", "dataset_name"), "")
    dataset_config = first_value(data, ("dataset_config", "dataset_config_name", "config_name"), "")
    if dataset and dataset_config:
        dataset = f"{dataset}/{dataset_config}"

    record = {
        "run_dir": run_dir_text,
        "summary_file": path.name,
        "stage": infer_stage(path, data),
        "timestamp": first_value(data, ("timestamp", "created_at", "run_timestamp"), run_timestamp(run_dir)),
        "model_name": first_value(data, ("model_name", "base_model", "model"), ""),
        "dtype": first_value(data, ("dtype", "torch_dtype"), ""),
        "dataset": dataset,
        "split": first_value(data, ("split", "eval_split", "test_split"), ""),
        "status": infer_status(data),
        "load_success": first_value(data, ("load_success",), ""),
        "generation_success": first_value(data, ("generation_success",), ""),
        "eval_success": first_value(data, ("eval_success", "evaluation_success"), ""),
        "train_success": first_value(data, ("train_success", "training_success"), ""),
        "inspect_success": first_value(data, ("inspect_success", "router_inspect_success"), ""),
        "perplexity": first_value(data, ("perplexity", "ppl", "eval_perplexity"), ""),
        "eval_loss": first_value(data, ("eval_loss", "validation_loss"), ""),
        "train_loss": first_value(data, ("train_loss", "final_train_loss"), ""),
        "loss": first_value(data, ("loss", "final_loss"), ""),
        "new_tokens": first_value(data, ("new_tokens",), ""),
        "tokens_per_sec": first_value(data, ("tokens_per_sec", "generation_tokens_per_sec"), ""),
        "has_cpu_offload": first_value(data, ("has_cpu_offload",), ""),
        "has_disk_offload": first_value(data, ("has_disk_offload",), ""),
        "peak_gpu_memory_gb": max_peak_gpu_memory(data),
        "error_type": first_value(data, ("error_type",), ""),
        "error_message": first_value(data, ("error_message",), ""),
        "summary_path": str(path),
    }
    return {field: record.get(field, "") for field in RESULT_FIELDS}


def collect_result_records(outputs_dir: Path) -> list[dict[str, Any]]:
    """Collect all recognized result summaries under an outputs directory."""
    records: list[dict[str, Any]] = []
    for path in iter_summary_paths(outputs_dir):
        try:
            data = load_json(path)
            records.append(normalize_record(path, data, outputs_dir))
        except Exception as exc:
            records.append(
                {
                    **{field: "" for field in RESULT_FIELDS},
                    "run_dir": str(path.parent),
                    "summary_file": path.name,
                    "stage": "unreadable",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "summary_path": str(path),
                }
            )
    return records


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    """Write result records to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def markdown_escape(value: Any) -> str:
    """Escape Markdown table separators in scalar values."""
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def write_markdown_summary(records: list[dict[str, Any]], path: Path) -> None:
    """Write a compact Markdown summary table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_dir",
        "stage",
        "status",
        "model_name",
        "dataset",
        "split",
        "perplexity",
        "eval_loss",
        "train_loss",
        "tokens_per_sec",
        "has_cpu_offload",
        "has_disk_offload",
        "peak_gpu_memory_gb",
    ]
    lines = [
        "# Results Summary",
        "",
        f"Collected runs: {len(records)}",
        "",
    ]
    if records:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for record in records:
            lines.append("| " + " | ".join(markdown_escape(record.get(col, "")) for col in columns) + " |")
    else:
        lines.append("No summary files found under `outputs/run_*`.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
