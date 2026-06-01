"""Utilities for aggregating p-MoE experiment result summaries."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable


RESULT_FIELDS = (
    "run_dir",
    "stage",
    "model_name",
    "method",
    "success",
    "failure",
    "ppl",
    "train_loss",
    "tokens_per_sec",
    "load_variance",
    "dead_expert_ratio",
    "expert_entropy",
    "has_cpu_offload",
    "has_disk_offload",
    "error_message",
)


@dataclass
class ResultRecord:
    """Normalized row for paper-facing result tables."""

    run_dir: str = ""
    stage: str = ""
    model_name: str = ""
    method: str = ""
    success: bool = False
    failure: bool = False
    ppl: float | None = None
    train_loss: float | None = None
    tokens_per_sec: float | None = None
    load_variance: float | None = None
    dead_expert_ratio: float | None = None
    expert_entropy: float | None = None
    has_cpu_offload: bool = False
    has_disk_offload: bool = False
    error_message: str = ""


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from a summary file."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def discover_summary_files(outputs_dir: Path) -> list[Path]:
    """Find summary files following the repository result conventions."""
    candidates: set[Path] = set()
    candidates.update(outputs_dir.glob("run_*/summary.json"))
    candidates.update(outputs_dir.glob("run_*/eval_summary.json"))
    candidates.update(outputs_dir.glob("**/train_summary.json"))
    return sorted(candidates)


def _first_value(data: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    """Return the first present non-empty value from a list of possible keys."""
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return default


def _as_float(value: Any) -> float | None:
    """Convert a JSON value to float when possible."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    """Convert common JSON scalar values to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def infer_stage(path: Path, data: dict[str, Any]) -> str:
    """Infer experiment stage from explicit metadata or run directory names."""
    explicit = _first_value(data, ("stage", "experiment_stage"))
    if explicit:
        return str(explicit)
    run_name = path.parent.name.lower()
    if "stage1" in run_name or "generate" in run_name:
        return "stage1_generate"
    if "stage2" in run_name or "wikitext" in run_name or path.name == "eval_summary.json":
        return "stage2_eval"
    if "stage3" in run_name or "qlora" in run_name or "lora" in run_name:
        return "stage3_train"
    if "stage4" in run_name or "router" in run_name:
        return "stage4_router_inspect"
    if "stage5" in run_name or "mini_moe" in run_name:
        return "stage5_mini_moe"
    return "unknown"


def normalize_record(path: Path, outputs_dir: Path) -> ResultRecord:
    """Normalize one summary JSON file into a stable result row."""
    data = load_json(path)
    metrics = data.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    final_eval = data.get("final_eval", {})
    if not isinstance(final_eval, dict):
        final_eval = {}
    model_config = data.get("model_config", {})
    if not isinstance(model_config, dict):
        model_config = {}

    merged = {**data, **metrics, **final_eval, **model_config}

    error_message = str(_first_value(merged, ("error_message", "error", "exception"), ""))
    success = _as_bool(_first_value(merged, ("success", "ok", "completed"), not error_message))
    failure = _as_bool(_first_value(merged, ("failure", "failed"), bool(error_message) or not success))

    relative_run_dir = path.parent
    try:
        relative_run_dir = path.parent.relative_to(outputs_dir)
    except ValueError:
        pass

    return ResultRecord(
        run_dir=str(relative_run_dir).replace("\\", "/"),
        stage=infer_stage(path, merged),
        model_name=str(_first_value(merged, ("model_name", "model", "base_model"), "")),
        method=str(_first_value(merged, ("method", "router_method", "routing_method", "experiment"), "")),
        success=success,
        failure=failure,
        ppl=_as_float(_first_value(merged, ("ppl", "perplexity", "eval_ppl"))),
        train_loss=_as_float(_first_value(merged, ("train_loss", "loss", "final_train_loss"))),
        tokens_per_sec=_as_float(_first_value(merged, ("tokens_per_sec", "tokens_per_second", "throughput"))),
        load_variance=_as_float(_first_value(merged, ("load_variance", "expert_load_variance"))),
        dead_expert_ratio=_as_float(_first_value(merged, ("dead_expert_ratio", "dead_experts_ratio"))),
        expert_entropy=_as_float(_first_value(merged, ("expert_entropy", "routing_entropy"))),
        has_cpu_offload=_as_bool(_first_value(merged, ("has_cpu_offload", "cpu_offload", "use_cpu_offload"), False)),
        has_disk_offload=_as_bool(_first_value(merged, ("has_disk_offload", "disk_offload", "use_disk_offload"), False)),
        error_message=error_message,
    )


def collect_result_records(outputs_dir: Path) -> list[ResultRecord]:
    """Collect all known result summaries below an outputs directory."""
    if not outputs_dir.exists():
        return []
    records: list[ResultRecord] = []
    for path in discover_summary_files(outputs_dir):
        try:
            records.append(normalize_record(path, outputs_dir))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            records.append(
                ResultRecord(
                    run_dir=str(path.parent).replace("\\", "/"),
                    stage="collection_error",
                    success=False,
                    failure=True,
                    error_message=f"{path.name}: {exc}",
                )
            )
    return records


def record_to_row(record: ResultRecord) -> dict[str, Any]:
    """Convert a result dataclass into a CSV-safe mapping."""
    return {field.name: getattr(record, field.name) for field in fields(ResultRecord)}


def write_results_table(records: list[ResultRecord], csv_path: Path) -> None:
    """Write normalized records as a CSV table."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record_to_row(record))


def _format_metric(value: float | None) -> str:
    """Format optional float values for markdown tables."""
    if value is None:
        return ""
    return f"{value:.6g}"


def write_results_markdown(records: list[ResultRecord], md_path: Path) -> None:
    """Write a compact markdown summary for quick paper-metric review."""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(records)
    failures = sum(1 for record in records if record.failure)
    successes = sum(1 for record in records if record.success)

    lines = [
        "# p-MoE Results Summary",
        "",
        f"- Total records: {total}",
        f"- Successful records: {successes}",
        f"- Failed records: {failures}",
        "",
        "| run_dir | stage | method | ppl | train_loss | load_variance | dead_expert_ratio | expert_entropy | status |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        status = "success" if record.success and not record.failure else "failure"
        lines.append(
            "| "
            + " | ".join(
                [
                    record.run_dir,
                    record.stage,
                    record.method,
                    _format_metric(record.ppl),
                    _format_metric(record.train_loss),
                    _format_metric(record.load_variance),
                    _format_metric(record.dead_expert_ratio),
                    _format_metric(record.expert_entropy),
                    status,
                ]
            )
            + " |"
        )

    failed = [record for record in records if record.failure and record.error_message]
    if failed:
        lines.extend(["", "## Errors", ""])
        for record in failed:
            lines.append(f"- `{record.run_dir}`: {record.error_message}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
