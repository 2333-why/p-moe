"""Training and evaluation helpers for Mini-MoE experiments."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .mini_moe_model import MiniMoEModelConfig, MiniMoELanguageModel


class TokenBlockDataset(Dataset):
    """Dataset of fixed-length causal LM token blocks."""

    def __init__(self, token_ids: List[int], block_size: int) -> None:
        self.block_size = block_size
        usable = max(0, len(token_ids) - block_size - 1)
        self.token_ids = token_ids[: usable + block_size + 1]
        self.length = usable // block_size

    def __len__(self) -> int:
        """Return number of non-overlapping token blocks."""

        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return input and next-token labels for one block."""

        start = idx * self.block_size
        chunk = self.token_ids[start : start + self.block_size + 1]
        return {
            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
            "labels": torch.tensor(chunk[1:], dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch RNG seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(path: str) -> Dict[str, object]:
    """Load a YAML config file without importing optional libraries elsewhere."""

    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_wikitext_tokens(
    dataset_name: str,
    dataset_config_name: str,
    split: str,
    tokenizer_name: str,
    max_chars: Optional[int] = None,
) -> List[int]:
    """Load WikiText text and tokenize it with a Hugging Face tokenizer."""

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Mini-MoE WikiText training requires datasets and transformers. "
            "Install the base requirements before running this script."
        ) from exc

    dataset = load_dataset(dataset_name, dataset_config_name, split=split)
    text = "\n\n".join(row["text"] for row in dataset if row.get("text"))
    if max_chars is not None and max_chars > 0:
        text = text[:max_chars]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def build_dataloader(token_ids: List[int], block_size: int, batch_size: int, shuffle: bool) -> DataLoader:
    """Build a PyTorch dataloader for fixed-length token blocks."""

    dataset = TokenBlockDataset(token_ids, block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Compute L2 norm of gradients for a parameter iterable."""

    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().norm(2).item() ** 2)
    return math.sqrt(total)


def router_parameters(model: MiniMoELanguageModel) -> Iterable[torch.nn.Parameter]:
    """Yield router projection parameters."""

    for name, param in model.named_parameters():
        if ".router.router." in name:
            yield param


def expert_parameters(model: MiniMoELanguageModel) -> Iterable[torch.nn.Parameter]:
    """Yield expert MLP parameters."""

    for name, param in model.named_parameters():
        if ".experts." in name:
            yield param


def tensor_to_float(value: object) -> object:
    """Convert scalar tensors to Python floats for JSON serialization."""

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    return value


def summarize_outputs(outputs: Dict[str, torch.Tensor]) -> Dict[str, object]:
    """Extract serializable loss and router metrics from model outputs."""

    keys = [
        "loss",
        "expert_load_variance",
        "dead_expert_ratio",
        "expert_entropy",
        "router_scores_mean",
        "router_scores_std",
        "router_temperature",
        "expert_load_by_layer",
    ]
    return {key: tensor_to_float(outputs[key]) for key in keys if key in outputs}


@torch.no_grad()
def evaluate(
    model: MiniMoELanguageModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate causal LM loss, perplexity, and router metrics."""

    model.eval()
    losses: List[float] = []
    metric_sums: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        losses.append(float(outputs["loss"].item()))
        for key, value in summarize_outputs(outputs).items():
            if key in {"loss", "expert_load_by_layer"}:
                continue
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            metric_counts[key] = metric_counts.get(key, 0) + 1
    mean_loss = sum(losses) / max(len(losses), 1)
    result = {"eval_loss": mean_loss, "eval_ppl": math.exp(min(mean_loss, 20.0))}
    for key, total in metric_sums.items():
        result[key] = total / max(metric_counts[key], 1)
    model.train()
    return result


def save_json(path: Path, payload: Dict[str, object]) -> None:
    """Write a JSON payload with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_checkpoint(
    model: MiniMoELanguageModel,
    output_dir: Path,
    step: int,
    config: MiniMoEModelConfig,
    metrics: Dict[str, object],
) -> None:
    """Save a model checkpoint and adjacent metadata."""

    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    save_json(checkpoint_dir / "config.json", asdict(config))
    save_json(checkpoint_dir / "metrics.json", metrics)


def train_mini_moe(config: Dict[str, object]) -> Dict[str, object]:
    """Run a Mini-MoE training loop and save paper-oriented metrics."""

    seed = int(config.get("seed", 42))
    set_seed(seed)
    output_dir = Path(str(config.get("output_dir", "outputs/mini_moe_run")))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tokens = load_wikitext_tokens(
        dataset_name=str(config.get("dataset_name", "wikitext")),
        dataset_config_name=str(config.get("dataset_config_name", "wikitext-2-raw-v1")),
        split=str(config.get("train_split", "train")),
        tokenizer_name=str(config.get("tokenizer_name", "gpt2")),
        max_chars=config.get("max_train_chars"),
    )
    eval_tokens = load_wikitext_tokens(
        dataset_name=str(config.get("dataset_name", "wikitext")),
        dataset_config_name=str(config.get("dataset_config_name", "wikitext-2-raw-v1")),
        split=str(config.get("eval_split", "validation")),
        tokenizer_name=str(config.get("tokenizer_name", "gpt2")),
        max_chars=config.get("max_eval_chars"),
    )

    model_config = MiniMoEModelConfig(**dict(config.get("model", {})))
    train_loader = build_dataloader(
        train_tokens,
        block_size=model_config.block_size,
        batch_size=int(config.get("batch_size", 4)),
        shuffle=True,
    )
    eval_loader = build_dataloader(
        eval_tokens,
        block_size=model_config.block_size,
        batch_size=int(config.get("eval_batch_size", config.get("batch_size", 4))),
        shuffle=False,
    )

    device = torch.device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    model = MiniMoELanguageModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3.0e-4)),
        weight_decay=float(config.get("weight_decay", 0.1)),
    )

    max_steps = int(config.get("max_steps", 1000))
    eval_steps = int(config.get("eval_steps", 100))
    save_steps = int(config.get("save_steps", 500))
    grad_clip = float(config.get("grad_clip", 1.0))
    history: List[Dict[str, object]] = []
    model.train()
    step = 0
    while step < max_steps:
        for batch in tqdm(train_loader, total=min(len(train_loader), max_steps - step), desc="mini-moe"):
            if step >= max_steps:
                break
            step += 1
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels, step=step)
            loss = outputs["loss"]
            loss.backward()
            router_norm = grad_norm(router_parameters(model))
            expert_norm = grad_norm(expert_parameters(model))
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            record = {
                "step": step,
                "train_loss": float(loss.detach().cpu().item()),
                "router_grad_norm": router_norm,
                "expert_grad_norm": expert_norm,
            }
            record.update({k: v for k, v in summarize_outputs(outputs).items() if k != "loss"})
            history.append(record)

            if eval_steps > 0 and step % eval_steps == 0:
                eval_metrics = evaluate(
                    model,
                    eval_loader,
                    device=device,
                    max_batches=config.get("eval_max_batches"),
                )
                history[-1].update(eval_metrics)
                save_json(output_dir / "metrics.json", {"history": history})

            if save_steps > 0 and step % save_steps == 0:
                save_checkpoint(model, output_dir, step, model_config, history[-1])

    final_eval = evaluate(model, eval_loader, device=device, max_batches=config.get("eval_max_batches"))
    summary = {
        "stage": "stage5_mini_moe",
        "max_steps": max_steps,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_eval": final_eval,
        "model_config": asdict(model_config),
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "metrics.json", {"history": history})
    save_json(output_dir / "train_summary.json", summary)
    save_checkpoint(model, output_dir, max_steps, model_config, summary)
    return summary


def load_model_for_eval(config_path: str, checkpoint_path: str, device: torch.device) -> MiniMoELanguageModel:
    """Load a Mini-MoE checkpoint using the model section from a YAML config."""

    config = load_yaml_config(config_path)
    model_config = MiniMoEModelConfig(**dict(config.get("model", {})))
    model = MiniMoELanguageModel(model_config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
