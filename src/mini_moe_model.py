"""MiniGPT/Mini-MoE language model for controlled routing experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mini_moe_layers import MiniMoERouterConfig, MiniTransformerBlock


@dataclass
class MiniMoEModelConfig:
    """Configuration for the MiniGPT/Mini-MoE language model."""

    vocab_size: int = 50257
    block_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    hidden_size: int = 256
    intermediate_size: int = 1024
    dropout: float = 0.1
    num_experts: int = 8
    top_k: int = 2
    routing_method: str = "pbit_load_competition"
    noisy_std: float = 1.0
    temperature: float = 1.0
    min_temperature: float = 0.2
    alpha: float = 0.1
    beta: float = 0.1
    load_ema_decay: float = 0.99
    dead_expert_threshold: float = 0.01
    warmup_steps: int = 0
    warmup_mode: str = "none"
    warmup_top_k: int = 4


class MiniMoELanguageModel(nn.Module):
    """Compact causal language model with MoE feed-forward transformer blocks."""

    def __init__(self, config: MiniMoEModelConfig) -> None:
        super().__init__()
        self.config = config
        router_config = MiniMoERouterConfig(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            routing_method=config.routing_method,
            noisy_std=config.noisy_std,
            temperature=config.temperature,
            min_temperature=config.min_temperature,
            alpha=config.alpha,
            beta=config.beta,
            load_ema_decay=config.load_ema_decay,
            dead_expert_threshold=config.dead_expert_threshold,
            warmup_steps=config.warmup_steps,
            warmup_mode=config.warmup_mode,
            warmup_top_k=config.warmup_top_k,
        )
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.block_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                MiniTransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    intermediate_size=config.intermediate_size,
                    dropout=config.dropout,
                    block_size=config.block_size,
                    router_config=router_config,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize trainable weights with GPT-style normal initialization."""

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute logits, optional LM loss, and aggregated routing metrics."""

        batch, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size {self.config.block_size}")
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        layer_stats = []
        for block in self.blocks:
            x, stats = block(x, step=step)
            layer_stats.append(stats)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        output: Dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            output["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        output.update(self.aggregate_router_stats(layer_stats))
        return output

    def aggregate_router_stats(self, layer_stats: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Average scalar router metrics across transformer layers."""

        metrics: Dict[str, torch.Tensor] = {}
        if not layer_stats:
            return metrics
        scalar_keys = [
            "expert_load_variance",
            "dead_expert_ratio",
            "expert_entropy",
            "router_scores_mean",
            "router_scores_std",
            "router_temperature",
        ]
        for key in scalar_keys:
            values = [stats[key].float() for stats in layer_stats if key in stats]
            if values:
                metrics[key] = torch.stack(values).mean()
        metrics["expert_load_by_layer"] = torch.stack(
            [stats["expert_load"].float() for stats in layer_stats if "expert_load" in stats]
        )
        return metrics

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively for small qualitative checks."""

        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.block_size :]
            logits = self(context)["logits"][:, -1, :] / max(temperature, 1.0e-8)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


def build_model_from_dict(config: Dict[str, object]) -> MiniMoELanguageModel:
    """Construct a MiniMoE language model from a plain configuration dictionary."""

    return MiniMoELanguageModel(MiniMoEModelConfig(**config))
