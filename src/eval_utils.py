"""Perplexity utilities for Stage 2 causal language model evaluation."""

from __future__ import annotations

import math
import time
from typing import Any

import torch
from tqdm import tqdm


def get_model_input_device(model: Any) -> torch.device:
    """
    Return the device where input tensors should be placed.
    """
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def tokenize_eval_text(tokenizer: Any, text: str) -> torch.Tensor:
    """
    Tokenize evaluation text and return a 2D input_ids tensor.
    """
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    if input_ids.ndim != 2 or input_ids.size(1) == 0:
        raise ValueError("Tokenized evaluation text is empty.")
    return input_ids


def compute_sliding_window_ppl(
    model: Any,
    tokenizer: Any,
    text: str,
    block_size: int = 2048,
    stride: int = 1024,
    max_eval_tokens: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Compute standard causal LM perplexity with overlapping sliding windows.

    The loss is accumulated only over the newly evaluated target tokens in each
    window, matching the common Hugging Face fixed-length PPL recipe.
    """
    if block_size <= 1:
        raise ValueError("block_size must be greater than 1.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if stride > block_size:
        raise ValueError("stride must be less than or equal to block_size.")

    input_ids = tokenize_eval_text(tokenizer, text)
    total_tokens = int(input_ids.size(1))
    if max_eval_tokens is not None:
        if max_eval_tokens <= 1:
            raise ValueError("max_eval_tokens must be greater than 1 when set.")
        input_ids = input_ids[:, : int(max_eval_tokens)]
    eval_tokens = int(input_ids.size(1))

    model.eval()
    device = get_model_input_device(model)
    nll_sum = 0.0
    target_token_count = 0
    prev_end_loc = 0
    start_time = time.perf_counter()

    window_starts = range(0, eval_tokens, stride)
    iterator = tqdm(window_starts, desc="Evaluating PPL", disable=not show_progress)
    with torch.no_grad():
        for begin_loc in iterator:
            end_loc = min(begin_loc + block_size, eval_tokens)
            trg_len = end_loc - prev_end_loc
            if trg_len <= 0:
                continue

            input_window = input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_window.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_window, labels=target_ids)
            neg_log_likelihood = float(outputs.loss.detach().cpu()) * trg_len
            nll_sum += neg_log_likelihood
            target_token_count += trg_len

            prev_end_loc = end_loc
            if end_loc >= eval_tokens:
                break

    elapsed = time.perf_counter() - start_time
    if target_token_count == 0:
        raise ValueError("No target tokens were evaluated.")

    mean_nll = nll_sum / target_token_count
    perplexity = math.exp(mean_nll)
    return {
        "ppl": perplexity,
        "perplexity": perplexity,
        "mean_nll": mean_nll,
        "total_nll": nll_sum,
        "total_tokens": total_tokens,
        "total_eval_tokens": target_token_count,
        "eval_tokens": eval_tokens,
        "target_tokens": target_token_count,
        "block_size": block_size,
        "stride": stride,
        "max_eval_tokens": max_eval_tokens,
        "eval_time_sec": elapsed,
        "tokens_per_sec": target_token_count / elapsed if elapsed > 0 else 0.0,
    }
