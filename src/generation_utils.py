"""Text generation utilities for Stage 1 inference."""

from __future__ import annotations

import time
from typing import Any

import torch


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
) -> dict[str, Any]:
    """
    Generate text and return:
    {
        "prompt": ...,
        "generated_text": ...,
        "input_tokens": ...,
        "output_tokens": ...,
        "new_tokens": ...,
        "generation_time_sec": ...,
        "tokens_per_sec": ...
    }
    """
    device = get_model_input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    input_tokens = int(inputs["input_ids"].shape[-1])

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "repetition_penalty": float(repetition_penalty),
    }
    if do_sample:
        generation_kwargs["temperature"] = float(temperature)
        generation_kwargs["top_p"] = float(top_p)
    if tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    elif tokenizer.eos_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generation_time_sec = time.perf_counter() - start_time

    output_tokens = int(output_ids.shape[-1])
    new_tokens = max(0, output_tokens - input_tokens)
    tokens_per_sec = new_tokens / generation_time_sec if generation_time_sec > 0 else 0.0
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "new_tokens": new_tokens,
        "generation_time_sec": generation_time_sec,
        "tokens_per_sec": tokens_per_sec,
    }


def get_model_input_device(model) -> torch.device:
    """
    Resolve the device to use for tokenized inputs.
    """
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return torch.device(model_device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
