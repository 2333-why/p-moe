"""Text generation helpers used by Stage 1 scripts."""

from __future__ import annotations

from typing import Any


def build_generation_kwargs(
    *,
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    do_sample: bool = True,
    repetition_penalty: float = 1.0,
) -> dict[str, Any]:
    """Build common generation kwargs with explicit sampling controls."""

    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
    }
    if do_sample:
        kwargs.update({"temperature": temperature, "top_p": top_p, "top_k": top_k})
    return kwargs


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    device: str | None = None,
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    do_sample: bool = True,
    repetition_penalty: float = 1.0,
) -> dict[str, Any]:
    """Generate text from a loaded model/tokenizer pair."""

    import torch

    encoded = tokenizer(prompt, return_tensors="pt")
    if device:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    else:
        try:
            first_param = next(model.parameters())
            if str(first_param.device) not in {"meta", "cpu"}:
                encoded = {key: value.to(first_param.device) for key, value in encoded.items()}
        except StopIteration:
            pass

    generation_kwargs = build_generation_kwargs(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
    )
    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)
    input_tokens = int(encoded["input_ids"].shape[-1])
    output_tokens = int(output_ids.shape[-1])
    return {
        "prompt": prompt,
        "text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "new_tokens": max(0, output_tokens - input_tokens),
        "generation_kwargs": generation_kwargs,
    }
