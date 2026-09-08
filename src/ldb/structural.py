"""Reversible, exclusive offline edits for eager GPT-2 experiments."""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def edit_gpt2(
    model: Any,
    *,
    layer: int | None = None,
    passes: int = 1,
    parameter: str | None = None,
    scale: float = 1.0,
) -> Iterator[None]:
    """Temporarily edit one GPT-2 layer or weight tensor; never use concurrently."""
    import torch
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

    if not isinstance(model, GPT2LMHeadModel):
        raise ValueError("Structural experiments currently require GPT2LMHeadModel")
    if model.training:
        raise ValueError("Structural experiments require eval mode")
    if (layer is None) == (parameter is None):
        raise ValueError("Choose exactly one layer or parameter")
    if passes < 0 or passes > 8 or not math.isfinite(scale):
        raise ValueError("Passes must be between 0 and 8 and scale must be finite")
    if getattr(model, "_ldb_structural_edit", False):
        raise ValueError("A structural edit is already active on this model")
    block = None
    weight = None
    snapshot = None
    previous_forward: Any = None
    had_forward = False
    if layer is not None:
        if not 0 <= layer < len(model.transformer.h):
            raise ValueError("Layer index is outside the model")
        block = model.transformer.h[layer]
        original = block.forward
        had_forward = "forward" in block.__dict__
        previous_forward = block.__dict__.get("forward")
        signature = inspect.signature(original)

        def forward(hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(hidden_states, *args, **kwargs)
            if bound.arguments.get("past_key_values") is not None or bound.arguments.get(
                "use_cache"
            ):
                raise ValueError("Layer edits require no KV cache")
            for _ in range(passes):
                hidden_states = original(hidden_states, *args, **kwargs)
                if not isinstance(hidden_states, torch.Tensor):
                    raise RuntimeError("Unsupported GPT-2 block output contract")
            return hidden_states

    else:
        assert parameter is not None
        weight = dict(model.named_parameters()).get(parameter)
        if weight is None:
            raise ValueError(f"Unknown parameter: {parameter}")
        snapshot = weight.detach().cpu().clone()

    def guard(module: Any, args: Any, kwargs: Any) -> None:
        if args or kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
            raise ValueError("Structural edits require keyword inputs and use_cache=False")

    cache_setting = model.config.use_cache
    handle = model.register_forward_pre_hook(guard, with_kwargs=True)
    model.__dict__["_ldb_structural_edit"] = True
    try:
        model.config.use_cache = False
        if block is not None:
            block.forward = forward
        if weight is not None:
            with torch.no_grad():
                weight.mul_(scale)
        yield
    finally:
        if block is not None:
            if had_forward:
                block.forward = previous_forward
            else:
                del block.forward
        if weight is not None:
            assert snapshot is not None
            with torch.no_grad():
                weight.copy_(snapshot)
        model.config.use_cache = cache_setting
        handle.remove()
        del model._ldb_structural_edit


def structural_experiment(model_id: str, *, device: str, prompt: str, layer: int) -> dict[str, Any]:
    import torch

    from ldb.adapters.transformers import TransformersAdapter

    adapter = TransformersAdapter(model_id, device=device)
    model = adapter.model
    inputs = adapter.tokenizer(prompt, return_tensors="pt").to(device)

    def logits() -> Any:
        with torch.inference_mode():
            return model(**inputs, use_cache=False).logits[:, -1].detach().float().cpu()

    baseline = logits()
    parameter = f"transformer.h.{layer}.mlp.c_proj.weight"
    operations = (
        ("weight_scale", {"parameter": parameter, "scale": 0.0}),
        ("skip_layer", {"layer": layer, "passes": 0}),
        ("repeat_layer", {"layer": layer, "passes": 2}),
    )
    results = []
    for name, config in operations:
        with edit_gpt2(model, **config):
            edited = logits()
        restored = logits()
        recovered = bool(torch.equal(baseline, restored))
        if not recovered:
            raise RuntimeError(f"{name}: baseline logits did not recover exactly")
        results.append(
            {
                "operation": name,
                "configuration": config,
                "max_logit_delta": float((edited - baseline).abs().max()),
                "restored_exactly": recovered,
            }
        )
    return {
        "model": model_id,
        "revision": adapter.revision,
        "device": str(next(model.parameters()).device),
        "prompt": prompt,
        "baseline_next_token": adapter.tokenizer.decode([int(baseline.argmax())]),
        "experiments": results,
    }
