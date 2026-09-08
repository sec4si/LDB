"""Argument binding and deferred residual contracts for audited serving decoders."""

from __future__ import annotations

import inspect
from typing import Any

from ldb.agent import AgentRuntime

DEFERRED_RESIDUAL_CLASSES = frozenset(
    {
        "vllm.model_executor.models.llama.LlamaDecoderLayer",
        "vllm.model_executor.models.qwen2.Qwen2DecoderLayer",
        "vllm.model_executor.models.qwen3.Qwen3DecoderLayer",
        "sglang.srt.models.llama.LlamaDecoderLayer",
        "sglang.srt.models.qwen2.Qwen2DecoderLayer",
    }
)
COMMUNICATOR_DECODER = "sglang.srt.models.qwen3.Qwen3DecoderLayer"


def validate_communicator(module: Any) -> None:
    communicator = getattr(module, "layer_communicator", None)
    context = getattr(communicator, "_context", None)
    modes = getattr(communicator, "layer_scatter_modes", None)
    if (
        any(
            getattr(context, key, None) != 1
            for key in ("tp_size", "attn_tp_size", "attn_dp_size", "attn_cp_size")
        )
        or getattr(communicator, "_sp_variant", None) is not None
        or getattr(communicator, "allow_reduce_scatter", True)
        or getattr(communicator, "enable_fused_ar_quant", True)
        or any(
            getattr(getattr(modes, key, None), "name", None) not in {"FULL", "TP_ATTN_FULL"}
            for key in (
                "layer_input_mode",
                "attn_mode",
                "mlp_mode",
                "middle_residual_mode",
                "layer_output_mode",
            )
        )
    ):
        raise ValueError(
            "SGLang Qwen3 communicator currently requires unsharded, unquantized "
            "TP1/DP1/CP1 execution"
        )


def has_deferred_residual(module: Any) -> bool:
    parameters = inspect.signature(module.forward).parameters
    identity = f"{type(module).__module__}.{type(module).__name__}"
    if identity == COMMUNICATOR_DECODER:
        if (
            not {"hidden_states", "residual", "positions", "post_residual_addition"}
            <= parameters.keys()
        ):
            raise ValueError("SGLang Qwen3 decoder signature changed")
        validate_communicator(module)
        return True
    if identity in DEFERRED_RESIDUAL_CLASSES:
        if not {"hidden_states", "residual", "positions"} <= parameters.keys():
            raise ValueError(f"Audited decoder signature changed: {identity}")
        return True
    if "residual" in parameters:
        raise ValueError(f"Unsupported decoder residual contract: {identity}")
    return False


def _combined(hidden: Any, residual: Any) -> Any:
    import torch

    if not isinstance(hidden, torch.Tensor) or not hidden.is_floating_point():
        raise ValueError("Decoder hidden_states must be a floating tensor")
    if residual is None:
        return hidden
    if (
        not isinstance(residual, torch.Tensor)
        or residual.shape != hidden.shape
        or residual.dtype != hidden.dtype
        or residual.device != hidden.device
    ):
        raise ValueError(
            "Deferred residual requires matching floating tensor shape, dtype and device"
        )
    return hidden + residual


def _redistribute(hidden: Any, residual: Any, combined: Any, modified: Any) -> tuple[Any, Any]:
    import torch

    if residual is None:
        return modified, None
    changed = modified != combined
    return (
        torch.where(changed, modified, hidden),
        torch.where(changed, torch.zeros_like(residual), residual),
    )


def make_input_hook(agent: AgentRuntime, hook: str, module: Any, *, deferred: bool = False) -> Any:
    signature = inspect.signature(module.forward)
    communicator = f"{type(module).__module__}.{type(module).__name__}" == COMMUNICATOR_DECODER

    def callback(_: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        import torch

        if communicator:
            validate_communicator(module)
        if deferred and not agent.needs_hook(hook):
            return args, kwargs
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        name = next(
            (key for key in ("hidden_states", "x", "value", "input") if key in bound.arguments),
            None,
        )
        if name is None:
            name = next(
                (
                    key
                    for key, value in bound.arguments.items()
                    if isinstance(value, torch.Tensor) and value.is_floating_point()
                ),
                None,
            )
        if name is None:
            raise ValueError(f"Cannot identify activation argument for {hook}")
        tensor = bound.arguments[name]
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise ValueError(f"Activation argument for {hook} is not a floating tensor")
        if deferred:
            if getattr(tensor, "_sglang_needs_allreduce_fusion", False):
                raise ValueError(
                    "Unreduced hidden_states require a separate all-reduce hook contract"
                )
            tensor = _combined(tensor, bound.arguments["residual"])
        addition = bound.arguments.get("post_residual_addition") if communicator else None
        if addition is not None and bound.arguments["residual"] is not None:
            tensor = _combined(tensor, addition)
        modified = agent.process_tensor(hook, tensor)
        if modified is tensor:
            return args, kwargs
        if deferred:
            bound.arguments[name], bound.arguments["residual"] = _redistribute(
                bound.arguments[name], bound.arguments["residual"], tensor, modified
            )
            if addition is not None and bound.arguments["residual"] is not None:
                bound.arguments["post_residual_addition"] = torch.where(
                    modified != tensor, torch.zeros_like(addition), addition
                )
        else:
            bound.arguments[name] = modified
        return bound.args, bound.kwargs

    return callback


def make_residual_output_hook(agent: AgentRuntime, hook: str) -> Any:
    def callback(_: Any, args: Any, output: Any) -> Any:
        if not agent.needs_hook(hook):
            return output
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("Audited decoder must return (hidden_states, residual)")
        tensor = _combined(*output)
        modified = agent.process_tensor(hook, tensor)
        if modified is tensor:
            return output
        # Preserve untouched requests' original decomposition and avoid subtractive
        # cancellation on changed elements consumed by the next fused add/norm.
        return _redistribute(*output, tensor, modified)

    return callback
