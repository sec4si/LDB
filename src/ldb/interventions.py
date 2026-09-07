"""Shared tensor intervention primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ldb.models import InterventionKind


def load_intervention_tensor(path_value: str | Path) -> Any:
    from safetensors.torch import load_file

    path = Path(path_value).expanduser()
    tensors = load_file(path)
    if "activation" in tensors:
        return tensors["activation"]
    if len(tensors) != 1:
        raise ValueError("Patch tensor file must contain one tensor or an 'activation' tensor")
    return next(iter(tensors.values()))


def apply_tensor_operation(
    tensor: Any,
    operation: InterventionKind,
    *,
    value: float | None = None,
    replacement: Any = None,
    expert: int | None = None,
    head: int | None = None,
    num_heads: int | None = None,
    feature: Any = None,
    token: int | None = None,
    token_axis: int | None = None,
    head_axis: int | None = None,
) -> Any:
    import torch

    if token is not None:
        if tensor.ndim == 0:
            raise ValueError("Token selection requires a tensor with at least one dimension")
        axis = token_axis if token_axis is not None else tensor.ndim - 2 if tensor.ndim >= 3 else 0
        axis %= tensor.ndim
        size = tensor.shape[axis]
        index = token if token >= 0 else size + token
        if index < 0 or index >= size:
            raise ValueError(f"Token {token} is outside token dimension of size {size}")
        selector: list[Any] = [slice(None)] * tensor.ndim
        selector[axis] = index
        location = tuple(selector)
        modified = tensor.clone()
        modified[location] = apply_tensor_operation(
            tensor[location],
            operation,
            value=value,
            replacement=replacement,
            expert=expert,
            head=head,
            num_heads=num_heads,
            feature=feature,
            head_axis=head_axis,
        )
        return modified

    if operation is InterventionKind.SCALE:
        assert value is not None
        return tensor * value
    if operation is InterventionKind.ADD:
        operand = _tensor_for(replacement, tensor) if replacement is not None else value
        assert operand is not None
        return tensor + operand
    if operation is InterventionKind.ABLATE:
        return tensor * 0
    if operation is InterventionKind.REPLACE:
        assert replacement is not None
        return torch.broadcast_to(_tensor_for(replacement, tensor), tensor.shape)
    if operation is InterventionKind.ROUTING_BIAS:
        assert value is not None
        assert expert is not None
        if not tensor.shape or expert >= tensor.shape[-1]:
            raise ValueError(f"Expert {expert} is outside router dimension {tensor.shape[-1:]}")
        modified = tensor.clone()
        modified[..., expert] += value
        return modified
    if operation is InterventionKind.HEAD_ABLATE:
        assert head is not None
        assert num_heads is not None
        if head_axis is not None:
            axis = head_axis % tensor.ndim
            if tensor.shape[axis] != num_heads:
                raise ValueError(
                    f"Head axis has size {tensor.shape[axis]}, expected {num_heads} heads"
                )
            selector = [slice(None)] * tensor.ndim
            selector[axis] = head
            modified = tensor.clone()
            modified[tuple(selector)] = 0
            return modified
        if not tensor.shape or tensor.shape[-1] % num_heads:
            raise ValueError(
                f"Activation width {tensor.shape[-1:]} is not divisible by {num_heads} heads"
            )
        width = tensor.shape[-1] // num_heads
        modified = tensor.clone()
        modified[..., head * width : (head + 1) * width] = 0
        return modified
    if operation is InterventionKind.FEATURE_SCALE:
        assert value is not None
        assert feature is not None
        return feature.scale(tensor, value)
    raise ValueError(f"Unsupported intervention operation: {operation}")


def _tensor_for(value: Any, target: Any) -> Any:
    return value.to(device=target.device, dtype=target.dtype)
