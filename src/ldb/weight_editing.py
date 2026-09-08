"""Export explicit parameter edits as reloadable Transformers model weights."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any


def tensor_hash(value: Any) -> str:
    import torch

    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    header = json.dumps([str(value.dtype), list(value.shape)]).encode()
    digest = hashlib.sha256(header)
    digest.update(raw)
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_weight_edit(
    model: Any,
    output: Path,
    *,
    parameter: str,
    operation: str,
    value: float | None = None,
    tensor: Path | None = None,
    axis: int | None = None,
    start: int | None = None,
    stop: int | None = None,
    tokenizer: Any = None,
    max_shard_size: str = "2GB",
) -> dict[str, Any]:
    """Edit an exclusive offline model for export, restoring its in-memory parameter.

    The new directory owns the permanent change. Existing directories are refused.
    Shared Parameter aliases are preserved, not silently untied.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import PreTrainedModel

    if not isinstance(model, PreTrainedModel):
        raise ValueError("Weight export requires a Transformers PreTrainedModel")
    if model.training or getattr(model, "_ldb_structural_edit", False):
        raise ValueError("Export requires an exclusive eval model without structural edits")
    if getattr(model, "is_quantized", False) or getattr(model, "peft_config", None):
        raise ValueError("Quantized and PEFT models require their own merge/export workflow")
    parameters = dict(model.named_parameters(remove_duplicate=False))
    if parameter not in parameters:
        raise ValueError(f"Unknown parameter: {parameter}")
    weight = parameters[parameter]
    if weight.device.type == "meta" or not weight.is_floating_point():
        raise ValueError("The parameter must be a materialized floating-point tensor")
    if weight.layout != torch.strided or not weight.is_contiguous():
        raise ValueError("Only contiguous dense parameters are supported")
    aliases = [name for name, candidate in parameters.items() if candidate is weight]
    for candidate in parameters.values():
        if (
            candidate is not weight
            and candidate.device == weight.device
            and (candidate.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr())
        ):
            raise ValueError("Distinct Parameters sharing storage are not supported")
    if operation not in {"scale", "add", "replace"}:
        raise ValueError("Operation must be scale, add, or replace")
    if operation == "replace":
        if tensor is None or value is not None:
            raise ValueError("replace requires --tensor and no scalar value")
    elif value is None or not math.isfinite(value) or tensor is not None:
        raise ValueError("scale/add require a finite scalar value and no tensor")
    selection: list[Any] = [slice(None)] * weight.ndim
    if axis is None:
        if start is not None or stop is not None:
            raise ValueError("A slice requires axis, start, and stop")
    else:
        if not 0 <= axis < weight.ndim or start is None or stop is None:
            raise ValueError("A slice requires a valid nonnegative axis, start, and stop")
        if not 0 <= start < stop <= weight.shape[axis]:
            raise ValueError("Slice bounds must select at least one existing element")
        selection[axis] = slice(start, stop)
    before = weight.detach().cpu().clone()
    edited = before.clone()
    part = edited[tuple(selection)]
    if operation == "replace":
        assert tensor is not None
        tensors = load_file(tensor)
        if len(tensors) != 1:
            raise ValueError("Replacement file must contain exactly one tensor")
        replacement = next(iter(tensors.values()))
        if replacement.shape != part.shape or not replacement.is_floating_point():
            raise ValueError("Replacement shape must exactly match the selected floating tensor")
        part.copy_(replacement)
    elif operation == "scale":
        assert value is not None
        part.mul_(value)
    else:
        assert value is not None
        part.add_(value)
    if not bool(torch.isfinite(edited).all()):
        raise ValueError("Edited weights contain NaN or infinity")
    manifest: dict[str, Any] = {
        "schema": "ldb.weight-edit/v1alpha1",
        "source_model": model.config.name_or_path,
        "source_revision": getattr(model.config, "_commit_hash", None),
        "parameter": parameter,
        "tied_aliases": aliases,
        "operation": operation,
        "value": value,
        "slice": {"axis": axis, "start": start, "stop": stop},
        "before_hash": tensor_hash(before),
        "after_hash": tensor_hash(edited),
        "replacement_file_sha256": file_hash(tensor) if tensor else None,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        try:
            with torch.no_grad():
                weight.copy_(edited)
            model.save_pretrained(output, max_shard_size=max_shard_size)
            if tokenizer is not None:
                tokenizer.save_pretrained(output)
            found = []
            for shard in sorted(output.glob("*.safetensors")):
                with safe_open(shard, framework="pt", device="cpu") as archive:
                    keys = set(archive.keys())
                    for alias in aliases:
                        if alias in keys:
                            if tensor_hash(archive.get_tensor(alias)) != manifest["after_hash"]:
                                raise RuntimeError(
                                    "Serialized parameter differs from the edited weight"
                                )
                            found.append(alias)
            if not found:
                raise RuntimeError("Edited parameter or tied alias missing from serialized weights")
            manifest["serialized_aliases"] = found
            manifest["files"] = {
                str(path.relative_to(output)): file_hash(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            }
            (output / "ldb-weight-edit.json").write_text(json.dumps(manifest, indent=2) + "\n")
        finally:
            with torch.no_grad():
                weight.copy_(before)
    except BaseException:
        shutil.rmtree(output)
        raise
    return manifest
