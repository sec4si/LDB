"""Portable activation checkpoints bound to exact model and forward inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from ldb.debug_edits import ActivationEdit
from ldb.models import FrozenModel


class LayerCheckpoint(FrozenModel):
    schema_version: Literal["ldb.layer-checkpoint/v1alpha1", "ldb.layer-checkpoint/v1alpha2"] = (
        "ldb.layer-checkpoint/v1alpha2"
    )
    identity: str
    next_layer: int = Field(ge=0)
    patches: dict[int, float]
    activation_hash: str
    edits: dict[str, ActivationEdit] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_version(self) -> Self:
        if self.schema_version == "ldb.layer-checkpoint/v1alpha1" and self.edits:
            raise ValueError("Named edits require checkpoint v1alpha2")
        return self


def fingerprint(value: Any) -> str:
    import torch

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            update((str(tensor.dtype), list(tensor.shape)))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item):
                update(key)
                update(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(f"sequence:{len(item)}".encode())
            for child in item:
                update(child)
        else:
            digest.update(json.dumps(item, allow_nan=False).encode() + b"\0")

    update(value)
    return digest.hexdigest()


def save_checkpoint(path: Path, checkpoint: LayerCheckpoint, hidden: Any) -> None:
    from safetensors.torch import save

    payload = save(
        {"hidden": hidden.detach().cpu().contiguous()},
        metadata={"checkpoint": checkpoint.model_dump_json()},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def load_checkpoint(path: Path) -> tuple[LayerCheckpoint, Any]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as stream:
        if set(stream.keys()) != {"hidden"}:
            raise ValueError("Checkpoint must contain exactly the hidden tensor")
        checkpoint = LayerCheckpoint.model_validate_json(
            (stream.metadata() or {}).get("checkpoint", "")
        )
        hidden = stream.get_tensor("hidden")
    if fingerprint(hidden) != checkpoint.activation_hash:
        raise ValueError("Checkpoint activation checksum mismatch")
    return checkpoint, hidden
