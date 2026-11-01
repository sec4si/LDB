"""Weight-only model diffing with a pretrained two-model crosscoder."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from ldb.models import FrozenModel
from ldb.sae import sha256_file


class CrosscoderFeature(FrozenModel):
    feature: int = Field(ge=0)
    relative_norm: float = Field(ge=0.0, le=1.0)
    norm_a: float = Field(ge=0.0)
    norm_b: float = Field(ge=0.0)
    cosine: float


class CrosscoderDiffReport(FrozenModel):
    schema_version: Literal["ldb.crosscoder-diff/v1alpha1"] = "ldb.crosscoder-diff/v1alpha1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dictionary_hash: str
    model_a: str
    model_b: str
    feature_count: int = Field(gt=0)
    dead_features: int = Field(ge=0)
    exclusive_threshold: float
    a_only: int = Field(ge=0)
    b_only: int = Field(ge=0)
    shared: int = Field(ge=0)
    # Ten equal-width bins of relative decoder norm, from 0 (model A only) to 1 (model B only).
    relative_norm_histogram: tuple[int, ...]
    shared_cosine_mean: float | None
    top_a_only: tuple[CrosscoderFeature, ...]
    top_b_only: tuple[CrosscoderFeature, ...]


def diff_crosscoder(
    path: Path,
    *,
    model_a: str = "A",
    model_b: str = "B",
    exclusive_threshold: float = 0.1,
    top_k: int = 20,
) -> CrosscoderDiffReport:
    """Classify features by how much of their decoder norm each model owns.

    The canonical layout stores ``decoder`` as ``[feature_count, 2, model_width]``.
    """

    import torch
    from safetensors.torch import load_file

    if not 0 < exclusive_threshold < 0.5 or top_k <= 0:
        raise ValueError("exclusive_threshold must be in (0, 0.5) and top_k positive")
    tensors = load_file(path)
    if "decoder" not in tensors:
        raise ValueError("Crosscoder dictionary is missing tensor 'decoder'")
    decoder = tensors["decoder"].float()
    if decoder.ndim != 3 or decoder.shape[1] != 2:
        raise ValueError("Crosscoder decoder must have shape [feature_count, 2, model_width]")

    norms = decoder.norm(dim=-1)
    total = norms.sum(-1)
    alive = total > 1e-8
    relative = torch.where(alive, norms[:, 1] / total.clamp_min(1e-8), torch.zeros_like(total))
    cosine = torch.nn.functional.cosine_similarity(decoder[:, 0], decoder[:, 1], dim=-1)
    a_only = alive & (relative < exclusive_threshold)
    b_only = alive & (relative > 1 - exclusive_threshold)
    shared = alive & ~a_only & ~b_only

    def top(mask: torch.Tensor, side: int) -> tuple[CrosscoderFeature, ...]:
        indices = mask.nonzero().flatten()
        ranked = indices[norms[indices, side].argsort(descending=True)][:top_k]
        return tuple(
            CrosscoderFeature(
                feature=int(index),
                relative_norm=float(relative[index]),
                norm_a=float(norms[index, 0]),
                norm_b=float(norms[index, 1]),
                cosine=float(cosine[index]),
            )
            for index in ranked
        )

    histogram = torch.histc(relative[alive], bins=10, min=0.0, max=1.0)
    return CrosscoderDiffReport(
        dictionary_hash=sha256_file(path),
        model_a=model_a,
        model_b=model_b,
        feature_count=int(decoder.shape[0]),
        dead_features=int((~alive).sum()),
        exclusive_threshold=exclusive_threshold,
        a_only=int(a_only.sum()),
        b_only=int(b_only.sum()),
        shared=int(shared.sum()),
        relative_norm_histogram=tuple(int(count) for count in histogram),
        shared_cosine_mean=float(cosine[shared].mean()) if bool(shared.any()) else None,
        top_a_only=top(a_only, 0),
        top_b_only=top(b_only, 1),
    )
