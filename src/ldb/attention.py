"""Structured Q/K/V, attention-pattern, and per-head analysis."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pydantic import Field

from ldb.models import FrozenModel, HookPoint
from ldb.runtime import ProbeCallback, ProbeHandle

_ATTENTION_HOOK = re.compile(
    r"^block\.(?P<layer>\d+)\.attention\.(?P<component>q|k|v|scores|pattern|z|result)$"
)


class AttentionRuntime(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def backend(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    revision: str | None

    def hook_points(self) -> Sequence[HookPoint]: ...

    def install_probe(self, hook: HookPoint, callback: ProbeCallback) -> ProbeHandle: ...

    def next_token_log_probability(self, prompt: str, target_token_id: int) -> float: ...

    def tokenize(self, prompt: str) -> Sequence[int]: ...


class AttentionEdge(FrozenModel):
    query_token: int = Field(ge=0)
    key_token: int = Field(ge=0)
    weight: float


class AttentionHead(FrozenModel):
    layer: int = Field(ge=0)
    head: int = Field(ge=0)
    q_norm: float | None = None
    k_norm: float | None = None
    v_norm: float | None = None
    z_norm: float | None = None
    result_norm: float | None = None
    score_mean: float | None = None
    score_std: float | None = None
    pattern_entropy: float | None = None
    maximum_attention: AttentionEdge | None = None


class AttentionReport(FrozenModel):
    schema_version: Literal["ldb.attention/v1alpha1"] = "ldb.attention/v1alpha1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_id: str
    model_revision: str | None = None
    backend: str
    backend_version: str
    prompt: str
    token_ids: tuple[int, ...]
    heads: tuple[AttentionHead, ...]


def analyze_attention(runtime: AttentionRuntime, *, prompt: str) -> AttentionReport:
    """Execute one prompt and summarize every exposed attention head."""

    hooks = tuple(hook for hook in runtime.hook_points() if _ATTENTION_HOOK.match(hook.name))
    if not hooks:
        raise ValueError("Runtime exposes no detailed attention hooks")
    activations: dict[tuple[int, str], Any] = {}

    def capture(hook: HookPoint, tensor: Any) -> None:
        match = _ATTENTION_HOOK.match(hook.name)
        assert match is not None
        activations[(int(match.group("layer")), match.group("component"))] = (
            tensor.detach().float().cpu()
        )

    handles: list[ProbeHandle] = []
    try:
        for hook in hooks:
            handles.append(runtime.install_probe(hook, capture))
        runtime.next_token_log_probability(prompt, 0)
    finally:
        for handle in handles:
            handle.remove()

    heads: list[AttentionHead] = []
    layers = sorted({layer for layer, _ in activations})
    for layer in layers:
        layer_values = {
            component: value
            for (candidate_layer, component), value in activations.items()
            if candidate_layer == layer
        }
        head_count = _head_count(layer_values)
        for head in range(head_count):
            pattern = layer_values.get("pattern")
            heads.append(
                AttentionHead(
                    layer=layer,
                    head=head,
                    q_norm=_head_norm(layer_values.get("q"), head, head_count),
                    k_norm=_head_norm(layer_values.get("k"), head, head_count),
                    v_norm=_head_norm(layer_values.get("v"), head, head_count),
                    z_norm=_head_norm(layer_values.get("z"), head, head_count),
                    result_norm=_head_norm(layer_values.get("result"), head, head_count),
                    score_mean=_score_stat(layer_values.get("scores"), head, "mean"),
                    score_std=_score_stat(layer_values.get("scores"), head, "std"),
                    pattern_entropy=_pattern_entropy(pattern, head),
                    maximum_attention=_maximum_attention(pattern, head),
                )
            )

    return AttentionReport(
        model_id=runtime.model_id,
        model_revision=runtime.revision,
        backend=runtime.backend,
        backend_version=runtime.backend_version,
        prompt=prompt,
        token_ids=tuple(runtime.tokenize(prompt)),
        heads=tuple(heads),
    )


def _head_count(values: dict[str, Any]) -> int:
    for component in ("q", "k", "v", "z", "result"):
        tensor = values.get(component)
        if tensor is not None and tensor.ndim == 4:
            return int(tensor.shape[2])
    pattern = values.get("pattern")
    if pattern is not None and pattern.ndim == 4:
        return int(pattern.shape[1])
    raise ValueError("Detailed attention hooks have no semantic head axis")


def _head_norm(tensor: Any, head: int, query_head_count: int) -> float | None:
    if tensor is None:
        return None
    if tensor.ndim != 4:
        raise ValueError("Per-head activation does not match [batch, token, head, channel]")
    tensor_head_count = int(tensor.shape[2])
    if query_head_count % tensor_head_count:
        raise ValueError("Query heads cannot be mapped onto the exposed key/value heads")
    tensor_head = head // (query_head_count // tensor_head_count)
    return float(tensor[:, :, tensor_head].norm().item())


def _pattern_entropy(pattern: Any, head: int) -> float | None:
    if pattern is None:
        return None
    probabilities = pattern[:, head].clamp_min(0)
    totals = probabilities.sum(dim=-1, keepdim=True)
    valid = totals.squeeze(-1) > 0
    normalized = probabilities / totals.clamp_min(1e-30)
    entropy = -(normalized * normalized.clamp_min(1e-30).log()).sum(dim=-1)
    return float(entropy[valid].mean().item()) if valid.any() else 0.0


def _score_stat(scores: Any, head: int, statistic: str) -> float | None:
    import torch

    if scores is None:
        return None
    values = scores[:, head]
    finite = torch.isfinite(values)
    if values.is_floating_point():
        finite &= values > torch.finfo(values.dtype).min / 2
    values = values[finite]
    if values.numel() == 0:
        return None
    if statistic == "mean":
        return float(values.mean().item())
    return float(values.std(unbiased=False).item())


def _maximum_attention(pattern: Any, head: int) -> AttentionEdge | None:
    if pattern is None:
        return None
    values = pattern[:, head]
    flat_index = int(values.reshape(-1).argmax().item())
    key_count = int(values.shape[-1])
    query_count = int(values.shape[-2])
    within_batch = flat_index % (query_count * key_count)
    query, key = divmod(within_batch, key_count)
    return AttentionEdge(
        query_token=query,
        key_token=key,
        weight=float(values.reshape(-1)[flat_index].item()),
    )
