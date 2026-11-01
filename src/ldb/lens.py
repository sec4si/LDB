"""Logit lens: read what each layer's residual stream would predict if decoding stopped there."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pydantic import Field

from ldb.models import FrozenModel, HookPoint
from ldb.runtime import ProbeCallback, ProbeHandle

_RESIDUAL_HOOK = re.compile(r"^block\.(?P<layer>\d+)\.residual\.post$")


class LensRuntime(Protocol):
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

    def unembed(self, residual: Any) -> Any: ...

    def decode(self, token_id: int) -> str: ...


class LensToken(FrozenModel):
    token_id: int = Field(ge=0)
    text: str
    probability: float


class LensLayer(FrozenModel):
    layer: int = Field(ge=0)
    top_tokens: tuple[LensToken, ...]
    target_probability: float | None = None
    target_rank: int | None = Field(default=None, ge=1)


class LogitLensReport(FrozenModel):
    schema_version: Literal["ldb.logit-lens/v1alpha1"] = "ldb.logit-lens/v1alpha1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_id: str
    model_revision: str | None = None
    backend: str
    backend_version: str
    prompt: str
    token_ids: tuple[int, ...]
    position: int
    target_token_id: int | None = None
    layers: tuple[LensLayer, ...]
    # Only defined when reading the last position: the final layer's lens must reproduce
    # the model's own next-token distribution, otherwise the unembedding path is wrong.
    final_layer_matches_model: bool | None = None


def logit_lens(
    runtime: LensRuntime,
    *,
    prompt: str,
    position: int = -1,
    top_k: int = 5,
    target_token_id: int | None = None,
    layers: Sequence[int] | None = None,
) -> LogitLensReport:
    """Project every residual.post activation at one position through the unembedding."""

    import torch

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    token_ids = tuple(runtime.tokenize(prompt))
    index = position if position >= 0 else len(token_ids) + position
    if not 0 <= index < len(token_ids):
        raise ValueError(f"Position {position} is outside the {len(token_ids)}-token prompt")
    residual_hooks = {
        int(match.group("layer")): hook
        for hook in runtime.hook_points()
        if (match := _RESIDUAL_HOOK.match(hook.name))
    }
    wanted = set(residual_hooks) if layers is None else set(layers)
    hooks = {layer: hook for layer, hook in residual_hooks.items() if layer in wanted}
    if not hooks:
        raise ValueError("Runtime exposes no residual.post hooks for the selected layers")

    residuals: dict[int, Any] = {}

    def capture(hook: HookPoint, tensor: Any) -> None:
        match = _RESIDUAL_HOOK.match(hook.name)
        assert match is not None
        residuals[int(match.group("layer"))] = tensor[0, index].detach().clone()

    handles = [runtime.install_probe(hook, capture) for hook in hooks.values()]
    probe_token = target_token_id if target_token_id is not None else 0
    try:
        model_log_probability = runtime.next_token_log_probability(prompt, probe_token)
    finally:
        for handle in handles:
            handle.remove()

    final_layer = max(residual_hooks)
    final_log_probability = None
    report_layers = []
    for layer in sorted(residuals):
        log_probabilities = torch.log_softmax(runtime.unembed(residuals[layer]).float(), dim=-1)
        if layer == final_layer:
            final_log_probability = float(log_probabilities[probe_token])
        top = torch.topk(log_probabilities, min(top_k, log_probabilities.shape[-1]))
        target_probability = target_rank = None
        if target_token_id is not None:
            target_log_probability = log_probabilities[target_token_id]
            target_probability = float(target_log_probability.exp())
            target_rank = int((log_probabilities > target_log_probability).sum()) + 1
        report_layers.append(
            LensLayer(
                layer=layer,
                top_tokens=tuple(
                    LensToken(
                        token_id=int(token),
                        text=runtime.decode(int(token)),
                        probability=float(value.exp()),
                    )
                    for value, token in zip(top.values, top.indices, strict=True)
                ),
                target_probability=target_probability,
                target_rank=target_rank,
            )
        )

    matches = None
    if index == len(token_ids) - 1 and final_log_probability is not None:
        matches = abs(final_log_probability - model_log_probability) < 1e-3
    return LogitLensReport(
        model_id=runtime.model_id,
        model_revision=runtime.revision,
        backend=runtime.backend,
        backend_version=runtime.backend_version,
        prompt=prompt,
        token_ids=token_ids,
        position=index,
        target_token_id=target_token_id,
        layers=tuple(report_layers),
        final_layer_matches_model=matches,
    )
