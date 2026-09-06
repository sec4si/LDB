"""Clean/corrupt activation patching for causal localization."""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import Field

from ldb.models import HOOK_SCHEMA_VERSION, FrozenModel, HookPoint
from ldb.runtime import InterventionCallback, ProbeCallback, ProbeHandle


class CausalRuntime(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def backend(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    @property
    def tokenizer_hash(self) -> str: ...

    revision: str | None

    def hook_points(self) -> Sequence[HookPoint]: ...

    def install_probe(self, hook: HookPoint, callback: ProbeCallback) -> ProbeHandle: ...

    def install_intervention(
        self, hook: HookPoint, callback: InterventionCallback
    ) -> ProbeHandle: ...

    def next_token_log_probability(self, prompt: str, target_token_id: int) -> float: ...


class CausalTrial(FrozenModel):
    hook: str
    patched_log_probability: float
    effect: float
    recovery: float | None = None


class CausalReport(FrozenModel):
    schema_version: str = "ldb.causal/v1alpha1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_id: str
    model_revision: str | None = None
    backend: str
    backend_version: str
    tokenizer_hash: str
    hook_schema_version: str = HOOK_SCHEMA_VERSION
    clean_prompt: str
    corrupt_prompt: str
    target_token_id: int = Field(ge=0)
    clean_log_probability: float
    corrupt_log_probability: float
    trials: tuple[CausalTrial, ...]


class CausalEdge(FrozenModel):
    source: str
    target: str
    activation_recovery: float


class CausalGraphReport(FrozenModel):
    schema_version: str = "ldb.causal-graph/v1alpha1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_id: str
    model_revision: str | None = None
    backend: str
    backend_version: str
    tokenizer_hash: str
    hook_schema_version: str = HOOK_SCHEMA_VERSION
    clean_prompt: str
    corrupt_prompt: str
    target_token_id: int = Field(ge=0)
    clean_log_probability: float
    corrupt_log_probability: float
    nodes: tuple[CausalTrial, ...]
    edges: tuple[CausalEdge, ...]


def causal_trace(
    runtime: CausalRuntime,
    *,
    clean_prompt: str,
    corrupt_prompt: str,
    target_token_id: int,
    hook_globs: Sequence[str] = ("block.*.residual.post",),
) -> CausalReport:
    """Patch clean activations into a corrupt run one hook at a time."""

    hooks = tuple(
        hook
        for hook in runtime.hook_points()
        if any(fnmatch.fnmatchcase(hook.name, pattern) for pattern in hook_globs)
    )
    if not hooks:
        raise ValueError("No hook points matched the requested patterns")

    clean_activations: dict[str, Any] = {}
    handles: list[ProbeHandle] = []

    def capture(hook: HookPoint, tensor: Any) -> None:
        clean_activations[hook.name] = tensor.detach().clone()

    try:
        handles = [runtime.install_probe(hook, capture) for hook in hooks]
        clean_score = runtime.next_token_log_probability(clean_prompt, target_token_id)
    finally:
        for handle in handles:
            handle.remove()

    missing = [hook.name for hook in hooks if hook.name not in clean_activations]
    if missing:
        raise RuntimeError(f"Clean run did not execute hooks: {', '.join(missing)}")

    corrupt_score = runtime.next_token_log_probability(corrupt_prompt, target_token_id)
    denominator = clean_score - corrupt_score
    trials: list[CausalTrial] = []
    for hook in hooks:
        clean_activation = clean_activations[hook.name]

        def replace(_: HookPoint, __: Any, value: Any = clean_activation) -> Any:
            return value

        handle = runtime.install_intervention(hook, replace)
        try:
            patched_score = runtime.next_token_log_probability(corrupt_prompt, target_token_id)
        finally:
            handle.remove()
        effect = patched_score - corrupt_score
        recovery = effect / denominator if denominator != 0.0 else None
        trials.append(
            CausalTrial(
                hook=hook.name,
                patched_log_probability=patched_score,
                effect=effect,
                recovery=recovery,
            )
        )

    trials.sort(key=lambda trial: trial.effect, reverse=True)
    return CausalReport(
        model_id=runtime.model_id,
        model_revision=runtime.revision,
        backend=runtime.backend,
        backend_version=runtime.backend_version,
        tokenizer_hash=runtime.tokenizer_hash,
        clean_prompt=clean_prompt,
        corrupt_prompt=corrupt_prompt,
        target_token_id=target_token_id,
        clean_log_probability=clean_score,
        corrupt_log_probability=corrupt_score,
        trials=tuple(trials),
    )


def causal_propagation_graph(
    runtime: CausalRuntime,
    *,
    clean_prompt: str,
    corrupt_prompt: str,
    target_token_id: int,
    hook_globs: Sequence[str] = ("block.*.residual.post",),
    minimum_recovery: float = 0.01,
) -> CausalGraphReport:
    """Measure directed downstream activation recovery after clean source patching."""

    hooks = tuple(
        hook
        for hook in runtime.hook_points()
        if any(fnmatch.fnmatchcase(hook.name, pattern) for pattern in hook_globs)
    )
    if not hooks:
        raise ValueError("No hook points matched the requested patterns")
    clean, clean_order, clean_score = _capture_run(runtime, hooks, clean_prompt, target_token_id)
    corrupt, corrupt_order, corrupt_score = _capture_run(
        runtime, hooks, corrupt_prompt, target_token_id
    )
    if clean_order != corrupt_order:
        raise ValueError("Clean and corrupt runs executed selected hooks in different orders")

    denominator = clean_score - corrupt_score
    nodes: list[CausalTrial] = []
    edges: list[CausalEdge] = []
    hook_by_name = {hook.name: hook for hook in hooks}
    for source_index, source_name in enumerate(clean_order):
        source = hook_by_name[source_name]

        def replace(_: HookPoint, __: Any, value: Any = clean[source_name]) -> Any:
            return value

        handle = runtime.install_intervention(source, replace)
        try:
            patched, patched_order, patched_score = _capture_run(
                runtime, hooks, corrupt_prompt, target_token_id
            )
        finally:
            handle.remove()
        if patched_order != clean_order:
            raise ValueError("Patched run executed selected hooks in a different order")
        effect = patched_score - corrupt_score
        nodes.append(
            CausalTrial(
                hook=source_name,
                patched_log_probability=patched_score,
                effect=effect,
                recovery=effect / denominator if denominator != 0.0 else None,
            )
        )
        for target_name in clean_order[source_index + 1 :]:
            recovery = _activation_recovery(
                clean[target_name], corrupt[target_name], patched[target_name]
            )
            if abs(recovery) >= minimum_recovery:
                edges.append(
                    CausalEdge(
                        source=source_name,
                        target=target_name,
                        activation_recovery=recovery,
                    )
                )

    nodes.sort(key=lambda trial: trial.effect, reverse=True)
    edges.sort(key=lambda edge: abs(edge.activation_recovery), reverse=True)
    return CausalGraphReport(
        model_id=runtime.model_id,
        model_revision=runtime.revision,
        backend=runtime.backend,
        backend_version=runtime.backend_version,
        tokenizer_hash=runtime.tokenizer_hash,
        clean_prompt=clean_prompt,
        corrupt_prompt=corrupt_prompt,
        target_token_id=target_token_id,
        clean_log_probability=clean_score,
        corrupt_log_probability=corrupt_score,
        nodes=tuple(nodes),
        edges=tuple(edges),
    )


def _capture_run(
    runtime: CausalRuntime,
    hooks: tuple[HookPoint, ...],
    prompt: str,
    target_token_id: int,
) -> tuple[dict[str, Any], tuple[str, ...], float]:
    activations: dict[str, Any] = {}
    order: list[str] = []

    def capture(hook: HookPoint, tensor: Any) -> None:
        if hook.name not in activations:
            order.append(hook.name)
        activations[hook.name] = tensor.detach().clone()

    handles = [runtime.install_probe(hook, capture) for hook in hooks]
    try:
        score = runtime.next_token_log_probability(prompt, target_token_id)
    finally:
        for handle in handles:
            handle.remove()
    missing = [hook.name for hook in hooks if hook.name not in activations]
    if missing:
        raise RuntimeError(f"Run did not execute hooks: {', '.join(missing)}")
    return activations, tuple(order), score


def _activation_recovery(clean: Any, corrupt: Any, patched: Any) -> float:
    baseline = (corrupt.float() - clean.float()).norm().item()
    if baseline == 0.0:
        return 0.0
    remaining = (patched.float() - clean.float()).norm().item()
    return float(1.0 - remaining / baseline)
