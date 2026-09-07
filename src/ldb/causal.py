"""Clean/corrupt activation patching for causal localization."""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pydantic import Field

from ldb.interventions import apply_tensor_operation
from ldb.models import (
    HOOK_SCHEMA_VERSION,
    FrozenModel,
    HookPoint,
    InterventionKind,
    TensorAxis,
)
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

    def continuation_log_probability(self, prompt: str, continuation: str) -> float: ...


class CausalTrial(FrozenModel):
    hook: str
    head: int | None = Field(default=None, ge=0)
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
    token_index: int = -1
    head_index: int | None = Field(default=None, ge=0)
    clean_log_probability: float
    corrupt_log_probability: float
    trials: tuple[CausalTrial, ...]


class CausalEdge(FrozenModel):
    source: str
    target: str
    activation_recovery: float
    mediated_log_probability: float
    mediated_effect: float
    mediated_recovery: float | None = None


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
    token_index: int = -1
    minimum_recovery: float
    max_path_trials: int = Field(gt=0)
    clean_log_probability: float
    corrupt_log_probability: float
    nodes: tuple[CausalTrial, ...]
    edges: tuple[CausalEdge, ...]


class AblationTrial(FrozenModel):
    hook: str
    head: int | None = Field(default=None, ge=0)
    ablated_log_probability: float
    effect: float


class AblationReport(FrozenModel):
    schema_version: Literal["ldb.ablation/v1alpha1"] = "ldb.ablation/v1alpha1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_id: str
    model_revision: str | None = None
    backend: str
    backend_version: str
    tokenizer_hash: str
    hook_schema_version: str = HOOK_SCHEMA_VERSION
    prompt: str
    target_token_id: int = Field(ge=0)
    token_index: int = -1
    baseline_log_probability: float
    trials: tuple[AblationTrial, ...]


def causal_ablation(
    runtime: CausalRuntime,
    *,
    prompt: str,
    target_token_id: int,
    token_index: int = -1,
    hook_globs: Sequence[str] = ("block.*.residual.post",),
    head_index: int | None = None,
) -> AblationReport:
    """Ablate one activation token at each selected hook and measure target impact."""

    hooks = tuple(
        hook
        for hook in runtime.hook_points()
        if any(fnmatch.fnmatchcase(hook.name, pattern) for pattern in hook_globs)
    )
    if not hooks:
        raise ValueError("No hook points matched the requested patterns")
    if head_index is not None:
        hooks = tuple(hook for hook in hooks if hook.axis(TensorAxis.HEAD) is not None)
        if not hooks:
            raise ValueError("Head selection requires a hook with a semantic head axis")

    baseline = runtime.next_token_log_probability(prompt, target_token_id)
    trials: list[AblationTrial] = []
    for hook in hooks:

        def ablate(selected_hook: HookPoint, tensor: Any) -> Any:
            head_axis = selected_hook.axis(TensorAxis.HEAD)
            if head_index is not None:
                assert head_axis is not None
                return _ablate_selection(
                    tensor,
                    token_index=token_index,
                    token_axis=_token_axis(selected_hook, tensor),
                    head_index=head_index,
                    head_axis=head_axis,
                )
            return apply_tensor_operation(
                tensor,
                InterventionKind.ABLATE,
                token=token_index,
                token_axis=_token_axis(selected_hook, tensor),
            )

        handle = runtime.install_intervention(hook, ablate)
        try:
            score = runtime.next_token_log_probability(prompt, target_token_id)
        finally:
            handle.remove()
        trials.append(
            AblationTrial(
                hook=hook.name,
                head=head_index,
                ablated_log_probability=score,
                effect=score - baseline,
            )
        )

    trials.sort(key=lambda trial: abs(trial.effect), reverse=True)
    return AblationReport(
        model_id=runtime.model_id,
        model_revision=runtime.revision,
        backend=runtime.backend,
        backend_version=runtime.backend_version,
        tokenizer_hash=runtime.tokenizer_hash,
        prompt=prompt,
        target_token_id=target_token_id,
        token_index=token_index,
        baseline_log_probability=baseline,
        trials=tuple(trials),
    )


def causal_trace(
    runtime: CausalRuntime,
    *,
    clean_prompt: str,
    corrupt_prompt: str,
    target_token_id: int,
    token_index: int = -1,
    head_index: int | None = None,
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
    if head_index is not None:
        hooks = tuple(hook for hook in hooks if hook.axis(TensorAxis.HEAD) is not None)
        if not hooks:
            raise ValueError("Head selection requires a hook with a semantic head axis")

    clean_activations: dict[str, Any] = {}
    handles: list[ProbeHandle] = []

    def capture(hook: HookPoint, tensor: Any) -> None:
        clean_activations[hook.name] = (
            _activation_slice(
                hook,
                tensor,
                token_index=token_index,
                head_index=head_index,
            )
            .detach()
            .clone()
        )

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

        def replace(selected_hook: HookPoint, tensor: Any, value: Any = clean_activation) -> Any:
            return _replace_activation(
                selected_hook,
                tensor,
                value,
                token_index=token_index,
                head_index=head_index,
            )

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
                head=head_index,
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
        token_index=token_index,
        head_index=head_index,
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
    token_index: int = -1,
    hook_globs: Sequence[str] = ("block.*.residual.post",),
    minimum_recovery: float = 0.01,
    max_path_trials: int = 256,
) -> CausalGraphReport:
    """Measure activation propagation and output mediation along candidate paths."""

    if minimum_recovery < 0.0:
        raise ValueError("Minimum recovery must be non-negative")
    if max_path_trials <= 0:
        raise ValueError("Maximum path trials must be positive")

    hooks = tuple(
        hook
        for hook in runtime.hook_points()
        if any(fnmatch.fnmatchcase(hook.name, pattern) for pattern in hook_globs)
    )
    if not hooks:
        raise ValueError("No hook points matched the requested patterns")
    clean, clean_order, clean_score = _capture_run(
        runtime, hooks, clean_prompt, target_token_id, token_index
    )
    corrupt, corrupt_order, corrupt_score = _capture_run(
        runtime, hooks, corrupt_prompt, target_token_id, token_index
    )
    if clean_order != corrupt_order:
        raise ValueError("Clean and corrupt runs executed selected hooks in different orders")

    denominator = clean_score - corrupt_score
    nodes: list[CausalTrial] = []
    candidate_paths: list[tuple[str, str, float, Any]] = []
    hook_by_name = {hook.name: hook for hook in hooks}
    for source_index, source_name in enumerate(clean_order):
        source = hook_by_name[source_name]

        def replace(selected_hook: HookPoint, tensor: Any, value: Any = clean[source_name]) -> Any:
            return _replace_token(tensor, value, token_index, _token_axis(selected_hook, tensor))

        handle = runtime.install_intervention(source, replace)
        try:
            patched, patched_order, patched_score = _capture_run(
                runtime, hooks, corrupt_prompt, target_token_id, token_index
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
                candidate_paths.append((source_name, target_name, recovery, patched[target_name]))

    candidate_paths.sort(key=lambda candidate: abs(candidate[2]), reverse=True)
    if len(candidate_paths) > max_path_trials:
        raise ValueError(
            f"Causal graph produced {len(candidate_paths)} path trials; limit is {max_path_trials}"
        )

    edges: list[CausalEdge] = []
    for source_name, target_name, activation_recovery, mediated_activation in candidate_paths:
        target = hook_by_name[target_name]

        def mediate(selected_hook: HookPoint, tensor: Any, value: Any = mediated_activation) -> Any:
            return _replace_token(tensor, value, token_index, _token_axis(selected_hook, tensor))

        handle = runtime.install_intervention(target, mediate)
        try:
            mediated_score = runtime.next_token_log_probability(corrupt_prompt, target_token_id)
        finally:
            handle.remove()
        effect = mediated_score - corrupt_score
        edges.append(
            CausalEdge(
                source=source_name,
                target=target_name,
                activation_recovery=activation_recovery,
                mediated_log_probability=mediated_score,
                mediated_effect=effect,
                mediated_recovery=effect / denominator if denominator != 0.0 else None,
            )
        )

    nodes.sort(key=lambda trial: trial.effect, reverse=True)
    edges.sort(key=lambda edge: abs(edge.mediated_effect), reverse=True)
    return CausalGraphReport(
        model_id=runtime.model_id,
        model_revision=runtime.revision,
        backend=runtime.backend,
        backend_version=runtime.backend_version,
        tokenizer_hash=runtime.tokenizer_hash,
        clean_prompt=clean_prompt,
        corrupt_prompt=corrupt_prompt,
        target_token_id=target_token_id,
        token_index=token_index,
        minimum_recovery=minimum_recovery,
        max_path_trials=max_path_trials,
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
    token_index: int,
) -> tuple[dict[str, Any], tuple[str, ...], float]:
    activations: dict[str, Any] = {}
    order: list[str] = []

    def capture(hook: HookPoint, tensor: Any) -> None:
        if hook.name not in activations:
            order.append(hook.name)
        activations[hook.name] = (
            _token_slice(tensor, token_index, _token_axis(hook, tensor)).detach().clone()
        )

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


def _token_axis(hook: HookPoint, tensor: Any) -> int:
    explicit = hook.axis(TensorAxis.TOKEN, TensorAxis.QUERY_TOKEN)
    if explicit is not None:
        if explicit >= tensor.ndim:
            raise ValueError(f"Hook {hook.name} declares an invalid token axis")
        return explicit
    return tensor.ndim - 2 if tensor.ndim >= 3 else 0


def _token_slice(tensor: Any, token_index: int, axis: int | None = None) -> Any:
    if tensor.ndim == 0:
        raise ValueError("Token selection requires a tensor with at least one dimension")
    axis = axis if axis is not None else tensor.ndim - 2 if tensor.ndim >= 3 else 0
    axis %= tensor.ndim
    size = tensor.shape[axis]
    index = token_index if token_index >= 0 else size + token_index
    if index < 0 or index >= size:
        raise ValueError(f"Token {token_index} is outside token dimension of size {size}")
    selector: list[Any] = [slice(None)] * tensor.ndim
    selector[axis] = index
    return tensor[tuple(selector)]


def _replace_token(tensor: Any, replacement: Any, token_index: int, axis: int | None = None) -> Any:
    selected = _token_slice(tensor, token_index, axis)
    value = replacement.to(device=selected.device, dtype=selected.dtype)
    if value.shape != selected.shape:
        raise ValueError(
            f"Clean token activation shape {tuple(value.shape)} does not match "
            f"corrupt token activation shape {tuple(selected.shape)}"
        )
    axis = axis if axis is not None else tensor.ndim - 2 if tensor.ndim >= 3 else 0
    axis %= tensor.ndim
    size = tensor.shape[axis]
    index = token_index if token_index >= 0 else size + token_index
    selector: list[Any] = [slice(None)] * tensor.ndim
    selector[axis] = index
    modified = tensor.clone()
    modified[tuple(selector)] = value
    return modified


def _ablate_selection(
    tensor: Any,
    *,
    token_index: int,
    token_axis: int,
    head_index: int,
    head_axis: int,
) -> Any:
    token_size = tensor.shape[token_axis]
    token = token_index if token_index >= 0 else token_size + token_index
    if token < 0 or token >= token_size:
        raise ValueError(f"Token {token_index} is outside token dimension of size {token_size}")
    head_size = tensor.shape[head_axis]
    if head_index < 0 or head_index >= head_size:
        raise ValueError(f"Head {head_index} is outside head dimension of size {head_size}")
    selector: list[Any] = [slice(None)] * tensor.ndim
    selector[token_axis] = token
    selector[head_axis] = head_index
    modified = tensor.clone()
    modified[tuple(selector)] = 0
    return modified


def _activation_slice(
    hook: HookPoint,
    tensor: Any,
    *,
    token_index: int,
    head_index: int | None,
) -> Any:
    token_axis = _token_axis(hook, tensor)
    token_size = tensor.shape[token_axis]
    token = token_index if token_index >= 0 else token_size + token_index
    if token < 0 or token >= token_size:
        raise ValueError(f"Token {token_index} is outside token dimension of size {token_size}")
    selector: list[Any] = [slice(None)] * tensor.ndim
    selector[token_axis] = token
    if head_index is not None:
        head_axis = hook.axis(TensorAxis.HEAD)
        if head_axis is None:
            raise ValueError(f"Hook {hook.name} has no semantic head axis")
        if head_index >= tensor.shape[head_axis]:
            raise ValueError(
                f"Head {head_index} is outside head dimension of size {tensor.shape[head_axis]}"
            )
        selector[head_axis] = head_index
    return tensor[tuple(selector)]


def _replace_activation(
    hook: HookPoint,
    tensor: Any,
    replacement: Any,
    *,
    token_index: int,
    head_index: int | None,
) -> Any:
    selected = _activation_slice(hook, tensor, token_index=token_index, head_index=head_index)
    value = replacement.to(device=selected.device, dtype=selected.dtype)
    if value.shape != selected.shape:
        raise ValueError(
            f"Clean activation shape {tuple(value.shape)} does not match "
            f"corrupt activation shape {tuple(selected.shape)}"
        )
    token_axis = _token_axis(hook, tensor)
    token_size = tensor.shape[token_axis]
    token = token_index if token_index >= 0 else token_size + token_index
    selector: list[Any] = [slice(None)] * tensor.ndim
    selector[token_axis] = token
    if head_index is not None:
        head_axis = hook.axis(TensorAxis.HEAD)
        assert head_axis is not None
        selector[head_axis] = head_index
    modified = tensor.clone()
    modified[tuple(selector)] = value
    return modified
