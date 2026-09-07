"""End-to-end causal localization and minimal patch search."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import Field

from ldb.attention import AttentionReport, AttentionRuntime, analyze_attention
from ldb.causal import CausalReport, CausalRuntime, CausalTrial, causal_trace
from ldb.models import (
    FrozenModel,
    HookPoint,
    Intervention,
    InterventionKind,
    NeuralPatch,
    TensorAxis,
)
from ldb.runtime import ProbeHandle
from ldb.sae import FeatureActivation, SAEDictionary, inspect_sae_features
from ldb.search import (
    BehaviorCase,
    CandidateSet,
    CombinationSearchReport,
    EvaluationSuite,
    PatchCandidate,
    SearchReport,
    patch_from_combination_search,
    patch_from_search,
    search_candidate_combinations,
    search_scale_patches,
)


class DiagnosisReport(FrozenModel):
    schema_version: Literal["ldb.diagnosis/v1alpha1"] = "ldb.diagnosis/v1alpha1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    causal: CausalReport
    attention: AttentionReport | None = None
    head_trials: tuple[CausalTrial, ...] = ()
    sae_features: tuple[FeatureActivation, ...] = ()
    selected_hooks: tuple[str, ...]
    search: SearchReport
    feature_search: CombinationSearchReport | None = None
    patch: NeuralPatch | None = None
    conclusion: str


class DiagnosisRuntime(CausalRuntime, AttentionRuntime, Protocol):
    pass


def diagnose_failure(
    runtime: DiagnosisRuntime,
    *,
    clean_prompt: str,
    failing_prompt: str,
    target_token_id: int,
    token_index: int = -1,
    hook_globs: Sequence[str] = (
        "block.*.residual.post",
        "block.*.attention.z",
        "block.*.mlp.output",
    ),
    scales: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
    max_hooks: int = 8,
    max_head_trials: int = 32,
    include_attention_report: bool = False,
    sae_hook: str | None = None,
    sae_dictionary: Path | None = None,
    max_sae_features: int = 8,
    feature_scales: Sequence[float] = (0.0, 0.5, 1.5),
    regression_weight: float = 1.0,
    complexity_weight: float = 0.01,
) -> DiagnosisReport:
    """Locate a failure and search bounded one-hook patches in one command."""

    if max_hooks <= 0 or max_head_trials < 0 or max_sae_features <= 0:
        raise ValueError("Diagnosis limits are invalid")
    if (sae_hook is None) != (sae_dictionary is None):
        raise ValueError("sae_hook and sae_dictionary must be provided together")
    causal = causal_trace(
        runtime,
        clean_prompt=clean_prompt,
        corrupt_prompt=failing_prompt,
        target_token_id=target_token_id,
        token_index=token_index,
        hook_globs=hook_globs,
    )
    ranked = sorted(causal.trials, key=lambda trial: abs(trial.effect), reverse=True)
    selected_hooks = tuple(trial.hook for trial in ranked[:max_hooks])
    attention, head_trials = _diagnose_heads(
        runtime,
        clean_prompt=clean_prompt,
        failing_prompt=failing_prompt,
        target_token_id=target_token_id,
        token_index=token_index,
        max_trials=max_head_trials,
        include_attention_report=include_attention_report,
    )
    suite = EvaluationSuite(
        target=(
            BehaviorCase(
                case_id="failure",
                prompt=failing_prompt,
                target_token_id=target_token_id,
                category="failure",
            ),
        ),
        regression=(
            BehaviorCase(
                case_id="clean-control",
                prompt=clean_prompt,
                target_token_id=target_token_id,
                category="locality",
            ),
        ),
    )
    search = search_scale_patches(
        runtime,
        suite,
        hook_globs=selected_hooks,
        scales=scales,
        token_index=token_index,
        regression_weight=regression_weight,
        complexity_weight=complexity_weight,
    )
    sae_features: tuple[FeatureActivation, ...] = ()
    feature_search = None
    if sae_hook is not None and sae_dictionary is not None:
        sae_features, feature_search = _diagnose_features(
            runtime,
            suite=suite,
            failing_prompt=failing_prompt,
            hook_name=sae_hook,
            dictionary_path=sae_dictionary,
            token_index=token_index,
            max_features=max_sae_features,
            scales=feature_scales,
            regression_weight=regression_weight,
            complexity_weight=complexity_weight,
        )
    feature_won = False
    try:
        patch = patch_from_search(search)
    except ValueError:
        patch = None
    if (
        feature_search is not None
        and feature_search.trials[0].objective > search.trials[0].objective
    ):
        try:
            patch = patch_from_combination_search(feature_search)
            feature_won = True
        except ValueError:
            pass
    if patch is None:
        conclusion = "The tested patch space did not outperform the no-op baseline."
    elif feature_won:
        assert feature_search is not None
        best_feature = feature_search.trials[0]
        intervention = best_feature.interventions[0]
        assert intervention.feature is not None
        assert intervention.value is not None
        conclusion = (
            f"Localized the strongest tested intervention to SAE feature "
            f"{intervention.feature} at {intervention.hook}; scale {intervention.value:.6g} "
            f"produced objective {best_feature.objective:+.6g}."
        )
    else:
        best = search.trials[0]
        conclusion = (
            f"Localized the strongest tested intervention to {best.hook}; "
            f"scale {best.scale:.6g} produced objective {best.objective:+.6g}."
        )
    return DiagnosisReport(
        causal=causal,
        attention=attention,
        head_trials=head_trials,
        sae_features=sae_features,
        selected_hooks=selected_hooks,
        search=search,
        feature_search=feature_search,
        patch=patch,
        conclusion=conclusion,
    )


def _diagnose_heads(
    runtime: DiagnosisRuntime,
    *,
    clean_prompt: str,
    failing_prompt: str,
    target_token_id: int,
    token_index: int,
    max_trials: int,
    include_attention_report: bool,
) -> tuple[AttentionReport | None, tuple[CausalTrial, ...]]:
    z_hooks = tuple(hook for hook in runtime.hook_points() if hook.name.endswith(".attention.z"))
    if not z_hooks or max_trials == 0 or not hasattr(runtime, "tokenize"):
        return None, ()
    attention = (
        analyze_attention(runtime, prompt=failing_prompt) if include_attention_report else None
    )
    available = (
        {(head.layer, head.head) for head in attention.heads}
        if attention is not None
        else _discover_head_coordinates(runtime, z_hooks, failing_prompt)
    )
    trials: list[CausalTrial] = []
    for hook in z_hooks:
        layer = int(hook.name.split(".")[1])
        for candidate_layer, head in sorted(available):
            if candidate_layer != layer or len(trials) >= max_trials:
                continue
            report = causal_trace(
                runtime,
                clean_prompt=clean_prompt,
                corrupt_prompt=failing_prompt,
                target_token_id=target_token_id,
                token_index=token_index,
                head_index=head,
                hook_globs=(hook.name,),
            )
            trials.extend(report.trials)
    trials.sort(key=lambda trial: abs(trial.effect), reverse=True)
    return attention, tuple(trials)


def _discover_head_coordinates(
    runtime: DiagnosisRuntime,
    hooks: tuple[HookPoint, ...],
    prompt: str,
) -> set[tuple[int, int]]:
    coordinates: set[tuple[int, int]] = set()

    def capture(hook: HookPoint, tensor: Any) -> None:
        axis = hook.axis(TensorAxis.HEAD)
        if axis is None or axis >= tensor.ndim:
            raise ValueError(f"Hook {hook.name} has no valid semantic head axis")
        layer = int(hook.name.split(".")[1])
        coordinates.update((layer, head) for head in range(int(tensor.shape[axis])))

    handles: list[ProbeHandle] = []
    try:
        for hook in hooks:
            handles.append(runtime.install_probe(hook, capture))
        runtime.next_token_log_probability(prompt, 0)
    finally:
        for handle in handles:
            handle.remove()
    return coordinates


def _diagnose_features(
    runtime: DiagnosisRuntime,
    *,
    suite: EvaluationSuite,
    failing_prompt: str,
    hook_name: str,
    dictionary_path: Path,
    token_index: int,
    max_features: int,
    scales: Sequence[float],
    regression_weight: float,
    complexity_weight: float,
) -> tuple[tuple[FeatureActivation, ...], CombinationSearchReport | None]:
    if not scales or not all(math.isfinite(scale) for scale in scales):
        raise ValueError("SAE feature scales must contain finite values")
    hook = next((hook for hook in runtime.hook_points() if hook.name == hook_name), None)
    if hook is None:
        raise ValueError(f"SAE hook did not match this runtime: {hook_name}")
    activation = _capture_activation(runtime, hook, failing_prompt)
    activation = _select_token(hook, activation, token_index)
    dictionary = SAEDictionary(dictionary_path)
    features = tuple(
        feature
        for feature in inspect_sae_features(activation, dictionary, top_k=max_features)
        if feature.peak > 0.0
    )
    candidates = tuple(
        PatchCandidate(
            candidate_id=f"feature-{feature.feature}-scale-{scale:g}",
            intervention=Intervention(
                hook=hook_name,
                operation=InterventionKind.FEATURE_SCALE,
                value=scale,
                dictionary=str(dictionary_path.resolve()),
                feature=feature.feature,
                token=token_index,
                request_scoped=True,
            ),
            complexity=abs(scale - 1.0),
        )
        for feature in features
        for scale in dict.fromkeys(scales)
        if scale != 1.0
    )
    if not features:
        return (), None
    if not candidates:
        raise ValueError("SAE feature search requires a non-no-op scale")
    report = search_candidate_combinations(
        runtime,
        suite,
        CandidateSet(candidates=candidates),
        resource_directory=dictionary_path.parent,
        max_interventions=1,
        regression_weight=regression_weight,
        complexity_weight=complexity_weight,
        max_trials=len(candidates) + 1,
    )
    return features, report


def _capture_activation(runtime: DiagnosisRuntime, hook: HookPoint, prompt: str) -> Any:
    captured: list[Any] = []

    def capture(_: HookPoint, tensor: Any) -> None:
        captured.append(tensor.detach().clone())

    handle: ProbeHandle = runtime.install_probe(hook, capture)
    try:
        runtime.next_token_log_probability(prompt, 0)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError(f"Failure run did not execute SAE hook: {hook.name}")
    return captured[-1]


def _select_token(hook: HookPoint, tensor: Any, token_index: int) -> Any:
    ndim = tensor.ndim
    axis = hook.axis(TensorAxis.TOKEN, TensorAxis.QUERY_TOKEN)
    axis = axis if axis is not None else ndim - 2 if ndim >= 3 else 0
    size = tensor.shape[axis]
    index = token_index if token_index >= 0 else size + token_index
    if index < 0 or index >= size:
        raise ValueError(f"Token {token_index} is outside token dimension of size {size}")
    selector = [slice(None)] * ndim
    selector[axis] = slice(index, index + 1)
    return tensor[tuple(selector)]
