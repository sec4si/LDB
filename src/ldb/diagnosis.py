"""End-to-end causal localization and minimal patch search."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import Field

from ldb.attention import AttentionReport, AttentionRuntime, analyze_attention
from ldb.causal import CausalReport, CausalRuntime, CausalTrial, causal_trace
from ldb.models import FrozenModel, NeuralPatch
from ldb.search import (
    BehaviorCase,
    EvaluationSuite,
    SearchReport,
    patch_from_search,
    search_scale_patches,
)


class DiagnosisReport(FrozenModel):
    schema_version: Literal["ldb.diagnosis/v1alpha1"] = "ldb.diagnosis/v1alpha1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    causal: CausalReport
    attention: AttentionReport | None = None
    head_trials: tuple[CausalTrial, ...] = ()
    selected_hooks: tuple[str, ...]
    search: SearchReport
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
    regression_weight: float = 1.0,
    complexity_weight: float = 0.01,
) -> DiagnosisReport:
    """Locate a failure and search bounded one-hook patches in one command."""

    if max_hooks <= 0 or max_head_trials < 0:
        raise ValueError("max_hooks must be positive and max_head_trials must be non-negative")
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
    try:
        patch = patch_from_search(search)
    except ValueError:
        patch = None
    if patch is None:
        conclusion = "The tested patch space did not outperform the no-op baseline."
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
        selected_hooks=selected_hooks,
        search=search,
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
) -> tuple[AttentionReport | None, tuple[CausalTrial, ...]]:
    z_hooks = tuple(hook for hook in runtime.hook_points() if hook.name.endswith(".attention.z"))
    if not z_hooks or max_trials == 0 or not hasattr(runtime, "tokenize"):
        return None, ()
    attention = analyze_attention(runtime, prompt=failing_prompt)
    available = {(head.layer, head.head) for head in attention.heads}
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
