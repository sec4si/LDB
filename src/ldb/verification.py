"""Offline Neural Patch verification against target and regression suites."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from ldb.causal import CausalRuntime
from ldb.models import FrozenModel, NeuralPatch
from ldb.offline import install_interventions, remove_interventions
from ldb.search import BehaviorCase, CaseScore, EvaluationSuite


class VerificationReport(FrozenModel):
    schema_version: Literal["ldb.verification/v1alpha1"] = "ldb.verification/v1alpha1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    patch_id: str
    model_id: str
    model_revision: str | None = None
    target_min_gain: float
    related_min_gain: float
    regression_max_drop: float
    target_success: float = Field(ge=0.0, le=1.0)
    related_generalization: float | None = Field(default=None, ge=0.0, le=1.0)
    unrelated_regression: float
    passed: bool
    cases: tuple[CaseScore, ...]


def verify_patch(
    runtime: CausalRuntime,
    patch: NeuralPatch,
    suite: EvaluationSuite,
    *,
    patch_directory: Path,
    target_min_gain: float = 0.0,
    related_min_gain: float = 0.0,
    regression_max_drop: float = 0.05,
) -> VerificationReport:
    """Apply a Neural Patch and compare desired-token scores with an unpatched run."""

    if regression_max_drop < 0.0:
        raise ValueError("Regression maximum drop must be non-negative")
    if patch.model_id != runtime.model_id:
        raise ValueError(f"Patch targets {patch.model_id!r}, runtime serves {runtime.model_id!r}")
    if runtime.revision is not None and patch.model_revision != runtime.revision:
        raise ValueError(
            f"Patch revision {patch.model_revision!r} does not match runtime {runtime.revision!r}"
        )
    if patch.tokenizer_hash != runtime.tokenizer_hash:
        raise ValueError("Patch tokenizer hash does not match the loaded runtime tokenizer")

    grouped_cases: tuple[
        tuple[Literal["target", "related", "regression"], tuple[BehaviorCase, ...]], ...
    ] = (
        ("target", suite.target),
        ("related", suite.related),
        ("regression", suite.regression),
    )
    baselines = {
        case.case_id: runtime.next_token_log_probability(case.prompt, case.target_token_id)
        for _, cases in grouped_cases
        for case in cases
    }

    handles = []
    try:
        handles = install_interventions(
            runtime, patch.interventions, resource_directory=patch_directory
        )

        case_scores = tuple(
            CaseScore(
                case_id=case.case_id,
                group=group,
                baseline_log_probability=baselines[case.case_id],
                patched_log_probability=(
                    patched := runtime.next_token_log_probability(case.prompt, case.target_token_id)
                ),
                delta=patched - baselines[case.case_id],
            )
            for group, cases in grouped_cases
            for case in cases
        )
    finally:
        remove_interventions(handles)

    target_scores = [score for score in case_scores if score.group == "target"]
    related_scores = [score for score in case_scores if score.group == "related"]
    regression_scores = [score for score in case_scores if score.group == "regression"]
    target_success = sum(score.delta >= target_min_gain for score in target_scores) / len(
        target_scores
    )
    related_success = (
        sum(score.delta >= related_min_gain for score in related_scores) / len(related_scores)
        if related_scores
        else None
    )
    worst_regression = min((score.delta for score in regression_scores), default=0.0)
    passed = (
        target_success == 1.0
        and related_success in {None, 1.0}
        and worst_regression >= -regression_max_drop
    )
    return VerificationReport(
        patch_id=patch.patch_id,
        model_id=runtime.model_id,
        model_revision=runtime.revision,
        target_min_gain=target_min_gain,
        related_min_gain=related_min_gain,
        regression_max_drop=regression_max_drop,
        target_success=target_success,
        related_generalization=related_success,
        unrelated_regression=worst_regression,
        passed=passed,
        cases=case_scores,
    )
