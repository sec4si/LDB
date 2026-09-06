"""Offline Neural Patch verification against target and regression suites."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field

from ldb.causal import CausalRuntime
from ldb.models import FrozenModel, NeuralPatch
from ldb.offline import install_interventions, remove_interventions
from ldb.search import BehaviorCase, CaseScore, EvaluationSuite, score_case


class CategorySummary(FrozenModel):
    case_count: int = Field(gt=0)
    success: float = Field(ge=0.0, le=1.0)
    mean_delta: float
    worst_delta: float


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
    categories: dict[str, CategorySummary]


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
        case.case_id: score_case(runtime, case) for _, cases in grouped_cases for case in cases
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
                category=case.category,
                baseline_log_probability=baselines[case.case_id],
                patched_log_probability=(patched := score_case(runtime, case)),
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
    categories: dict[str, CategorySummary] = {}
    for category in sorted({score.category for score in case_scores}):
        scores = [score for score in case_scores if score.category == category]
        successful = sum(
            score.delta
            >= (
                target_min_gain
                if score.group == "target"
                else related_min_gain
                if score.group == "related"
                else -regression_max_drop
            )
            for score in scores
        )
        categories[category] = CategorySummary(
            case_count=len(scores),
            success=successful / len(scores),
            mean_delta=sum(score.delta for score in scores) / len(scores),
            worst_delta=min(score.delta for score in scores),
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
        categories=categories,
    )
