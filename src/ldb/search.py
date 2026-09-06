"""Minimal intervention search with target and regression scoring."""

from __future__ import annotations

import fnmatch
import hashlib
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from ldb.causal import CausalRuntime
from ldb.models import (
    HOOK_SCHEMA_VERSION,
    EvidenceLevel,
    FrozenModel,
    HookPoint,
    Intervention,
    InterventionKind,
    NeuralPatch,
)


class BehaviorCase(FrozenModel):
    case_id: str
    prompt: str
    target_token_id: int = Field(ge=0)
    weight: float = Field(default=1.0, gt=0.0)


class EvaluationSuite(FrozenModel):
    schema_version: Literal["ldb.suite/v1alpha1"] = "ldb.suite/v1alpha1"
    target: tuple[BehaviorCase, ...] = Field(min_length=1)
    related: tuple[BehaviorCase, ...] = ()
    regression: tuple[BehaviorCase, ...] = ()

    @model_validator(mode="after")
    def unique_case_ids(self) -> EvaluationSuite:
        case_ids = [case.case_id for case in (*self.target, *self.related, *self.regression)]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Case IDs must be unique across the evaluation suite")
        return self


class CaseScore(FrozenModel):
    case_id: str
    group: Literal["target", "related", "regression"]
    baseline_log_probability: float
    patched_log_probability: float
    delta: float


class SearchTrial(FrozenModel):
    hook: str
    scale: float
    target_gain: float
    related_gain: float | None = None
    regression_loss: float
    complexity: float
    objective: float
    cases: tuple[CaseScore, ...]


class SearchReport(FrozenModel):
    schema_version: Literal["ldb.search/v1alpha1"] = "ldb.search/v1alpha1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_id: str
    model_revision: str | None = None
    backend: str
    backend_version: str
    tokenizer_hash: str
    hook_schema_version: str = HOOK_SCHEMA_VERSION
    regression_weight: float
    complexity_weight: float
    trials: tuple[SearchTrial, ...]


def load_suite(path: Path) -> EvaluationSuite:
    return EvaluationSuite.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def search_scale_patches(
    runtime: CausalRuntime,
    suite: EvaluationSuite,
    *,
    hook_globs: Sequence[str],
    scales: Sequence[float],
    regression_weight: float = 1.0,
    complexity_weight: float = 0.01,
) -> SearchReport:
    """Evaluate one scalar residual intervention at a time."""

    if regression_weight < 0.0 or complexity_weight < 0.0:
        raise ValueError("Search weights must be non-negative")
    if not scales:
        raise ValueError("At least one scale candidate is required")
    if not all(math.isfinite(scale) for scale in scales):
        raise ValueError("Scale candidates must be finite")
    scale_candidates = tuple(dict.fromkeys((*scales, 1.0)))
    hooks = tuple(
        hook
        for hook in runtime.hook_points()
        if any(fnmatch.fnmatchcase(hook.name, pattern) for pattern in hook_globs)
    )
    if not hooks:
        raise ValueError("No hook points matched the requested patterns")

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
    trials: list[SearchTrial] = []
    for hook in hooks:
        for scale in scale_candidates:

            def apply_scale(_: HookPoint, tensor: Any, factor: float = scale) -> Any:
                return tensor * factor

            handle = runtime.install_intervention(hook, apply_scale)
            try:
                case_scores = tuple(
                    CaseScore(
                        case_id=case.case_id,
                        group=group,
                        baseline_log_probability=baselines[case.case_id],
                        patched_log_probability=(
                            patched := runtime.next_token_log_probability(
                                case.prompt, case.target_token_id
                            )
                        ),
                        delta=patched - baselines[case.case_id],
                    )
                    for group, cases in grouped_cases
                    for case in cases
                )
            finally:
                handle.remove()

            target_gain = _weighted_mean(case_scores, suite.target, group="target", loss=False)
            related_gain = (
                _weighted_mean(case_scores, suite.related, group="related", loss=False)
                if suite.related
                else None
            )
            regression_loss = _weighted_mean(
                case_scores, suite.regression, group="regression", loss=True
            )
            complexity = abs(scale - 1.0)
            objective = (
                target_gain - regression_weight * regression_loss - complexity_weight * complexity
            )
            trials.append(
                SearchTrial(
                    hook=hook.name,
                    scale=scale,
                    target_gain=target_gain,
                    related_gain=related_gain,
                    regression_loss=regression_loss,
                    complexity=complexity,
                    objective=objective,
                    cases=case_scores,
                )
            )

    trials.sort(key=lambda trial: (trial.objective, -trial.complexity), reverse=True)
    return SearchReport(
        model_id=runtime.model_id,
        model_revision=runtime.revision,
        backend=runtime.backend,
        backend_version=runtime.backend_version,
        tokenizer_hash=runtime.tokenizer_hash,
        regression_weight=regression_weight,
        complexity_weight=complexity_weight,
        trials=tuple(trials),
    )


def patch_from_search(report: SearchReport) -> NeuralPatch:
    if not report.trials:
        raise ValueError("Search report contains no trials")
    best = report.trials[0]
    if best.scale == 1.0 or best.objective <= 0.0:
        raise ValueError("No tested intervention outperformed the no-op baseline")
    identity = hashlib.sha256(
        f"{report.model_id}\0{report.model_revision}\0{best.hook}\0{best.scale}".encode()
    ).hexdigest()[:12]
    if report.model_revision is None:
        raise ValueError("Cannot export a patch without a resolved model revision")
    return NeuralPatch(
        patch_id=f"ldb-{identity}",
        description=f"Scale {best.hook} by {best.scale:.6g}.",
        model_id=report.model_id,
        model_revision=report.model_revision,
        tokenizer_hash=report.tokenizer_hash,
        adapter="ldb.adapters.transformers",
        interventions=(
            Intervention(
                hook=best.hook,
                operation=InterventionKind.SCALE,
                value=best.scale,
                request_scoped=True,
            ),
        ),
        evidence=EvidenceLevel.CAUSALLY_TESTED,
    )


def _weighted_mean(
    scores: Sequence[CaseScore],
    cases: Sequence[BehaviorCase],
    *,
    group: Literal["target", "related", "regression"],
    loss: bool,
) -> float:
    if not cases:
        return 0.0
    weights = {case.case_id: case.weight for case in cases}
    values = [score for score in scores if score.group == group]
    numerator = sum(
        weights[score.case_id] * (max(0.0, -score.delta) if loss else score.delta)
        for score in values
    )
    return numerator / sum(weights.values())
