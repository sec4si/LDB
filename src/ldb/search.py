"""Minimal intervention search with target and regression scoring."""

from __future__ import annotations

import fnmatch
import hashlib
import itertools
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from ldb.causal import CausalRuntime
from ldb.interventions import apply_tensor_operation
from ldb.models import (
    HOOK_SCHEMA_VERSION,
    EvidenceLevel,
    FrozenModel,
    HookPoint,
    Intervention,
    InterventionKind,
    NeuralPatch,
    TensorAxis,
)
from ldb.offline import install_interventions, remove_interventions


class BehaviorCase(FrozenModel):
    case_id: str
    prompt: str
    target_token_id: int | None = Field(default=None, ge=0)
    target_text: str | None = None
    category: str = "unspecified"
    weight: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def exactly_one_target(self) -> BehaviorCase:
        if (self.target_token_id is None) == (self.target_text is None):
            raise ValueError("A case requires exactly one of target_token_id or target_text")
        if self.target_text == "":
            raise ValueError("target_text must not be empty")
        return self


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
    category: str = "unspecified"
    baseline_log_probability: float
    patched_log_probability: float
    delta: float


class SearchTrial(FrozenModel):
    hook: str
    scale: float
    token: int | None = None
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


class PatchCandidate(FrozenModel):
    candidate_id: str
    intervention: Intervention
    complexity: float = Field(default=1.0, ge=0.0)


class CandidateSet(FrozenModel):
    schema_version: Literal["ldb.candidates/v1alpha1"] = "ldb.candidates/v1alpha1"
    candidates: tuple[PatchCandidate, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_candidate_ids(self) -> CandidateSet:
        identities = [candidate.candidate_id for candidate in self.candidates]
        if len(identities) != len(set(identities)):
            raise ValueError("Candidate IDs must be unique")
        return self


class CombinationTrial(FrozenModel):
    candidate_ids: tuple[str, ...]
    interventions: tuple[Intervention, ...]
    target_gain: float
    related_gain: float | None = None
    regression_loss: float
    complexity: float
    objective: float
    cases: tuple[CaseScore, ...]


class CombinationSearchReport(FrozenModel):
    schema_version: Literal["ldb.combination-search/v1alpha1"] = "ldb.combination-search/v1alpha1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_id: str
    model_revision: str | None = None
    backend: str
    backend_version: str
    tokenizer_hash: str
    hook_schema_version: str = HOOK_SCHEMA_VERSION
    regression_weight: float
    complexity_weight: float
    max_interventions: int = Field(gt=0)
    trials: tuple[CombinationTrial, ...]


def _adapter_for_backend(backend: str) -> str:
    return {
        "transformerlens": "ldb.adapters.transformer_lens",
        "transformers": "ldb.adapters.transformers",
    }.get(backend, f"ldb.adapters.{backend}")


def load_suite(path: Path) -> EvaluationSuite:
    return EvaluationSuite.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_candidate_set(path: Path) -> CandidateSet:
    return CandidateSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def search_scale_patches(
    runtime: CausalRuntime,
    suite: EvaluationSuite,
    *,
    hook_globs: Sequence[str],
    scales: Sequence[float],
    token_index: int | None = None,
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
        case.case_id: score_case(runtime, case) for _, cases in grouped_cases for case in cases
    }
    trials: list[SearchTrial] = []
    for hook in hooks:
        for scale in scale_candidates:

            def apply_scale(selected_hook: HookPoint, tensor: Any, factor: float = scale) -> Any:
                return apply_tensor_operation(
                    tensor,
                    InterventionKind.SCALE,
                    value=factor,
                    token=token_index,
                    token_axis=selected_hook.axis(TensorAxis.TOKEN, TensorAxis.QUERY_TOKEN),
                )

            handle = runtime.install_intervention(hook, apply_scale)
            try:
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
                    token=token_index,
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
        f"{report.model_id}\0{report.model_revision}\0{best.hook}\0{best.scale}\0{best.token}".encode()
    ).hexdigest()[:12]
    if report.model_revision is None:
        raise ValueError("Cannot export a patch without a resolved model revision")
    return NeuralPatch(
        patch_id=f"ldb-{identity}",
        description=f"Scale {best.hook} by {best.scale:.6g}.",
        model_id=report.model_id,
        model_revision=report.model_revision,
        tokenizer_hash=report.tokenizer_hash,
        adapter=_adapter_for_backend(report.backend),
        interventions=(
            Intervention(
                hook=best.hook,
                operation=InterventionKind.SCALE,
                value=best.scale,
                token=best.token,
                request_scoped=True,
            ),
        ),
        evidence=EvidenceLevel.CAUSALLY_TESTED,
    )


def search_candidate_combinations(
    runtime: CausalRuntime,
    suite: EvaluationSuite,
    candidate_set: CandidateSet,
    *,
    resource_directory: Path,
    max_interventions: int = 2,
    regression_weight: float = 1.0,
    complexity_weight: float = 0.01,
    max_trials: int = 10_000,
) -> CombinationSearchReport:
    if max_interventions <= 0:
        raise ValueError("max_interventions must be positive")
    if regression_weight < 0.0 or complexity_weight < 0.0:
        raise ValueError("Search weights must be non-negative")
    candidate_count = len(candidate_set.candidates)
    depth = min(max_interventions, candidate_count)
    trial_count = 1 + sum(math.comb(candidate_count, size) for size in range(1, depth + 1))
    if trial_count > max_trials:
        raise ValueError(
            f"Candidate search would require {trial_count} trials; limit is {max_trials}"
        )

    grouped_cases = _grouped_cases(suite)
    baselines = {
        case.case_id: score_case(runtime, case) for _, cases in grouped_cases for case in cases
    }
    baseline_scores = tuple(
        CaseScore(
            case_id=case.case_id,
            group=group,
            category=case.category,
            baseline_log_probability=baselines[case.case_id],
            patched_log_probability=baselines[case.case_id],
            delta=0.0,
        )
        for group, cases in grouped_cases
        for case in cases
    )
    trials = [
        CombinationTrial(
            candidate_ids=(),
            interventions=(),
            target_gain=0.0,
            related_gain=0.0 if suite.related else None,
            regression_loss=0.0,
            complexity=0.0,
            objective=0.0,
            cases=baseline_scores,
        )
    ]

    for size in range(1, depth + 1):
        for combination in itertools.combinations(candidate_set.candidates, size):
            interventions = tuple(candidate.intervention for candidate in combination)
            handles = install_interventions(
                runtime, interventions, resource_directory=resource_directory
            )
            try:
                case_scores = _score_cases(runtime, grouped_cases, baselines)
            finally:
                remove_interventions(handles)
            target_gain = _weighted_mean(case_scores, suite.target, group="target", loss=False)
            related_gain = (
                _weighted_mean(case_scores, suite.related, group="related", loss=False)
                if suite.related
                else None
            )
            regression_loss = _weighted_mean(
                case_scores, suite.regression, group="regression", loss=True
            )
            complexity = sum(candidate.complexity for candidate in combination)
            objective = (
                target_gain - regression_weight * regression_loss - complexity_weight * complexity
            )
            trials.append(
                CombinationTrial(
                    candidate_ids=tuple(candidate.candidate_id for candidate in combination),
                    interventions=interventions,
                    target_gain=target_gain,
                    related_gain=related_gain,
                    regression_loss=regression_loss,
                    complexity=complexity,
                    objective=objective,
                    cases=case_scores,
                )
            )

    trials.sort(key=lambda trial: (trial.objective, -trial.complexity), reverse=True)
    return CombinationSearchReport(
        model_id=runtime.model_id,
        model_revision=runtime.revision,
        backend=runtime.backend,
        backend_version=runtime.backend_version,
        tokenizer_hash=runtime.tokenizer_hash,
        regression_weight=regression_weight,
        complexity_weight=complexity_weight,
        max_interventions=max_interventions,
        trials=tuple(trials),
    )


def patch_from_combination_search(report: CombinationSearchReport) -> NeuralPatch:
    if not report.trials or not report.trials[0].interventions:
        raise ValueError("No tested intervention combination outperformed the no-op baseline")
    best = report.trials[0]
    if best.objective <= 0.0:
        raise ValueError("No tested intervention combination outperformed the no-op baseline")
    if report.model_revision is None:
        raise ValueError("Cannot export a patch without a resolved model revision")
    serialized = "\0".join(intervention.model_dump_json() for intervention in best.interventions)
    identity = hashlib.sha256(
        f"{report.model_id}\0{report.model_revision}\0{serialized}".encode()
    ).hexdigest()[:12]
    return NeuralPatch(
        patch_id=f"ldb-{identity}",
        description="Combine candidates: " + ", ".join(best.candidate_ids) + ".",
        model_id=report.model_id,
        model_revision=report.model_revision,
        tokenizer_hash=report.tokenizer_hash,
        adapter=_adapter_for_backend(report.backend),
        interventions=best.interventions,
        evidence=EvidenceLevel.CAUSALLY_TESTED,
    )


def _grouped_cases(
    suite: EvaluationSuite,
) -> tuple[tuple[Literal["target", "related", "regression"], tuple[BehaviorCase, ...]], ...]:
    return (
        ("target", suite.target),
        ("related", suite.related),
        ("regression", suite.regression),
    )


def _score_cases(
    runtime: CausalRuntime,
    grouped_cases: tuple[
        tuple[Literal["target", "related", "regression"], tuple[BehaviorCase, ...]], ...
    ],
    baselines: dict[str, float],
) -> tuple[CaseScore, ...]:
    return tuple(
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


def score_case(runtime: CausalRuntime, case: BehaviorCase) -> float:
    if case.target_token_id is not None:
        return runtime.next_token_log_probability(case.prompt, case.target_token_id)
    assert case.target_text is not None
    return runtime.continuation_log_probability(case.prompt, case.target_text)
