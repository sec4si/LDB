"""Offline Neural Patch verification against target and regression suites."""

from __future__ import annotations

import fnmatch
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from ldb.causal import CausalRuntime
from ldb.interventions import apply_tensor_operation, load_intervention_tensor
from ldb.models import FrozenModel, HookPoint, NeuralPatch
from ldb.runtime import ProbeHandle
from ldb.sae import SAEDictionary
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

    handles: list[ProbeHandle] = []
    try:
        for intervention in patch.interventions:
            matches = tuple(
                hook
                for hook in runtime.hook_points()
                if fnmatch.fnmatchcase(hook.name, intervention.hook)
            )
            if not matches:
                raise ValueError(f"Patch hook did not match this runtime: {intervention.hook}")
            replacement = None
            if intervention.dictionary is not None:
                dictionary_path = Path(intervention.dictionary)
                if not dictionary_path.is_absolute():
                    dictionary_path = patch_directory / dictionary_path
                assert intervention.feature is not None
                replacement = SAEDictionary(dictionary_path).feature(intervention.feature)
            elif intervention.tensor is not None:
                tensor_path = Path(intervention.tensor)
                if not tensor_path.is_absolute():
                    tensor_path = patch_directory / tensor_path
                replacement = load_intervention_tensor(tensor_path)

            def apply(
                _: HookPoint,
                tensor: Any,
                operation: Any = intervention.operation,
                value: float | None = intervention.value,
                loaded: Any = replacement,
                expert: int | None = intervention.expert,
                head: int | None = intervention.head,
                num_heads: int | None = intervention.num_heads,
            ) -> Any:
                return apply_tensor_operation(
                    tensor,
                    operation,
                    value=value,
                    replacement=loaded,
                    expert=expert,
                    head=head,
                    num_heads=num_heads,
                    feature=loaded,
                )

            handles.extend(runtime.install_intervention(hook, apply) for hook in matches)

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
        for handle in reversed(handles):
            handle.remove()

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
