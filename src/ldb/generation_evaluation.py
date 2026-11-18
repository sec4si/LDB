"""Explicit behavioral checks on free generation, separate from log-probability suites."""

from __future__ import annotations

import fnmatch
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import yaml
from pydantic import Field, model_validator

from ldb.causal import CausalRuntime
from ldb.generation import GenerationInput, GenerationResult, GenerationSettings
from ldb.models import HOOK_SCHEMA_VERSION, FrozenModel, NeuralPatch
from ldb.offline import install_interventions, remove_interventions
from ldb.sae import sha256_file

Verdict = Literal["pass", "fail", "inconclusive"]
Group = Literal["attack", "benign"]


class OutputRule(FrozenModel):
    """A literal, case-sensitive check; no semantic or safety-judge inference."""

    kind: Literal["exact", "contains", "not_contains"]
    value: str = Field(min_length=1)


class GenerationCase(FrozenModel):
    case_id: str = Field(min_length=1)
    group: Group
    category: str = Field(default="unspecified", min_length=1)
    description: str = ""
    input: GenerationInput
    rules: tuple[OutputRule, ...] = Field(min_length=1)


class GenerationSuite(FrozenModel):
    schema_version: Literal["ldb.generation-suite/v1alpha1"] = "ldb.generation-suite/v1alpha1"
    description: str = ""
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    cases: tuple[GenerationCase, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_cases(self) -> GenerationSuite:
        identifiers = [case.case_id for case in self.cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Case IDs must be unique")
        if {case.group for case in self.cases} != {"attack", "benign"}:
            raise ValueError("A generation suite requires both attack and benign cases")
        return self


class RuleResult(FrozenModel):
    rule: OutputRule
    verdict: Verdict


class GenerationObservation(FrozenModel):
    verdict: Verdict
    generation: GenerationResult | None = None
    rules: tuple[RuleResult, ...] = ()
    error: str | None = None


class GenerationCaseResult(FrozenModel):
    case_id: str
    group: Group
    category: str
    baseline: GenerationObservation
    patched: GenerationObservation | None = None
    comparison_error: str | None = None


class OutcomeCounts(FrozenModel):
    total: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    inconclusive: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)
    failure_rate: float = Field(ge=0, le=1)
    inconclusive_rate: float = Field(ge=0, le=1)


class PairedChanges(FrozenModel):
    fixed: int = 0
    regressed: int = 0
    unresolved: int = 0


class GenerationEvaluationReport(FrozenModel):
    schema_version: Literal["ldb.generation-evaluation/v1alpha1"] = (
        "ldb.generation-evaluation/v1alpha1"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_id: str
    model_revision: str | None
    tokenizer_hash: str
    backend: str
    backend_version: str
    adapter_version: str
    device: str
    hook_schema_version: str = HOOK_SCHEMA_VERSION
    suite_sha256: str
    suite: GenerationSuite
    patch: NeuralPatch | None
    patch_resource_sha256: dict[str, str]
    # Group then category then arm; metrics always use all cases in their group.
    groups: dict[str, dict[str, OutcomeCounts]]
    categories: dict[str, dict[str, dict[str, OutcomeCounts]]]
    paired_changes: dict[str, PairedChanges]
    cases: tuple[GenerationCaseResult, ...]
    passed: bool
    limitations: tuple[str, ...] = (
        "Literal output rules only; no semantic safety judge or tool execution.",
        "One greedy generation per arm; no confidence interval or stochastic robustness claim.",
        "Length-limited or unknown stops cannot pass; witnessed forbidden strings still fail.",
        "Patch hooks remain active for every matching invocation throughout generation.",
        "No model weight hash or GPU determinism guarantee; pin and archive model artifacts.",
    )


class GenerationRuntime(CausalRuntime, Protocol):
    adapter_version: str
    device: str

    def generate_result(
        self, request: GenerationInput, *, settings: GenerationSettings
    ) -> GenerationResult: ...


def load_generation_suite(path: Path) -> GenerationSuite:
    return GenerationSuite.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def evaluate_rules(
    result: GenerationResult, rules: tuple[OutputRule, ...]
) -> tuple[Verdict, tuple[RuleResult, ...]]:
    """Do not promote a truncated answer, error, or empty output into a success."""

    outcomes: list[RuleResult] = []
    text = result.output_text
    complete = result.finish_reason == "eos" and bool(text.strip())
    for rule in rules:
        witnessed_violation = rule.kind == "not_contains" and (
            rule.value in text or rule.value in result.raw_content_text
        )
        if witnessed_violation:
            verdict: Verdict = "fail"
        elif not complete:
            verdict = "inconclusive"
        else:
            matches = {
                "exact": text == rule.value,
                "contains": rule.value in text,
                "not_contains": rule.value not in text,
            }
            verdict = "pass" if matches[rule.kind] else "fail"
        outcomes.append(RuleResult(rule=rule, verdict=verdict))
    verdicts = {outcome.verdict for outcome in outcomes}
    overall: Verdict = (
        "fail" if "fail" in verdicts else "inconclusive" if "inconclusive" in verdicts else "pass"
    )
    return overall, tuple(outcomes)


def _observe(
    runtime: GenerationRuntime, case: GenerationCase, settings: GenerationSettings
) -> GenerationObservation:
    try:
        result = runtime.generate_result(case.input, settings=settings)
    except Exception as exc:
        # Keep failed cases in the denominator and preserve a bounded diagnostic.
        return GenerationObservation(
            verdict="inconclusive", error=f"{type(exc).__name__}: {exc}"[:2000]
        )
    verdict, rules = evaluate_rules(result, case.rules)
    return GenerationObservation(verdict=verdict, generation=result, rules=rules)


def _counts(observations: list[GenerationObservation]) -> OutcomeCounts:
    total = len(observations)
    passed = sum(item.verdict == "pass" for item in observations)
    failed = sum(item.verdict == "fail" for item in observations)
    inconclusive = total - passed - failed
    return OutcomeCounts(
        total=total,
        passed=passed,
        failed=failed,
        inconclusive=inconclusive,
        pass_rate=passed / total if total else 0,
        failure_rate=failed / total if total else 0,
        inconclusive_rate=inconclusive / total if total else 0,
    )


def _arms(cases: list[GenerationCaseResult]) -> dict[str, OutcomeCounts]:
    arms = {"baseline": _counts([case.baseline for case in cases])}
    patched = [case.patched for case in cases if case.patched is not None]
    if patched:
        arms["patched"] = _counts(patched)
    return arms


def _validate_patch(runtime: GenerationRuntime, patch: NeuralPatch) -> None:
    if patch.model_id != runtime.model_id:
        raise ValueError("Patch model ID does not match the runtime")
    if runtime.revision is None or patch.model_revision != runtime.revision:
        raise ValueError("Patch requires a known, matching runtime revision")
    if patch.tokenizer_hash != runtime.tokenizer_hash:
        raise ValueError("Patch tokenizer hash does not match the runtime")
    if patch.adapter != f"ldb.adapters.{runtime.backend}":
        raise ValueError("Patch adapter does not match the runtime")
    if patch.hook_schema_version != HOOK_SCHEMA_VERSION:
        raise ValueError("Patch hook schema does not match the runtime")
    if any(operation.token is not None for operation in patch.interventions):
        raise ValueError(
            "Token-scoped patches are not supported by generation evaluation: "
            "prefill/decode positions have different semantics"
        )
    hooks = runtime.hook_points()
    for operation in patch.interventions:
        if not any(fnmatch.fnmatchcase(hook.name, operation.hook) for hook in hooks):
            raise ValueError(f"Patch hook did not match this runtime: {operation.hook}")


def _resource_hashes(patch: NeuralPatch, directory: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for operation in patch.interventions:
        for resource in (operation.tensor, operation.dictionary):
            if resource is not None and resource not in hashes:
                path = Path(resource)
                hashes[resource] = sha256_file(path if path.is_absolute() else directory / path)
    return hashes


def _comparison_error(
    baseline: GenerationObservation, patched: GenerationObservation | None
) -> str | None:
    if baseline.generation is None or patched is None or patched.generation is None:
        return None  # Missing observations already carry an inconclusive verdict.
    first, second = baseline.generation, patched.generation
    if (
        first.prompt_token_ids != second.prompt_token_ids
        or first.rendered_prompt != second.rendered_prompt
        or first.chat_template_sha256 != second.chat_template_sha256
    ):
        return "Baseline and patched inputs differ; no paired attribution is valid"
    if (
        first.generation_config != second.generation_config
        or first.generation_config_scope != second.generation_config_scope
    ):
        return "Baseline and patched generation settings differ; no paired attribution is valid"
    return None


def evaluate_generation(
    runtime: GenerationRuntime,
    suite: GenerationSuite,
    *,
    patch: NeuralPatch | None = None,
    patch_directory: Path | None = None,
) -> GenerationEvaluationReport:
    """Compare independent full generations; this never rewrites patch verification scores."""

    resource_hashes: dict[str, str] = {}
    if patch is not None:
        _validate_patch(runtime, patch)
        if patch_directory is None:
            raise ValueError("patch_directory is required when evaluating a patch")
        resource_hashes = _resource_hashes(patch, patch_directory)
    baselines = [_observe(runtime, case, suite.generation) for case in suite.cases]
    patched: list[GenerationObservation] = []
    if patch is not None:
        assert patch_directory is not None
        handles = []
        try:
            handles = install_interventions(
                runtime, patch.interventions, resource_directory=patch_directory
            )
            if _resource_hashes(patch, patch_directory) != resource_hashes:
                raise ValueError("Patch resources changed during evaluation")
            patched = [_observe(runtime, case, suite.generation) for case in suite.cases]
            if _resource_hashes(patch, patch_directory) != resource_hashes:
                raise ValueError("Patch resources changed during evaluation")
        finally:
            remove_interventions(handles)
    cases = tuple(
        GenerationCaseResult(
            case_id=case.case_id,
            group=case.group,
            category=case.category,
            baseline=baselines[index],
            patched=patched[index] if patch is not None else None,
            comparison_error=_comparison_error(
                baselines[index], patched[index] if patch is not None else None
            ),
        )
        for index, case in enumerate(suite.cases)
    )
    groups = {}
    categories = {}
    changes = {}
    for group in ("attack", "benign"):
        selected = [case for case in cases if case.group == group]
        groups[group] = _arms(selected)
        categories[group] = {
            category: _arms([case for case in selected if case.category == category])
            for category in sorted({case.category for case in selected})
        }
        if patch is not None:
            pairs = [
                (case.baseline, case.patched)
                for case in selected
                if case.patched is not None and case.comparison_error is None
            ]
            changes[group] = PairedChanges(
                fixed=sum(a.verdict == "fail" and b.verdict == "pass" for a, b in pairs),
                regressed=sum(a.verdict == "pass" and b.verdict == "fail" for a, b in pairs),
                unresolved=(
                    sum("inconclusive" in (a.verdict, b.verdict) for a, b in pairs)
                    + sum(case.comparison_error is not None for case in selected)
                ),
            )
    evaluated = patched if patch is not None else baselines
    return GenerationEvaluationReport(
        model_id=runtime.model_id,
        model_revision=runtime.revision,
        tokenizer_hash=runtime.tokenizer_hash,
        backend=runtime.backend,
        backend_version=runtime.backend_version,
        adapter_version=runtime.adapter_version,
        device=runtime.device,
        suite_sha256=hashlib.sha256(suite.model_dump_json().encode("utf-8")).hexdigest(),
        suite=suite,
        patch=patch,
        patch_resource_sha256=resource_hashes,
        groups=groups,
        categories=categories,
        paired_changes=changes,
        cases=cases,
        passed=(
            all(item.verdict == "pass" for item in evaluated)
            and all(case.comparison_error is None for case in cases)
        ),
    )
