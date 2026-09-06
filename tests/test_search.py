from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ldb.models import HookPhase, HookPoint, Intervention, InterventionKind
from ldb.search import (
    BehaviorCase,
    CandidateSet,
    EvaluationSuite,
    PatchCandidate,
    patch_from_combination_search,
    patch_from_search,
    search_candidate_combinations,
    search_scale_patches,
)

torch = pytest.importorskip("torch")


class Handle:
    def __init__(self, remove: Callable[[], None]) -> None:
        self._remove = remove

    def remove(self) -> None:
        self._remove()


class SearchRuntime:
    model_id = "fake-model"
    revision = "revision"
    backend = "fake"
    backend_version = "1"
    tokenizer_hash = "tokenizer-hash"

    def __init__(self) -> None:
        self.hook = HookPoint(
            name="block.0.residual.post", module_path="block", phase=HookPhase.POST
        )
        self.interventions: list[Callable[[HookPoint, Any], Any]] = []

    def hook_points(self) -> tuple[HookPoint, ...]:
        return (self.hook,)

    def install_probe(self, hook: HookPoint, callback: Callable[[HookPoint, Any], None]) -> Handle:
        raise AssertionError("not used by patch search")

    def install_intervention(
        self, hook: HookPoint, callback: Callable[[HookPoint, Any], Any]
    ) -> Handle:
        self.interventions.append(callback)
        return Handle(lambda: self.interventions.remove(callback))

    def next_token_log_probability(self, prompt: str, target_token_id: int) -> float:
        value = 1.0
        for intervention in self.interventions:
            value = float(intervention(self.hook, value))
        if prompt == "target":
            return -((value - 0.5) ** 2)
        return -((value - 1.0) ** 2)

    def continuation_log_probability(self, prompt: str, continuation: str) -> float:
        assert continuation == " answer"
        return self.next_token_log_probability(prompt, 1)


class TokenSearchRuntime(SearchRuntime):
    def next_token_log_probability(self, prompt: str, target_token_id: int) -> float:
        value = torch.tensor([[[1.0], [1.0]]])
        for intervention in self.interventions:
            value = intervention(self.hook, value)
        scalar = float(value[0, -1, 0])
        if prompt == "target":
            return -((scalar - 0.5) ** 2)
        return -((scalar - 1.0) ** 2)


def test_search_balances_target_gain_regression_and_complexity() -> None:
    suite = EvaluationSuite(
        target=(BehaviorCase(case_id="failure", prompt="target", target_token_id=1),),
        regression=(BehaviorCase(case_id="control", prompt="control", target_token_id=2),),
    )
    report = search_scale_patches(
        SearchRuntime(),
        suite,
        hook_globs=["block.*"],
        scales=[0.5, 0.75, 1.0],
        regression_weight=0.5,
        complexity_weight=0.01,
    )

    assert report.trials[0].scale == 0.75
    assert report.trials[0].target_gain == 0.1875
    assert report.trials[0].regression_loss == 0.0625
    assert report.trials[0].objective == 0.15375
    patch = patch_from_search(report)
    assert patch.model_revision == "revision"
    assert patch.tokenizer_hash == "tokenizer-hash"
    assert patch.interventions[0].value == 0.75


def test_suite_rejects_duplicate_case_ids() -> None:
    duplicate = BehaviorCase(case_id="same", prompt="prompt", target_token_id=1)
    with pytest.raises(ValidationError, match="Case IDs must be unique"):
        EvaluationSuite(target=(duplicate,), regression=(duplicate,))


def test_case_accepts_exactly_one_token_or_text_target() -> None:
    text_case = BehaviorCase(case_id="text", prompt="target", target_text=" answer")
    suite = EvaluationSuite(target=(text_case,))
    report = search_scale_patches(SearchRuntime(), suite, hook_globs=["block.*"], scales=[0.75])
    assert report.trials[0].target_gain > 0

    with pytest.raises(ValidationError, match="exactly one"):
        BehaviorCase(case_id="missing", prompt="prompt")
    with pytest.raises(ValidationError, match="exactly one"):
        BehaviorCase(case_id="both", prompt="prompt", target_token_id=1, target_text=" answer")


def test_search_always_compares_with_noop() -> None:
    suite = EvaluationSuite(
        target=(BehaviorCase(case_id="failure", prompt="target", target_token_id=1),)
    )
    report = search_scale_patches(SearchRuntime(), suite, hook_globs=["block.*"], scales=[1.5])
    assert report.trials[0].scale == 1.0
    with pytest.raises(ValueError, match="no-op baseline"):
        patch_from_search(report)


def test_search_exports_token_scoped_scale() -> None:
    suite = EvaluationSuite(
        target=(BehaviorCase(case_id="failure", prompt="target", target_token_id=1),)
    )
    report = search_scale_patches(
        TokenSearchRuntime(),
        suite,
        hook_globs=["block.*"],
        scales=[0.5],
        token_index=-1,
        complexity_weight=0.0,
    )

    assert report.trials[0].token == -1
    assert patch_from_search(report).interventions[0].token == -1


def test_combination_search_finds_multi_intervention_patch(tmp_path: Path) -> None:
    suite = EvaluationSuite(
        target=(BehaviorCase(case_id="failure", prompt="target", target_token_id=1),)
    )
    candidates = CandidateSet(
        candidates=tuple(
            PatchCandidate(
                candidate_id=f"scale-{index}",
                intervention=Intervention(
                    hook="block.0.residual.post",
                    operation=InterventionKind.SCALE,
                    value=0.75,
                ),
                complexity=0.0,
            )
            for index in range(2)
        )
    )
    report = search_candidate_combinations(
        SearchRuntime(),
        suite,
        candidates,
        resource_directory=tmp_path,
        max_interventions=2,
        regression_weight=0.0,
        complexity_weight=0.0,
    )
    assert report.trials[0].candidate_ids == ("scale-0", "scale-1")
    assert len(patch_from_combination_search(report).interventions) == 2


def test_combination_search_enforces_trial_limit(tmp_path: Path) -> None:
    suite = EvaluationSuite(
        target=(BehaviorCase(case_id="failure", prompt="target", target_token_id=1),)
    )
    candidates = CandidateSet(
        candidates=tuple(
            PatchCandidate(
                candidate_id=f"scale-{index}",
                intervention=Intervention(
                    hook="block.0.residual.post",
                    operation=InterventionKind.SCALE,
                    value=0.75,
                ),
            )
            for index in range(4)
        )
    )
    with pytest.raises(ValueError, match="limit is 2"):
        search_candidate_combinations(
            SearchRuntime(),
            suite,
            candidates,
            resource_directory=tmp_path,
            max_interventions=2,
            max_trials=2,
        )
