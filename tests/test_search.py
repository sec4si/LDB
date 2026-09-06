from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from ldb.models import HookPhase, HookPoint
from ldb.search import BehaviorCase, EvaluationSuite, patch_from_search, search_scale_patches


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
        self.intervention: Callable[[HookPoint, Any], Any] | None = None

    def hook_points(self) -> tuple[HookPoint, ...]:
        return (self.hook,)

    def install_probe(self, hook: HookPoint, callback: Callable[[HookPoint, Any], None]) -> Handle:
        raise AssertionError("not used by patch search")

    def install_intervention(
        self, hook: HookPoint, callback: Callable[[HookPoint, Any], Any]
    ) -> Handle:
        self.intervention = callback
        return Handle(lambda: setattr(self, "intervention", None))

    def next_token_log_probability(self, prompt: str, target_token_id: int) -> float:
        value = 1.0
        if self.intervention is not None:
            value = float(self.intervention(self.hook, value))
        if prompt == "target":
            return -((value - 0.5) ** 2)
        return -((value - 1.0) ** 2)


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


def test_search_always_compares_with_noop() -> None:
    suite = EvaluationSuite(
        target=(BehaviorCase(case_id="failure", prompt="target", target_token_id=1),)
    )
    report = search_scale_patches(SearchRuntime(), suite, hook_globs=["block.*"], scales=[1.5])
    assert report.trials[0].scale == 1.0
    with pytest.raises(ValueError, match="no-op baseline"):
        patch_from_search(report)
