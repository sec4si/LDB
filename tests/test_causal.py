from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ldb.causal import causal_trace
from ldb.models import HookPhase, HookPoint

torch = pytest.importorskip("torch")


class Handle:
    def __init__(self, remove: Callable[[], None]) -> None:
        self._remove = remove

    def remove(self) -> None:
        self._remove()


class FakeRuntime:
    def __init__(self) -> None:
        self.model_id = "fake-model"
        self.backend = "fake"
        self.backend_version = "1"
        self.revision: str | None = "fake-revision"
        self.hooks = (
            HookPoint(name="block.0.residual.post", module_path="a", phase=HookPhase.POST),
            HookPoint(name="block.1.residual.post", module_path="b", phase=HookPhase.POST),
        )
        self.probes: dict[str, list[Callable[[HookPoint, Any], None]]] = {}
        self.interventions: dict[str, list[Callable[[HookPoint, Any], Any]]] = {}

    def hook_points(self) -> tuple[HookPoint, ...]:
        return self.hooks

    def install_probe(self, hook: HookPoint, callback: Callable[[HookPoint, Any], None]) -> Handle:
        callbacks = self.probes.setdefault(hook.name, [])
        callbacks.append(callback)
        return Handle(lambda: callbacks.remove(callback))

    def install_intervention(
        self, hook: HookPoint, callback: Callable[[HookPoint, Any], Any]
    ) -> Handle:
        callbacks = self.interventions.setdefault(hook.name, [])
        callbacks.append(callback)
        return Handle(lambda: callbacks.remove(callback))

    def next_token_log_probability(self, prompt: str, target_token_id: int) -> float:
        assert target_token_id == 7
        clean = prompt == "clean"
        gains = (2.0, 0.5)
        score = -0.5 if clean else -3.0
        for index, hook in enumerate(self.hooks):
            value = torch.tensor([1.0 if clean else 0.0])
            for callback in self.probes.get(hook.name, []):
                callback(hook, value)
            for callback in self.interventions.get(hook.name, []):
                value = callback(hook, value)
            if not clean and value.item() == 1.0:
                score += gains[index]
        return score


def test_causal_trace_ranks_clean_activation_recovery() -> None:
    report = causal_trace(
        FakeRuntime(),
        clean_prompt="clean",
        corrupt_prompt="corrupt",
        target_token_id=7,
    )

    assert report.clean_log_probability == -0.5
    assert report.corrupt_log_probability == -3.0
    assert report.model_id == "fake-model"
    assert [trial.hook for trial in report.trials] == [
        "block.0.residual.post",
        "block.1.residual.post",
    ]
    assert report.trials[0].effect == 2.0
    assert report.trials[0].recovery == pytest.approx(0.8)
