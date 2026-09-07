from types import SimpleNamespace
from typing import Any

from ldb.diagnosis import diagnose_failure
from ldb.models import HookPhase, HookPoint, TensorAxis


class Runtime:
    model_id = "model"
    backend = "test"
    backend_version = "1"
    revision = "commit"
    tokenizer_hash = "tokenizer"

    def __init__(self) -> None:
        self.hook = HookPoint(
            name="block.0.residual.post",
            module_path="block",
            phase=HookPhase.POST,
            axes=(TensorAxis.BATCH, TensorAxis.TOKEN, TensorAxis.CHANNEL),
        )
        self.probes: list[Any] = []
        self.interventions: list[Any] = []

    def hook_points(self) -> tuple[HookPoint, ...]:
        return (self.hook,)

    def install_probe(self, hook: HookPoint, callback: Any) -> Any:
        del hook
        self.probes.append(callback)
        return SimpleNamespace(remove=lambda: self.probes.remove(callback))

    def install_intervention(self, hook: HookPoint, callback: Any) -> Any:
        del hook
        self.interventions.append(callback)
        return SimpleNamespace(remove=lambda: self.interventions.remove(callback))

    def next_token_log_probability(self, prompt: str, target_token_id: int) -> float:
        import torch

        del target_token_id
        value = torch.tensor([[[2.0 if prompt == "clean" else 1.0]]])
        for probe in self.probes:
            probe(self.hook, value)
        for intervention in self.interventions:
            value = intervention(self.hook, value)
        return float(value.item())

    def continuation_log_probability(self, prompt: str, continuation: str) -> float:
        del continuation
        return self.next_token_log_probability(prompt, 0)


def test_diagnosis_localizes_and_exports_best_patch() -> None:
    report = diagnose_failure(
        Runtime(),
        clean_prompt="clean",
        failing_prompt="failure",
        target_token_id=0,
        scales=(1.0, 1.5),
    )

    assert report.selected_hooks == ("block.0.residual.post",)
    assert report.patch is not None
    assert report.patch.interventions[0].value == 1.5
    assert report.search.trials[0].target_gain == 0.5
