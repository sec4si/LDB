from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors.torch import save_file

from ldb.cli import _portable_diagnosis_report
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

    def tokenize(self, prompt: str) -> tuple[int, ...]:
        del prompt
        return (0,)


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


def test_diagnosis_searches_active_sae_features(tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.safetensors"
    save_file(
        {"encoder": torch.ones(1, 1), "decoder": torch.ones(1, 1)},
        dictionary,
    )

    report = diagnose_failure(
        Runtime(),
        clean_prompt="clean",
        failing_prompt="failure",
        target_token_id=0,
        scales=(1.0, 1.25),
        sae_hook="block.0.residual.post",
        sae_dictionary=dictionary,
        max_sae_features=1,
        feature_scales=(0.5, 1.5),
    )

    assert report.sae_features[0].feature == 0
    assert report.feature_search is not None
    feature_trials = [trial for trial in report.feature_search.trials if trial.interventions]
    assert len(feature_trials) == 2
    assert feature_trials[0].interventions[0].operation.value == "feature_scale"

    portable = _portable_diagnosis_report(tmp_path / "patch.yaml", report)
    assert portable.patch is not None
    resource = portable.patch.interventions[0].dictionary
    assert resource is not None and not Path(resource).is_absolute()
    assert portable.feature_search is not None
    assert all(
        intervention.dictionary == resource
        for trial in portable.feature_search.trials
        for intervention in trial.interventions
    )
