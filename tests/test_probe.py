import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ldb.models import HookPhase, HookPoint, TensorAxis
from ldb.probe import ProbeExample, load_examples, train_probes

torch = pytest.importorskip("torch")


class Runtime:
    model_id = "model"
    backend = "test"
    backend_version = "1"
    revision = "commit"

    def __init__(self) -> None:
        self.hook = HookPoint(
            name="block.0.residual.post",
            module_path="layers.0",
            phase=HookPhase.POST,
            axes=(TensorAxis.BATCH, TensorAxis.TOKEN, TensorAxis.CHANNEL),
        )
        self.callbacks: dict[str, Any] = {}

    def hook_points(self) -> tuple[HookPoint, ...]:
        return (self.hook,)

    def install_probe(self, hook: HookPoint, callback: Any) -> Any:
        self.callbacks[hook.name] = callback
        return SimpleNamespace(remove=lambda: self.callbacks.pop(hook.name, None))

    def next_token_log_probability(self, prompt: str, target_token_id: int) -> float:
        del target_token_id
        generator = torch.Generator().manual_seed(sum(map(ord, prompt)))
        noise = torch.randn(4, generator=generator)
        signal = 3.0 if prompt.endswith("cat") else -3.0
        last = torch.cat([torch.tensor([signal]), noise])
        self.callbacks[self.hook.name](self.hook, torch.stack([torch.zeros(5), last])[None])
        return 0.0


def _examples() -> list[ProbeExample]:
    return [
        ProbeExample(text=f"sentence {index} {animal}", label=animal)
        for index in range(20)
        for animal in ("cat", "dog")
    ]


def test_probe_reads_a_linearly_decodable_label_above_the_control() -> None:
    report = train_probes(Runtime(), _examples(), controls=3, seed=1)

    layer = report.layers[0]
    assert report.labels == ("cat", "dog")
    assert report.train_examples + report.test_examples == 40
    assert layer.accuracy == 1.0
    assert layer.control_accuracy < 0.9
    assert layer.selectivity == pytest.approx(layer.accuracy - layer.control_accuracy)


def test_probe_requires_two_labels() -> None:
    examples = [ProbeExample(text="a cat", label="cat"), ProbeExample(text="b cat", label="cat")]

    with pytest.raises(ValueError, match="two distinct labels"):
        train_probes(Runtime(), examples)


def test_probe_examples_load_from_json_lines(tmp_path: Path) -> None:
    dataset = tmp_path / "examples.jsonl"
    rows = [{"text": "a cat", "label": "cat"}, {"text": "a dog", "label": "dog"}]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8")

    assert [example.label for example in load_examples(dataset)] == ["cat", "dog"]
