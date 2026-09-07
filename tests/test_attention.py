from types import SimpleNamespace
from typing import Any

import pytest

from ldb.attention import analyze_attention
from ldb.models import HookPhase, HookPoint, TensorAxis

torch = pytest.importorskip("torch")


class Runtime:
    model_id = "model"
    backend = "test"
    backend_version = "1"
    revision = "commit"

    def __init__(self) -> None:
        per_head = (
            TensorAxis.BATCH,
            TensorAxis.TOKEN,
            TensorAxis.HEAD,
            TensorAxis.CHANNEL,
        )
        pattern = (
            TensorAxis.BATCH,
            TensorAxis.HEAD,
            TensorAxis.QUERY_TOKEN,
            TensorAxis.KEY_TOKEN,
        )
        self.hooks = tuple(
            HookPoint(
                name=f"block.0.attention.{name}",
                module_path=name,
                phase=HookPhase.POST,
                axes=pattern if name == "pattern" else per_head,
            )
            for name in ("q", "k", "v", "scores", "pattern", "z", "result")
        )
        self.callbacks: dict[str, Any] = {}

    def hook_points(self) -> tuple[HookPoint, ...]:
        return self.hooks

    def install_probe(self, hook: HookPoint, callback: Any) -> Any:
        self.callbacks[hook.name] = callback
        return SimpleNamespace(remove=lambda: self.callbacks.pop(hook.name, None))

    def next_token_log_probability(self, prompt: str, target_token_id: int) -> float:
        del prompt, target_token_id
        per_head = torch.ones(1, 2, 2, 3)
        pattern = torch.tensor([[[[1.0, 0.0], [0.25, 0.75]], [[1.0, 0.0], [0.5, 0.5]]]])
        for hook in self.hooks:
            value = pattern if hook.name.endswith(("pattern", "scores")) else per_head
            self.callbacks[hook.name](hook, value)
        return 0.0

    def tokenize(self, prompt: str) -> tuple[int, ...]:
        del prompt
        return (10, 11)


def test_attention_analysis_reports_every_head() -> None:
    report = analyze_attention(Runtime(), prompt="hello")

    assert report.token_ids == (10, 11)
    assert len(report.heads) == 2
    assert report.heads[0].q_norm == pytest.approx(6**0.5)
    assert report.heads[0].maximum_attention is not None
    assert report.heads[0].maximum_attention.query_token == 0
    assert report.heads[0].maximum_attention.key_token == 0
    assert report.heads[0].maximum_attention.weight == 1.0
    assert report.heads[0].score_mean == pytest.approx(0.5)
