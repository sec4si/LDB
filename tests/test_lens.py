from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ldb.lens import TunedLens, logit_lens
from ldb.models import HookPhase, HookPoint, TensorAxis

torch = pytest.importorskip("torch")
save_file = pytest.importorskip("safetensors.torch").save_file

RESIDUAL_AXES = (TensorAxis.BATCH, TensorAxis.TOKEN, TensorAxis.CHANNEL)


class Runtime:
    model_id = "model"
    backend = "test"
    backend_version = "1"
    revision = "commit"

    def __init__(self, *, model_offset: float = 0.0) -> None:
        self.hooks = tuple(
            HookPoint(
                name=f"block.{layer}.residual.post",
                module_path=f"layers.{layer}",
                phase=HookPhase.POST,
                axes=RESIDUAL_AXES,
            )
            for layer in (0, 1)
        )
        # Layer 0 favours token 1, layer 1 favours token 2 at the last position.
        self.residuals = {
            0: torch.tensor([[[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]]),
            1: torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 4.0]]]),
        }
        self.model_offset = model_offset
        self.callbacks: dict[str, Any] = {}

    def hook_points(self) -> tuple[HookPoint, ...]:
        return self.hooks

    def install_probe(self, hook: HookPoint, callback: Any) -> Any:
        self.callbacks[hook.name] = callback
        return SimpleNamespace(remove=lambda: self.callbacks.pop(hook.name, None))

    def next_token_log_probability(self, prompt: str, target_token_id: int) -> float:
        del prompt
        for hook in self.hooks:
            if hook.name in self.callbacks:
                self.callbacks[hook.name](hook, self.residuals[int(hook.name.split(".")[1])])
        final = torch.log_softmax(self.unembed(self.residuals[1][0, -1]), dim=-1)
        return float(final[target_token_id]) + self.model_offset

    def tokenize(self, prompt: str) -> tuple[int, ...]:
        del prompt
        return (7, 8)

    def unembed(self, residual: Any) -> Any:
        return residual

    def decode(self, token_id: int) -> str:
        return f"<{token_id}>"


def test_logit_lens_reads_every_layer_and_checks_the_final_layer() -> None:
    report = logit_lens(Runtime(), prompt="hello", top_k=2, target_token_id=2)

    assert [layer.layer for layer in report.layers] == [0, 1]
    assert report.layers[0].top_tokens[0].token_id == 1
    assert report.layers[1].top_tokens[0].text == "<2>"
    assert report.layers[0].target_rank == 2
    assert report.layers[1].target_rank == 1
    assert report.position == 1
    assert report.final_layer_matches_model is True


def test_logit_lens_flags_an_unembedding_that_disagrees_with_the_model() -> None:
    report = logit_lens(Runtime(model_offset=0.5), prompt="hello")

    assert report.final_layer_matches_model is False


def test_logit_lens_skips_the_final_check_away_from_the_last_position() -> None:
    report = logit_lens(Runtime(), prompt="hello", position=0, layers=[0])

    assert report.position == 0
    assert [layer.layer for layer in report.layers] == [0]
    assert report.final_layer_matches_model is None


def test_logit_lens_rejects_positions_outside_the_prompt() -> None:
    with pytest.raises(ValueError, match="outside"):
        logit_lens(Runtime(), prompt="hello", position=5)


def test_zero_tuned_lens_reproduces_the_logit_lens(tmp_path: Path) -> None:
    path = tmp_path / "lens.safetensors"
    save_file(
        {f"layer.{n}.weight": torch.zeros(3, 3) for n in (0, 1)}
        | {f"layer.{n}.bias": torch.zeros(3) for n in (0, 1)},
        path,
    )

    plain = logit_lens(Runtime(), prompt="hello", target_token_id=2)
    tuned = logit_lens(Runtime(), prompt="hello", target_token_id=2, tuned_lens=TunedLens(path))

    assert [layer.model_dump() for layer in tuned.layers] == [
        layer.model_dump() for layer in plain.layers
    ]
    assert tuned.translated_layers == (0, 1)
    assert tuned.tuned_lens_hash is not None


def test_tuned_lens_translates_only_its_layers_and_keeps_the_raw_final_check(
    tmp_path: Path,
) -> None:
    path = tmp_path / "lens.safetensors"
    save_file(
        {"layer.1.weight": torch.zeros(3, 3), "layer.1.bias": torch.tensor([10.0, 0, 0])}, path
    )

    plain = logit_lens(Runtime(), prompt="hello", target_token_id=2)
    report = logit_lens(Runtime(), prompt="hello", target_token_id=2, tuned_lens=TunedLens(path))

    assert report.translated_layers == (1,)
    assert report.layers[0] == plain.layers[0]
    assert plain.layers[1].top_tokens[0].token_id == 2
    assert report.layers[1].top_tokens[0].token_id == 0
    assert report.final_layer_matches_model is True


def test_tuned_lens_rejects_malformed_translators(tmp_path: Path) -> None:
    save_file({"layer.0.weight": torch.zeros(2, 3), "layer.0.bias": torch.zeros(2)}, tmp_path / "a")
    save_file({"translator": torch.zeros(2)}, tmp_path / "b")

    with pytest.raises(ValueError, match="square weight"):
        TunedLens(tmp_path / "a")
    with pytest.raises(ValueError, match="Unexpected tensor"):
        TunedLens(tmp_path / "b")
