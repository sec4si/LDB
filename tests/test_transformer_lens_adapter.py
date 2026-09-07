from types import SimpleNamespace
from typing import Any

import pytest
import torch

from ldb.adapters.transformer_lens import TransformerLensAdapter


class FakeHookPoint:
    def __init__(self) -> None:
        self.fwd_hooks: list[Any] = []

    def add_hook(self, user_hook: Any) -> None:
        removable = SimpleNamespace(remove=lambda: None)
        self.fwd_hooks.append(SimpleNamespace(hook=removable, user_hook=user_hook))

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        value = tensor
        for handle in tuple(self.fwd_hooks):
            replacement = handle.user_hook(value, hook=self)
            if replacement is not None:
                value = replacement
        return value


class FakeBackendTokenizer:
    def to_str(self) -> str:
        return '{"model":{"type":"BPE"}}'


class FakeBridge:
    def __init__(self) -> None:
        names = (
            "blocks.0.hook_in",
            "blocks.0.hook_resid_mid",
            "blocks.0.mlp.hook_in",
            "blocks.0.mlp.hook_pre",
            "blocks.0.mlp.hook_post",
            "blocks.0.attn.hook_out",
            "blocks.0.mlp.hook_out",
            "blocks.0.hook_out",
        )
        self.hook_dict = {name: FakeHookPoint() for name in names}
        self.original_model = SimpleNamespace(
            config=SimpleNamespace(_commit_hash="commit", is_encoder_decoder=False)
        )
        self.adapter = SimpleNamespace(supports_generation=True)
        self.tokenizer = SimpleNamespace(
            backend_tokenizer=FakeBackendTokenizer(),
            special_tokens_map={"eos_token": "<eos>"},
            chat_template=None,
        )

    def to_tokens(self, text: str, **_: Any) -> torch.Tensor:
        values = {
            "prompt": [1, 2],
            "prompt continuation": [1, 2, 3],
        }
        return torch.tensor([values[text]])

    def generate(self, prompt: str, **_: Any) -> str:
        return f"{prompt} generated"

    def __call__(self, value: Any, **_: Any) -> torch.Tensor:
        length = 2 if isinstance(value, str) else value.shape[-1]
        logits = torch.zeros((1, length, 5))
        logits[..., 3] = 2.0
        return logits


def _adapter() -> TransformerLensAdapter:
    return TransformerLensAdapter(
        "test-model",
        revision="requested",
        _bridge=FakeBridge(),
        _backend_version="3.8.1",
    )


def test_transformer_lens_adapter_discovers_normalized_hooks() -> None:
    adapter = _adapter()

    assert [hook.name for hook in adapter.hook_points()] == [
        "block.0.residual.pre",
        "block.0.residual.mid",
        "block.0.attention.output",
        "block.0.mlp.input",
        "block.0.mlp.pre",
        "block.0.mlp.activation",
        "block.0.mlp.output",
        "block.0.residual.post",
    ]
    assert adapter.backend == "transformerlens"
    assert adapter.backend_version == "3.8.1"
    assert adapter.revision == "commit"
    assert len(adapter.tokenizer_hash) == 64


def test_transformer_lens_adapter_probes_and_intervenes() -> None:
    adapter = _adapter()
    hook = adapter.hook_points()[0]
    source = adapter.model.hook_dict[hook.module_path]
    observed: list[torch.Tensor] = []
    probe = adapter.install_probe(hook, lambda _, tensor: observed.append(tensor.clone()))
    intervention = adapter.install_intervention(hook, lambda _, tensor: tensor * 2)

    output = source.apply(torch.tensor([1.0, 2.0]))
    assert observed[0].tolist() == [1.0, 2.0]
    assert output.tolist() == [2.0, 4.0]

    probe.remove()
    intervention.remove()
    assert not source.fwd_hooks


def test_transformer_lens_adapter_scores_and_generates() -> None:
    adapter = _adapter()

    assert adapter.tokenize("prompt") == (1, 2)
    assert adapter.generate("prompt", max_new_tokens=1) == "prompt generated"
    assert adapter.next_token_log_probability("prompt", 3) == pytest.approx(-0.432653)
    assert adapter.continuation_log_probability("prompt", " continuation") == pytest.approx(
        -0.432653
    )


def test_transformer_lens_adapter_rejects_encoder_decoder_models() -> None:
    bridge = FakeBridge()
    bridge.original_model.config.is_encoder_decoder = True

    with pytest.raises(ValueError, match="decoder-only"):
        TransformerLensAdapter(
            "test-model",
            _bridge=bridge,
            _backend_version="3.8.1",
        )
