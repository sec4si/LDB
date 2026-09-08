import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from ldb.structural import edit_gpt2


def test_layer_skip_repeat_and_exception_restore() -> None:
    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=16)).eval()
    inputs = {"input_ids": torch.tensor([[1, 2]]), "use_cache": False}
    calls = []
    handle = model.transformer.h[0].mlp.register_forward_hook(lambda *args: calls.append(1))
    original = model.transformer.h[0].forward
    with edit_gpt2(model, layer=0, passes=0):
        model(**inputs)
    assert calls == []
    with (
        pytest.raises(RuntimeError, match="experiment failed"),
        edit_gpt2(model, layer=0, passes=2),
    ):
        model(**inputs)
        assert len(calls) == 2
        raise RuntimeError("experiment failed")
    assert model.transformer.h[0].forward == original
    assert model.config.use_cache is True
    handle.remove()


def test_weight_edit_restores_exactly_and_rejects_cache() -> None:
    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=16)).eval()
    weight = model.transformer.h[0].mlp.c_proj.weight
    original = weight.detach().clone()
    with (
        pytest.raises(ValueError, match="use_cache=False"),
        edit_gpt2(model, parameter="transformer.h.0.mlp.c_proj.weight", scale=0),
    ):
        assert torch.count_nonzero(weight) == 0
        model(input_ids=torch.tensor([[1]]), use_cache=True)
    assert torch.equal(weight, original)
