from types import SimpleNamespace

import pytest

from ldb.adapters.transformers import TransformersAdapter, _replace_first_tensor

torch = pytest.importorskip("torch")


class SizedModule:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


def adapter_with_modules(modules: dict[str, object]) -> TransformersAdapter:
    adapter = object.__new__(TransformersAdapter)
    adapter._module_by_path = modules
    return adapter


def test_discovers_gpt2_hook_points() -> None:
    adapter = adapter_with_modules(
        {
            "transformer.h": SizedModule(2),
            "transformer.h.0": SimpleNamespace(),
            "transformer.h.0.attn": SimpleNamespace(),
            "transformer.h.0.attn.c_proj": SimpleNamespace(),
            "transformer.h.0.mlp": SimpleNamespace(),
            "transformer.h.0.mlp.experts": SizedModule(2),
            "transformer.h.0.mlp.experts.0": SimpleNamespace(),
            "transformer.h.0.mlp.experts.1": SimpleNamespace(),
            "transformer.h.1": SimpleNamespace(),
            "transformer.h.1.attn": SimpleNamespace(),
            "transformer.h.1.attn.c_proj": SimpleNamespace(),
            "transformer.h.1.mlp": SimpleNamespace(),
        }
    )

    hooks = adapter._discover_hooks()

    assert len(hooks) == 12
    assert hooks[0].name == "block.0.residual.pre"
    assert hooks[5].name == "block.0.moe.expert.0.output"
    assert hooks[6].name == "block.0.moe.expert.1.output"


def test_replace_first_tensor_preserves_nested_output() -> None:
    output, replaced = _replace_first_tensor(
        (torch.tensor([[1.0, 2.0]]), ("cache",)), torch.tensor([[3.0, 4.0]])
    )
    assert replaced is True
    assert output[0].tolist() == [[3.0, 4.0]]
    assert output[1] == ("cache",)


def test_replace_first_tensor_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Replacement shape"):
        _replace_first_tensor(torch.zeros(2), torch.zeros(3))


def test_continuation_score_uses_contextual_tokenization() -> None:
    adapter = object.__new__(TransformersAdapter)
    adapter.device = "cpu"

    class Tokenizer:
        def __call__(self, text):
            return {
                "input_ids": {
                    "prompt": [1],
                    "prompt continuation": [1, 2, 3],
                    " continuation": [4],
                }[text]
            }

    class Model:
        def __call__(self, *, input_ids):
            assert input_ids.tolist() == [[1, 2, 3]]
            logits = torch.zeros(1, 3, 5)
            logits[0, 0, 2] = 10
            logits[0, 1, 3] = 10
            return SimpleNamespace(logits=logits)

    adapter.tokenizer = Tokenizer()
    adapter.model = Model()

    score = adapter.continuation_log_probability("prompt", " continuation")
    assert score == pytest.approx(-0.0001816, abs=1e-6)


def test_continuation_score_rejects_unstable_token_boundary() -> None:
    adapter = object.__new__(TransformersAdapter)
    adapter.device = "cpu"

    class Tokenizer:
        def __call__(self, text):
            return {"input_ids": [1, 2] if text == "prompt" else [1, 3]}

    adapter.tokenizer = Tokenizer()

    with pytest.raises(ValueError, match="token boundary"):
        adapter.continuation_log_probability("prompt", "continuation")
