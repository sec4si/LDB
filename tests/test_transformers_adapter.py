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
            "transformer.h.1": SimpleNamespace(),
            "transformer.h.1.attn": SimpleNamespace(),
            "transformer.h.1.attn.c_proj": SimpleNamespace(),
            "transformer.h.1.mlp": SimpleNamespace(),
        }
    )

    hooks = adapter._discover_hooks()

    assert len(hooks) == 10
    assert hooks[0].name == "block.0.residual.pre"
    assert hooks[-1].name == "block.1.attention.heads"


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
