import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from ldb.weight_editing import export_weight_edit, tensor_hash


def tiny():
    torch.manual_seed(5)
    return GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=8, vocab_size=16)).eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_export_tied_weight_reload_and_restore(tmp_path, dtype):
    model = tiny().to(dtype=dtype)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    inputs = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        baseline = model(inputs).logits
    output = tmp_path / "edited"
    report = export_weight_edit(
        model, output, parameter="lm_head.weight", operation="scale", value=0
    )
    assert set(report["tied_aliases"]) == {"lm_head.weight", "transformer.wte.weight"}
    assert report["before_hash"] != report["after_hash"]
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    reloaded = AutoModelForCausalLM.from_pretrained(output, dtype="auto").eval()
    assert reloaded.lm_head.weight is reloaded.transformer.wte.weight
    assert torch.count_nonzero(reloaded.lm_head.weight) == 0
    with torch.no_grad():
        assert torch.equal(model(inputs).logits, baseline)
        assert not torch.equal(reloaded(inputs).logits, baseline)
    assert tensor_hash(reloaded.lm_head.weight) == report["after_hash"]
    assert json.loads((output / "ldb-weight-edit.json").read_text()) == report


def test_qwen_slice_replacement_preserves_other_parameters(tmp_path):
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    ).eval()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    name = "model.layers.0.self_attn.o_proj.weight"
    replacement = torch.full((16, 8), 0.25)
    path = tmp_path / "replace.safetensors"
    save_file({"weight": replacement}, str(path))
    output = tmp_path / "qwen"
    export_weight_edit(
        model,
        output,
        parameter=name,
        operation="replace",
        tensor=path,
        axis=1,
        start=0,
        stop=8,
        max_shard_size="1KB",
    )
    new = AutoModelForCausalLM.from_pretrained(output).state_dict()
    for key, value in original.items():
        expected = value.clone()
        if key == name:
            expected[:, :8] = replacement
        assert torch.equal(new[key], expected), key
        assert torch.equal(model.state_dict()[key], value), key


def test_export_failure_and_existing_output_leave_model_untouched(tmp_path, monkeypatch):
    model = tiny()
    before = model.transformer.wte.weight.detach().clone()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(model, "save_pretrained", fail)
    with pytest.raises(OSError, match="disk full"):
        export_weight_edit(
            model, tmp_path / "failed", parameter="lm_head.weight", operation="add", value=1
        )
    assert not (tmp_path / "failed").exists()
    assert torch.equal(model.transformer.wte.weight, before)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_text("original")
    with pytest.raises(FileExistsError):
        export_weight_edit(model, existing, parameter="lm_head.weight", operation="scale", value=0)
    assert (existing / "keep").read_text() == "original"
    assert torch.equal(model.transformer.wte.weight, before)


@pytest.mark.parametrize(
    "options",
    [
        {"parameter": "missing", "operation": "scale", "value": 1},
        {"operation": "scale", "value": float("nan")},
        {"operation": "replace"},
        {"operation": "scale", "value": 1, "axis": 2, "start": 0, "stop": 2},
        {"operation": "add", "value": 1, "axis": 0, "start": 3, "stop": 2},
        {"operation": "add", "value": 1, "start": 1},
    ],
)
def test_invalid_edits_do_not_publish(tmp_path, options):
    model = tiny()
    config = {"parameter": "lm_head.weight", **options}
    with pytest.raises(ValueError):
        export_weight_edit(model, tmp_path / "result", **config)
    assert not (tmp_path / "result").exists()


def test_overflow_and_shared_view_rejected(tmp_path):
    model = tiny().half()
    with pytest.raises(ValueError, match="NaN or infinity"):
        export_weight_edit(
            model, tmp_path / "overflow", parameter="lm_head.weight", operation="add", value=1e10
        )
    model.register_parameter(
        "view_weight", torch.nn.Parameter(model.transformer.wte.weight.view(-1))
    )
    with pytest.raises(ValueError, match="sharing storage"):
        export_weight_edit(
            model, tmp_path / "shared", parameter="lm_head.weight", operation="scale", value=0
        )
