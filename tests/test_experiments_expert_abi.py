"""Intervention, finite-capacity interface and document-mask regression checks."""

import pytest
import torch
from transformers import MixtralConfig, MixtralForCausalLM

from ldb.experiments.expert_abi.common import aligned_layers, summarize
from ldb.experiments.expert_abi.evaluate import evaluate
from ldb.experiments.expert_abi.interfaces import cost, fit, load_maps, mapping, save_maps
from ldb.experiments.expert_abi.model import body, replace


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    config = MixtralConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_local_experts=4,
        num_experts_per_tok=2,
        head_dim=8,
        attn_implementation="eager",
        experts_implementation="eager",
    )
    return MixtralForCausalLM(config).eval()


def test_routed_replacement_changes_output_and_restores_on_error(model):
    tokens = torch.tensor([[1, 3, 4, 5]])
    bank = model.model.layers[0].mlp.experts
    original = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        baseline = model(tokens, use_cache=False).logits
        replacements = [(0, e, lambda x: torch.ones_like(x)) for e in range(4)]
        with pytest.raises(RuntimeError, match="intentional"), replace(model, replacements):
            assert not torch.equal(model(tokens, use_cache=False).logits, baseline)
            raise RuntimeError("intentional")
        assert torch.equal(model(tokens, use_cache=False).logits, baseline)
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
    assert not bank._forward_hooks


def test_native_body_reconstruction_and_multiple_slots(model):
    torch.manual_seed(11)
    x = torch.randn(7, 16)
    bank = model.model.layers[0].mlp.experts
    ids = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3], [1, 3], [0, 2], [2, 1]])
    w = torch.rand(7, 2).softmax(-1)
    with torch.inference_mode():
        baseline = bank(x, ids, w)
        payloads = [(bank.gate_up_proj[e].clone(), bank.down_proj[e].clone()) for e in range(4)]
        replacement = [(0, e, lambda z, p=p: body(z, p)) for e, p in enumerate(payloads)]
        with replace(model, replacement):
            torch.testing.assert_close(bank(x, ids, w), baseline, rtol=1e-5, atol=1e-6)
        with replace(model, [(0, e, None) for e in range(4)]):
            assert torch.count_nonzero(bank(x, ids, w)) == 0
        assert torch.equal(bank(x, ids, w), baseline)


def test_low_rank_fit_export_and_unseen_samples(tmp_path):
    torch.manual_seed(13)
    x = torch.randn(300, 12)
    left, right = torch.randn(12, 2) * 0.1, torch.randn(2, 12) * 0.1
    bias = torch.randn(12) * 0.01
    y = x + x @ left @ right + bias
    params = fit(x[:200], y[:200], "rank2")
    predicted = mapping(x[200:], params)
    assert float((predicted - y[200:]).square().mean()) < 1e-7
    path = tmp_path / "interface.safetensors"
    save_maps(path, {"H": params, "T": params})
    loaded = load_maps(path)
    assert all(torch.equal(value, loaded["H"][key]) for key, value in params.items())
    # Contiguous file layout can change the BLAS path, but cannot change stored values.
    torch.testing.assert_close(predicted, mapping(x[200:], loaded["H"]), rtol=1e-6, atol=1e-7)
    assert sum(v.numel() for k, v in params.items() if k != "identity") == 2 * 12 * 2 + 12


def test_zero_cost_scale_and_masked_metrics(model, tmp_path):
    x = torch.randn(20, 8)
    w = torch.rand(20)
    assert cost(x, [torch.zeros_like(x)], w) == [1.0]
    assert cost(x, [x], w) == [0.0]
    batch = dict(
        input_ids=torch.tensor([[1, 2, 3, 4], [1, 2, 0, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
    )
    with torch.inference_mode():
        original = evaluate(model, [batch], tmp_path / "baseline", create=True)
        replay = evaluate(model, [batch], tmp_path / "baseline")
        evaluate(model, [batch], tmp_path / "baseline", replay=True)
    assert [r["tokens"] for r in original] == [3, 1]
    result = summarize(replay, original)
    assert result["quality_pass"] and result["mean_kl"] == 0 and result["delta_nll"] == 0


def test_depth_folds_are_unique_and_disjoint():
    pairs = aligned_layers(10, 12)
    assert len(set(a for a, _ in pairs)) == len(set(b for _, b in pairs)) == 10
    assert not set(pairs[::2]) & set(pairs[1::2])
    assert not {a for a, _ in pairs[::2]} & {a for a, _ in pairs[1::2]}
    assert not {b for _, b in pairs[::2]} & {b for _, b in pairs[1::2]}
    assert pairs[0] == (0, 0) and pairs[-1] == (9, 11)


def test_invalid_weights_rejected():
    with pytest.raises(ValueError, match="Invalid regression"):
        fit(torch.ones(4, 3), torch.ones(4, 3), "rank1", torch.zeros(4))


def test_layer_fit_svd_fallback_preserves_objective(monkeypatch):
    from ldb.experiments.expert_abi.layer_interfaces import fit as layer_fit

    torch.manual_seed(23)
    x = torch.randn(150, 12)
    y = x + x @ (torch.randn(12, 2) * 0.1) @ (torch.randn(2, 12) * 0.1)
    expected = mapping(x, fit(x, y, "rank2"))

    def fail(*args, **kwargs):
        raise torch._C._LinAlgError("SVD convergence failure")

    monkeypatch.setattr(torch.linalg, "svd", fail)
    actual = mapping(x, layer_fit(x, y, "rank2"))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
