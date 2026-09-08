"""CPU contracts matching audited serving forward signatures, not GPU acceptance."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from ldb.client import AgentClient
from ldb.context import BatchContext, RequestSpan, reset_batch_context, set_batch_context
from ldb.hooks import close_agents, install_model_runtime
from ldb.serving_hooks import DEFERRED_RESIDUAL_CLASSES


class DeferredBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(2, 2, bias=False)
        self.mlp = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.self_attn.weight.copy_(2 * torch.eye(2))
            self.mlp.weight.copy_(3 * torch.eye(2))

    def forward(self, positions, hidden_states, residual=None, forward_batch=None):
        assert positions.dtype == torch.long
        combined = hidden_states if residual is None else hidden_states + residual
        residual = combined + self.self_attn(combined)
        return self.mlp(residual), residual


class ServingModel(nn.Module):
    def __init__(self, identity):
        super().__init__()
        namespace, name = identity.rsplit(".", 1)
        block = type(name, (DeferredBlock,), {"__module__": namespace})
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([block(), block()])

    def forward(self, value):
        positions = torch.arange(value.shape[0])
        hidden, residual = self.model.layers[0](positions, value, None)
        hidden, residual = self.model.layers[1](
            positions=positions, hidden_states=hidden, residual=residual, forward_batch=None
        )
        return hidden + residual


def install(model, tmp_path):
    return install_model_runtime(
        model,
        backend="test-serving-contract",
        backend_version="1",
        model_id="contract-model",
        config={
            "socket_dir": tmp_path / "agents",
            "output_dir": tmp_path / "runs",
            "request_context": True,
        },
    )


@pytest.mark.parametrize("identity", sorted(DEFERRED_RESIDUAL_CLASSES))
def test_normalized_residual_is_sum_for_positional_and_keyword_calls(tmp_path: Path, identity):
    model = ServingModel(identity)
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    baseline = model(values)
    agent = install(model, tmp_path)
    try:
        assert torch.equal(model(values), baseline)
        client = AgentClient(agent.socket_path)
        client.call("arm", {"hooks": ["block.*.residual.*"], "capture_tensors": True})
        assert torch.equal(model(values), baseline)
        events = {event["hook"]: event for event in client.call("events")}
        for hook, expected in {
            "block.0.residual.pre": values,
            "block.0.residual.post": values * 12,
            "block.1.residual.pre": values * 12,
            "block.1.residual.post": baseline,
        }.items():
            tensors = load_file(events[hook]["tensor_file"])
            assert torch.equal(next(iter(tensors.values())), expected)
    finally:
        close_agents()


@pytest.mark.parametrize(
    "hook", ["block.0.residual.pre", "block.0.residual.post", "block.1.residual.pre"]
)
def test_request_token_patch_changes_logical_residual_and_preserves_other_request(
    tmp_path: Path, hook
):
    model = ServingModel("vllm.model_executor.models.llama.LlamaDecoderLayer")
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    baseline = model(values)
    agent = install(model, tmp_path)
    context = set_batch_context(
        BatchContext("prefill", (RequestSpan("a", 0, 2), RequestSpan("b", 2, 3)))
    )
    try:
        client = AgentClient(agent.socket_path)
        client.call(
            "patch.set",
            {
                "patch_id": "selected",
                "hook": hook,
                "operation": "scale",
                "value": 0.5,
                "request_ids": ["a"],
                "token": -1,
            },
        )
        expected = baseline.clone()
        expected[1] *= 0.5
        assert torch.equal(model(values), expected)
        client.call("patch.remove", {"patch_id": "selected"})
        assert torch.equal(model(values), baseline)
    finally:
        reset_batch_context(context)
        close_agents()


def test_noop_preserves_split_tensors_and_active_unknown_layout_fails(tmp_path: Path):
    model = ServingModel("vllm.model_executor.models.qwen2.Qwen2DecoderLayer")
    block = model.model.layers[0]
    value = torch.ones(2, 2)
    positions = torch.arange(2)
    expected = block(positions, value, None)
    agent = install(model, tmp_path)
    try:
        client = AgentClient(agent.socket_path)
        client.call(
            "patch.set",
            {
                "patch_id": "noop",
                "hook": "block.0.residual.post",
                "operation": "scale",
                "value": 1.0,
            },
        )
        output = block(positions, value, None)
        assert all(torch.equal(a, b) for a, b in zip(output, expected, strict=True))
        client.call("arm", {"hooks": ["block.0.residual.pre"]})
        with pytest.raises(ValueError, match="matching floating"):
            block(positions, value, torch.ones(1, 2))
    finally:
        close_agents()


def test_unknown_residual_contract_and_cuda_graph_reject_before_socket_creation(tmp_path: Path):
    for identity, graph in [
        ("sglang.srt.models.qwen3.Qwen3DecoderLayer", False),
        ("vllm.model_executor.models.llama.LlamaDecoderLayer", True),
    ]:
        with pytest.raises(ValueError, match="residual|eager|signature"):
            install_model_runtime(
                ServingModel(identity),
                backend="test",
                backend_version="1",
                model_id="test",
                config={"socket_dir": tmp_path / "agents", "graph_control": graph},
            )
        assert not (tmp_path / "agents").exists()


def test_projection_keyword_input_hook_ignores_position_tensor(tmp_path: Path):
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.o_proj.weight.copy_(torch.eye(2))

        def forward(self, value):
            return self.o_proj(input=value)

    model = ServingModel("vllm.model_executor.models.qwen3.Qwen3DecoderLayer")
    model.model.layers[0].self_attn = Attention()
    agent = install(model, tmp_path)
    try:
        client = AgentClient(agent.socket_path)
        client.call(
            "patch.set",
            {
                "patch_id": "head",
                "hook": "block.0.attention.heads",
                "operation": "head_ablate",
                "head": 1,
                "num_heads": 2,
            },
        )
        value = model.model.layers[0].self_attn(torch.ones(1, 2))
        assert value.tolist() == [[1.0, 0.0]]
    finally:
        close_agents()


class Communicator:
    def __init__(self):
        self._context = SimpleNamespace(tp_size=1, attn_tp_size=1, attn_dp_size=1, attn_cp_size=1)
        self._sp_variant = None
        self.allow_reduce_scatter = False
        self.enable_fused_ar_quant = False
        self.layer_scatter_modes = SimpleNamespace(
            **{
                key: SimpleNamespace(name="TP_ATTN_FULL")
                for key in (
                    "layer_input_mode",
                    "attn_mode",
                    "mlp_mode",
                    "middle_residual_mode",
                    "layer_output_mode",
                )
            }
        )

    def prepare_attn(self, hidden_states, residual, forward_batch, post_residual_addition=None):
        if residual is not None:
            hidden_states = hidden_states + residual
            if post_residual_addition is not None:
                hidden_states = hidden_states + post_residual_addition
        return hidden_states, hidden_states

    def prepare_mlp(self, hidden_states, residual, forward_batch):
        combined = hidden_states + residual
        return combined, combined

    def postprocess_layer(self, hidden_states, residual, forward_batch):
        return hidden_states, residual


def communicator_forward(
    self, positions, hidden_states, forward_batch, residual, post_residual_addition=None
):
    hidden_states, residual = self.layer_communicator.prepare_attn(
        hidden_states, residual, forward_batch, post_residual_addition=post_residual_addition
    )
    hidden_states = self.self_attn(hidden_states)
    hidden_states, residual = self.layer_communicator.prepare_mlp(
        hidden_states, residual, forward_batch
    )
    hidden_states = self.mlp(hidden_states)
    return self.layer_communicator.postprocess_layer(hidden_states, residual, forward_batch)


def qwen3_model():
    model = ServingModel("sglang.srt.models.qwen3.Qwen3DecoderLayer")
    for block in model.model.layers:
        block.layer_communicator = Communicator()
        type(block).forward = communicator_forward
    return model


def test_qwen3_additional_residual_is_patched_once_with_request_isolation(tmp_path: Path):
    model = qwen3_model()
    agent = install(model, tmp_path)
    value, residual, addition = torch.ones(2, 2), torch.full((2, 2), 2.0), torch.full((2, 2), 3.0)
    context = set_batch_context(
        BatchContext("decode", (RequestSpan("a", 0, 1), RequestSpan("b", 1, 2)))
    )
    try:
        client = AgentClient(agent.socket_path)
        client.call(
            "patch.set",
            {
                "patch_id": "halve",
                "hook": "block.0.residual.pre",
                "operation": "scale",
                "value": 0.5,
                "request_ids": ["a"],
            },
        )
        output = model.model.layers[0](
            positions=torch.arange(2),
            hidden_states=value,
            forward_batch=None,
            residual=residual,
            post_residual_addition=addition,
        )
        assert torch.equal(output[0] + output[1], torch.tensor([[36.0, 36.0], [72.0, 72.0]]))
        assert torch.equal(residual, torch.full((2, 2), 2.0))
        assert torch.equal(addition, torch.full((2, 2), 3.0))
        client.call("patch.remove", {"patch_id": "halve"})
        output = model.model.layers[0](torch.arange(2), value, None, residual, addition)
        assert torch.equal(output[0] + output[1], torch.full((2, 2), 72.0))
    finally:
        reset_batch_context(context)
        close_agents()


def test_qwen3_scattered_communicator_rejected_before_installation(tmp_path: Path):
    model = qwen3_model()
    model.model.layers[0].layer_communicator.layer_scatter_modes.mlp_mode.name = "SCATTERED"
    with pytest.raises(ValueError, match="unsharded"):
        install(model, tmp_path)
    assert not (tmp_path / "agents").exists()


@pytest.mark.parametrize(
    "namespace", ["vllm.model_executor.models.utils", "sglang.srt.layers.utils.common"]
)
def test_pipeline_missing_layers_do_not_advertise_phantom_hooks(tmp_path: Path, namespace):
    model = ServingModel("vllm.model_executor.models.llama.LlamaDecoderLayer")
    placeholder = type("PPMissingLayer", (nn.Identity,), {"__module__": namespace})
    model.model.layers[0] = placeholder()
    agent = install(model, tmp_path)
    try:
        hooks = AgentClient(agent.socket_path).call("hooks")
        assert hooks and all(hook.startswith("block.1.") for hook in hooks)
        assert not model.model.layers[0]._forward_pre_hooks
        assert not model.model.layers[0]._forward_hooks
    finally:
        close_agents()
