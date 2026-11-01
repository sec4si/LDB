from pathlib import Path

import pytest

from ldb.client import AgentClient
from ldb.hooks import close_agents, install_model_runtime

torch = pytest.importorskip("torch")
nn = torch.nn


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Linear(2, 2, bias=False)
        self.mlp = nn.Linear(2, 2, bias=False)

    def forward(self, value):
        return value + self.mlp(self.self_attn(value))


class Core(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Block()])

    def forward(self, value):
        return self.layers[0](value)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Core()

    def forward(self, value):
        return self.model(value)


class MoEMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.gate = nn.Linear(2, 2)

    def forward(self, value):
        self.gate(value)
        return sum((expert(value) for expert in self.experts), torch.zeros_like(value))


class MoEBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = MoEMLP()

    def forward(self, value):
        return value + self.mlp(value)


class MoEModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([MoEBlock()])

    def forward(self, value):
        return self.model.layers[0](value)


class UnsupportedModel(nn.Module):
    def forward(self, value):
        return value


def test_installed_model_runtime_uses_normalized_hooks(tmp_path: Path) -> None:
    model = Model()
    agent = install_model_runtime(
        model,
        backend="test-hooks",
        backend_version="1",
        model_id="test-model",
        config={"socket_dir": tmp_path / "agents", "output_dir": tmp_path / "runs"},
    )
    try:
        client = AgentClient(agent.socket_path)
        available = set(client.call("hooks"))
        assert available == {
            "block.0.residual.pre",
            "block.0.attention.output",
            "block.0.mlp.output",
            "block.0.residual.post",
        }
        client.call("arm", {"hooks": ["block.0.*"], "capture_tensors": False})
        model(torch.ones(1, 2))
        hooks = {event["hook"] for event in client.call("events")}
        assert hooks == {
            "block.0.residual.pre",
            "block.0.attention.output",
            "block.0.mlp.output",
            "block.0.residual.post",
        }
    finally:
        close_agents()


def test_installed_runtime_rejects_unknown_hook_patterns(tmp_path: Path) -> None:
    model = Model()
    agent = install_model_runtime(
        model,
        backend="test-hook-validation",
        backend_version="1",
        model_id="test-model",
        config={"socket_dir": tmp_path / "agents", "output_dir": tmp_path / "runs"},
    )
    try:
        client = AgentClient(agent.socket_path)
        with pytest.raises(RuntimeError, match="matches no installed hooks"):
            client.call("arm", {"hooks": ["block.99.*"]})
        with pytest.raises(RuntimeError, match="matches no installed hooks"):
            client.call(
                "patch.set",
                {
                    "patch_id": "missing",
                    "hook": "block.99.residual.post",
                    "operation": "ablate",
                },
            )
        with pytest.raises(RuntimeError, match="matches no installed hooks"):
            client.call(
                "break.set",
                {"breakpoint_id": "missing", "hook": "block.99.residual.post"},
            )
    finally:
        close_agents()


def test_runtime_installation_is_idempotent(tmp_path: Path) -> None:
    model = Model()
    first = install_model_runtime(
        model,
        backend="test-hooks",
        backend_version="1",
        model_id="test-model",
        config={"socket_dir": tmp_path / "agents", "output_dir": tmp_path / "runs"},
    )
    second = install_model_runtime(
        model,
        backend="test-hooks",
        backend_version="1",
        model_id="test-model",
        config={"socket_dir": tmp_path / "agents", "output_dir": tmp_path / "runs"},
    )
    try:
        assert first is second
    finally:
        close_agents()


def test_runtime_is_not_exposed_before_topology_validation(tmp_path: Path) -> None:
    socket_dir = tmp_path / "agents"
    with pytest.raises(ValueError, match="no decoder block container"):
        install_model_runtime(
            UnsupportedModel(),
            backend="test-invalid-hooks",
            backend_version="1",
            model_id="test-model",
            config={"socket_dir": socket_dir, "output_dir": tmp_path / "runs"},
        )

    assert not socket_dir.exists()


def test_process_agent_reuse_requires_consistent_identity(tmp_path: Path) -> None:
    first = install_model_runtime(
        Model(),
        backend="test-agent-identity",
        backend_version="1",
        model_id="model-a",
        config={"socket_dir": tmp_path / "agents", "output_dir": tmp_path / "runs"},
    )
    try:
        with pytest.raises(RuntimeError, match="incompatible identity: model_id"):
            install_model_runtime(
                Model(),
                backend="test-agent-identity",
                backend_version="1",
                model_id="model-b",
                config={
                    "socket_dir": tmp_path / "agents",
                    "output_dir": tmp_path / "runs",
                },
            )
    finally:
        first.close()
        close_agents()


def test_process_agents_are_isolated_by_instance(tmp_path: Path) -> None:
    agents = tuple(
        install_model_runtime(
            Model(),
            backend="test-agent-instance",
            backend_version="1",
            model_id="test-model",
            config={
                "instance_id": instance_id,
                "socket_dir": tmp_path / "agents",
                "output_dir": tmp_path / "runs",
            },
        )
        for instance_id in ("blue", "green")
    )
    try:
        assert agents[0] is not agents[1]
        assert agents[0].socket_path != agents[1].socket_path
        assert {agent.descriptor.instance_id for agent in agents} == {"blue", "green"}
    finally:
        close_agents()


def test_installed_runtime_prepares_graph_control_slots(tmp_path: Path) -> None:
    model = Model()
    agent = install_model_runtime(
        model,
        backend="test-graph-hooks",
        backend_version="1",
        model_id="test-model",
        config={
            "socket_dir": tmp_path / "agents",
            "output_dir": tmp_path / "runs",
            "graph_control": True,
        },
    )
    try:
        client = AgentClient(agent.socket_path)
        baseline = model(torch.ones(1, 2))
        client.call(
            "patch.set",
            {
                "patch_id": "ablate-output",
                "hook": "block.0.residual.post",
                "operation": "ablate",
            },
        )
        patched = model(torch.ones(1, 2))
        assert baseline.abs().sum().item() > 0
        assert patched.tolist() == [[0.0, 0.0]]
    finally:
        close_agents()


def test_installed_runtime_traces_visible_moe_experts(tmp_path: Path) -> None:
    model = MoEModel()
    agent = install_model_runtime(
        model,
        backend="test-moe-hooks",
        backend_version="1",
        model_id="test-model",
        config={"socket_dir": tmp_path / "agents", "output_dir": tmp_path / "runs"},
    )
    try:
        client = AgentClient(agent.socket_path)
        client.call("arm", {"hooks": ["block.0.moe.*"], "capture_tensors": False})
        model(torch.ones(1, 2))
        hooks = {event["hook"] for event in client.call("events")}
        assert hooks == {
            "block.0.moe.router.logits",
            "block.0.moe.expert.0.output",
            "block.0.moe.expert.1.output",
        }
    finally:
        close_agents()


class FusedRouter(nn.Module):
    """Mimics vLLM's FusedTopKBiasRouter: the bias only steers expert choice."""

    def __init__(self) -> None:
        super().__init__()
        self.e_score_correction_bias = nn.Parameter(torch.zeros(4), requires_grad=False)

    def _compute_routing(self, hidden_states, router_logits, indices_type=None, *, input_ids=None):
        scores = router_logits.sigmoid()
        ids = torch.topk(scores + self.e_score_correction_bias.data, 2).indices
        return scores.gather(-1, ids), ids.to(torch.int32)


class FusedRunner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = FusedRouter()


class FusedMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = FusedRunner()


class MHCDecoderLayer(nn.Module):
    """Mirrors DeepseekV4DecoderLayer: the previous layer's mean stream comes back last."""

    def __init__(self, *, sequence_parallel: bool = False) -> None:
        super().__init__()
        self.ffn = FusedMoE()
        self.use_sequence_parallel = sequence_parallel
        self.capture_requests: list[bool] = []

    def forward(self, x, positions, input_ids, residual=None, *, capture_previous_aux=False):
        self.capture_requests.append(capture_previous_aux)
        aux = x * 2 if capture_previous_aux else None
        return x, x, x, x, x, aux


MHCDecoderLayer.__module__ = "vllm.models.deepseek_v41.nvidia.model"
MHCDecoderLayer.__name__ = "DeepseekV4DecoderLayer"


class MHCModel(nn.Module):
    def __init__(self, layers: int = 1, *, sequence_parallel: bool = False) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [MHCDecoderLayer(sequence_parallel=sequence_parallel) for _ in range(layers)]
        )


def test_mhc_decoders_expose_routing_hooks_only(tmp_path: Path) -> None:
    model = MHCModel()
    router = model.model.layers[0].ffn.experts.router
    logits = torch.tensor([[1.0, 5.0, 4.0, 0.0]])
    agent = install_model_runtime(
        model,
        backend="test-mhc-hooks",
        backend_version="1",
        model_id="test-model",
        config={"socket_dir": tmp_path / "agents", "output_dir": tmp_path / "runs"},
    )
    try:
        client = AgentClient(agent.socket_path)
        assert set(client.call("hooks")) == {
            "block.0.moe.router.logits",
            "block.0.moe.router.topk_ids",
            "block.0.moe.router.topk_weights",
        }
        client.call("arm", {"hooks": ["block.0.moe.router.topk_*"], "capture_tensors": False})
        assert router._compute_routing(None, logits)[1].tolist() == [[1, 2]]
        assert {event["hook"] for event in client.call("events")} == {
            "block.0.moe.router.topk_ids",
            "block.0.moe.router.topk_weights",
        }

        client.call(
            "patch.set",
            {
                "patch_id": "drop-1",
                "hook": "block.0.moe.router.logits",
                "operation": "routing_bias",
                "value": -1e4,
                "expert": 1,
            },
        )
        weights, ids = router._compute_routing(None, logits)
        assert ids.tolist() == [[2, 0]]
        assert weights[0, 0].item() == pytest.approx(torch.tensor(4.0).sigmoid().item())
        assert router.e_score_correction_bias.abs().sum().item() == 0

        client.call("patch.remove", {"patch_id": "drop-1"})
        assert router._compute_routing(None, logits)[1].tolist() == [[1, 2]]
    finally:
        close_agents()
    assert "_compute_routing" not in router.__dict__


def test_mhc_residual_is_read_through_the_next_layers_aux_capture(tmp_path: Path) -> None:
    model = MHCModel(layers=2)
    layer = model.model.layers[1]
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    agent = install_model_runtime(
        model,
        backend="test-mhc-residual",
        backend_version="1",
        model_id="test-model",
        config={"socket_dir": tmp_path / "agents", "output_dir": tmp_path / "runs"},
    )
    try:
        client = AgentClient(agent.socket_path)
        hooks = set(client.call("hooks"))
        assert "block.0.residual.post" in hooks
        assert not any(hook.startswith("block.1.residual") for hook in hooks)

        assert layer(x, None, None)[-1] is None
        assert layer.capture_requests == [False]

        client.call("arm", {"hooks": ["block.0.residual.post"], "capture_tensors": False})
        assert layer(x, None, None)[-1] is None
        assert layer.capture_requests[-1] is True
        event = client.call("events")[0]
        assert event["hook"] == "block.0.residual.post"
        assert event["summary"]["mean"] == pytest.approx(5.0)

        requested = layer(x, None, None, capture_previous_aux=True)[-1]
        assert torch.equal(requested, x * 2)

        with pytest.raises(RuntimeError, match="Observe-only"):
            client.call(
                "patch.set",
                {
                    "patch_id": "scale",
                    "hook": "block.0.residual.post",
                    "operation": "scale",
                    "value": 2.0,
                },
            )
    finally:
        close_agents()


def test_mhc_residual_is_not_exposed_for_sequence_parallel_shards(tmp_path: Path) -> None:
    agent = install_model_runtime(
        MHCModel(layers=2, sequence_parallel=True),
        backend="test-mhc-sp",
        backend_version="1",
        model_id="test-model",
        config={"socket_dir": tmp_path / "agents", "output_dir": tmp_path / "runs"},
    )
    try:
        assert "block.0.residual.post" not in set(AgentClient(agent.socket_path).call("hooks"))
    finally:
        close_agents()
