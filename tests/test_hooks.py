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
