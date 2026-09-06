from pathlib import Path

from typer.testing import CliRunner

from ldb.agent import AgentRuntime
from ldb.cli import _parse_patch_arguments, app

runner = CliRunner()


def test_attach_executes_command_against_live_agent(tmp_path: Path) -> None:
    agent = AgentRuntime(
        backend="test",
        backend_version="1",
        model_id="test-model",
        socket_dir=tmp_path / "agents",
        output_dir=tmp_path / "runs",
    ).start()
    try:
        result = runner.invoke(app, ["attach", str(agent.socket_path), "-e", "status"])
        assert result.exit_code == 0
        assert '"backend": "test"' in result.stdout
        assert '"armed": false' in result.stdout
    finally:
        agent.close()


def test_parse_patch_arguments_supports_tensor_expert_and_request() -> None:
    value, tensor, expert, request_ids = _parse_patch_arguments(
        ["-0.25", "--expert", "7", "--request", "request-a"]
    )
    assert value == -0.25
    assert tensor is None
    assert expert == 7
    assert request_ids == ["request-a"]

    value, tensor, expert, request_ids = _parse_patch_arguments(
        ["--tensor", "/tmp/steering.safetensors"]
    )
    assert value is None
    assert tensor == "/tmp/steering.safetensors"
    assert expert is None
    assert request_ids == []
