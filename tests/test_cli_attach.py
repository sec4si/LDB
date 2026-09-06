from pathlib import Path

from typer.testing import CliRunner

from ldb.agent import AgentRuntime
from ldb.cli import app

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
