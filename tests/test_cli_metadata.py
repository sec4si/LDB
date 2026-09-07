from typer.testing import CliRunner

from ldb.cli import app

runner = CliRunner()


def test_runtimes_lists_audited_serving_versions() -> None:
    result = runner.invoke(app, ["runtimes"])

    assert result.exit_code == 0
    assert '"sglang": [' in result.stdout
    assert '"0.5.19"' in result.stdout
    assert '"vllm": [' in result.stdout
    assert '"0.28.0"' in result.stdout
