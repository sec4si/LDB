from pathlib import Path

import torch
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


def test_attach_loads_versioned_patch_into_live_agent(tmp_path: Path) -> None:
    agent = AgentRuntime(
        backend="test",
        backend_version="1",
        model_id="test-model",
        socket_dir=tmp_path / "agents",
        output_dir=tmp_path / "runs",
    ).start()
    patch_file = tmp_path / "patch.yaml"
    patch_file.write_text(
        """schema_version: ldb.patch/v1alpha1
patch_id: test-patch
description: Test patch.
model_id: test-model
model_revision: revision
tokenizer_hash: hash
adapter: test
interventions:
  - hook: block.0.residual.post
    operation: scale
    value: 0.5
    request_scoped: false
evidence: causally_tested
""",
        encoding="utf-8",
    )
    try:
        result = runner.invoke(
            app,
            ["attach", str(agent.socket_path), "-e", f"patch-load {patch_file}"],
        )
        assert result.exit_code == 0
        output = agent.process_tensor("block.0.residual.post", torch.tensor([2.0]))
        assert output.tolist() == [1.0]
    finally:
        agent.close()
