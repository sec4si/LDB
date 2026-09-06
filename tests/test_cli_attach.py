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
        model_revision="revision",
        tokenizer_hash="hash",
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
    parsed = _parse_patch_arguments(["-0.25", "--expert", "7", "--request", "request-a"])
    assert parsed.value == -0.25
    assert parsed.tensor is None
    assert parsed.expert == 7
    assert parsed.request_ids == ["request-a"]

    parsed = _parse_patch_arguments(["--tensor", "/tmp/steering.safetensors"])
    assert parsed.value is None
    assert parsed.tensor == "/tmp/steering.safetensors"
    assert parsed.expert is None
    assert parsed.request_ids == []

    parsed = _parse_patch_arguments(["--head", "3", "--heads", "8"])
    assert parsed.head == 3
    assert parsed.num_heads == 8


def test_attach_loads_versioned_patch_into_live_agent(tmp_path: Path) -> None:
    agent = AgentRuntime(
        backend="test",
        backend_version="1",
        model_id="test-model",
        model_revision="revision",
        tokenizer_hash="hash",
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


def test_attach_rejects_versioned_patch_without_runtime_identity(tmp_path: Path) -> None:
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
""",
        encoding="utf-8",
    )
    try:
        result = runner.invoke(
            app,
            ["attach", str(agent.socket_path), "-e", f"patch-load {patch_file}"],
        )
        assert result.exit_code == 1
        assert "requires model revision and tokenizer hash" in str(result.exception)
    finally:
        agent.close()
