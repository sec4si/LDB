import stat
import threading
import time
from pathlib import Path

import pytest

from ldb.agent import AgentRuntime
from ldb.client import AgentClient, AgentGroupClient
from ldb.context import BatchContext, RequestSpan, reset_batch_context, set_batch_context

torch = pytest.importorskip("torch")


def test_live_agent_trace_and_patch(tmp_path: Path) -> None:
    agent = AgentRuntime(
        backend="test",
        backend_version="1",
        model_id="test-model",
        socket_dir=tmp_path / "agents",
        output_dir=tmp_path / "runs",
    ).start()
    try:
        client = AgentClient(agent.socket_path)
        hello = client.call("hello")
        assert hello["backend"] == "test"
        assert stat.S_IMODE(agent.socket_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE(agent.socket_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(agent.descriptor_path.stat().st_mode) == 0o600

        client.call("arm", {"hooks": ["block.*"], "capture_tensors": True})
        client.call(
            "patch.set",
            {
                "patch_id": "scale-1",
                "hook": "block.0.residual.post",
                "operation": "scale",
                "value": 0.5,
            },
        )
        output = agent.process_tensor("block.0.residual.post", torch.tensor([2.0, 4.0]))
        assert output.tolist() == [1.0, 2.0]

        events = client.call("events")
        assert events[0]["patch_ids"] == ["scale-1"]
        assert events[0]["summary"]["norm"] == pytest.approx(4.472136)
        assert Path(events[0]["tensor_file"]).is_file()
    finally:
        agent.close()


def test_agent_group_discovers_descriptors(tmp_path: Path) -> None:
    agent = AgentRuntime(
        backend="test",
        backend_version="1",
        model_id="test-model",
        socket_dir=tmp_path / "agents",
        output_dir=tmp_path / "runs",
    ).start()
    try:
        group = AgentGroupClient.connect(tmp_path / "agents")
        results = group.call("status")
        assert len(results) == 1
        assert results[0]["armed"] is False
    finally:
        agent.close()


def test_agent_rejects_unsupported_patch(tmp_path: Path) -> None:
    agent = AgentRuntime(
        backend="test",
        backend_version="1",
        model_id="test-model",
        socket_dir=tmp_path / "agents",
        output_dir=tmp_path / "runs",
    ).start()
    try:
        client = AgentClient(agent.socket_path)
        with pytest.raises(RuntimeError, match="does not support replace"):
            client.call(
                "patch.set",
                {
                    "patch_id": "replace-1",
                    "hook": "block.0.residual.post",
                    "operation": "replace",
                },
            )
    finally:
        agent.close()


def test_request_scoped_patch_only_changes_selected_tokens(tmp_path: Path) -> None:
    agent = AgentRuntime(
        backend="test",
        backend_version="1",
        model_id="test-model",
        socket_dir=tmp_path / "agents",
        output_dir=tmp_path / "runs",
        request_context=True,
    ).start()
    token = set_batch_context(
        BatchContext(
            phase="prefill",
            requests=(
                RequestSpan("request-a", 0, 2),
                RequestSpan("request-b", 2, 3),
            ),
        )
    )
    try:
        client = AgentClient(agent.socket_path)
        client.call(
            "arm",
            {
                "hooks": ["block.0.*"],
                "request_ids": ["request-b"],
                "capture_tensors": False,
            },
        )
        client.call(
            "patch.set",
            {
                "patch_id": "request-b-scale",
                "hook": "block.0.residual.post",
                "operation": "scale",
                "value": 10,
                "request_ids": ["request-b"],
            },
        )
        output = agent.process_tensor("block.0.residual.post", torch.tensor([[1.0], [2.0], [3.0]]))
        assert output.tolist() == [[1.0], [2.0], [30.0]]
        event = client.call("events")[0]
        assert event["request_ids"] == ["request-b"]
        assert event["phase"] == "prefill"
        assert event["summary"]["mean"] == 3.0
        assert event["patched_summary"]["mean"] == 30.0
    finally:
        reset_batch_context(token)
        agent.close()


def test_breakpoint_accepts_patch_before_continue(tmp_path: Path) -> None:
    agent = AgentRuntime(
        backend="test",
        backend_version="1",
        model_id="test-model",
        socket_dir=tmp_path / "agents",
        output_dir=tmp_path / "runs",
    ).start()
    client = AgentClient(agent.socket_path)
    outputs = []
    try:
        client.call(
            "break.set",
            {
                "breakpoint_id": "stop-1",
                "hook": "block.0.residual.post",
                "timeout_seconds": 2,
            },
        )
        worker = threading.Thread(
            target=lambda: outputs.append(
                agent.process_tensor("block.0.residual.post", torch.tensor([2.0]))
            )
        )
        worker.start()

        deadline = time.monotonic() + 1
        status = client.call("status")
        while not status["stopped"] and time.monotonic() < deadline:
            time.sleep(0.01)
            status = client.call("status")
        assert status["stopped"][0]["breakpoint_id"] == "stop-1"

        client.call(
            "patch.set",
            {
                "patch_id": "late-scale",
                "hook": "block.0.residual.post",
                "operation": "scale",
                "value": 3,
            },
        )
        assert client.call("continue")["released"] == 1
        worker.join(timeout=1)
        assert not worker.is_alive()
        assert outputs[0].tolist() == [6.0]
    finally:
        client.call("continue")
        agent.close()
