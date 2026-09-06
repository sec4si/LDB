import stat
import threading
import time
from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file

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


def test_agent_rejects_duplicate_patch_ids(tmp_path: Path) -> None:
    agent = AgentRuntime(
        backend="test",
        backend_version="1",
        model_id="test-model",
        socket_dir=tmp_path / "agents",
        output_dir=tmp_path / "runs",
    ).start()
    try:
        client = AgentClient(agent.socket_path)
        patch = {
            "patch_id": "scale-1",
            "hook": "block.0.residual.post",
            "operation": "scale",
            "value": 0.5,
        }
        client.call("patch.set", patch)
        with pytest.raises(RuntimeError, match="Patch ID already exists"):
            client.call("patch.set", patch | {"value": 0.25})

        output = agent.process_tensor("block.0.residual.post", torch.tensor([2.0]))
        assert output.tolist() == [1.0]
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


def test_group_validates_hooks_across_pipeline_ranks(tmp_path: Path) -> None:
    agents = tuple(
        AgentRuntime(
            backend="test",
            backend_version="1",
            model_id="test-model",
            socket_dir=tmp_path / "agents",
            output_dir=tmp_path / "runs",
            rank=rank,
            world_size=2,
        ).start()
        for rank in range(2)
    )
    agents[0].register_hook("block.0.residual.post")
    agents[1].register_hook("block.1.residual.post")
    group = AgentGroupClient(tuple(AgentClient(agent.socket_path) for agent in agents))
    try:
        group.call("arm", {"hooks": ["block.1.*"]})
        assert all(result["armed"] for result in group.call("status"))
        with pytest.raises(ValueError, match="across attached ranks"):
            group.call("arm", {"hooks": ["block.99.*"]})
    finally:
        for agent in agents:
            agent.close()


def test_agent_replaces_activation_from_safetensors(tmp_path: Path) -> None:
    agent = AgentRuntime(
        backend="test",
        backend_version="1",
        model_id="test-model",
        socket_dir=tmp_path / "agents",
        output_dir=tmp_path / "runs",
    ).start()
    try:
        client = AgentClient(agent.socket_path)
        tensor_file = tmp_path / "replacement.safetensors"
        save_file({"activation": torch.tensor([9.0, 8.0])}, tensor_file)
        client.call(
            "patch.set",
            {
                "patch_id": "replace-1",
                "hook": "block.0.residual.post",
                "operation": "replace",
                "tensor": str(tensor_file),
            },
        )
        output = agent.process_tensor("block.0.residual.post", torch.tensor([1.0, 2.0]))
        assert output.tolist() == [9.0, 8.0]
    finally:
        agent.close()


def test_agent_adds_tensor_and_routing_bias(tmp_path: Path) -> None:
    agent = AgentRuntime(
        backend="test",
        backend_version="1",
        model_id="test-model",
        socket_dir=tmp_path / "agents",
        output_dir=tmp_path / "runs",
    ).start()
    try:
        client = AgentClient(agent.socket_path)
        tensor_file = tmp_path / "steering.safetensors"
        save_file({"steering": torch.tensor([0.5, -0.5])}, tensor_file)
        client.call(
            "patch.set",
            {
                "patch_id": "steer-1",
                "hook": "block.0.residual.post",
                "operation": "add",
                "tensor": str(tensor_file),
            },
        )
        client.call(
            "patch.set",
            {
                "patch_id": "router-1",
                "hook": "block.0.moe.router",
                "operation": "routing_bias",
                "value": -0.25,
                "expert": 1,
            },
        )
        steered = agent.process_tensor("block.0.residual.post", torch.tensor([1.0, 2.0]))
        biased = agent.process_tensor("block.0.moe.router", torch.tensor([[1.0, 2.0]]))
        assert steered.tolist() == [1.5, 1.5]
        assert biased.tolist() == [[1.0, 1.75]]
    finally:
        agent.close()


def test_agent_ablates_one_attention_head(tmp_path: Path) -> None:
    agent = AgentRuntime(
        backend="test",
        backend_version="1",
        model_id="test-model",
        socket_dir=tmp_path / "agents",
        output_dir=tmp_path / "runs",
    ).start()
    try:
        client = AgentClient(agent.socket_path)
        client.call(
            "patch.set",
            {
                "patch_id": "disable-head-1",
                "hook": "block.0.attention.heads",
                "operation": "head_ablate",
                "head": 1,
                "num_heads": 2,
            },
        )
        output = agent.process_tensor(
            "block.0.attention.heads", torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        )
        assert output.tolist() == [[1.0, 2.0, 0.0, 0.0]]
    finally:
        agent.close()


def test_agent_scales_one_sae_feature(tmp_path: Path) -> None:
    dictionary_path = tmp_path / "dictionary.safetensors"
    save_file(
        {
            "encoder": torch.eye(2),
            "decoder": torch.eye(2),
            "encoder_bias": torch.zeros(2),
            "decoder_bias": torch.zeros(2),
        },
        dictionary_path,
    )
    agent = AgentRuntime(
        backend="test",
        backend_version="1",
        model_id="test-model",
        socket_dir=tmp_path / "agents",
        output_dir=tmp_path / "runs",
    ).start()
    try:
        client = AgentClient(agent.socket_path)
        client.call(
            "patch.set",
            {
                "patch_id": "feature-0",
                "hook": "block.0.residual.post",
                "operation": "feature_scale",
                "value": 0.0,
                "dictionary": str(dictionary_path),
                "feature": 0,
            },
        )
        output = agent.process_tensor("block.0.residual.post", torch.tensor([[2.0, 3.0]]))
        assert output.tolist() == [[0.0, 3.0]]
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
        hit_id = status["stopped"][0]["hit_id"]
        inspection = client.call("break.inspect", {"hit_id": hit_id, "capture": True})
        assert inspection["hit"]["summary"]["mean"] == 2.0
        captured = load_file(inspection["tensor_file"])["activation"]
        assert captured.tolist() == [2.0]

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
        assert client.call("break.inspect", {"hit_id": hit_id})["hit"] is None
    finally:
        client.call("continue")
        agent.close()


def test_graph_control_composes_scalar_patches(tmp_path: Path) -> None:
    agent = AgentRuntime(
        backend="test",
        backend_version="1",
        model_id="test-model",
        socket_dir=tmp_path / "agents",
        output_dir=tmp_path / "runs",
        graph_control=True,
    ).start()
    try:
        client = AgentClient(agent.socket_path)
        capabilities = client.call("hello")["capabilities"]
        assert "patch.cuda_graph" in capabilities
        agent.prepare_graph_hook("block.0.residual.post", torch.ones(1))
        assert agent.process_tensor("block.0.residual.post", torch.tensor([2.0])).tolist() == [2.0]

        client.call(
            "patch.set",
            {
                "patch_id": "scale",
                "hook": "block.0.residual.post",
                "operation": "scale",
                "value": 0.5,
            },
        )
        client.call(
            "patch.set",
            {
                "patch_id": "bias",
                "hook": "block.0.residual.post",
                "operation": "add",
                "value": 1.0,
            },
        )
        output = agent.process_tensor("block.0.residual.post", torch.tensor([2.0]))
        assert output.tolist() == [2.0]

        client.call("patch.remove", {"patch_id": "scale"})
        output = agent.process_tensor("block.0.residual.post", torch.tensor([2.0]))
        assert output.tolist() == [3.0]
    finally:
        agent.close()


def test_graph_trace_reports_replays_after_arm(tmp_path: Path) -> None:
    agent = AgentRuntime(
        backend="test",
        backend_version="1",
        model_id="test-model",
        socket_dir=tmp_path / "agents",
        output_dir=tmp_path / "runs",
        graph_control=True,
    ).start()
    hook = "block.0.residual.post"
    agent.register_hook(hook)
    agent.prepare_graph_hook(hook, torch.ones(1))
    try:
        client = AgentClient(agent.socket_path)
        agent.process_tensor(hook, torch.tensor([10.0, 20.0]))
        client.call("arm", {"hooks": [hook]})
        assert client.call("events") == []

        agent.process_tensor(hook, torch.tensor([1.0, 3.0]))
        agent.process_tensor(hook, torch.tensor([2.0, 4.0]))
        event = client.call("events")[0]
        assert event["source"] == "cuda_graph"
        assert event["occurrences"] == 2
        assert event["summary"]["mean"] == 3.0
        assert event["patched_summary"]["mean"] == 3.0
        assert event["request_ids"] == []
    finally:
        agent.close()


def test_graph_trace_rejects_dynamic_debug_controls(tmp_path: Path) -> None:
    agent = AgentRuntime(
        backend="test",
        backend_version="1",
        model_id="test-model",
        socket_dir=tmp_path / "agents",
        output_dir=tmp_path / "runs",
        graph_control=True,
    ).start()
    agent.register_hook("block.0.residual.post")
    try:
        client = AgentClient(agent.socket_path)
        with pytest.raises(RuntimeError, match="scalar summaries only"):
            client.call("arm", {"hooks": ["*"], "capture_tensors": True})
        with pytest.raises(RuntimeError, match="cannot be request-scoped"):
            client.call("arm", {"hooks": ["*"], "request_ids": ["request-a"]})
        with pytest.raises(RuntimeError, match="unavailable during CUDA Graph replay"):
            client.call(
                "break.set",
                {"breakpoint_id": "stop", "hook": "block.0.residual.post"},
            )
    finally:
        agent.close()


def test_graph_control_rejects_request_scoped_patch(tmp_path: Path) -> None:
    agent = AgentRuntime(
        backend="test",
        backend_version="1",
        model_id="test-model",
        socket_dir=tmp_path / "agents",
        output_dir=tmp_path / "runs",
        request_context=True,
        graph_control=True,
    ).start()
    try:
        client = AgentClient(agent.socket_path)
        with pytest.raises(RuntimeError, match="cannot be request-scoped"):
            client.call(
                "patch.set",
                {
                    "patch_id": "request-scale",
                    "hook": "block.0.residual.post",
                    "operation": "scale",
                    "value": 0.5,
                    "request_ids": ["request-a"],
                },
            )
    finally:
        agent.close()
