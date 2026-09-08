import copy

import pytest
import torch
from safetensors.torch import load_file

from ldb.operators import OperatorDebugger


def test_pauses_before_operation_and_steps_exactly_once(tmp_path):
    value = torch.ones(2)

    def run():
        value.add_(2)
        value.mul_(3)
        return value

    with OperatorDebugger(run) as debugger:
        assert debugger.inspect()["completed_operations"] == 0
        assert any(entry["function"] == "run" for entry in debugger.command("where"))
        assert torch.equal(value, torch.ones(2))
        debugger.capture(tmp_path / "before.safetensors")
        first = debugger.command("step")
        assert first["completed_operations"] == 1
        assert torch.equal(value, torch.full((2,), 3.0))
        assert "mul_" in first["current"]["operator"]
        debugger.capture(tmp_path / "after.safetensors", outputs=True)
        second = debugger.command("step")
        assert second["state"] == "finished"
        assert second["completed_operations"] == 2
        assert torch.equal(debugger.result, torch.full((2,), 9.0))
    assert debugger.thread is not None and not debugger.thread.is_alive()
    assert torch.equal(
        next(iter(load_file(tmp_path / "before.safetensors").values())), torch.ones(2)
    )
    assert torch.equal(
        next(iter(load_file(tmp_path / "after.safetensors").values())), torch.full((2,), 3.0)
    )


def test_breakpoints_backward_and_result_match_direct_execution():
    torch.manual_seed(0)
    baseline = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
    model = copy.deepcopy(baseline)
    x = torch.randn(2, 3)
    expected = baseline(x).square().sum()
    expected.backward()

    def run():
        loss = model(x).square().sum()
        loss.backward()
        return loss.detach()

    with OperatorDebugger(run, max_history=1000) as debugger:
        debugger.command("break aten.relu.*")
        stopped = debugger.command("continue")
        assert stopped["state"] == "paused"
        assert stopped["current"]["operator"].startswith("aten.relu.")
        debugger.command("delete aten.relu.*")
        assert debugger.command("continue")["state"] == "finished"
        assert torch.equal(debugger.result, expected)
        assert any("backward" in e["operator"] for e in debugger.history)
    assert all(
        torch.equal(a.grad, b.grad)
        for a, b in zip(baseline.parameters(), model.parameters(), strict=True)
    )


def test_cancel_error_limit_and_thread_cleanup():
    value = torch.ones(1)
    with OperatorDebugger(lambda: value.add_(9)) as debugger:
        pass
    assert value.item() == 1
    assert debugger.state == "cancelled"
    assert not debugger.thread.is_alive()

    def fail():
        value.add_(1)
        raise ValueError("model failed")

    with OperatorDebugger(fail) as failed, pytest.raises(RuntimeError, match="model failed"):
        failed.command("continue")
    assert not failed.thread.is_alive()
    with (
        OperatorDebugger(lambda: value.add(1).mul(2), max_operations=1) as bounded,
        pytest.raises(RuntimeError, match="limit reached"),
    ):
        bounded.command("continue")
    assert not bounded.thread.is_alive()


def test_caller_grad_mode_and_empty_execution():
    x = torch.ones(1, requires_grad=True)
    with torch.no_grad(), OperatorDebugger(lambda: x * 2) as debugger:
        debugger.command("continue")
        assert not debugger.result.requires_grad
    with OperatorDebugger(lambda: 7) as no_ops:
        assert no_ops.state == "finished" and no_ops.result == 7


def test_timeout_does_not_claim_the_worker_stopped():
    import threading

    gate = threading.Event()
    value = torch.ones(1)

    def run():
        gate.wait()
        return value.add_(1)

    debugger = OperatorDebugger(run, timeout=0.02)
    try:
        with pytest.raises(TimeoutError, match="still running"):
            debugger.start()
        with pytest.raises(TimeoutError, match="has not stopped"):
            debugger.close()
        assert debugger.thread.is_alive()
    finally:
        gate.set()
        debugger.thread.join(timeout=2)
        debugger.close()
    assert not debugger.thread.is_alive()
    assert value.item() == 1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires Apple MPS")
def test_mps_operator_and_gradient_execution():
    from ldb.gradients import GradientTracer

    model = torch.nn.Linear(4, 2).to("mps")
    x = torch.ones(2, 4, device="mps")
    with torch.no_grad():
        expected = model(x).relu()
        with OperatorDebugger(lambda: model(x).relu()) as debugger:
            debugger.command("step")
            debugger.command("continue")
            assert torch.equal(expected, debugger.result)
    with GradientTracer(model, capture=True) as tracer:
        model(x).sum().backward()
    assert all(e["stats"]["device"] == "mps:0" for e in tracer.events)
    assert tracer.tensors


def test_registered_fused_operator_is_one_visible_boundary():
    @torch.library.custom_op("ldb_debugger_tests::fused", mutates_args=())
    def fused(x: torch.Tensor) -> torch.Tensor:
        return x.sin().cos()

    value = torch.ones(2)
    with OperatorDebugger(lambda: fused(value)) as debugger:
        assert debugger.inspect()["current"]["operator"] == "ldb_debugger_tests.fused.default"
        assert debugger.command("step")["state"] == "finished"
        assert debugger.count == 1
        assert torch.equal(debugger.result, value.sin().cos())
