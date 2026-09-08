import copy
import json

import pytest
import torch
from safetensors.torch import load_file

from ldb.gradients import GradientTracer


def test_tracer_preserves_training_gradients_and_updates(tmp_path):
    torch.manual_seed(0)
    reference = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
    observed = copy.deepcopy(reference)
    inputs = torch.randn(5, 3)
    optimizers = [torch.optim.SGD(m.parameters(), lr=0.1) for m in (reference, observed)]
    tracer = GradientTracer(observed, modules=["0"], capture=True)
    reference(inputs).square().sum().backward()
    with tracer:
        tracer.mark_step(0)
        observed(inputs).square().sum().backward()
        states = tracer.snapshot("before_update")
        assert set(states.values()) == {"present"}
        for a, b in zip(reference.parameters(), observed.parameters(), strict=True):
            assert torch.equal(a.grad, b.grad)
        optimizers[1].step()
    optimizers[0].step()
    assert all(
        torch.equal(a, b)
        for a, b in zip(reference.parameters(), observed.parameters(), strict=True)
    )
    assert {e["kind"] for e in tracer.events} >= {
        "incoming_gradient",
        "accumulated_gradient",
        "activation",
        "activation_gradient",
    }
    count = len(tracer.events)
    observed.zero_grad()
    observed(inputs).sum().backward()
    assert len(tracer.events) == count
    report = tracer.save(tmp_path / "trace")
    tensors = load_file(tmp_path / "trace/tensors.safetensors")
    assert all(e["tensor_key"] in tensors for e in report["events"])
    assert json.loads((tmp_path / "trace/trace.json").read_text()) == report


def test_accumulation_scale_and_nonfinite_are_distinct():
    model = torch.nn.Linear(1, 1, bias=False)
    tracer = GradientTracer(model, norm_threshold=10)
    with tracer:
        tracer.mark_step("microbatch", loss_scale=8)
        for _ in range(2):
            (model(torch.ones(1, 1)).sum() * 8).backward()
    incoming = [e["stats"]["norm"] for e in tracer.events if e["kind"] == "incoming_gradient"]
    accumulated = [e["stats"]["norm"] for e in tracer.events if e["kind"] == "accumulated_gradient"]
    assert incoming == [8, 8] and accumulated == [8, 16]
    assert tracer.events[-1]["alerts"] == ["norm_threshold"]
    assert all(e["loss_scale"] == 8 for e in tracer.events)
    model.zero_grad()
    with GradientTracer(model) as bad:
        (model(torch.ones(1, 1)).sum() * float("nan")).backward()
    assert torch.isnan(model.weight.grad).all()
    assert all(e["alerts"] == ["nonfinite"] for e in bad.events)
    json.dumps(bad.events, allow_nan=False)


def test_frozen_unused_aliases_budgets_and_failure_cleanup(tmp_path):
    model = torch.nn.Module()
    model.register_parameter("used", torch.nn.Parameter(torch.ones(2)))
    model.register_parameter("alias", model.used)
    model.register_parameter("unused", torch.nn.Parameter(torch.ones(2)))
    model.register_parameter("frozen", torch.nn.Parameter(torch.ones(2), requires_grad=False))
    tracer = GradientTracer(model, max_events=2, capture=True, max_tensor_bytes=0)
    with pytest.raises(RuntimeError, match="body failed"), tracer:
        model.used.sum().backward()
        assert tracer.snapshot() == {"used": "present", "unused": "none", "frozen": "frozen"}
        raise RuntimeError("body failed")
    assert not tracer.active and not tracer.handles
    assert len(tracer.events) == 2 and tracer.dropped_events == 1
    assert tracer.skipped_tensors == 2 and not tracer.tensors
    tracer.save(tmp_path / "bounded")
    with pytest.raises(ValueError, match="single-use"), tracer:
        pass
    with pytest.raises(ValueError, match="matches no names"):
        GradientTracer(model, parameters=["not_a_parameter"])
