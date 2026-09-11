"""Budget infeasibility, surrogate optimization and capability gate regressions."""

from copy import deepcopy

import pytest

from ldb.experiments.budget_builder.common import validate_spec
from ldb.experiments.budget_builder.run import choose, grade
from ldb.experiments.expert_compaction.selection import allocate_layers


@pytest.fixture
def spec():
    return dict(
        weight_bytes=1000,
        memory_bytes=2000,
        context_tokens=64,
        max_new_tokens=8,
        candidate_limit=2,
        timeout_seconds=30,
        concurrency=1,
        memory_headroom=0.1,
        min_task_accuracy=0.75,
        max_task_drop=0.125,
        max_delta_nll=0.03,
        max_delta_nll_upper=0.05,
        min_decode_tps=1.0,
    )


def measurement(accuracy=1.0, nll=1.0, weight=900, memory=1500):
    tasks = [
        {"id": str(i), "kind": "arithmetic", "correct": i < round(accuracy * 8)} for i in range(8)
    ]
    accuracy = sum(t["correct"] for t in tasks) / len(tasks)
    r = {
        "tasks": tasks,
        "documents": [{"nll": nll, "tokens": 10} for _ in range(8)],
        "task_accuracy": accuracy,
        "bits_per_byte": 2.0,
        "weight_file_bytes": weight,
        "decode_tps": 10.0,
        "capacity_context_tokens": 64,
    }
    return {"returncode": 0, "peak_physical_footprint_bytes": memory, "result": r}


def test_capability_floor_rejects_tiny_weak_model(spec):
    baseline = measurement(accuracy=0.125)["result"]
    outcome = grade(measurement(accuracy=0.125, weight=10), baseline, spec)
    assert not outcome["pass"] and "task_floor" in outcome["reasons"]


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"weight": 1001}, "weight"),
        ({"memory": 1801}, "memory"),
        ({"nll": 1.1}, "text_retention"),
        ({"accuracy": 0.75}, "task_retention"),
    ],
)
def test_independent_gates(spec, change, reason):
    result = grade(measurement(**change), measurement()["result"], spec)
    assert reason in result["reasons"]


def test_no_feasible_candidate_is_not_forced(spec):
    rows = [
        {
            "candidate": {"id": "too-weak"},
            "process": measurement(0.1),
            "grade": grade(measurement(0.1), measurement()["result"], spec),
        }
    ]
    assert choose(rows) is None


def test_rank_by_task_before_size(spec):
    rows = []
    for name, accuracy, size in [("small", 0.875, 100), ("strong", 1.0, 900)]:
        process = measurement(accuracy, weight=size)
        rows.append(
            {
                "candidate": {"id": name},
                "process": process,
                "grade": grade(process, measurement()["result"], spec),
            }
        )
    assert choose(rows) == "strong"


def test_dp_beats_greedy_and_rejects_discrete_impossibility():
    costs = [[0.0, 0.01, 0.9], [0.0, 0.2, 0.21]]
    loss, allocation = allocate_layers(costs, [0, 4, 8], 8)
    assert loss == pytest.approx(0.21) and sum(allocation) == 8
    with pytest.raises(ValueError):
        allocate_layers(costs, [0, 4, 8], 7)


def test_bad_metrics_and_configuration_fail_closed(spec):
    a = measurement()
    a["result"]["documents"][0]["nll"] = float("nan")
    assert not grade(a, measurement()["result"], spec)["pass"]
    invalid = deepcopy(spec)
    invalid["memory_headroom"] = float("nan")
    with pytest.raises(ValueError):
        validate_spec(invalid)
    invalid = deepcopy(spec)
    invalid["concurrency"] = 2
    with pytest.raises(ValueError):
        validate_spec(invalid)


def test_physical_memory_watchdog_and_fresh_worker(tmp_path):
    import os
    import platform
    import sys

    from ldb.experiments.budget_builder.resources import run_worker

    if platform.system() != "Darwin":
        pytest.skip("macOS footprint adapter")
    output = tmp_path / "ok.json"
    command = [
        sys.executable,
        "-c",
        "import json,sys,time; time.sleep(.1); "
        'open(sys.argv[1],"w").write(json.dumps({"ok":True}))',
        str(output),
    ]
    result = run_worker(command, tmp_path, output, None, 5, dict(os.environ))
    assert result["returncode"] == 0 and result["result"]["ok"] and result["samples"] > 0
    output = tmp_path / "killed.json"
    result = run_worker(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        tmp_path,
        output,
        1,
        5,
        dict(os.environ),
    )
    assert result["returncode"] != 0 and result["reason"] == "memory_watchdog_exceeded"


def test_exact_accuracy_and_typed_thresholds(spec):
    spec["min_task_accuracy"] = 1.0
    validate_spec(spec)
    assert grade(measurement(), measurement()["result"], spec)["pass"]
    assert not grade(measurement(0.875), measurement()["result"], spec)["pass"]
    for key in ["min_task_accuracy", "concurrency", "memory_headroom"]:
        invalid = dict(spec, **{key: True})
        with pytest.raises(ValueError):
            validate_spec(invalid)


@pytest.mark.parametrize(
    "fault", ["summary", "duplicate", "identity", "nan_speed", "zero_memory", "fractional_tokens"]
)
def test_untrustworthy_measurement_rejected(spec, fault):
    process = measurement()
    result = process["result"]
    if fault == "summary":
        result["tasks"][0]["correct"] = False
    elif fault == "duplicate":
        result["tasks"][1]["id"] = result["tasks"][0]["id"]
    elif fault == "identity":
        result["tasks"][0]["kind"] = "extract"
    elif fault == "nan_speed":
        result["decode_tps"] = float("nan")
    elif fault == "zero_memory":
        process["peak_physical_footprint_bytes"] = 0
    else:
        result["documents"][0]["tokens"] = 10.5
    assert grade(process, measurement()["result"], spec)["reasons"] == ["invalid_measurements"]


def test_delivery_ignores_caches_and_is_atomic(tmp_path, spec):
    from ldb.experiments.budget_builder.common import sha
    from ldb.experiments.budget_builder.delivery import export_delivery

    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"fixture weights")
    (source / "__pycache__").mkdir()
    (source / "__pycache__/module.pyc").write_bytes(b"volatile")
    (source / "budget-model.json").write_text("{}")
    host = {
        "backend": "torch",
        "weight_sha256": {"model.safetensors": sha(source / "model.safetensors")},
    }
    target = tmp_path / "delivery"
    contract = export_delivery(
        source, target, host, spec, {"operation": "original"}, {"pass": True}
    )
    assert not (target / "__pycache__").exists()
    assert all("__pycache__" not in name for name in contract["files"])
    with pytest.raises(FileExistsError):
        export_delivery(source, target, host, spec, {"operation": "original"}, {"pass": True})
    host["weight_sha256"]["model.safetensors"] = "0" * 64
    with pytest.raises(ValueError, match="weights"):
        export_delivery(
            source, tmp_path / "bad", host, spec, {"operation": "original"}, {"pass": True}
        )
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob("bad.staging-*"))
