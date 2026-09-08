import subprocess
import sys
from pathlib import Path

import pytest

from ldb.live import AgentCapability, AgentDescriptor, GpuSmokeReport


def _descriptor(rank: int, world_size: int) -> AgentDescriptor:
    return AgentDescriptor(
        backend="vllm",
        backend_version="0.28.0",
        instance_id="gpu-smoke",
        model_id="model",
        pid=100 + rank,
        rank=rank,
        world_size=world_size,
        socket_path=f"/tmp/worker-{rank}.sock",
        capabilities=(AgentCapability.HOOK_DISCOVERY,),
    )


def test_gpu_report_records_hot_attachment_evidence() -> None:
    report = GpuSmokeReport(
        ldb_commit="commit",
        backend="vllm",
        backend_version="0.28.0",
        model_id="model",
        mode="eager",
        attachment="hot",
        injected_pids=(101,),
        python_version="3.11",
        torch_version="2.0",
        cuda_version="12.0",
        devices=({"index": 0, "name": "GPU", "capability": (8, 0)},),
        agents=(_descriptor(0, 1),),
        trace_observed=True,
        patch_observed=True,
    )

    assert report.schema_version == "ldb.gpu-smoke/v1alpha2"
    assert report.attachment == "hot"
    assert report.patch_observed


def test_gpu_report_rejects_unproven_hot_attachment() -> None:
    with pytest.raises(ValueError, match="live patch"):
        GpuSmokeReport(
            ldb_commit="commit",
            backend="vllm",
            backend_version="0.28.0",
            model_id="model",
            mode="eager",
            attachment="hot",
            injected_pids=(101,),
            python_version="3.11",
            torch_version="2.0",
            devices=({"index": 0, "name": "GPU", "capability": (8, 0)},),
            agents=(_descriptor(0, 1),),
            trace_observed=True,
        )


def test_gpu_smoke_parses_harness_options_after_model() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "gpu_smoke.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "vllm",
            "model",
            "--attachment",
            "hot",
            "--mode",
            "graph",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "hot attachment requires eager mode" in result.stderr


def test_eager_evidence_checks_only_owners_and_requires_every_owner() -> None:
    from scripts.gpu_smoke import _trace_on_owners

    event = {"hook": "block.0.residual.post", "patch_ids": ["p"], "patched_summary": {}}
    assert _trace_on_owners([[event], [], [event]], (0, 2), event["hook"], "p")
    assert not _trace_on_owners([[event], [], []], (0, 2), event["hook"], "p")
    assert not _trace_on_owners([[event]], (), event["hook"])
    assert not _trace_on_owners([[event]], (0,), event["hook"], "another")


@pytest.mark.parametrize("outcome", ["restored", "no_effect", "wrong_restore"])
def test_eager_harness_requires_effect_and_always_removes_patch(tmp_path, monkeypatch, outcome):
    from scripts import gpu_smoke

    calls = []
    baseline = {"choices": [{"text": "base"}]}
    patched = baseline if outcome == "no_effect" else {"choices": [{"text": "patched"}]}
    restored = patched if outcome == "wrong_restore" else baseline
    responses = iter([baseline, patched, restored])
    event = {"hook": "block.0.residual.post"}
    trace = iter(
        [
            [[event], []],
            [[{**event, "patch_ids": ["smoke-ablate"], "patched_summary": {"norm": 0}}], []],
        ]
    )

    def ldb_json(agents, command):
        if command == "hooks":
            return [[event["hook"]], ["block.1.residual.post"]]
        return next(trace)

    monkeypatch.setattr(gpu_smoke, "_ldb_json", ldb_json)
    monkeypatch.setattr(gpu_smoke, "_ldb", lambda agents, command: calls.append(command))
    monkeypatch.setattr(gpu_smoke, "_completion", lambda *args: next(responses))
    if outcome == "restored":
        assert gpu_smoke._exercise_eager(tmp_path, 1, "model", tmp_path) == (baseline, patched)
        assert '"restoration_verified": true' in (tmp_path / "eager-evidence.json").read_text()
    else:
        message = "did not change" if outcome == "no_effect" else "did not restore"
        with pytest.raises(RuntimeError, match=message):
            gpu_smoke._exercise_eager(tmp_path, 1, "model", tmp_path)
    assert calls[-1] == "patch-remove smoke-ablate"


@pytest.mark.parametrize(
    "outcome", ["restored", "eager_trace", "missing_owner", "no_effect", "wrong_restore"]
)
def test_graph_harness_checks_owning_ranks_and_restoration(tmp_path, monkeypatch, outcome):
    from scripts import gpu_smoke

    calls = []
    baseline = {"choices": [{"text": "base"}]}
    patched = baseline if outcome == "no_effect" else {"choices": [{"text": "patched"}]}
    responses = iter([baseline, patched, patched if outcome == "wrong_restore" else baseline])
    event = {"hook": "block.0.residual.post", "source": "cuda_graph"}
    baseline_events = [[event], [], [event]]
    if outcome == "eager_trace":
        baseline_events = [[{**event, "source": "eager"}], [], [event]]
    patched_events = [[event], [], [] if outcome == "missing_owner" else [event]]
    trace = iter([baseline_events, patched_events])

    def ldb_json(agents, command):
        if command == "hooks":
            return [[event["hook"]], ["block.1.residual.post"], [event["hook"]]]
        if command == "patches":
            return [[{"patch_id": "graph-ablate"}], [], [{"patch_id": "graph-ablate"}]]
        return next(trace)

    monkeypatch.setattr(gpu_smoke, "_ldb_json", ldb_json)
    monkeypatch.setattr(gpu_smoke, "_ldb", lambda agents, command: calls.append(command))
    monkeypatch.setattr(gpu_smoke, "_completion", lambda *args: next(responses))
    if outcome == "restored":
        assert gpu_smoke._exercise_graph(tmp_path, 1, "model", tmp_path) == (baseline, patched)
        assert '"restoration_verified": true' in (tmp_path / "graph-evidence.json").read_text()
    else:
        with pytest.raises(RuntimeError, match="owning|control-buffer|restore"):
            gpu_smoke._exercise_graph(tmp_path, 1, "model", tmp_path)
    if outcome == "eager_trace":
        assert not any(command.startswith("patch-set") for command in calls)
    else:
        assert calls[-1] == "patch-remove graph-ablate"
