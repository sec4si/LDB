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
