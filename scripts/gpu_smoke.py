"""Run an end-to-end LDB smoke test against a local GPU serving runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal, cast

from ldb.live import AgentDescriptor, GpuDevice, GpuSmokeReport


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("vllm", "sglang"))
    parser.add_argument("model")
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    temporary = tempfile.TemporaryDirectory(prefix="ldb-gpu-smoke-")
    try:
        root = args.artifacts or Path(temporary.name)
        root.mkdir(parents=True, exist_ok=True)
        agents = root / "agents"
        runs = root / "runs"
        log_path = root / "server.log"
        environment = os.environ.copy()
        environment.update(
            {
                "LDB_ENABLE": "1",
                "LDB_INSTANCE_ID": "gpu-smoke",
                "LDB_SOCKET_DIR": str(agents),
                "LDB_OUTPUT_DIR": str(runs),
            }
        )
        if args.backend == "vllm":
            environment["VLLM_PLUGINS"] = "ldb"
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                args.model,
                "--port",
                str(args.port),
            ]
            if args.mode == "eager":
                command.append("--enforce-eager")
        else:
            environment["SGLANG_PLUGINS"] = "ldb"
            command = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                args.model,
                "--port",
                str(args.port),
            ]
            if args.mode == "eager":
                command.append("--disable-cuda-graph")
        if args.mode == "graph":
            environment["LDB_CUDA_GRAPH"] = "1"
        command.extend(args.server_args)

        with log_path.open("wb") as log:
            server = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        baseline: dict[str, object] | None = None
        patched: dict[str, object] | None = None
        try:
            _wait_ready(args.port, server, log_path)
            statuses = _ldb_json(agents, "status")
            descriptors = tuple(
                AgentDescriptor.model_validate(status["descriptor"]) for status in statuses
            )
            if args.mode == "eager":
                _ldb(agents, "arm 'block.0.residual.post'")
                _completion(args.port, args.model)
                events = _events(_ldb_json(agents, "events"))
                if not any(event.get("hook") == "block.0.residual.post" for event in events):
                    raise RuntimeError(
                        "Eager smoke test did not receive the normalized trace event"
                    )
            else:
                for capability in ("trace.cuda_graph", "patch.cuda_graph"):
                    if not all(
                        capability in descriptor.capabilities for descriptor in descriptors
                    ):
                        raise RuntimeError(
                            f"Every agent must advertise CUDA Graph capability {capability}"
                        )
                _ldb(agents, "arm 'block.0.residual.post'")
                baseline = _completion(args.port, args.model)
                events = _events(_ldb_json(agents, "events"))
                if not any(
                    event.get("hook") == "block.0.residual.post"
                    and event.get("source") == "cuda_graph"
                    for event in events
                ):
                    raise RuntimeError("CUDA Graph replay did not update the trace summary slot")
                _ldb(agents, "patch-set graph-ablate block.0.residual.post ablate")
                patched = _completion(args.port, args.model)
                if baseline["choices"] == patched["choices"]:
                    raise RuntimeError(
                        "CUDA Graph replay did not observe the control-buffer update"
                    )
                patches = _ldb_json(agents, "patches")
                if not all(
                    any(patch.get("patch_id") == "graph-ablate" for patch in rank_patches)
                    for rank_patches in patches
                ):
                    raise RuntimeError("CUDA Graph patch was not retained by the live agent")
            _write_report(
                root / "result.json",
                backend=args.backend,
                model=args.model,
                mode=args.mode,
                descriptors=descriptors,
                baseline=baseline,
                patched=patched,
            )
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=10)
    finally:
        temporary.cleanup()


def _wait_ready(port: int, server: subprocess.Popen[bytes], log_path: Path) -> None:
    deadline = time.monotonic() + 600
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"Server exited with {server.returncode}:\n"
                + log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError(f"Server did not become ready at {url}")


def _completion(port: int, model: str) -> dict[str, object]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": "The Eiffel Tower is in",
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 1,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"Completion failed with HTTP {response.status}")
        return cast(dict[str, object], json.loads(response.read()))


def _ldb(agents: Path, command: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "ldb.cli", "attach", str(agents), "-e", command],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout


def _ldb_json(agents: Path, command: str) -> list[Any]:
    return cast(list[Any], json.loads(_ldb(agents, command)))


def _events(rank_events: list[Any]) -> list[dict[str, Any]]:
    return [cast(dict[str, Any], event) for events in rank_events for event in events]


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_report(
    path: Path,
    *,
    backend: Literal["vllm", "sglang"],
    model: str,
    mode: Literal["eager", "graph"],
    descriptors: tuple[AgentDescriptor, ...],
    baseline: dict[str, object] | None,
    patched: dict[str, object] | None,
) -> None:
    import torch

    report = GpuSmokeReport(
        ldb_commit=os.environ.get("GITHUB_SHA", "unknown"),
        backend=backend,
        backend_version=importlib.metadata.version(backend),
        model_id=model,
        mode=mode,
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        devices=tuple(_gpu_device(index) for index in range(torch.cuda.device_count())),
        agents=descriptors,
        trace_observed=True,
        graph_patch_changed=(baseline != patched) if mode == "graph" else None,
        baseline_digest=_digest(baseline) if baseline is not None else None,
        patched_digest=_digest(patched) if patched is not None else None,
    )
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _gpu_device(index: int) -> GpuDevice:
    import torch

    major, minor = torch.cuda.get_device_capability(index)
    return GpuDevice(
        index=index,
        name=torch.cuda.get_device_name(index),
        capability=(major, minor),
    )


if __name__ == "__main__":
    main()
