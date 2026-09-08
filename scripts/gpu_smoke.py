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

from ldb.injector import inject_process
from ldb.live import AgentDescriptor, GpuDevice, GpuSmokeReport


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("vllm", "sglang"))
    parser.add_argument("model")
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--attachment", choices=("prepared", "hot"), default="prepared")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--artifacts", type=Path)
    args, server_args = parser.parse_known_args()
    if server_args[:1] == ["--"]:
        server_args = server_args[1:]
    if args.attachment == "hot" and args.mode != "eager":
        parser.error(
            "hot attachment requires eager mode because existing CUDA Graphs are immutable"
        )

    temporary = tempfile.TemporaryDirectory(prefix="ldb-gpu-smoke-")
    try:
        root = args.artifacts or Path(temporary.name)
        root.mkdir(parents=True, exist_ok=True)
        agents = root / "agents"
        runs = root / "runs"
        log_path = root / "server.log"
        environment = os.environ.copy()
        for variable in (
            "LDB_ENABLE",
            "LDB_INSTANCE_ID",
            "LDB_SOCKET_DIR",
            "LDB_OUTPUT_DIR",
            "LDB_CUDA_GRAPH",
            "VLLM_PLUGINS",
            "SGLANG_PLUGINS",
        ):
            environment.pop(variable, None)
        if args.attachment == "prepared":
            environment.update(
                {
                    "LDB_ENABLE": "1",
                    "LDB_INSTANCE_ID": "gpu-smoke",
                    "LDB_SOCKET_DIR": str(agents),
                    "LDB_OUTPUT_DIR": str(runs),
                }
            )
        if args.backend == "vllm":
            if args.attachment == "prepared":
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
            if args.attachment == "prepared":
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
        if args.mode == "graph" and args.attachment == "prepared":
            environment["LDB_CUDA_GRAPH"] = "1"
        command.extend(server_args)

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
        injected_pids: tuple[int, ...] = ()
        patch_observed = False
        try:
            _wait_ready(args.port, server, log_path)
            if args.attachment == "hot":
                injected_pids = _hot_attach_workers(
                    server.pid,
                    backend=args.backend,
                    model=args.model,
                    agents=agents,
                    runs=runs,
                )
            statuses = _ldb_json(agents, "status")
            descriptors = tuple(
                AgentDescriptor.model_validate(status["descriptor"]) for status in statuses
            )
            if args.mode == "eager":
                baseline, patched = _exercise_eager(agents, args.port, args.model, root)
                patch_observed = True
            else:
                for capability in ("trace.cuda_graph", "patch.cuda_graph"):
                    if not all(capability in descriptor.capabilities for descriptor in descriptors):
                        raise RuntimeError(
                            f"Every agent must advertise CUDA Graph capability {capability}"
                        )
                baseline, patched = _exercise_graph(agents, args.port, args.model, root)
                patch_observed = True
            _write_report(
                root / "result.json",
                backend=args.backend,
                model=args.model,
                mode=args.mode,
                attachment=args.attachment,
                injected_pids=injected_pids,
                descriptors=descriptors,
                baseline=baseline,
                patched=patched,
                patch_observed=patch_observed,
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


def _trace_on_owners(
    events: list[Any],
    owners: tuple[int, ...],
    hook: str,
    patch_id: str | None = None,
    *,
    source: str | None = None,
) -> bool:
    return bool(owners) and all(
        index < len(events)
        and any(
            event.get("hook") == hook
            and (source is None or event.get("source") == source)
            and (
                patch_id is None
                or (
                    patch_id in event.get("patch_ids", [])
                    and event.get("patched_summary") is not None
                )
            )
            for event in events[index]
        )
        for index in owners
    )


def _exercise_eager(
    agents: Path, port: int, model: str, root: Path
) -> tuple[dict[str, object], dict[str, object]]:
    hook, patch_id = "block.0.residual.post", "smoke-ablate"
    inventories = _ldb_json(agents, "hooks")
    owners = tuple(index for index, hooks in enumerate(inventories) if hook in hooks)
    if not owners:
        raise RuntimeError("No rank owns the eager acceptance hook")
    evidence: dict[str, Any] = {
        "hook": hook,
        "owner_indices": owners,
        "hook_inventories": inventories,
    }

    def record(key: str, value: Any) -> None:
        evidence[key] = value
        (root / "eager-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")

    _ldb(agents, f"arm '{hook}'")
    baseline = _completion(port, model)
    record("baseline", baseline)
    events = _ldb_json(agents, "events")
    record("baseline_events", events)
    if not _trace_on_owners(events, owners, hook):
        raise RuntimeError("Every owning rank must produce the eager trace event")
    _ldb(agents, "clear")
    _ldb(agents, f"patch-set {patch_id} {hook} ablate")
    try:
        patched = _completion(port, model)
        record("patched", patched)
        events = _ldb_json(agents, "events")
        record("patched_events", events)
        if not _trace_on_owners(events, owners, hook, patch_id):
            raise RuntimeError("Eager patch did not execute on every owning rank")
        if baseline["choices"] == patched["choices"]:
            raise RuntimeError("Eager patch did not change completion or token log probability")
    finally:
        _ldb(agents, f"patch-remove {patch_id}")
    restored = _completion(port, model)
    record("restored", restored)
    if baseline["choices"] != restored["choices"]:
        raise RuntimeError("Eager patch removal did not restore deterministic baseline choices")
    record("restoration_verified", True)
    return baseline, patched


def _exercise_graph(
    agents: Path, port: int, model: str, root: Path
) -> tuple[dict[str, object], dict[str, object]]:
    hook, patch_id = "block.0.residual.post", "graph-ablate"
    inventories = _ldb_json(agents, "hooks")
    owners = tuple(index for index, hooks in enumerate(inventories) if hook in hooks)
    if not owners:
        raise RuntimeError("No rank owns the CUDA Graph acceptance hook")
    evidence: dict[str, Any] = {
        "hook": hook,
        "owner_indices": owners,
        "hook_inventories": inventories,
    }

    def record(key: str, value: Any) -> None:
        evidence[key] = value
        (root / "graph-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")

    _ldb(agents, f"arm '{hook}'")
    baseline = _completion(port, model)
    record("baseline", baseline)
    events = _ldb_json(agents, "events")
    record("baseline_events", events)
    if not _trace_on_owners(events, owners, hook, source="cuda_graph"):
        raise RuntimeError("CUDA Graph replay did not update every owning rank's trace slot")
    _ldb(agents, "clear")
    _ldb(agents, f"patch-set {patch_id} {hook} ablate")
    try:
        patched = _completion(port, model)
        record("patched", patched)
        events = _ldb_json(agents, "events")
        record("patched_events", events)
        if not _trace_on_owners(events, owners, hook, source="cuda_graph"):
            raise RuntimeError("Patched CUDA Graph replay did not update every owning rank")
        if baseline["choices"] == patched["choices"]:
            raise RuntimeError("CUDA Graph replay did not observe the control-buffer update")
        patches = _ldb_json(agents, "patches")
        record("patches", patches)
        if not all(
            index < len(patches)
            and any(patch.get("patch_id") == patch_id for patch in patches[index])
            for index in owners
        ):
            raise RuntimeError("CUDA Graph patch was not retained by every owning agent")
    finally:
        _ldb(agents, f"patch-remove {patch_id}")
    restored = _completion(port, model)
    record("restored", restored)
    if baseline["choices"] != restored["choices"]:
        raise RuntimeError("CUDA Graph patch removal did not restore baseline choices")
    record("restoration_verified", True)
    return baseline, patched


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


def _hot_attach_workers(
    server_pid: int,
    *,
    backend: Literal["vllm", "sglang"],
    model: str,
    agents: Path,
    runs: Path,
) -> tuple[int, ...]:
    attempts: list[str] = []
    attached: list[int] = []
    for pid in _worker_candidates(server_pid, backend):
        try:
            inject_process(
                pid,
                backend=backend,
                model_id=model,
                instance_id="gpu-smoke",
                socket_dir=agents,
                output_dir=runs,
                timeout=10.0,
            )
            attached.append(pid)
        except Exception as exc:
            attempts.append(f"PID {pid}: {exc}")
        descriptors = _local_descriptors(agents)
        if descriptors and _complete_group(descriptors):
            return tuple(attached)
    detail = "; ".join(attempts) or "no descendant CPython processes were found"
    raise RuntimeError(f"Could not hot-attach a complete serving worker group: {detail}")


def _worker_candidates(server_pid: int, backend: Literal["vllm", "sglang"]) -> tuple[int, ...]:
    descendants = _descendant_pids(server_pid)
    candidates = []
    for pid in (server_pid, *descendants):
        process = Path(f"/proc/{pid}")
        try:
            executable = (process / "exe").resolve().name.lower()
            command = (
                (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            )
        except OSError:
            continue
        if "python" not in executable:
            continue
        text = command.lower()
        keywords = (
            ("gpu_model_runner", "enginecore", "worker", "vllm")
            if backend == "vllm"
            else ("modelrunner", "scheduler", "worker", "sglang")
        )
        score = sum(
            (len(keywords) - index) * (keyword in text) for index, keyword in enumerate(keywords)
        )
        candidates.append((score, pid))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return tuple(pid for _, pid in candidates)


def _descendant_pids(root_pid: int) -> tuple[int, ...]:
    pending = [root_pid]
    seen = {root_pid}
    descendants = []
    while pending:
        parent = pending.pop()
        children: list[int] = []
        for children_path in Path(f"/proc/{parent}/task").glob("*/children"):
            try:
                children.extend(int(value) for value in children_path.read_text().split())
            except (OSError, ValueError):
                continue
        for child in children:
            if child in seen:
                continue
            seen.add(child)
            descendants.append(child)
            pending.append(child)
    return tuple(descendants)


def _local_descriptors(directory: Path) -> tuple[AgentDescriptor, ...]:
    descriptors = []
    for path in directory.glob("*.json"):
        try:
            descriptor = AgentDescriptor.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if Path(descriptor.socket_path).is_socket():
            descriptors.append(descriptor)
    return tuple(descriptors)


def _complete_group(descriptors: tuple[AgentDescriptor, ...]) -> bool:
    world_sizes = {descriptor.world_size for descriptor in descriptors}
    if len(world_sizes) != 1:
        return False
    world_size = next(iter(world_sizes))
    ranks = {descriptor.rank for descriptor in descriptors}
    return len(descriptors) == world_size and ranks == set(range(world_size))


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


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_report(
    path: Path,
    *,
    backend: Literal["vllm", "sglang"],
    model: str,
    mode: Literal["eager", "graph"],
    attachment: Literal["prepared", "hot"],
    injected_pids: tuple[int, ...],
    descriptors: tuple[AgentDescriptor, ...],
    baseline: dict[str, object] | None,
    patched: dict[str, object] | None,
    patch_observed: bool,
) -> None:
    import torch

    report = GpuSmokeReport(
        ldb_commit=os.environ.get("GITHUB_SHA", "unknown"),
        backend=backend,
        backend_version=importlib.metadata.version(backend),
        model_id=model,
        mode=mode,
        attachment=attachment,
        injected_pids=injected_pids,
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        devices=tuple(_gpu_device(index) for index in range(torch.cuda.device_count())),
        agents=descriptors,
        trace_observed=True,
        patch_observed=patch_observed,
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
