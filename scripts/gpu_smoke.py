"""Run an end-to-end LDB smoke test against a local GPU serving runtime."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("vllm", "sglang"))
    parser.add_argument("model")
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="ldb-gpu-smoke-") as temporary:
        root = Path(temporary)
        agents = root / "agents"
        runs = root / "runs"
        log_path = root / "server.log"
        environment = os.environ.copy()
        environment.update(
            {
                "LDB_ENABLE": "1",
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
        try:
            _wait_ready(args.port, server, log_path)
            if args.mode == "eager":
                _ldb(agents, "arm 'block.0.residual.post'")
                _completion(args.port, args.model)
                events = _ldb(agents, "events")
                if "block.0.residual.post" not in events:
                    raise RuntimeError(
                        "Eager smoke test did not receive the normalized trace event"
                    )
            else:
                hello = _ldb(agents, "status")
                for capability in ("trace.cuda_graph", "patch.cuda_graph"):
                    if capability not in hello:
                        raise RuntimeError(
                            f"Agent did not advertise CUDA Graph capability {capability}"
                        )
                _ldb(agents, "arm 'block.0.residual.post'")
                baseline = _completion(args.port, args.model)
                events = _ldb(agents, "events")
                if "block.0.residual.post" not in events or "cuda_graph" not in events:
                    raise RuntimeError("CUDA Graph replay did not update the trace summary slot")
                _ldb(agents, "patch-set graph-ablate block.0.residual.post ablate")
                patched = _completion(args.port, args.model)
                if baseline["choices"] == patched["choices"]:
                    raise RuntimeError(
                        "CUDA Graph replay did not observe the control-buffer update"
                    )
                status = _ldb(agents, "patches")
                if "graph-ablate" not in status:
                    raise RuntimeError("CUDA Graph patch was not retained by the live agent")
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=10)


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
        return json.loads(response.read())


def _ldb(agents: Path, command: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "ldb.cli", "attach", str(agents), "-e", command],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.stdout


if __name__ == "__main__":
    main()
