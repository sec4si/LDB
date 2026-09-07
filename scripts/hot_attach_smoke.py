"""Exercise shared-object injection and the live control handshake on Linux."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from ldb.client import AgentGroupClient
from ldb.injector import inject_process


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for backend in ("vllm", "sglang"):
        with tempfile.TemporaryDirectory(prefix=f"ldb-{backend}-smoke-") as temporary:
            target = subprocess.Popen(
                [sys.executable, str(root / "scripts/hot_attach_smoke_target.py"), backend],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                assert target.stdout is not None
                ready = target.stdout.readline().strip()
                if not ready.startswith("ready "):
                    raise RuntimeError(f"Hot-attach target failed to start: {ready}")
                directory = Path(temporary)
                result = inject_process(
                    target.pid,
                    backend=backend,
                    socket_dir=directory / "agents",
                    output_dir=directory / "runs",
                    timeout=15,
                )
                socket = Path(result["agents"][0]["socket_path"])
                status = AgentGroupClient.connect(socket).call("status")
                if status[0]["descriptor"]["backend"] != backend:
                    raise RuntimeError("Injected agent reported the wrong backend")
                print(f"{backend}: hot attachment and control handshake passed")
            finally:
                target.terminate()
                try:
                    target.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    target.kill()
                    target.wait(timeout=5)


if __name__ == "__main__":
    main()
