"""Bootstrap LDB inside an already-running serving worker."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any


def bootstrap_payload(path_value: str) -> None:
    """Install runtime hooks from an injected JSON payload and report the outcome."""

    path = Path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    status_path = Path(payload["status_path"])
    try:
        agents = install_into_running_process(
            backend=str(payload["backend"]),
            model_id=payload.get("model_id"),
            config=dict(payload.get("config", {})),
        )
        status = {
            "ok": True,
            "pid": os.getpid(),
            "agents": [agent.descriptor.model_dump(mode="json") for agent in agents],
        }
    except Exception as exc:
        status = {
            "ok": False,
            "pid": os.getpid(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    temporary = status_path.with_suffix(status_path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(status_path)


def install_into_running_process(
    *, backend: str, model_id: str | None, config: dict[str, Any]
) -> list[Any]:
    if backend == "vllm":
        from ldb.adapters.vllm import install_running_runtimes

        return install_running_runtimes(model_id=model_id, config=config)
    if backend == "sglang":
        from ldb.adapters.sglang import install_running_runtimes

        return install_running_runtimes(model_id=model_id, config=config)
    raise ValueError(f"Unsupported hot-attach backend: {backend}")
