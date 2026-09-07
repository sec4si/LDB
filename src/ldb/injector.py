"""Linux CPython shared-object injector used for true hot attachment."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

_INJECTOR_SOURCE = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

typedef int (*gil_ensure_fn)(void);
typedef void (*gil_release_fn)(int);
typedef int (*run_string_fn)(const char *);

static const char *bootstrap_code = BOOTSTRAP_CODE;

static void *ldb_bootstrap(void *unused) {
    (void)unused;
    gil_ensure_fn ensure = (gil_ensure_fn)dlsym(RTLD_DEFAULT, "PyGILState_Ensure");
    gil_release_fn release = (gil_release_fn)dlsym(RTLD_DEFAULT, "PyGILState_Release");
    run_string_fn run = (run_string_fn)dlsym(RTLD_DEFAULT, "PyRun_SimpleString");
    if (!ensure || !release || !run) return NULL;
    int state = ensure();
    run(bootstrap_code);
    release(state);
    return NULL;
}

__attribute__((constructor)) static void ldb_load(void) {
    pthread_t thread;
    if (pthread_create(&thread, NULL, ldb_bootstrap, NULL) == 0) pthread_detach(thread);
}
"""


def inject_process(
    pid: int,
    *,
    backend: str,
    model_id: str | None = None,
    model_revision: str | None = None,
    tokenizer_hash: str | None = None,
    instance_id: str = "default",
    socket_dir: Path | None = None,
    output_dir: Path | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Inject a deferred GIL-safe bootstrap thread into one live CPython process."""

    _validate_target(pid, backend)
    compiler = shutil.which("cc") or shutil.which("gcc")
    gdb = shutil.which("gdb")
    if compiler is None or gdb is None:
        raise RuntimeError("Hot attachment requires both a C compiler and gdb")
    package_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="ldb-inject-") as temporary_value:
        temporary = Path(temporary_value)
        payload_path = temporary / "payload.json"
        status_path = temporary / "status.json"
        source_path = temporary / "inject.c"
        library_path = temporary / "inject.so"
        config: dict[str, Any] = {"instance_id": instance_id, "graph_control": False}
        if model_revision is not None:
            config["model_revision"] = model_revision
        if tokenizer_hash is not None:
            config["tokenizer_hash"] = tokenizer_hash
        if socket_dir is not None:
            config["socket_dir"] = str(socket_dir.resolve())
        if output_dir is not None:
            config["output_dir"] = str(output_dir.resolve())
        payload_path.write_text(
            json.dumps(
                {
                    "backend": backend,
                    "model_id": model_id,
                    "config": config,
                    "status_path": str(status_path),
                }
            ),
            encoding="utf-8",
        )
        os.chmod(payload_path, 0o600)
        bootstrap_code = (
            f"import sys;sys.path.insert(0,{str(package_root)!r});"
            "from ldb.hot import bootstrap_payload;"
            f"bootstrap_payload({str(payload_path)!r})"
        )
        source = _INJECTOR_SOURCE.replace("BOOTSTRAP_CODE", json.dumps(bootstrap_code))
        source_path.write_text(source, encoding="utf-8")
        _run(
            [
                compiler,
                "-shared",
                "-fPIC",
                "-pthread",
                "-o",
                library_path,
                source_path,
                "-ldl",
            ]
        )
        _run(
            [
                gdb,
                "--batch",
                "--nx",
                "--pid",
                str(pid),
                "--eval-command",
                "set scheduler-locking off",
                "--eval-command",
                f'call (void *) dlopen("{library_path}", 2)',
                "--eval-command",
                "detach",
            ]
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if status_path.exists():
                result = json.loads(status_path.read_text(encoding="utf-8"))
                if not result.get("ok"):
                    raise RuntimeError(f"Injected bootstrap failed: {result.get('error')}")
                return dict(result)
            if not Path(f"/proc/{pid}").exists():
                raise RuntimeError("Target process exited during hot attachment")
            time.sleep(0.05)
        raise TimeoutError("Timed out waiting for the injected LDB agent")


def _validate_target(pid: int, backend: str) -> None:
    if platform.system() != "Linux":
        raise RuntimeError("True hot attachment is supported only on Linux")
    if backend not in {"vllm", "sglang"}:
        raise ValueError(f"Unsupported backend: {backend}")
    process = Path(f"/proc/{pid}")
    if pid <= 1 or not process.is_dir():
        raise ValueError(f"PID {pid} is not a running process")
    status = (process / "status").read_text(encoding="utf-8")
    owner_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
    owner = int(owner_line.split()[1])
    if owner != os.geteuid() and os.geteuid() != 0:
        raise PermissionError("Hot attachment requires the same effective UID or root")
    executable = (process / "exe").resolve().name.lower()
    if "python" not in executable:
        raise ValueError(f"PID {pid} is not a CPython process")


def _run(arguments: list[Any]) -> None:
    result = subprocess.run(
        [str(argument) for argument in arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Hot-attach command failed: {detail}")
