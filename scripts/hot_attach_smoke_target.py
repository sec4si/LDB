"""Minimal live CPython target for the Linux hot-attach acceptance test."""

from __future__ import annotations

import ctypes
import signal
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any

import ldb.adapters.sglang as sglang_adapter
import ldb.adapters.vllm as adapter


class Handle:
    def remove(self) -> None:
        pass


class Block:
    def forward(self, hidden_states: Any) -> Any:
        return hidden_states

    def register_forward_pre_hook(self, callback: Any, *, with_kwargs: bool = False) -> Handle:
        self.pre = callback
        return Handle()

    def register_forward_hook(self, callback: Any) -> Handle:
        self.post = callback
        return Handle()


class Model:
    def __init__(self) -> None:
        self.config = SimpleNamespace(_commit_hash="smoke-commit")
        self.layers = [Block()]

    def named_modules(self) -> list[tuple[str, Any]]:
        return [
            ("model.layers", self.layers),
            ("model.layers.0", self.layers[0]),
        ]


class GPUModelRunner:
    def __init__(self) -> None:
        self.model = Model()
        self.rank = 0
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(model="hot-smoke-model"),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
            ),
        )

    def execute_model(self, scheduler_output: Any) -> Any:
        return scheduler_output


class ModelRunner:
    def __init__(self) -> None:
        self.model = Model()
        self.server_args = SimpleNamespace(model_path="hot-smoke-model")
        self.tp_rank = 0
        self.tp_size = 1

    def forward(self, forward_batch: Any) -> Any:
        return forward_batch


def main() -> None:
    _allow_test_ptrace()
    backend = sys.argv[1] if len(sys.argv) > 1 else "vllm"
    if backend == "vllm":
        module_name = "vllm.v1.worker.gpu_model_runner"
        module = ModuleType(module_name)
        module.GPUModelRunner = GPUModelRunner  # type: ignore[attr-defined]
        sys.modules[module_name] = module
        adapter.__dict__["_package_version"] = lambda _: "0.28.0"
        runner: Any = GPUModelRunner()
    elif backend == "sglang":
        module_name = "sglang.srt.model_executor.model_runner"
        module = ModuleType(module_name)
        module.ModelRunner = ModelRunner  # type: ignore[attr-defined]
        sys.modules[module_name] = module
        sglang_adapter.__dict__["_package_version"] = lambda _: "0.5.19"
        runner = ModelRunner()
    else:
        raise ValueError(f"Unsupported smoke backend: {backend}")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    print(f"ready {runner.model.config._commit_hash}", flush=True)
    while True:
        time.sleep(1)


def _allow_test_ptrace() -> None:
    if not sys.platform.startswith("linux"):
        return
    libc = ctypes.CDLL(None, use_errno=True)
    pr_set_ptracer = 0x59616D61
    if libc.prctl(pr_set_ptracer, ctypes.c_ulong(-1), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PTRACER_ANY failed")


if __name__ == "__main__":
    main()
