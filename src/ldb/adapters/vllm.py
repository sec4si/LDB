"""vLLM plugin entry point for the live LDB runtime."""

from __future__ import annotations

import functools
import importlib
import os
from typing import Any

from ldb.context import context_from_vllm, reset_batch_context, set_batch_context
from ldb.hooks import _package_version, install_model_runtime


def register_plugin() -> None:
    """Patch model loading when the operator explicitly enables the agent."""

    if os.environ.get("LDB_ENABLE") != "1":
        return

    worker_module = importlib.import_module("vllm.v1.worker.gpu_worker")
    worker_class = worker_module.Worker

    for module_path in (
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu.model_runner",
    ):
        _wrap_model_runner(module_path)

    original = worker_class.load_model
    if getattr(original, "_ldb_wrapped", False):
        return

    @functools.wraps(original)
    def load_model(worker: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(worker, *args, **kwargs)
        runner = worker.model_runner
        model = runner.get_model() if hasattr(runner, "get_model") else runner.model
        model_config = worker.vllm_config.model_config
        parallel = worker.vllm_config.parallel_config
        world_size = (
            int(getattr(parallel, "tensor_parallel_size", 1))
            * int(getattr(parallel, "pipeline_parallel_size", 1))
            * int(getattr(parallel, "data_parallel_size", 1))
        )
        install_model_runtime(
            model,
            backend="vllm",
            backend_version=_package_version("vllm"),
            model_id=str(model_config.model),
            config={
                "request_context": True,
                "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
                "rank": int(getattr(worker, "rank", 0)),
                "world_size": world_size,
                "topology": {
                    "tp_size": int(getattr(parallel, "tensor_parallel_size", 1)),
                    "pp_size": int(getattr(parallel, "pipeline_parallel_size", 1)),
                    "dp_size": int(getattr(parallel, "data_parallel_size", 1)),
                },
            },
        )
        return result

    load_model._ldb_wrapped = True  # type: ignore[attr-defined]
    worker_class.load_model = load_model


def _wrap_model_runner(module_path: str) -> None:
    try:
        module = importlib.import_module(module_path)
        runner_class = module.GPUModelRunner
    except (ImportError, AttributeError):
        return
    original = runner_class.execute_model
    if getattr(original, "_ldb_wrapped", False):
        return

    @functools.wraps(original)
    def execute_model(runner: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        token = set_batch_context(lambda: context_from_vllm(runner, scheduler_output))
        try:
            return original(runner, scheduler_output, *args, **kwargs)
        finally:
            reset_batch_context(token)

    execute_model._ldb_wrapped = True  # type: ignore[attr-defined]
    runner_class.execute_model = execute_model
