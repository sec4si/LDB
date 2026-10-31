"""vLLM plugin entry point for the live LDB runtime."""

from __future__ import annotations

import functools
import gc
import importlib
import os
from typing import Any

from ldb.compatibility import require_audited_runtime
from ldb.context import context_from_vllm, reset_batch_context, set_batch_context
from ldb.hooks import _package_version, install_model_runtime


def register_plugin() -> None:
    """Patch model loading when the operator explicitly enables the agent."""

    if os.environ.get("LDB_ENABLE") != "1":
        return

    backend_version = _package_version("vllm")
    require_audited_runtime("vllm", backend_version)
    worker_module = importlib.import_module("vllm.v1.worker.gpu_worker")
    worker_class = worker_module.Worker

    runner_boundaries = [
        _wrap_model_runner(module_path)
        for module_path in (
            "vllm.v1.worker.gpu_model_runner",
            "vllm.v1.worker.gpu.model_runner",
        )
    ]
    if not any(runner_boundaries):
        raise RuntimeError(
            "Unsupported vLLM runtime: no known GPUModelRunner.execute_model boundary"
        )

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
        tp_size = int(getattr(parallel, "tensor_parallel_size", 1))
        tp_rank = 0
        if tp_size > 1:
            distributed = importlib.import_module("vllm.distributed")
            tp_rank = int(distributed.get_tensor_model_parallel_rank())
        world_size = (
            tp_size
            * int(getattr(parallel, "pipeline_parallel_size", 1))
            * int(getattr(parallel, "data_parallel_size", 1))
        )
        install_model_runtime(
            model,
            backend="vllm",
            backend_version=backend_version,
            model_id=str(model_config.model),
            config={
                "request_context": True,
                "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
                "rank": int(getattr(worker, "rank", 0)),
                "world_size": world_size,
                "topology": {
                    "tp_rank": tp_rank,
                    "tp_size": tp_size,
                    "pp_size": int(getattr(parallel, "pipeline_parallel_size", 1)),
                    "dp_size": int(getattr(parallel, "data_parallel_size", 1)),
                },
            },
        )
        return result

    load_model._ldb_wrapped = True  # type: ignore[attr-defined]
    worker_class.load_model = load_model


def _wrap_model_runner(module_path: str) -> bool:
    try:
        module = importlib.import_module(module_path)
        runner_class = module.GPUModelRunner
    except (ImportError, AttributeError):
        return False
    original = runner_class.execute_model
    if getattr(original, "_ldb_wrapped", False):
        return True

    @functools.wraps(original)
    def execute_model(runner: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        # The newer vllm.v1.worker.gpu runner has no InputBatch; traces stay global there.
        if not hasattr(runner, "input_batch"):
            return original(runner, scheduler_output, *args, **kwargs)
        token = set_batch_context(lambda: context_from_vllm(runner, scheduler_output))
        try:
            return original(runner, scheduler_output, *args, **kwargs)
        finally:
            reset_batch_context(token)

    execute_model._ldb_wrapped = True  # type: ignore[attr-defined]
    runner_class.execute_model = execute_model
    return True


def install_running_runtimes(
    *, model_id: str | None = None, config: dict[str, Any] | None = None
) -> list[Any]:
    """Discover already-created GPUModelRunner objects and instrument their models."""

    backend_version = _package_version("vllm")
    require_audited_runtime("vllm", backend_version)
    runner_classes: list[type[Any]] = []
    for module_path in (
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu.model_runner",
    ):
        if _wrap_model_runner(module_path):
            runner_class = importlib.import_module(module_path).GPUModelRunner
            if runner_class not in runner_classes:
                runner_classes.append(runner_class)
    if not runner_classes:
        raise RuntimeError("No supported vLLM GPUModelRunner class is loaded")

    agents = []
    for candidate in gc.get_objects():
        try:
            is_runner = isinstance(candidate, tuple(runner_classes))
        except ReferenceError:
            continue
        if not is_runner:
            continue
        model = candidate.get_model() if hasattr(candidate, "get_model") else candidate.model
        runtime_config = _running_config(candidate, config or {})
        configured_model = model_id or _running_model_id(candidate)
        agents.append(
            install_model_runtime(
                model,
                backend="vllm",
                backend_version=backend_version,
                model_id=configured_model,
                config=runtime_config,
            )
        )
    if not agents:
        raise RuntimeError("No live vLLM GPUModelRunner instance was found in this process")
    return agents


def _running_model_id(runner: Any) -> str:
    values = (
        getattr(getattr(getattr(runner, "vllm_config", None), "model_config", None), "model", None),
        getattr(getattr(runner, "model_config", None), "model", None),
    )
    return str(next((value for value in values if value is not None), "unknown"))


def _running_config(runner: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    values = dict(overrides)
    values["request_context"] = True
    model = runner.get_model() if hasattr(runner, "get_model") else runner.model
    values.setdefault(
        "model_revision", getattr(getattr(model, "config", None), "_commit_hash", None)
    )
    parallel = getattr(getattr(runner, "vllm_config", None), "parallel_config", None)
    if parallel is not None:
        tp_size = int(getattr(parallel, "tensor_parallel_size", 1))
        pp_size = int(getattr(parallel, "pipeline_parallel_size", 1))
        dp_size = int(getattr(parallel, "data_parallel_size", 1))
        rank = int(getattr(runner, "rank", os.environ.get("RANK", 0)))
        tp_rank = getattr(runner, "tp_rank", None)
        if tp_rank is None and tp_size > 1:
            distributed = importlib.import_module("vllm.distributed")
            tp_rank = distributed.get_tensor_model_parallel_rank()
        values.setdefault("rank", rank)
        values.setdefault("world_size", max(rank + 1, tp_size * pp_size * dp_size))
        topology = {"tp_size": tp_size, "pp_size": pp_size, "dp_size": dp_size}
        if tp_rank is not None:
            topology["tp_rank"] = int(tp_rank)
        values.setdefault("topology", topology)
    return values
