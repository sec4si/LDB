"""SGLang plugin entry point for the live LDB runtime."""

from __future__ import annotations

import functools
import gc
import importlib
import os
from typing import Any, cast

from ldb.compatibility import require_audited_runtime
from ldb.context import context_from_sglang, reset_batch_context, set_batch_context
from ldb.hooks import _package_version, install_model_runtime


def register_plugin() -> None:
    """Register LDB hooks when the operator explicitly enables the agent."""

    if os.environ.get("LDB_ENABLE") != "1":
        return

    backend_version = _package_version("sglang")
    require_audited_runtime("sglang", backend_version)
    registry = importlib.import_module("sglang.srt.plugins.hook_registry")
    worker_module = importlib.import_module("sglang.srt.managers.tp_worker")
    runner_module = importlib.import_module("sglang.srt.model_executor.model_runner")
    if not hasattr(getattr(worker_module, "TpModelWorker", None), "__init__"):
        raise RuntimeError("Unsupported SGLang runtime: TpModelWorker.__init__ is unavailable")
    if not hasattr(getattr(runner_module, "ModelRunner", None), "forward"):
        raise RuntimeError("Unsupported SGLang runtime: ModelRunner.forward is unavailable")

    registry.HookRegistry.register(
        "sglang.srt.managers.tp_worker.TpModelWorker.__init__",
        functools.partial(_after_worker_init, backend_version=backend_version),
        registry.HookType.AFTER,
    )
    registry.HookRegistry.register(
        "sglang.srt.model_executor.model_runner.ModelRunner.forward",
        _around_model_forward,
        registry.HookType.AROUND,
    )


def _after_worker_init(
    result: Any,
    worker: Any,
    *_: Any,
    backend_version: str,
    **__: Any,
) -> Any:
    runner = worker.model_runner
    model = runner.model
    server_args = runner.server_args
    model_id = str(getattr(server_args, "model_path", "unknown"))
    ps = worker.ps
    tp_rank = int(getattr(ps, "tp_rank", 0))
    pp_rank = int(getattr(ps, "pp_rank", 0))
    dp_rank = int(getattr(ps, "dp_rank", 0))
    tp_size = int(getattr(ps, "tp_size", 1))
    pp_size = int(getattr(ps, "pp_size", 1))
    dp_size = int(getattr(ps, "dp_size", 1))
    rank = tp_rank + tp_size * (pp_rank + pp_size * dp_rank)
    install_model_runtime(
        model,
        backend="sglang",
        backend_version=backend_version,
        model_id=model_id,
        config={
            "request_context": True,
            "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
            "rank": rank,
            "world_size": tp_size * pp_size * dp_size,
            "topology": {
                "tp_rank": tp_rank,
                "tp_size": tp_size,
                "pp_rank": pp_rank,
                "pp_size": pp_size,
                "dp_rank": dp_rank,
                "dp_size": dp_size,
            },
        },
    )
    return result


def _around_model_forward(
    original: Any, runner: Any, forward_batch: Any, *args: Any, **kwargs: Any
) -> Any:
    token = set_batch_context(context_from_sglang(forward_batch))
    try:
        return original(runner, forward_batch, *args, **kwargs)
    finally:
        reset_batch_context(token)


def _wrap_running_model_runner() -> type[Any]:
    module = importlib.import_module("sglang.srt.model_executor.model_runner")
    runner_class = cast(type[Any], module.ModelRunner)
    original = runner_class.forward
    if getattr(original, "_ldb_wrapped", False):
        return runner_class

    @functools.wraps(original)
    def forward(runner: Any, forward_batch: Any, *args: Any, **kwargs: Any) -> Any:
        token = set_batch_context(context_from_sglang(forward_batch))
        try:
            return original(runner, forward_batch, *args, **kwargs)
        finally:
            reset_batch_context(token)

    forward._ldb_wrapped = True  # type: ignore[attr-defined]
    runner_class.forward = forward
    return runner_class


def install_running_runtimes(
    *, model_id: str | None = None, config: dict[str, Any] | None = None
) -> list[Any]:
    """Discover already-created ModelRunner objects and instrument their models."""

    backend_version = _package_version("sglang")
    require_audited_runtime("sglang", backend_version)
    runner_class = _wrap_running_model_runner()
    agents = []
    for candidate in gc.get_objects():
        try:
            is_runner = isinstance(candidate, runner_class)
        except ReferenceError:
            continue
        if (
            not is_runner
            or not hasattr(candidate, "model")
            or bool(getattr(candidate, "is_draft_worker", False))
        ):
            continue
        values = dict(config or {})
        values["request_context"] = True
        values.setdefault(
            "model_revision",
            getattr(getattr(candidate.model, "config", None), "_commit_hash", None),
        )
        parallel = getattr(candidate, "ps", candidate)
        tp_rank = int(getattr(parallel, "tp_rank", 0))
        pp_rank = int(getattr(parallel, "pp_rank", 0))
        dp_rank = int(getattr(parallel, "dp_rank", 0))
        tp_size = int(getattr(parallel, "tp_size", 1))
        pp_size = int(getattr(parallel, "pp_size", 1))
        dp_size = int(getattr(parallel, "dp_size", 1))
        inferred_rank = tp_rank + tp_size * (pp_rank + pp_size * dp_rank)
        rank = int(os.environ.get("RANK", inferred_rank))
        world_size = int(os.environ.get("WORLD_SIZE", tp_size * pp_size * dp_size))
        values.setdefault("rank", rank)
        values.setdefault("world_size", max(rank + 1, world_size))
        values.setdefault(
            "topology",
            {
                "tp_rank": tp_rank,
                "tp_size": tp_size,
                "pp_rank": pp_rank,
                "pp_size": pp_size,
                "dp_rank": dp_rank,
                "dp_size": dp_size,
            },
        )
        configured_model = model_id or str(
            getattr(
                getattr(candidate, "model_config", None),
                "model_path",
                getattr(getattr(candidate, "server_args", None), "model_path", "unknown"),
            )
        )
        agents.append(
            install_model_runtime(
                candidate.model,
                backend="sglang",
                backend_version=backend_version,
                model_id=configured_model,
                config=values,
            )
        )
    if not agents:
        raise RuntimeError("No live SGLang ModelRunner instance was found in this process")
    return agents
