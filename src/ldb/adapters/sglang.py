"""SGLang plugin entry point for the live LDB runtime."""

from __future__ import annotations

import importlib
import os
from typing import Any

from ldb.context import context_from_sglang, reset_batch_context, set_batch_context
from ldb.hooks import _package_version, install_model_runtime


def register_plugin() -> None:
    """Register LDB hooks when the operator explicitly enables the agent."""

    if os.environ.get("LDB_ENABLE") != "1":
        return

    registry = importlib.import_module("sglang.srt.plugins.hook_registry")
    worker_module = importlib.import_module("sglang.srt.managers.tp_worker")
    runner_module = importlib.import_module("sglang.srt.model_executor.model_runner")
    if not hasattr(getattr(worker_module, "TpModelWorker", None), "__init__"):
        raise RuntimeError("Unsupported SGLang runtime: TpModelWorker.__init__ is unavailable")
    if not hasattr(getattr(runner_module, "ModelRunner", None), "forward"):
        raise RuntimeError("Unsupported SGLang runtime: ModelRunner.forward is unavailable")

    registry.HookRegistry.register(
        "sglang.srt.managers.tp_worker.TpModelWorker.__init__",
        _after_worker_init,
        registry.HookType.AFTER,
    )
    registry.HookRegistry.register(
        "sglang.srt.model_executor.model_runner.ModelRunner.forward",
        _around_model_forward,
        registry.HookType.AROUND,
    )


def _after_worker_init(result: Any, worker: Any, *_: Any, **__: Any) -> Any:
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
        backend_version=_package_version("sglang"),
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
