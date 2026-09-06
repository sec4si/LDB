"""Runtime-neutral PyTorch hook installation."""

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ldb.agent import AgentRuntime

_AGENTS: dict[tuple[str, int], AgentRuntime] = {}


def _environment_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _configured_string(values: Mapping[str, Any], key: str, environment_name: str) -> str | None:
    value = values.get(key)
    if value is None:
        value = os.environ.get(environment_name)
    return str(value) if value is not None else None


def get_or_create_agent(
    *,
    backend: str,
    backend_version: str,
    model_id: str,
    config: Mapping[str, Any] | None = None,
) -> AgentRuntime:
    """Return the process-local runtime agent for a backend."""

    values = config or {}
    rank = int(values.get("rank", _environment_int("RANK", 0)))
    world_size = int(values.get("world_size", _environment_int("WORLD_SIZE", 1)))
    key = (backend, rank)
    agent = _AGENTS.get(key)
    if agent is not None:
        return agent

    default_root = f"/tmp/ldb-{os.getuid()}"
    socket_dir = Path(
        str(values.get("socket_dir", os.environ.get("LDB_SOCKET_DIR", f"{default_root}/agents")))
    )
    output_dir = Path(
        str(values.get("output_dir", os.environ.get("LDB_OUTPUT_DIR", f"{default_root}/runs")))
    )
    agent = AgentRuntime(
        backend=backend,
        backend_version=backend_version,
        model_id=model_id,
        model_revision=_configured_string(values, "model_revision", "LDB_MODEL_REVISION"),
        tokenizer_hash=_configured_string(values, "tokenizer_hash", "LDB_TOKENIZER_HASH"),
        socket_dir=socket_dir,
        output_dir=output_dir,
        rank=rank,
        world_size=world_size,
        request_context=bool(values.get("request_context", False)),
        topology=dict(values.get("topology", {})),
        graph_control=bool(values.get("graph_control", os.environ.get("LDB_CUDA_GRAPH") == "1")),
    ).start()
    _AGENTS[key] = agent
    return agent


def install_model_runtime(
    model: Any,
    *,
    backend: str,
    backend_version: str,
    model_id: str,
    config: Mapping[str, Any] | None = None,
) -> AgentRuntime:
    """Install normalized dormant probes onto a decoder-only model."""

    marker = "_ldb_agent_runtime"
    existing = getattr(model, marker, None)
    if isinstance(existing, AgentRuntime):
        return existing

    agent = get_or_create_agent(
        backend=backend,
        backend_version=backend_version,
        model_id=model_id,
        config=config,
    )
    module_by_path = dict(model.named_modules())
    block_path = _find_block_path(module_by_path)
    blocks = module_by_path[block_path]

    for index in range(len(blocks)):
        prefix = f"{block_path}.{index}"
        block = module_by_path[prefix]
        _prepare_graph_hook(agent, f"block.{index}.residual.pre", block)
        agent.add_handle(
            block.register_forward_pre_hook(_make_pre_hook(agent, f"block.{index}.residual.pre"))
        )
        _prepare_graph_hook(agent, f"block.{index}.residual.post", block)
        agent.add_handle(
            block.register_forward_hook(_make_post_hook(agent, f"block.{index}.residual.post"))
        )
        _install_first_post_hook(
            agent,
            module_by_path,
            prefix,
            index,
            "attention.output",
            ("self_attn", "attn"),
        )
        _install_first_post_hook(
            agent,
            module_by_path,
            prefix,
            index,
            "mlp.output",
            ("mlp",),
        )
        _install_first_post_hook(
            agent,
            module_by_path,
            prefix,
            index,
            "moe.router.logits",
            ("mlp.gate", "block_sparse_moe.gate"),
        )
    setattr(model, marker, agent)
    return agent


def sglang_forward_hook_factory(config: dict[str, Any]) -> Any:
    """Create a hook compatible with SGLang's forward_hooks interface."""

    backend_version = _package_version("sglang")
    hook_name = str(config["hook"])
    agent = get_or_create_agent(
        backend="sglang",
        backend_version=backend_version,
        model_id=str(config.get("model_id", "unknown")),
        config=config,
    )

    def hook(_: Any, __: tuple[Any, ...], output: Any) -> Any:
        return _process_output(agent, hook_name, output)

    return hook


def close_agents() -> None:
    """Close all process-local agents. Intended for orderly shutdown and tests."""

    for agent in _AGENTS.values():
        agent.close()
    _AGENTS.clear()


def _find_block_path(module_by_path: Mapping[str, Any]) -> str:
    candidates = (
        "model.layers",
        "transformer.h",
        "model.model.layers",
        "language_model.model.layers",
    )
    for path in candidates:
        module = module_by_path.get(path)
        if module is not None and hasattr(module, "__len__"):
            return path
    raise ValueError("Unsupported model topology: no decoder block container was found")


def _first_tensor_index(value: Any) -> tuple[int, Any] | None:
    import torch

    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            if isinstance(item, torch.Tensor):
                return index, item
    return None


def _process_output(agent: AgentRuntime, hook: str, output: Any) -> Any:
    import torch

    if isinstance(output, torch.Tensor):
        return agent.process_tensor(hook, output)
    found = _first_tensor_index(output)
    if found is None:
        return output
    index, tensor = found
    modified = agent.process_tensor(hook, tensor)
    if modified is tensor:
        return output
    if isinstance(output, tuple):
        values = list(output)
        values[index] = modified
        return tuple(values)
    values = list(output)
    values[index] = modified
    return values


def _make_pre_hook(agent: AgentRuntime, hook: str) -> Any:
    def callback(_: Any, inputs: tuple[Any, ...]) -> Any:
        return _process_output(agent, hook, inputs)

    return callback


def _make_post_hook(agent: AgentRuntime, hook: str) -> Any:
    def callback(_: Any, __: tuple[Any, ...], output: Any) -> Any:
        return _process_output(agent, hook, output)

    return callback


def _install_first_post_hook(
    agent: AgentRuntime,
    module_by_path: Mapping[str, Any],
    prefix: str,
    index: int,
    normalized: str,
    suffixes: Sequence[str],
) -> None:
    for suffix in suffixes:
        module = module_by_path.get(f"{prefix}.{suffix}")
        if module is not None:
            _prepare_graph_hook(agent, f"block.{index}.{normalized}", module)
            agent.add_handle(
                module.register_forward_hook(_make_post_hook(agent, f"block.{index}.{normalized}"))
            )
            return


def _floating_parameter(module: Any) -> Any:
    parameter = next(
        (value for value in module.parameters(recurse=True) if value.is_floating_point()),
        None,
    )
    if parameter is None:
        raise ValueError("Cannot prepare a CUDA Graph control slot without a floating parameter")
    return parameter


def _prepare_graph_hook(agent: AgentRuntime, hook: str, module: Any) -> None:
    if agent.graph_control:
        agent.prepare_graph_hook(hook, _floating_parameter(module))


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
