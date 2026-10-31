"""Runtime-neutral PyTorch hook installation."""

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ldb.agent import AgentRuntime
from ldb.serving_hooks import has_deferred_residual, make_input_hook, make_residual_output_hook

_AGENTS: dict[tuple[str, str, int], AgentRuntime] = {}

# Decoders whose residual stream is not a single hidden tensor (DeepSeek V4.1 mHC):
# only their fused-MoE routing is observed.
ROUTING_ONLY_DECODERS = frozenset({"vllm.models.deepseek_v41.nvidia.model.DeepseekV4DecoderLayer"})


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
    _start: bool = True,
) -> AgentRuntime:
    """Return the process-local runtime agent for a backend."""

    values = config or {}
    rank = int(values.get("rank", _environment_int("RANK", 0)))
    world_size = int(values.get("world_size", _environment_int("WORLD_SIZE", 1)))
    model_revision = _configured_string(values, "model_revision", "LDB_MODEL_REVISION")
    tokenizer_hash = _configured_string(values, "tokenizer_hash", "LDB_TOKENIZER_HASH")
    instance_id = _configured_string(values, "instance_id", "LDB_INSTANCE_ID") or "default"
    request_context = bool(values.get("request_context", False))
    topology = dict(values.get("topology", {}))
    graph_control = bool(values.get("graph_control", os.environ.get("LDB_CUDA_GRAPH") == "1"))
    default_root = f"/tmp/ldb-{os.getuid()}"
    socket_dir = Path(
        str(values.get("socket_dir", os.environ.get("LDB_SOCKET_DIR", f"{default_root}/agents")))
    )
    output_dir = Path(
        str(values.get("output_dir", os.environ.get("LDB_OUTPUT_DIR", f"{default_root}/runs")))
    )
    key = (backend, instance_id, rank)
    agent = _AGENTS.get(key)
    if agent is not None:
        _validate_existing_agent(
            agent,
            instance_id=instance_id,
            backend_version=backend_version,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_hash=tokenizer_hash,
            world_size=world_size,
            request_context=request_context,
            topology=topology,
            graph_control=graph_control,
            socket_dir=socket_dir,
            output_dir=output_dir,
        )
        return agent.start() if _start else agent

    agent = AgentRuntime(
        backend=backend,
        backend_version=backend_version,
        instance_id=instance_id,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_hash=tokenizer_hash,
        socket_dir=socket_dir,
        output_dir=output_dir,
        rank=rank,
        world_size=world_size,
        request_context=request_context,
        topology=topology,
        graph_control=graph_control,
    )
    _AGENTS[key] = agent
    return agent.start() if _start else agent


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

    module_by_path = dict(model.named_modules())
    block_path = _find_block_path(module_by_path)
    blocks = module_by_path[block_path]
    local_layers = {
        index: blocks[index]
        for index in range(len(blocks))
        if f"{type(blocks[index]).__module__}.{type(blocks[index]).__name__}"
        not in {
            "vllm.model_executor.models.utils.PPMissingLayer",
            "sglang.srt.layers.utils.common.PPMissingLayer",
        }
    }
    routing_only = {
        index
        for index, block in local_layers.items()
        if f"{type(block).__module__}.{type(block).__name__}" in ROUTING_ONLY_DECODERS
    }
    deferred = {
        index: has_deferred_residual(block)
        for index, block in local_layers.items()
        if index not in routing_only
    }
    graph_control = bool(
        (config or {}).get("graph_control", os.environ.get("LDB_CUDA_GRAPH") == "1")
    )
    if graph_control and any(deferred.values()):
        raise ValueError("Deferred-residual decoder hooks currently require eager execution")
    if graph_control and routing_only:
        raise ValueError("Router capture hooks currently require eager execution")
    agent = get_or_create_agent(
        backend=backend,
        backend_version=backend_version,
        model_id=model_id,
        config=config,
        _start=False,
    )

    for index, block in local_layers.items():
        prefix = f"{block_path}.{index}"
        if index in routing_only:
            _install_router_hooks(agent, module_by_path, prefix, index)
            continue
        _prepare_graph_hook(agent, f"block.{index}.residual.pre", block)
        pre_hook = f"block.{index}.residual.pre"
        agent.add_handle(
            block.register_forward_pre_hook(
                make_input_hook(agent, pre_hook, block, deferred=deferred[index]),
                with_kwargs=True,
            ),
            hook=pre_hook,
        )
        _prepare_graph_hook(agent, f"block.{index}.residual.post", block)
        post_hook = f"block.{index}.residual.post"
        agent.add_handle(
            block.register_forward_hook(
                make_residual_output_hook(agent, post_hook)
                if deferred[index]
                else _make_post_hook(agent, post_hook)
            ),
            hook=post_hook,
        )
        _install_first_post_hook(
            agent,
            module_by_path,
            prefix,
            index,
            "attention.output",
            ("self_attn", "attn"),
        )
        _install_first_pre_hook(
            agent,
            module_by_path,
            prefix,
            index,
            "attention.heads",
            ("self_attn.o_proj", "attn.c_proj"),
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
        _install_expert_hooks(agent, module_by_path, prefix, index)
    setattr(model, marker, agent)
    return agent.start()


def sglang_forward_hook_factory(config: dict[str, Any]) -> Any:
    """Create a hook compatible with SGLang's forward_hooks interface."""

    backend_version = _package_version("sglang")
    hook_name = str(config["hook"])
    agent = get_or_create_agent(
        backend="sglang",
        backend_version=backend_version,
        model_id=str(config.get("model_id", "unknown")),
        config=config,
        _start=False,
    )
    agent.register_hook(hook_name)
    agent.start()

    def hook(_: Any, __: tuple[Any, ...], output: Any) -> Any:
        return _process_output(agent, hook_name, output)

    return hook


def close_agents() -> None:
    """Close all process-local agents. Intended for orderly shutdown and tests."""

    for agent in _AGENTS.values():
        agent.close()
    _AGENTS.clear()


def _validate_existing_agent(
    agent: AgentRuntime,
    *,
    instance_id: str,
    backend_version: str,
    model_id: str,
    model_revision: str | None,
    tokenizer_hash: str | None,
    world_size: int,
    request_context: bool,
    topology: dict[str, Any],
    graph_control: bool,
    socket_dir: Path,
    output_dir: Path,
) -> None:
    expected = {
        "instance_id": instance_id,
        "backend_version": backend_version,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_hash": tokenizer_hash,
        "world_size": world_size,
        "request_context": request_context,
        "topology": topology,
        "graph_control": graph_control,
        "socket_dir": socket_dir,
        "output_dir": output_dir,
    }
    mismatched = [field for field, value in expected.items() if getattr(agent, field) != value]
    if mismatched:
        raise RuntimeError(
            "Refusing to reuse an LDB agent with incompatible identity: " + ", ".join(mismatched)
        )


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
            hook = f"block.{index}.{normalized}"
            agent.add_handle(module.register_forward_hook(_make_post_hook(agent, hook)), hook=hook)
            return


def _install_first_pre_hook(
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
            hook = f"block.{index}.{normalized}"
            _prepare_graph_hook(agent, hook, module)
            agent.add_handle(
                module.register_forward_pre_hook(
                    make_input_hook(agent, hook, module), with_kwargs=True
                ),
                hook=hook,
            )
            return


def _install_expert_hooks(
    agent: AgentRuntime,
    module_by_path: Mapping[str, Any],
    prefix: str,
    index: int,
) -> None:
    experts_path = next(
        (
            path
            for path in (f"{prefix}.mlp.experts", f"{prefix}.block_sparse_moe.experts")
            if path in module_by_path and hasattr(module_by_path[path], "__len__")
        ),
        None,
    )
    if experts_path is None:
        return
    for expert in range(len(module_by_path[experts_path])):
        module = module_by_path[f"{experts_path}.{expert}"]
        hook = f"block.{index}.moe.expert.{expert}.output"
        _prepare_graph_hook(agent, hook, module)
        agent.add_handle(module.register_forward_hook(_make_post_hook(agent, hook)), hook=hook)


def _install_router_hooks(
    agent: AgentRuntime,
    module_by_path: Mapping[str, Any],
    prefix: str,
    index: int,
) -> None:
    """Observe fused-MoE routing where no gate module forward is ever called."""

    router = next(
        (
            module.router
            for path, module in module_by_path.items()
            if path.startswith(f"{prefix}.")
            and hasattr(getattr(module, "router", None), "_compute_routing")
        ),
        None,
    )
    if router is None:
        raise ValueError(f"No fused-MoE router found under {prefix}")
    ids_hook = f"block.{index}.moe.router.topk_ids"
    weights_hook = f"block.{index}.moe.router.topk_weights"
    # noaux_tc adds this bias to the scores used for expert choice only, so a
    # routing_bias patch on it changes which experts are picked, never their weights.
    logits_hook = f"block.{index}.moe.router.logits"
    bias = getattr(router, "e_score_correction_bias", None)
    original = router._compute_routing

    def compute_routing(*args: Any, **kwargs: Any) -> Any:
        saved = bias.data if bias is not None else None
        if bias is not None:
            bias.data = agent.process_tensor(logits_hook, saved)
        try:
            topk_weights, topk_ids = original(*args, **kwargs)
        finally:
            if bias is not None:
                bias.data = saved
        return (
            agent.process_tensor(weights_hook, topk_weights),
            agent.process_tensor(ids_hook, topk_ids),
        )

    router._compute_routing = compute_routing
    agent.add_handle(_AttributeRestore(router, "_compute_routing"), hook=ids_hook)
    agent.register_hook(weights_hook)
    if bias is not None:
        agent.register_hook(logits_hook)


class _AttributeRestore:
    """Hook handle that drops an instance-level override."""

    def __init__(self, owner: Any, name: str) -> None:
        self.owner = owner
        self.name = name

    def remove(self) -> None:
        self.owner.__dict__.pop(self.name, None)


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
