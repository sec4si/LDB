"""In-process debugger agent exposed over a local Unix socket."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import socketserver
import stat
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from safetensors.torch import save_file

from ldb.context import BatchContext, current_batch_context
from ldb.graph import GraphControlSlot
from ldb.interventions import apply_tensor_operation, load_intervention_tensor
from ldb.live import (
    AgentCapability,
    AgentDescriptor,
    AgentRequest,
    AgentResponse,
    ArmConfig,
    BreakpointHit,
    LiveBreakpoint,
    LiveIntervention,
    LiveTraceEvent,
)
from ldb.models import InterventionKind
from ldb.trace import summarize_tensor

_MAX_REQUEST_BYTES = 1024 * 1024


class _ThreadingUnixServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, path: str, runtime: AgentRuntime) -> None:
        self.runtime = runtime
        super().__init__(path, _AgentRequestHandler)


class _AgentRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        line = self.rfile.readline(_MAX_REQUEST_BYTES + 1)
        if len(line) > _MAX_REQUEST_BYTES:
            self._write_error("", "Request exceeds the protocol size limit")
            return
        try:
            request = AgentRequest.model_validate_json(line)
            result = cast(_ThreadingUnixServer, self.server).runtime.dispatch(
                request.method, request.params
            )
            response = AgentResponse(request_id=request.request_id, ok=True, result=result)
        except Exception as exc:
            request_id = request.request_id if "request" in locals() else ""
            response = AgentResponse(request_id=request_id, ok=False, error=str(exc))
        self.wfile.write(response.model_dump_json().encode() + b"\n")

    def _write_error(self, request_id: str, error: str) -> None:
        response = AgentResponse(request_id=request_id, ok=False, error=error)
        self.wfile.write(response.model_dump_json().encode() + b"\n")


class AgentRuntime:
    """Own live trace state and apply reversible activation interventions."""

    def __init__(
        self,
        *,
        backend: str,
        backend_version: str,
        model_id: str,
        model_revision: str | None = None,
        tokenizer_hash: str | None = None,
        socket_dir: Path,
        output_dir: Path,
        rank: int = 0,
        world_size: int = 1,
        max_events: int = 4096,
        request_context: bool = False,
        topology: Mapping[str, int] | None = None,
        graph_control: bool = False,
    ) -> None:
        if rank < 0 or world_size <= rank:
            raise ValueError("rank must be within world_size")
        self.backend = backend
        self.backend_version = backend_version
        self.model_id = model_id
        self.model_revision = model_revision
        self.tokenizer_hash = tokenizer_hash
        self.rank = rank
        self.world_size = world_size
        self.socket_dir = socket_dir
        self.output_dir = output_dir
        self.max_events = max_events
        self.request_context = request_context
        self.topology = dict(topology or {})
        self.graph_control = graph_control
        self._events: deque[LiveTraceEvent] = deque(maxlen=max_events)
        self._patches: dict[str, LiveIntervention] = {}
        self._replacement_tensors: dict[str, Any] = {}
        self._graph_slots: dict[str, GraphControlSlot] = {}
        self._breakpoints: dict[str, LiveBreakpoint] = {}
        self._stopped: dict[str, BreakpointHit] = {}
        self._arm = ArmConfig(hooks=())
        self._sequence = 0
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._server: _ThreadingUnixServer | None = None
        self._thread: threading.Thread | None = None
        self._handles: list[Any] = []
        self._run_id = uuid.uuid4().hex

        backend_id = hashlib.sha256(backend.encode()).hexdigest()[:8]
        identity = f"worker-{os.getpid()}-r{rank}-{backend_id}"
        self.socket_path = _socket_path(socket_dir, identity)
        self.descriptor_path = socket_dir / f"{identity}.json"
        self.tensor_dir = output_dir / self._run_id / identity / "tensors"

    @property
    def descriptor(self) -> AgentDescriptor:
        capabilities = [
            AgentCapability.TRACE_SUMMARY,
            AgentCapability.TRACE_TENSOR,
            AgentCapability.PATCH_ACTIVATION,
            AgentCapability.BREAKPOINT_BATCH,
            AgentCapability.EAGER_ONLY,
        ]
        if self.request_context:
            capabilities.append(AgentCapability.PATCH_REQUEST_SCOPED)
        if self.model_revision is not None and self.tokenizer_hash is not None:
            capabilities.append(AgentCapability.MODEL_IDENTITY)
        if self.graph_control:
            capabilities.append(AgentCapability.PATCH_CUDA_GRAPH)
        return AgentDescriptor(
            backend=self.backend,
            backend_version=self.backend_version,
            model_id=self.model_id,
            model_revision=self.model_revision,
            tokenizer_hash=self.tokenizer_hash,
            pid=os.getpid(),
            rank=self.rank,
            world_size=self.world_size,
            topology=self.topology,
            socket_path=str(self.socket_path),
            capabilities=tuple(capabilities),
        )

    def start(self) -> AgentRuntime:
        if self._server is not None:
            return self
        _prepare_private_directory(self.socket_dir)
        _prepare_private_directory(self.socket_path.parent)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _remove_owned_path(self.socket_path, socket_only=True)
        server = _ThreadingUnixServer(str(self.socket_path), self)
        os.chmod(self.socket_path, 0o600)
        self._server = server
        _remove_owned_path(self.descriptor_path, socket_only=False)
        _write_private_file(
            self.descriptor_path,
            self.descriptor.model_dump_json(indent=2) + "\n",
        )
        thread = threading.Thread(target=server.serve_forever, name="ldb-agent", daemon=True)
        self._thread = thread
        thread.start()
        return self

    def close(self) -> None:
        with self._condition:
            self._stopped.clear()
            self._condition.notify_all()
        server = self._server
        if server is not None:
            server.shutdown()
            server.server_close()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.socket_path.unlink(missing_ok=True)
        self.descriptor_path.unlink(missing_ok=True)
        self._server = None
        self._thread = None

    def add_handle(self, handle: Any) -> None:
        self._handles.append(handle)

    def prepare_graph_hook(self, hook: str, reference: Any) -> None:
        if not self.graph_control:
            return
        with self._lock:
            if hook not in self._graph_slots:
                self._graph_slots[hook] = GraphControlSlot.create(reference)

    def dispatch(self, method: str, params: Mapping[str, Any]) -> Any:
        if method == "hello":
            return self.descriptor.model_dump(mode="json")
        if method == "status":
            with self._lock:
                return {
                    "descriptor": self.descriptor.model_dump(mode="json"),
                    "armed": bool(self._arm.hooks),
                    "arm": self._arm.model_dump(mode="json"),
                    "event_count": len(self._events),
                    "patches": [patch.model_dump(mode="json") for patch in self._patches.values()],
                    "breakpoints": [
                        breakpoint.model_dump(mode="json")
                        for breakpoint in self._breakpoints.values()
                    ],
                    "stopped": [hit.model_dump(mode="json") for hit in self._stopped.values()],
                    "run_id": self._run_id,
                }
        if method == "arm":
            arm = ArmConfig.model_validate(params)
            with self._lock:
                self._arm = arm
            return arm.model_dump(mode="json")
        if method == "disarm":
            with self._lock:
                self._arm = ArmConfig(hooks=())
            return {"armed": False}
        if method == "events":
            after = int(params.get("after", -1))
            with self._lock:
                events = [event for event in self._events if event.sequence > after]
            return [event.model_dump(mode="json") for event in events]
        if method == "clear":
            with self._lock:
                self._events.clear()
            return {"event_count": 0}
        if method == "patch.set":
            patch = LiveIntervention.model_validate(params)
            if patch.request_ids and not self.request_context:
                raise ValueError("This runtime does not provide authoritative request context")
            _validate_live_patch(patch)
            if self.graph_control:
                _validate_graph_patch(patch)
            replacement = load_intervention_tensor(patch.tensor) if patch.tensor else None
            with self._lock:
                self._patches[patch.patch_id] = patch
                if replacement is None:
                    self._replacement_tensors.pop(patch.patch_id, None)
                else:
                    self._replacement_tensors[patch.patch_id] = replacement
            self._refresh_graph_slots()
            return patch.model_dump(mode="json")
        if method == "patch.remove":
            patch_id = str(params["patch_id"])
            with self._lock:
                removed = self._patches.pop(patch_id, None)
                self._replacement_tensors.pop(patch_id, None)
            self._refresh_graph_slots()
            return {"removed": removed is not None}
        if method == "patch.list":
            with self._lock:
                return [patch.model_dump(mode="json") for patch in self._patches.values()]
        if method == "break.set":
            breakpoint = LiveBreakpoint.model_validate(params)
            if breakpoint.request_ids and not self.request_context:
                raise ValueError("This runtime does not provide authoritative request context")
            with self._lock:
                self._breakpoints[breakpoint.breakpoint_id] = breakpoint
            return breakpoint.model_dump(mode="json")
        if method == "break.remove":
            breakpoint_id = str(params["breakpoint_id"])
            with self._lock:
                removed_breakpoint = self._breakpoints.pop(breakpoint_id, None)
            return {"removed": removed_breakpoint is not None}
        if method == "break.list":
            with self._lock:
                return [item.model_dump(mode="json") for item in self._breakpoints.values()]
        if method == "continue":
            hit_id = str(params["hit_id"]) if "hit_id" in params else None
            with self._lock:
                if hit_id is None:
                    released = len(self._stopped)
                    self._stopped.clear()
                else:
                    released = int(self._stopped.pop(hit_id, None) is not None)
                self._condition.notify_all()
            return {"released": released}
        raise ValueError(f"Unknown agent method: {method}")

    def process_tensor(self, hook: str, tensor: Any) -> Any:
        context = current_batch_context()
        with self._lock:
            arm = self._arm
            breakpoints = tuple(
                breakpoint
                for breakpoint in self._breakpoints.values()
                if breakpoint.enabled
                and fnmatch.fnmatchcase(hook, breakpoint.hook)
                and _request_matches(context, breakpoint.request_ids)
            )
            hook_selected = any(fnmatch.fnmatchcase(hook, pattern) for pattern in arm.hooks)

        selected_spans = _selected_spans(context, arm.request_ids)
        request_selected = not arm.request_ids or bool(selected_spans)
        layout_valid = not arm.request_ids or _spans_fit(tensor, selected_spans)
        should_record = hook_selected and request_selected and layout_valid

        for breakpoint in breakpoints:
            spans = _selected_spans(context, breakpoint.request_ids)
            if breakpoint.request_ids and not _spans_fit(tensor, spans):
                continue
            observed = _select_tensor(tensor, spans) if breakpoint.request_ids else tensor
            self._stop_at_breakpoint(breakpoint, hook, observed, context, spans)

        with self._lock:
            patches = tuple(
                patch
                for patch in self._patches.values()
                if patch.enabled and fnmatch.fnmatchcase(hook, patch.hook)
            )

        if self.graph_control:
            modified = self._apply_graph_control(hook, tensor)
            patch_ids = [patch.patch_id for patch in patches]
        else:
            modified = tensor
            patch_ids = []
            for patch in patches:
                with self._lock:
                    replacement = self._replacement_tensors.get(patch.patch_id)
                modified, applied = _apply_intervention(
                    modified,
                    patch,
                    context,
                    replacement=replacement,
                )
                if applied:
                    patch_ids.append(patch.patch_id)

        if should_record:
            observed = _select_tensor(tensor, selected_spans) if arm.request_ids else tensor
            patched = _select_tensor(modified, selected_spans) if arm.request_ids else modified
            summary = summarize_tensor(observed)
            patched_summary = summarize_tensor(patched) if patch_ids else None
            with self._lock:
                sequence = self._sequence
                self._sequence += 1
            tensor_file = self._capture_tensor(sequence, patched) if arm.capture_tensors else None
            event = LiveTraceEvent(
                sequence=sequence,
                hook=hook,
                rank=self.rank,
                phase=context.phase if context else None,
                request_ids=tuple(span.request_id for span in selected_spans),
                summary=summary,
                patched_summary=patched_summary,
                patch_ids=tuple(patch_ids),
                tensor_file=tensor_file,
            )
            with self._lock:
                self._events.append(event)
        return modified

    def _apply_graph_control(self, hook: str, tensor: Any) -> Any:
        with self._lock:
            slot = self._graph_slots.get(hook)
            if slot is None:
                slot = GraphControlSlot.create(tensor)
                self._graph_slots[hook] = slot
                patches = tuple(self._patches.values())
            else:
                patches = ()
        if patches:
            scale, bias = _graph_patch_parameters(hook, patches)
            slot.update(scale=scale, bias=bias)
        return slot.apply(tensor)

    def _refresh_graph_slots(self) -> None:
        if not self.graph_control:
            return
        with self._lock:
            slots = tuple(self._graph_slots.items())
            patches = tuple(self._patches.values())
        for hook, slot in slots:
            scale, bias = _graph_patch_parameters(hook, patches)
            slot.update(scale=scale, bias=bias)

    def _capture_tensor(self, sequence: int, tensor: Any) -> str:
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        path = self.tensor_dir / f"{sequence:08d}.safetensors"
        save_file({"activation": tensor.detach().contiguous().cpu()}, path)
        return str(path)

    def _stop_at_breakpoint(
        self,
        breakpoint: LiveBreakpoint,
        hook: str,
        tensor: Any,
        context: BatchContext | None,
        spans: tuple[Any, ...],
    ) -> None:
        hit = BreakpointHit(
            hit_id=uuid.uuid4().hex,
            breakpoint_id=breakpoint.breakpoint_id,
            hook=hook,
            rank=self.rank,
            phase=context.phase if context else None,
            request_ids=tuple(span.request_id for span in spans),
            summary=summarize_tensor(tensor),
        )
        deadline = time.monotonic() + breakpoint.timeout_seconds
        with self._condition:
            self._stopped[hit.hit_id] = hit
            while hit.hit_id in self._stopped:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._stopped.pop(hit.hit_id, None)
                    break
                self._condition.wait(timeout=remaining)


def _socket_path(directory: Path, identity: str) -> Path:
    candidate = directory / f"{identity}.sock"
    if len(os.fsencode(candidate)) <= 100:
        return candidate
    digest = hashlib.sha256(os.fsencode(directory)).hexdigest()[:16]
    return Path("/tmp") / f"ldb-sockets-{os.getuid()}" / digest / f"{identity}.sock"


def _prepare_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = path.stat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise PermissionError(f"LDB agent directory is not owned by the current user: {path}")
    os.chmod(path, 0o700)


def _remove_owned_path(path: Path, *, socket_only: bool) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if metadata.st_uid != os.getuid():
        raise PermissionError(f"Refusing to replace a path owned by another user: {path}")
    if socket_only and not stat.S_ISSOCK(metadata.st_mode):
        raise FileExistsError(f"Refusing to replace a non-socket path: {path}")
    path.unlink()


def _write_private_file(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)


def _selected_spans(context: BatchContext | None, request_ids: tuple[str, ...]) -> tuple[Any, ...]:
    if context is None:
        return ()
    return context.spans_for(request_ids)


def _request_matches(context: BatchContext | None, request_ids: tuple[str, ...]) -> bool:
    return not request_ids or bool(_selected_spans(context, request_ids))


def _select_tensor(tensor: Any, spans: tuple[Any, ...]) -> Any:
    import torch

    if not spans:
        return tensor[:0]
    if max(span.end for span in spans) > tensor.shape[0]:
        raise ValueError("Request token layout exceeds the activation's leading dimension")
    values = [tensor[span.start : span.end] for span in spans]
    return values[0] if len(values) == 1 else torch.cat(values, dim=0)


def _spans_fit(tensor: Any, spans: tuple[Any, ...]) -> bool:
    return (
        bool(spans) and len(tensor.shape) > 0 and max(span.end for span in spans) <= tensor.shape[0]
    )


def _apply_intervention(
    tensor: Any,
    patch: LiveIntervention,
    context: BatchContext | None,
    *,
    replacement: Any = None,
) -> tuple[Any, bool]:
    spans = _selected_spans(context, patch.request_ids)
    if patch.request_ids and not spans:
        return tensor, False
    if not patch.request_ids:
        return _apply_operation(tensor, patch, replacement), True
    if not _spans_fit(tensor, spans):
        return tensor, False

    modified = tensor.clone()
    for span in spans:
        modified[span.start : span.end] = _apply_operation(
            tensor[span.start : span.end], patch, replacement
        )
    return modified, True


def _apply_operation(tensor: Any, patch: LiveIntervention, replacement: Any) -> Any:
    return apply_tensor_operation(
        tensor,
        patch.operation,
        value=patch.value,
        replacement=replacement,
        expert=patch.expert,
        head=patch.head,
        num_heads=patch.num_heads,
    )


def _validate_live_patch(patch: LiveIntervention) -> None:
    if patch.operation is InterventionKind.SCALE:
        if (
            patch.value is None
            or patch.tensor is not None
            or patch.expert is not None
            or patch.head is not None
            or patch.num_heads is not None
        ):
            raise ValueError("scale requires only a numeric value")
    elif patch.operation is InterventionKind.ROUTING_BIAS:
        if (
            patch.value is None
            or patch.expert is None
            or patch.tensor is not None
            or patch.head is not None
            or patch.num_heads is not None
        ):
            raise ValueError("routing_bias requires a numeric value and expert index")
    elif patch.operation is InterventionKind.ADD:
        if (
            (patch.value is None) == (patch.tensor is None)
            or patch.expert is not None
            or patch.head is not None
            or patch.num_heads is not None
        ):
            raise ValueError("add requires exactly one of value or tensor")
    elif patch.operation is InterventionKind.REPLACE:
        if (
            patch.tensor is None
            or patch.value is not None
            or patch.expert is not None
            or patch.head is not None
            or patch.num_heads is not None
        ):
            raise ValueError("replace requires only a tensor file")
    elif patch.operation is InterventionKind.HEAD_ABLATE:
        if (
            patch.value is not None
            or patch.tensor is not None
            or patch.expert is not None
            or patch.head is None
            or patch.num_heads is None
            or patch.head >= patch.num_heads
        ):
            raise ValueError("head_ablate requires a valid head and num_heads")
    elif patch.operation is InterventionKind.ABLATE and (
        patch.value is not None
        or patch.tensor is not None
        or patch.expert is not None
        or patch.head is not None
        or patch.num_heads is not None
    ):
        raise ValueError("ablate does not accept a value or tensor")


def _validate_graph_patch(patch: LiveIntervention) -> None:
    if patch.request_ids:
        raise ValueError("CUDA Graph control patches cannot be request-scoped")
    if patch.operation not in {
        InterventionKind.SCALE,
        InterventionKind.ADD,
        InterventionKind.ABLATE,
    }:
        raise ValueError(f"CUDA Graph control does not support {patch.operation.value}")
    if patch.tensor is not None:
        raise ValueError("CUDA Graph control supports scalar interventions only")


def _graph_patch_parameters(
    hook: str, patches: tuple[LiveIntervention, ...]
) -> tuple[float, float]:
    scale = 1.0
    bias = 0.0
    for patch in patches:
        if not patch.enabled or not fnmatch.fnmatchcase(hook, patch.hook):
            continue
        if patch.operation is InterventionKind.SCALE:
            assert patch.value is not None
            scale *= patch.value
            bias *= patch.value
        elif patch.operation is InterventionKind.ADD:
            assert patch.value is not None
            bias += patch.value
        elif patch.operation is InterventionKind.ABLATE:
            scale = 0.0
            bias = 0.0
    return scale, bias
