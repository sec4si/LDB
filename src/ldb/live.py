"""Models for the live debugger control protocol."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from ldb.models import FrozenModel, InterventionKind, TensorSummary

AGENT_PROTOCOL_VERSION = "ldb.agent/v1alpha1"


class AgentCapability(StrEnum):
    TRACE_SUMMARY = "trace.summary"
    TRACE_TENSOR = "trace.tensor"
    PATCH_ACTIVATION = "patch.activation"
    PATCH_REQUEST_SCOPED = "patch.request_scoped"
    BREAKPOINT_BATCH = "breakpoint.batch"
    EAGER_ONLY = "execution.eager_only"
    MODEL_IDENTITY = "model.identity"
    PATCH_CUDA_GRAPH = "patch.cuda_graph"


class AgentDescriptor(FrozenModel):
    protocol_version: Literal["ldb.agent/v1alpha1"] = "ldb.agent/v1alpha1"
    backend: str
    backend_version: str
    model_id: str
    model_revision: str | None = None
    tokenizer_hash: str | None = None
    pid: int = Field(gt=0)
    rank: int = Field(ge=0)
    world_size: int = Field(gt=0)
    topology: dict[str, int] = Field(default_factory=dict)
    socket_path: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    capabilities: tuple[AgentCapability, ...]


class ArmConfig(FrozenModel):
    hooks: tuple[str, ...] = ("*",)
    request_ids: tuple[str, ...] = ()
    capture_tensors: bool = False


class LiveIntervention(FrozenModel):
    patch_id: str
    hook: str
    operation: InterventionKind
    value: float | None = None
    tensor: str | None = None
    expert: int | None = Field(default=None, ge=0)
    head: int | None = Field(default=None, ge=0)
    num_heads: int | None = Field(default=None, gt=0)
    request_ids: tuple[str, ...] = ()
    enabled: bool = True


class LiveBreakpoint(FrozenModel):
    breakpoint_id: str
    hook: str
    request_ids: tuple[str, ...] = ()
    timeout_seconds: float = Field(default=300.0, gt=0.0, le=3600.0)
    enabled: bool = True


class BreakpointHit(FrozenModel):
    hit_id: str
    breakpoint_id: str
    hook: str
    rank: int = Field(ge=0)
    phase: str | None = None
    request_ids: tuple[str, ...] = ()
    summary: TensorSummary
    stopped_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LiveTraceEvent(FrozenModel):
    sequence: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    hook: str
    rank: int = Field(ge=0)
    phase: str | None = None
    request_ids: tuple[str, ...] = ()
    summary: TensorSummary
    patched_summary: TensorSummary | None = None
    patch_ids: tuple[str, ...] = ()
    tensor_file: str | None = None


class AgentRequest(FrozenModel):
    protocol_version: Literal["ldb.agent/v1alpha1"] = "ldb.agent/v1alpha1"
    request_id: str
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(FrozenModel):
    protocol_version: Literal["ldb.agent/v1alpha1"] = "ldb.agent/v1alpha1"
    request_id: str
    ok: bool
    result: Any = None
    error: str | None = None
