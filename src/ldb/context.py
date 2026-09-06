"""Request layout context propagated through model forwards."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass(frozen=True, slots=True)
class RequestSpan:
    request_id: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class BatchContext:
    phase: str
    requests: tuple[RequestSpan, ...]

    def spans_for(self, request_ids: tuple[str, ...]) -> tuple[RequestSpan, ...]:
        if not request_ids:
            return self.requests
        selected = set(request_ids)
        return tuple(span for span in self.requests if span.request_id in selected)


BatchContextProvider: TypeAlias = Callable[[], BatchContext]
BatchContextValue: TypeAlias = BatchContext | BatchContextProvider | None

_CURRENT_BATCH: ContextVar[BatchContextValue] = ContextVar("ldb_batch", default=None)


def current_batch_context() -> BatchContext | None:
    value = _CURRENT_BATCH.get()
    return value() if callable(value) else value


def set_batch_context(context: BatchContext | BatchContextProvider) -> Token[BatchContextValue]:
    return _CURRENT_BATCH.set(context)


def reset_batch_context(token: Token[BatchContextValue]) -> None:
    _CURRENT_BATCH.reset(token)


def context_from_sglang(forward_batch: Any) -> BatchContext:
    """Build token spans from SGLang's public ForwardBatch attributes."""

    request_ids = tuple(str(value) for value in (getattr(forward_batch, "rids", None) or ()))
    mode = getattr(forward_batch, "forward_mode", None)
    phase = _sglang_phase(mode)
    lengths = _sglang_lengths(forward_batch, phase, len(request_ids))
    if len(lengths) != len(request_ids):
        raise ValueError("SGLang request IDs and token lengths have different sizes")

    offset = 0
    spans: list[RequestSpan] = []
    for request_id, length in zip(request_ids, lengths, strict=True):
        if length < 0:
            raise ValueError("SGLang token lengths must not be negative")
        spans.append(RequestSpan(request_id=request_id, start=offset, end=offset + length))
        offset += length
    return BatchContext(phase=phase, requests=tuple(spans))


def context_from_vllm(runner: Any, scheduler_output: Any) -> BatchContext:
    """Build token spans after vLLM has finalized its internal InputBatch order."""

    scheduled = scheduler_output.num_scheduled_tokens
    request_ids = tuple(
        str(request_id)
        for request_id in runner.input_batch.req_ids
        if request_id in scheduled and int(scheduled[request_id]) > 0
    )
    lengths = tuple(int(scheduled[request_id]) for request_id in request_ids)
    spans: list[RequestSpan] = []
    offset = 0
    for request_id, length in zip(request_ids, lengths, strict=True):
        spans.append(RequestSpan(request_id=request_id, start=offset, end=offset + length))
        offset += length

    phases = _vllm_phases(runner, request_ids)
    phase = phases.pop() if len(phases) == 1 else "mixed"
    return BatchContext(phase=phase, requests=tuple(spans))


def _sglang_phase(mode: Any) -> str:
    for method, phase in (("is_extend", "prefill"), ("is_decode", "decode")):
        predicate = getattr(mode, method, None)
        if callable(predicate) and predicate():
            return phase
    return str(mode).lower() if mode is not None else "unknown"


def _sglang_lengths(forward_batch: Any, phase: str, count: int) -> tuple[int, ...]:
    if phase == "prefill":
        values = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if values is not None:
            return tuple(int(value) for value in values[:count])
    return (1,) * count


def _vllm_phases(runner: Any, request_ids: tuple[str, ...]) -> set[str]:
    phases: set[str] = set()
    batch = runner.input_batch
    for request_id in request_ids:
        index = batch.req_id_to_index[request_id]
        computed = int(batch.num_computed_tokens_cpu[index])
        prompt = int(batch.num_prompt_tokens[index])
        phases.add("prefill" if computed < prompt else "decode")
    return phases or {"unknown"}
