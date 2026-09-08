"""Live stepping at eager PyTorch dispatcher boundaries, without graph replay."""

from __future__ import annotations

import fnmatch
import math
import shlex
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ldb.gradients import tensor_leaves


class _Cancelled(BaseException):
    pass


def describe(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": value.requires_grad,
        }
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


class OperatorDebugger:
    """Run one callable once, pausing before each selected visible ATen operation.

    A private worker preserves the live Python/autograd stack. The caller must own
    the model exclusively. This does not step instructions inside fused kernels.
    """

    def __init__(
        self,
        function: Callable[[], Any],
        *,
        timeout: float = 30.0,
        max_history: int = 128,
        max_operations: int = 100000,
        synchronize: bool = True,
    ) -> None:
        import torch

        if not math.isfinite(timeout) or timeout <= 0 or max_history < 1 or max_operations < 1:
            raise ValueError("Invalid debugger limits")
        self.function = function
        self.timeout = timeout
        self.max_operations = max_operations
        self.synchronize = synchronize
        self.condition = threading.Condition()
        self.history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self.breakpoints: set[str] = set()
        self.current: dict[str, Any] | None = None
        self.arguments: Any = None
        self.outputs: Any = None
        self.state = "new"
        self.count = 0
        self.error: BaseException | None = None
        self.result: Any = None
        self._pause_next = True
        self._allowed = False
        self._cancelled = False
        self._grad_enabled = torch.is_grad_enabled()
        self._inference = torch.is_inference_mode_enabled()
        self.thread: threading.Thread | None = None

    def _dispatch(self, func: Any, args: Any, kwargs: Any) -> Any:
        with self.condition:
            if self._cancelled:
                raise _Cancelled
            if self.count >= self.max_operations:
                raise RuntimeError("Operator execution limit reached")
            event = {
                "index": self.count,
                "operator": str(func),
                "args": describe(args),
                "kwargs": describe(kwargs),
            }
            self.current = event
            self.arguments = (args, kwargs)
            if self._pause_next or any(fnmatch.fnmatchcase(str(func), p) for p in self.breakpoints):
                self.state = "paused"
                self.condition.notify_all()
                self.condition.wait_for(lambda: self._allowed or self._cancelled)
                self._allowed = False
            if self._cancelled:
                raise _Cancelled
            self.state = "running"
        output = func(*args, **kwargs)
        if self.synchronize:
            import torch

            devices = {t.device for _, t in tensor_leaves((args, kwargs, output))}
            for device in devices:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elif device.type == "mps":
                    torch.mps.synchronize()
        with self.condition:
            event["outputs"] = describe(output)
            self.outputs = output
            self.history.append(event)
            self.count += 1
            self.current = None
            self.arguments = None
        return output

    def _execute(self) -> Any:
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode

        owner = self

        class Mode(TorchDispatchMode):  # type: ignore[no-untyped-call]
            def __torch_dispatch__(
                self, func: Any, types: Any, args: Any = (), kwargs: Any = None
            ) -> Any:
                return owner._dispatch(func, args, kwargs or {})

        mode = Mode()  # type: ignore[no-untyped-call]
        with (
            torch.inference_mode(self._inference),
            torch.set_grad_enabled(self._grad_enabled),
            mode,
        ):
            return self.function()

    def _run(self) -> None:
        try:
            result = self._execute()
            with self.condition:
                self.result = result
                self.state = "finished"
        except _Cancelled:
            with self.condition:
                self.state = "cancelled"
        except BaseException as error:
            with self.condition:
                self.error = error
                self.state = "failed"
        finally:
            with self.condition:
                self.current = None
                self.arguments = None
                self.condition.notify_all()

    def _wait(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        with self.condition:
            while self.state in {"starting", "running"}:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Execution is still running; inspect status or close this session"
                    )
                self.condition.wait(remaining)
            if self.error is not None:
                raise RuntimeError(f"Operator execution failed: {self.error}") from self.error
            return self.inspect()

    def start(self) -> dict[str, Any]:
        with self.condition:
            if self.state != "new":
                raise ValueError("This execution has already started")
            self.state = "starting"
            self.thread = threading.Thread(target=self._run, name="ldb-operator", daemon=True)
            self.thread.start()
        return self._wait()

    def inspect(self) -> dict[str, Any]:
        with self.condition:
            return {
                "state": self.state,
                "completed_operations": self.count,
                "current": self.current,
                "last": self.history[-1] if self.history else None,
                "breakpoints": sorted(self.breakpoints),
                "error": str(self.error) if self.error else None,
            }

    def advance(self, *, single: bool) -> dict[str, Any]:
        with self.condition:
            if self.state != "paused":
                raise ValueError("Execution must be paused")
            self._pause_next = single
            self._allowed = True
            self.state = "running"
            self.condition.notify_all()
        return self._wait()

    def capture(self, path: Path, *, outputs: bool = False) -> dict[str, Any]:
        from safetensors.torch import save_file

        with self.condition:
            if self.state not in {"paused", "finished"}:
                raise ValueError("Capture requires a paused or completed execution")
            tensors = dict(tensor_leaves(self.outputs if outputs else self.arguments, "tensor"))
            if not tensors:
                raise ValueError("No tensors at this boundary")
            if path.exists():
                raise FileExistsError(path)
            copied = {
                key: value.detach().cpu().contiguous().clone() for key, value in tensors.items()
            }
            save_file(copied, str(path))
            return {"path": str(path), "keys": list(copied), "outputs": outputs}

    def where(self) -> list[dict[str, Any]]:
        with self.condition:
            if self.state != "paused" or self.thread is None:
                raise ValueError("Stack inspection requires a paused execution")
            frame = sys._current_frames().get(self.thread.ident or -1)
            if frame is None:
                raise RuntimeError("Worker frame is unavailable")
            return [
                {"file": entry.filename, "line": entry.lineno, "function": entry.name}
                for entry in traceback.extract_stack(frame, limit=30)
            ]

    def close(self) -> None:
        with self.condition:
            self._cancelled = True
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=self.timeout)
            if self.thread.is_alive():
                raise TimeoutError("Worker is inside an operation/Python call; it has not stopped")
        self.arguments = self.outputs = None

    def __enter__(self) -> OperatorDebugger:
        try:
            self.start()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def command(self, text: str) -> Any:
        args = shlex.split(text)
        match args:
            case ["step"] | ["next"]:
                return self.advance(single=True)
            case ["continue"]:
                return self.advance(single=False)
            case ["inspect"] | ["status"]:
                return self.inspect()
            case ["history"]:
                with self.condition:
                    return list(self.history)
            case ["where"]:
                return self.where()
            case ["break", pattern]:
                with self.condition:
                    self.breakpoints.add(pattern)
                return self.inspect()
            case ["delete", pattern]:
                with self.condition:
                    self.breakpoints.discard(pattern)
                return self.inspect()
            case ["capture", path]:
                return self.capture(Path(path))
            case ["capture-output", path]:
                return self.capture(Path(path), outputs=True)
            case _:
                raise ValueError(
                    "Commands: step, continue, inspect, where, history, break/delete PATTERN, "
                    "capture/capture-output PATH"
                )
