"""CUDA Graph-compatible intervention and trace slots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ldb.models import TensorSummary


@dataclass
class GraphControlSlot:
    """A fixed-address scale/bias buffer read by captured tensor operations."""

    control: Any

    @classmethod
    def create(cls, tensor: Any) -> GraphControlSlot:
        import torch

        if not tensor.is_floating_point():
            raise TypeError("Graph control slots require floating-point activations")
        return cls(torch.tensor([1.0, 0.0], device=tensor.device, dtype=tensor.dtype))

    def apply(self, tensor: Any) -> Any:
        return tensor * self.control[0] + self.control[1]

    def update(self, *, scale: float, bias: float) -> None:
        import torch

        value = torch.tensor([scale, bias], device=self.control.device, dtype=self.control.dtype)
        self.control.copy_(value)
        if self.control.device.type == "cuda":
            torch.cuda.synchronize(self.control.device)


@dataclass
class GraphTraceSlot:
    """Fixed-address summaries updated by eager calls or captured graph replay."""

    counter: Any
    observed: Any
    patched: Any
    shape: Any
    dtype: str
    device: str

    @classmethod
    def create(cls, tensor: Any) -> GraphTraceSlot:
        import torch

        if not tensor.is_floating_point():
            raise TypeError("Graph trace slots require floating-point activations")
        return cls(
            counter=torch.zeros((), device=tensor.device, dtype=torch.int64),
            observed=torch.zeros(3, device=tensor.device, dtype=torch.float32),
            patched=torch.zeros(3, device=tensor.device, dtype=torch.float32),
            shape=torch.full((9,), -1, device=tensor.device, dtype=torch.int64),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
        )

    def observe(self, tensor: Any, patched: Any) -> None:
        if tensor.ndim > len(self.shape) - 1:
            raise ValueError("Graph trace slots support tensors with at most 8 dimensions")
        self.shape.fill_(-1)
        self.shape[0].fill_(tensor.ndim)
        for index, size in enumerate(tensor.shape, start=1):
            self.shape[index].fill_(size)
        _write_summary(self.observed, tensor)
        _write_summary(self.patched, patched)
        self.counter.add_(1)

    def snapshot(self, *, synchronize: bool = True) -> tuple[int, TensorSummary, TensorSummary]:
        if synchronize and self.counter.device.type == "cuda":
            import torch

            torch.cuda.synchronize(self.counter.device)
        count = int(self.counter.item())
        return count, self._summary(self.observed), self._summary(self.patched)

    def _summary(self, values: Any) -> TensorSummary:
        mean, std, norm = (float(value) for value in values.tolist())
        shape_values = [int(value) for value in self.shape.tolist()]
        rank = shape_values[0]
        return TensorSummary(
            shape=tuple(shape_values[1 : rank + 1]) if rank >= 0 else (),
            dtype=self.dtype,
            device=self.device,
            mean=mean,
            std=std,
            norm=norm,
        )


def _write_summary(output: Any, tensor: Any) -> None:
    values = tensor.detach().float()
    output[0].copy_(values.mean())
    output[1].copy_(values.std(correction=0))
    output[2].copy_(values.norm())
