"""CUDA Graph-compatible scalar intervention control slots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
