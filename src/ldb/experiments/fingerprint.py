"""Group equal gate vectors by their full output-input coefficient matrix.

The logical descriptor is (gate, sum_i down_i outer up_i), sorted by gate.
Factors are retained as a storage/evaluation implementation, not as a unique
byte serialization. Equality below compares the logical coefficients.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GroupedFingerprint:
    gates: torch.Tensor
    groups: list
    input_dim: int
    output_dim: int

    @classmethod
    def from_weights(cls, gate_up, down):
        gate, up = gate_up.detach().cpu().double().chunk(2, 0)
        down = down.detach().cpu().double()
        if down.shape[1] != len(gate):
            raise ValueError("Mismatched intermediate dimensions")
        gates, inverse = np.unique(gate.numpy(), axis=0, return_inverse=True)
        result_gates = []
        groups = []
        for i, g in enumerate(gates):
            indices = np.flatnonzero(inverse == i)
            u, d = up[indices], down[:, indices]
            if not np.any(g) or not torch.any(u) or not torch.any(d):
                continue
            if len(indices) > 1 and not torch.any(coefficient(u, d)):
                continue
            result_gates.append(torch.from_numpy(g.copy()))
            groups.append((u, d))
        gs = (
            torch.stack(result_gates)
            if result_gates
            else torch.empty((0, gate.shape[1]), dtype=torch.float64)
        )
        return cls(gs, groups, gate.shape[1], down.shape[0])

    def __call__(self, x):
        x = x.double()
        y = torch.zeros((len(x), self.output_dim), dtype=torch.float64)
        if not self.groups:
            return y
        gates = torch.cat(
            [
                g[None, :].expand(len(u), -1)
                for g, (u, _) in zip(self.gates, self.groups, strict=True)
            ]
        )
        up = torch.cat([u for u, _ in self.groups])
        down = torch.cat([d for _, d in self.groups], 1)
        return (torch.nn.functional.silu(x @ gates.T) * (x @ up.T)) @ down.T

    def discrepancy(self, other):
        if not torch.equal(self.gates, other.gates):
            return float("inf")
        num = 0.0
        den = 0.0
        for (u, d), (v, e) in zip(self.groups, other.groups, strict=True):
            # Exact factor identity avoids expanding unchanged groups.
            if torch.equal(u, v) and torch.equal(d, e):
                den += (
                    float(d.norm().square() * u.norm().square())
                    if len(u) == 1
                    else float(coefficient(u, d).square().sum())
                )
                continue
            a, b = coefficient(u, d), coefficient(v, e)
            num += float((a - b).square().sum())
            den += float(a.square().sum())
        return (num / max(den, 1e-300)) ** 0.5


def coefficient(up, down):
    matrix = torch.zeros((down.shape[0], up.shape[1]), dtype=torch.float64)
    for u, d in zip(up, down.T, strict=True):
        matrix += d[:, None] * u[None, :]
    return matrix
