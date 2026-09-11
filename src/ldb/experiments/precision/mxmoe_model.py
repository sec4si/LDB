"""Model adapter for upstream mixed-precision grouped GEMM correctness testing."""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


class PaperBank(nn.Module):
    def __init__(self, experts, plan, bridge, kernel, down_rotation=None, device=None):
        super().__init__()
        if len(plan) != experts.gate_up_proj.shape[0]:
            raise ValueError("Incomplete expert plan")
        self.bridge, self.kernel = bridge, kernel
        self.grouped_routing = False
        self.reuse_prepared = False
        self.prepared_calls = OrderedDict()
        self.prepared_hits = 0
        self.prepared_misses = 0
        self.register_buffer("down_rotation", down_rotation)
        self.shapes = {}
        for e, choice in enumerate(plan):
            for part, weight in [("gu", experts.gate_up_proj[e]), ("down", experts.down_proj[e])]:
                if device is not None:
                    weight = weight.to(device)
                bits = choice[part] if isinstance(choice, dict) else choice
                if bits not in (4, 8):
                    raise ValueError("Expected INT4 or INT8")
                if part == "down" and down_rotation is not None:
                    weight = weight.float() @ down_rotation.float()
                if weight.is_meta:
                    n, k = weight.shape
                    packed = torch.empty(n * k // (16 // bits), device="meta", dtype=torch.float16)
                    scale = torch.empty(n * 2, device="meta", dtype=torch.float16)
                else:
                    (packed, scale, n, k, bits), _ = bridge.pack(
                        weight.detach().half().contiguous(), bits
                    )
                self.register_buffer(f"{part}_{e}_packed", packed)
                self.register_buffer(f"{part}_{e}_scale", scale)
                self.shapes[part, e] = (n, k, bits)
        self.num_experts = len(plan)

    def matrix(self, part, expert):
        return (
            getattr(self, f"{part}_{expert}_packed"),
            getattr(self, f"{part}_{expert}_scale"),
            *self.shapes[part, expert],
        )

    def run_group(self, part, active, inputs):
        matrices = [self.matrix(part, e) for e in active]
        if not self.reuse_prepared or sum(x.shape[0] for x in inputs) > 32:
            return self.bridge.run(matrices, inputs, self.kernel)
        key = (
            part,
            self.kernel,
            tuple(
                (m[0].data_ptr(), m[1].data_ptr(), *m[2:], tuple(x.shape))
                for m, x in zip(matrices, inputs)
            ),
        )
        call = self.prepared_calls.pop(key, None)
        if call is None:
            self.prepared_misses += 1
            call = self.bridge.prepare(matrices, [x.clone() for x in inputs], self.kernel)
        else:
            self.prepared_hits += 1
            for buffer, x in zip(call.inputs, inputs):
                buffer.copy_(x)
        self.prepared_calls[key] = call
        if len(self.prepared_calls) > 32:
            self.prepared_calls.popitem(last=False)
        return call()

    def forward(self, x, indices, weights):
        if torch.is_grad_enabled():
            raise ValueError("Paper backend is inference-only")
        if self.grouped_routing:
            return self.forward_grouped(x, indices, weights)
        active = []
        tokens = []
        scores = []
        inputs = []
        for expert in range(self.num_experts):
            token, slot = torch.where(indices == expert)
            if not token.numel():
                continue
            active.append(expert)
            tokens.append(token)
            scores.append(weights[token, slot, None])
            inputs.append(x[token].half().contiguous())
        result = torch.zeros_like(x)
        if not active:
            return result
        gu = self.run_group("gu", active, inputs)
        hidden = []
        for value in gu:
            gate, up = value.chunk(2, dim=-1)
            activation = F.silu(gate) * up
            if self.down_rotation is not None:
                activation = (activation.float() @ self.down_rotation.float()).half()
            hidden.append(activation.contiguous())
        outputs = self.run_group("down", active, hidden)
        for token, score, value in zip(tokens, scores, outputs):
            result.index_add_(0, token, (value.float() * score).to(x.dtype))
        return result

    def forward_grouped(self, x, indices, weights):
        flat = indices.flatten()
        if not flat.numel():
            return torch.zeros_like(x)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=self.num_experts).cpu().tolist()
        active = [e for e, c in enumerate(counts) if c]
        sizes = [counts[e] for e in active]
        tokens = order // indices.shape[1]
        inputs = x[tokens].half().contiguous().split(sizes)
        gu = torch.cat(self.run_group("gu", active, inputs))
        gate, up = gu.chunk(2, -1)
        hidden = F.silu(gate) * up
        hidden = hidden.contiguous().split(sizes)
        if self.down_rotation is not None:
            # Preserve per-expert GEMM shapes: changing M can change FP16 rounding.
            hidden = [(value.float() @ self.down_rotation.float()).half() for value in hidden]
        outputs = torch.cat(self.run_group("down", active, hidden))
        contributions = torch.empty_like(outputs, dtype=x.dtype)
        contributions[order] = (outputs.float() * weights.flatten()[order, None]).to(x.dtype)
        contributions = contributions.view(x.shape[0], indices.shape[1], x.shape[1])
        expert_order = indices.argsort(dim=1)
        contributions = contributions.gather(1, expert_order[:, :, None].expand_as(contributions))
        result = torch.zeros_like(x)
        # Match the original expert loop's rounding after every contribution.
        for slot in range(indices.shape[1]):
            result.add_(contributions[:, slot])
        return result


def convert(model, plan, bridge):
    if len(plan) != len(model.model.layers):
        raise ValueError("Incomplete layer plan")
    kernel = next(
        i for i, n in enumerate(bridge.names) if "w4a16_g-1_asym:" in n and "w8a16_g-1_asym:" in n
    )
    for layer, choices in zip(model.model.layers, plan):
        layer.mlp.experts = PaperBank(layer.mlp.experts, choices, bridge, kernel)
    return model
