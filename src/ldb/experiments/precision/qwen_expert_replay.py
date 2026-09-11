"""Reuse unchanged expert contributions while preserving native FP16 accumulation."""

import torch
from torch.nn import functional as F


def expert_contribution(experts, x, indices, weights, expert):
    slot, token = torch.where(indices.T == expert)
    selected = x[token]
    gate, up = F.linear(selected, experts.gate_up_proj[expert]).chunk(2, dim=-1)
    value = F.linear(experts.act_fn(gate) * up, experts.down_proj[expert])
    return token, (value * weights[token, slot, None]).to(x.dtype)


@torch.inference_mode()
def capture(experts, x, indices, weights):
    return {
        e: expert_contribution(experts, x, indices, weights, e)
        for e in range(experts.num_experts)
        if bool((indices == e).any())
    }


@torch.inference_mode()
def replay(x, contributions, replacement=None):
    result = torch.zeros_like(x)
    for expert, (token, value) in contributions.items():
        if replacement is not None and replacement[0] == expert:
            token, value = replacement[1]
        result.index_add_(0, token, value)
    return result


def layer_output(expert_output, shared_output, residual):
    return residual + (expert_output + shared_output).reshape_as(residual)


@torch.inference_mode()
def ordered_contributions(x, indices, contributions):
    values = torch.empty((*indices.shape, x.shape[-1]), device=x.device, dtype=x.dtype)
    for expert, (token, value) in contributions.items():
        rank = (indices[token] < expert).sum(dim=1)
        values[token, rank] = value
    return values


@torch.inference_mode()
def perturbed_layer_rows(values, indices, expert, changed, shared, residual):
    token, value = changed
    selected = values[token].clone()
    rank = (indices[token] < expert).sum(dim=1)
    selected[torch.arange(token.numel(), device=token.device), rank] = value
    result = torch.zeros_like(value)
    for slot in range(indices.shape[1]):
        result.add_(selected[:, slot])
    return token, residual.reshape(-1, result.shape[-1])[token] + (result + shared[token])
