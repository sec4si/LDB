"""MLX local model implementation for physically shared Qwen MoE experts.

The original router and its scores remain unchanged. Several router slots can
address one stored expert; their contributions are still accumulated separately.
"""

from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.qwen2_moe import Model as QwenModel
from mlx_lm.models.qwen2_moe import ModelArgs as QwenArgs
from mlx_lm.models.switch_layers import SwitchGLU


def validate_map(mapping, num_experts):
    if len(mapping) != num_experts or any(
        type(e) is not int or not 0 <= e < num_experts for e in mapping
    ):
        raise ValueError("Expert map must contain one valid integer donor per router slot")
    retained = sorted(set(mapping))
    if any(mapping[e] != e for e in retained):
        raise ValueError("Donors must be retained original experts, without chains")
    return retained


class RemappedSwitchGLU(SwitchGLU):
    def __init__(self, input_dims, hidden_dims, mapping, compact=True):
        retained = validate_map(mapping, len(mapping))
        slots = {e: i for i, e in enumerate(retained)}
        super().__init__(input_dims, hidden_dims, len(retained) if compact else len(mapping))
        self._expert_indices = mx.array([slots[e] for e in mapping] if compact else mapping)

    def __call__(self, x, indices):
        return super().__call__(x, self._expert_indices[indices])


@dataclass
class ModelArgs(QwenArgs):
    expert_maps: list[list[int]] | None = None


class Model(QwenModel):
    def __init__(self, args):
        super().__init__(args)
        if args.expert_maps is None or len(args.expert_maps) != args.num_hidden_layers:
            raise ValueError("A frozen expert map is required for every layer")
        for layer, mapping in zip(self.model.layers, args.expert_maps):
            validate_map(mapping, args.num_experts)
            layer.mlp.switch_mlp = RemappedSwitchGLU(
                args.hidden_size,
                args.moe_intermediate_size,
                mapping,
            )
