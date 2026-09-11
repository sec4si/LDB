"""Independent gate/up/down formats with fused GU when their formats agree."""

import torch
from torch import nn

from .mxmoe_activation_probe import pack8, prepare
from .mxmoe_model import PaperBank


class LinearBank(PaperBank):
    def __init__(self, experts, plan, bridge, kernel):
        nn.Module.__init__(self)
        if len(plan) != experts.gate_up_proj.shape[0]:
            raise ValueError("Incomplete expert plan")
        if any(
            set(p) != {"gate", "up", "down"} or any(v not in ("w4a16", "w8a8") for v in p.values())
            for p in plan
        ):
            raise ValueError("Expected independent gate/up/down W4A16 or W8A8 formats")
        self.bridge, self.kernel = bridge, kernel
        self.num_experts = len(plan)
        self.formats = plan
        self.shapes = {}
        self.grouped_routing = True
        self.reuse_prepared = False
        self.fused_activation_pack = False
        self.coalesce_metadata = False
        self.register_buffer("down_rotation", None)
        self.layouts = [("gu",) if p["gate"] == p["up"] else ("gate", "up") for p in plan]
        for e, layout in enumerate(self.layouts):
            gate, up = experts.gate_up_proj[e].chunk(2, 0)
            weights = {
                "gu": experts.gate_up_proj[e],
                "gate": gate,
                "up": up,
                "down": experts.down_proj[e],
            }
            for part in (*layout, "down"):
                weight = weights[part]
                fmt = plan[e]["gate" if part == "gu" else part]
                n, k = weight.shape
                bits = 4 if fmt == "w4a16" else 8
                if weight.is_meta:
                    packed = torch.empty(
                        (n * k // 4,) if bits == 4 else (n, k // 2),
                        device="meta",
                        dtype=torch.float16,
                    )
                    scale = torch.empty(
                        n * (2 if bits == 4 else 1), device="meta", dtype=torch.float16
                    )
                elif bits == 4:
                    (packed, scale, _, _, _), _ = bridge.pack(weight.half().contiguous(), 4)
                else:
                    packed, scale, _ = pack8(weight.half().contiguous())
                self.register_buffer(f"{part}_{e}_packed", packed)
                self.register_buffer(f"{part}_{e}_scale", scale)
                self.shapes[part, e] = (n, k, bits)

    def run_group(self, part, active, inputs):
        jobs = [
            (e, p, x)
            for e, x in zip(active, inputs, strict=True)
            for p in (self.layouts[e] if part == "gu" else ("down",))
        ]
        matrices = []
        packed = []
        scales = []
        abits = []
        for e, p, x in jobs:
            matrix = self.matrix(p, e)
            matrices.append(matrix)
            if matrix[-1] == 8:
                if self.fused_activation_pack:
                    from .mxmoe_activation_pack import pack8_fused

                    value, scale = pack8_fused(x)
                else:
                    value, scale, _ = pack8(x)
                packed.append(value)
                scales.append(scale)
                abits.append(8)
            else:
                packed.append(x)
                scales.append(matrix[1])
                abits.append(16)
        outputs = iter(
            prepare(
                self.bridge,
                self.kernel,
                matrices,
                packed,
                scales,
                abits,
                coalesce=self.coalesce_metadata,
            )()
        )
        return [
            torch.cat([next(outputs), next(outputs)], -1)
            if part == "gu" and len(self.layouts[e]) == 2
            else next(outputs)
            for e in active
        ]
