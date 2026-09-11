"""Experimental routed W4A16/W8A8 bank using the corrected mixed kernel."""

from .mxmoe_activation_probe import pack8, prepare
from .mxmoe_model import PaperBank


class ActivationBank(PaperBank):
    def __init__(self, experts, plan, bridge, kernel):
        if any(
            set(p) != {"gu", "down"} or any(v not in ("w4a16", "w8a8") for v in p.values())
            for p in plan
        ):
            raise ValueError("Expected explicit GU/down W4A16 or W8A8 formats")
        bits = [{part: 4 if fmt == "w4a16" else 8 for part, fmt in p.items()} for p in plan]
        super().__init__(experts, bits, bridge, kernel)
        self.formats = plan
        self.grouped_routing = True
        self.fused_activation_pack = False
        self.coalesce_metadata = False
        for e, p in enumerate(plan):
            for part, weight in (("gu", experts.gate_up_proj[e]), ("down", experts.down_proj[e])):
                if p[part] == "w8a8":
                    packed, scale, _ = pack8(weight.half().contiguous())
                    setattr(self, f"{part}_{e}_packed", packed)
                    setattr(self, f"{part}_{e}_scale", scale)

    def run_group(self, part, active, inputs):
        matrices = [self.matrix(part, e) for e in active]
        packed = []
        scales = []
        abits = []
        for e, x, matrix in zip(active, inputs, matrices):
            if self.formats[e][part] == "w8a8":
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
        return prepare(
            self.bridge,
            self.kernel,
            matrices,
            packed,
            scales,
            abits,
            coalesce=self.coalesce_metadata,
        )()
