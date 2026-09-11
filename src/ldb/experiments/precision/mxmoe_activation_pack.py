"""Fused per-row INT8 packing matching the mixed-kernel FP16 contract."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(X, Y, S, K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * K + col, col < K, other=0).to(tl.float32)
    scale = tl.div_rn(tl.max(tl.abs(x), 0), 127.0).to(tl.float16).to(tl.float32)
    scale = tl.where(scale == 0, 1.0, scale)
    q = tl.div_rn(x, scale).to(tl.float16).to(tl.float32)
    q = tl.extra.cuda.libdevice.nearbyint(q)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int32)
    # CUDA is little endian; the first code occupies the high byte.
    tl.store(Y + row * K + (col ^ 1), q.to(tl.int8), col < K)
    tl.store(S + row, scale)


def pack8_fused(value):
    if not value.is_cuda or value.dtype != torch.float16 or value.ndim != 2:
        raise ValueError("Expected a CUDA FP16 matrix")
    if not value.is_contiguous() or value.shape[1] % 2 or value.shape[1] == 0:
        raise ValueError("Expected contiguous rows with positive even K")
    m, k = value.shape
    packed = torch.empty((m, k // 2), device=value.device, dtype=torch.float16)
    scale = torch.empty(m, device=value.device, dtype=torch.float16)
    if m:
        with torch.cuda.device(value.device):
            _pack[(m,)](value, packed.view(torch.int8), scale, k, triton.next_power_of_2(k))
    return packed, scale
