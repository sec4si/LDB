"""Inference-only W4A16/W8A16: fused unpacking and BF16 tensor-core GEMM."""

import torch
import triton
import triton.language as tl


@triton.jit
def _linear(
    X,
    W,
    S,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    INT4: tl.constexpr,
    BM: tl.constexpr = 16,
    BN: tl.constexpr = 32,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, 64)
    acc = tl.full((BM, BN), 0, tl.float32)
    for group in range(K // 64):
        k = group * 64 + inner
        x = tl.load(X + rows[:, None] * K + k[None, :], rows[:, None] < M, other=0).to(tl.bfloat16)
        offset = cols[None, :] * K + k[:, None]
        if INT4:
            packed = tl.load(W + offset // 2, cols[None, :] < N, other=0)
            quant = ((packed >> ((offset % 2) * 4)) & 15).to(tl.float32) - 8
        else:
            quant = tl.load(W + offset, cols[None, :] < N, other=0).to(tl.float32)
        scale = tl.load(S + cols * (K // 64) + group, cols < N, other=0)
        w = (quant * scale[None, :]).to(tl.bfloat16)
        acc = tl.dot(x, w, acc)
    tl.store(Y + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))


def linear(x, matrix):
    if not x.is_cuda or x.device != matrix.weight.device:
        raise ValueError("Triton backend requires inputs and weights on the same CUDA device")
    if x.requires_grad and torch.is_grad_enabled():
        raise ValueError("Triton precision backend is inference-only")
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Expected floating-point activations")
    n, k = matrix.shape
    if x.shape[-1] != k or k % 64:
        raise ValueError("Activation width must match the group-64 packed matrix")
    if matrix.precision == "bf16":
        return torch.nn.functional.linear(x.to(torch.bfloat16), matrix.weight).to(x.dtype)
    if matrix.precision not in ("int4", "int8"):
        raise ValueError(matrix.precision)
    flat = x.reshape(-1, k).contiguous()
    output = torch.empty((flat.shape[0], n), device=x.device, dtype=x.dtype)
    if flat.shape[0]:
        with torch.cuda.device(x.device):
            _linear[(triton.cdiv(flat.shape[0], 16), triton.cdiv(n, 32))](
                flat,
                matrix.weight,
                matrix.scale,
                output,
                flat.shape[0],
                n,
                k,
                matrix.precision == "int4",
            )
    return output.reshape(*x.shape[:-1], n)


def enable(model):
    """Select explicitly; never silently fall back to dense FP32 execution."""
    from .expert_precision import PackedMatrix

    matrices = [m for m in model.modules() if isinstance(m, PackedMatrix)]
    if not matrices:
        raise ValueError("Model has no packed expert matrices")
    if any(not m.weight.is_cuda for m in matrices):
        raise ValueError("Move the model to CUDA before enabling the Triton backend")
    for matrix in matrices:
        matrix.backend = "triton"
    return model
