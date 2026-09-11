"""Mixed W4A16/W8A8 numerical probe; no inference performance claim."""

import argparse
import ctypes as C
from pathlib import Path

import torch

from .helpers import sha, write
from .mxmoe_bridge import Bridge, PreparedCall


def pack8(value):
    if value.dtype != torch.float16 or value.ndim != 2 or value.shape[1] % 2:
        raise ValueError("Expected FP16 matrix with even K")
    scale = value.abs().amax(1, keepdim=True) / 127
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    codes = (value / scale).round().clamp(-127, 127).to(torch.int32)
    pairs = codes.reshape(value.shape[0], -1, 2) & 255
    packed = ((pairs[:, :, 0] << 8) | pairs[:, :, 1]).to(torch.int16).view(torch.float16)
    return packed.contiguous(), scale.flatten().contiguous(), codes


def prepare(bridge, kernel, matrices, inputs, activation_scales, abits, coalesce=False):
    bridge.default_stream(inputs[0].device)
    if coalesce:
        return prepare_coalesced(bridge, kernel, matrices, inputs, activation_scales, abits)

    def ptrs(tensors):
        return torch.tensor([t.data_ptr() for t in tensors], device="cuda", dtype=torch.int64)

    dims = torch.tensor(
        [[x.shape[0], m[2], m[3]] for m, x in zip(matrices, inputs)], dtype=torch.int32
    )
    quant = torch.tensor(
        [[a, m[4], -1, int(a == 8)] for m, a in zip(matrices, abits)], dtype=torch.int32
    )
    outputs = [
        torch.empty((x.shape[0], m[2]), device="cuda", dtype=torch.float16)
        for m, x in zip(matrices, inputs)
    ]
    ks = torch.tensor([m[3] for m in matrices], device="cuda", dtype=torch.int64)
    ns = torch.tensor([m[2] for m in matrices], device="cuda", dtype=torch.int64)
    op = ptrs(outputs)
    args = [
        ptrs(inputs),
        ptrs([m[0] for m in matrices]),
        ptrs(activation_scales),
        ptrs([m[1] for m in matrices]),
        op,
        op,
        ks,
        ks,
        ns,
        ns,
        dims.cuda(),
        dims,
        quant.cuda(),
        quant,
    ]
    addresses = (C.c_void_p * 14)(*[t.data_ptr() for t in args])
    call = PreparedCall(bridge, kernel, matrices, inputs, outputs, args, addresses)
    call.activation_scales = activation_scales
    return call


def prepare_coalesced(bridge, kernel, matrices, inputs, activation_scales, abits):
    count = len(matrices)
    outputs = [
        torch.empty((x.shape[0], m[2]), device=x.device, dtype=torch.float16)
        for m, x in zip(matrices, inputs)
    ]
    pointers = [
        [t.data_ptr() for t in tensors]
        for tensors in (
            inputs,
            [m[0] for m in matrices],
            activation_scales,
            [m[1] for m in matrices],
            outputs,
        )
    ]
    pointers.extend(([m[3] for m in matrices], [m[2] for m in matrices]))
    dims = torch.tensor(
        [[x.shape[0], m[2], m[3]] for m, x in zip(matrices, inputs)], dtype=torch.int32
    )
    quant = torch.tensor(
        [[a, m[4], -1, int(a == 8)] for m, a in zip(matrices, abits)], dtype=torch.int32
    )
    # Preserve 16-byte alignment for kernel vector loads at every array start.
    pitch = (count + 3) // 4 * 4
    host = torch.zeros(21 * pitch, dtype=torch.int32)
    host[: 14 * pitch].view(torch.int64).reshape(7, pitch)[:, :count].copy_(
        torch.tensor(pointers, dtype=torch.int64)
    )
    host[14 * pitch : 14 * pitch + 3 * count].copy_(dims.flatten())
    host[17 * pitch : 17 * pitch + 4 * count].copy_(quant.flatten())
    device = host.to(inputs[0].device)
    xp, wp, ap, sp, op, ks, ns = (
        device[: 14 * pitch].view(torch.int64).reshape(7, pitch)[:, :count].unbind()
    )
    gd = device[14 * pitch : 14 * pitch + 3 * count].reshape(count, 3)
    gq = device[17 * pitch : 17 * pitch + 4 * count].reshape(count, 4)
    args = [xp, wp, ap, sp, op, op, ks, ks, ns, ns, gd, dims, gq, quant]
    addresses = (C.c_void_p * 14)(*[t.data_ptr() for t in args])
    call = PreparedCall(bridge, kernel, matrices, inputs, outputs, args, addresses)
    call.activation_scales = activation_scales
    call.metadata_storage = (host, device)
    return call


@torch.inference_mode()
def run(library, output, coalesce=False):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260911)
    bridge = Bridge(library)
    write(
        output / "freeze.json",
        dict(
            library_sha256=sha(library),
            script_sha256=sha(Path(__file__)),
            kernels=bridge.names,
            coalesce_metadata=coalesce,
            scope="Synthetic mixed-format packing and GEMM correctness only",
        ),
    )
    rows = []
    for m, n, k in (
        (1, 384, 768),
        (17, 384, 768),
        (128, 1536, 384),
        (17, 2816, 2048),
        (128, 2048, 1408),
        (257, 2048, 1408),
        (513, 2816, 2048),
    ):
        x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
        w = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.03
        w4, fake = bridge.pack(w, 4)
        a8, sa, qa = pack8(x)
        b8, sb, qb = pack8(w)
        w8 = (b8, sb, n, k, 8)
        refs = [
            (x.float() @ fake.float().T).half(),
            ((qa.float() @ qb.float().T) * (sa[:, None] * sb[None, :]).float()).half(),
        ]
        for kernel, name in enumerate(bridge.names):
            call = prepare(
                bridge, kernel, [w4, w8], [x, a8], [w4[1], sa], [16, 8], coalesce=coalesce
            )
            actual = call()
            torch.cuda.synchronize()
            errors = [float((a - b).abs().max()) for a, b in zip(actual, refs)]
            rows.append(dict(M=m, N=n, K=k, kernel=name, max_abs_errors=errors))
            write(output / "partial.json", rows)
            for a, b in zip(actual, refs):
                torch.testing.assert_close(a, b, atol=0.001, rtol=0.002)
        print("passed", m, n, k, flush=True)
    write(
        output / "result.json",
        dict(
            rows=rows,
            scope="INT8 reference uses integer dot product and FP16 scale-product arithmetic matching upstream; not model quality or acceleration",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("library", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--coalesce", action="store_true")
    run(**vars(p.parse_args()))
