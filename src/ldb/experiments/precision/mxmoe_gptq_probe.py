"""Exercise upstream GPTQ definitions without importing model-specific adapters."""

import ast
import contextlib
import io
import math
import time
from pathlib import Path

import torch
from torch import nn

from .helpers import sha, write


def upstream_definitions(path):
    tree = ast.parse(path.read_text())
    names = {"quantize", "Quantizer", "GPTQ"}
    nodes = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    if {n.name for n in nodes} != names:
        raise ValueError("Upstream API changed")
    scope = dict(torch=torch, nn=nn, math=math, time=time, DEBUG=False)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


@torch.inference_mode()
def run(source, output):
    if output.exists():
        raise FileExistsError(output)
    torch.backends.cuda.matmul.allow_tf32 = False
    api = upstream_definitions(source)
    torch.manual_seed(42)
    results = []
    for n, k in [(384, 768), (1536, 384)]:
        # Independent calibration/evaluation samples with correlated feature pairs.
        x = torch.randn(1024, k, device="cuda")
        x[:, 1::2] = 0.95 * x[:, ::2] + 0.05 * x[:, 1::2]
        layer = nn.Linear(k, n, bias=False, device="cuda")
        original = layer.weight.detach().clone()
        for bits in (4, 8):
            layer.weight.copy_(original)
            gptq = api["GPTQ"](layer)
            gptq.quantizer = api["Quantizer"]()
            gptq.quantizer.configure(bits, perchannel=True, sym=False, mse=False)
            gptq.add_batch(x[:512], None)
            q = api["Quantizer"]()
            q.configure(bits, perchannel=True, sym=False, mse=False)
            q.find_params(original, weight=True)
            rtn = q.quantize(original)
            with contextlib.redirect_stdout(io.StringIO()):
                gptq.fasterquant(percdamp=0.01, groupsize=-1, actorder=True, static_groups=True)
            quantized = layer.weight.detach()
            codes = torch.round(quantized / gptq.quantizer.scale + gptq.quantizer.zero)
            reconstructed = gptq.quantizer.scale * (codes - gptq.quantizer.zero)
            torch.testing.assert_close(quantized, reconstructed, atol=1e-7, rtol=1e-5)
            assert bool(torch.isfinite(quantized).all())
            assert bool(((codes >= 0) & (codes < 2**bits)).all())
            reference = x[512:] @ original.T
            results.append(
                dict(
                    N=n,
                    K=k,
                    bits=bits,
                    rtn_mse=float(((x[512:] @ rtn.T) - reference).square().mean()),
                    gptq_mse=float(((x[512:] @ quantized.T) - reference).square().mean()),
                )
            )
    write(
        output,
        dict(
            upstream_sha256=sha(source),
            script_sha256=sha(Path(__file__)),
            results=results,
            scope="Exact upstream GPTQ definitions, synthetic correlated calibration; no model-quality or runtime-speed claim",
        ),
    )
    print(results)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.output)
