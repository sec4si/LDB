# WSL2 single-GPU acceptance record (2026-09-08)

Test environment: Ubuntu 24.04 / WSL2, RTX 4070 12GB, Windows NVIDIA driver
591.86. The project code was copied to a separate directory, and each backend
used its own Python 3.12 virtual environment. This run used real HTTP
completions and the LDB agent inside the worker.

The model is `Qwen/Qwen3-0.6B`, pinned to revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Scope: prepared attachment,
single-GPU eager mode, unquantized weights, prefix caching disabled. This does
not mean that hot attachment, CUDA Graph, TP/PP/DP multi-GPU, other models, or
training workflows have passed acceptance.

## vLLM 0.28.0

**Passed.** PyTorch `2.13.0+cu130`, FlashAttention 2, V1 model runner,
PyTorch sampler. Global ablation of `block.0.residual.post`:

| Stage | Output token | Token log probability |
|---|---|---:|
| Baseline | ` Paris` | -0.8785509467124939 |
| Ablated | `!` | -11.931214332580566 |
| Reverted | ` Paris` | -0.8785509467124939 |

Rank 0, which owns the hook, produced the raw trace and a patched summary
containing the patch ID; after reverting, the full `choices` matched the
baseline. This strong ablation validates the intervention path; it does not
demonstrate semantic specialization of a particular layer or localized
knowledge editing.

The default V2 runner reports `UVA is not available` in this WSL2 environment.
The V1 runner's default FlashInfer sampler fails to locate the CUDA toolkit
during JIT, so this run explicitly used the PyTorch sampler. vLLM source was
not modified.

```bash
source .venv-vllm/bin/activate
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 \
python scripts/gpu_smoke.py vllm Qwen/Qwen3-0.6B \
  --mode eager --port 18976 --artifacts "$PWD/evidence/vllm-qwen3-eager-fixed" -- \
  --host 127.0.0.1 --gpu-memory-utilization 0.45 \
  --max-model-len 1024 --max-num-seqs 4 --no-enable-prefix-caching \
  --revision c1899de289a04d12100db370d81485cdf75e47ca
```

`HF_HUB_OFFLINE=1` requires the full model to be cached. The port must be free.

## SGLang 0.5.19

**Passed.** PyTorch `2.13.0+cu130`, Triton attention, PyTorch sampler.
With the same prompt and hook the result was ` Paris → ! → Paris`; the baseline
and restored token log probabilities were both `-0.9114812612533569`, and the
ablated value was `-11.931214332580566`. The rank 0 trace, patch ID, patched
summary, and full `choices` restoration checks all passed. The baseline log
probabilities differ between the two backends; this experiment does not
demonstrate numerical equivalence across backends.

Installing the pinned version requires allowing the pre-release dependencies
that SGLang declares:

```bash
uv venv --python 3.12 .venv-sglang
uv pip install --python .venv-sglang/bin/python --prerelease=allow \
  'sglang==0.5.19' -e .
source .venv-sglang/bin/activate
export CUDA_HOME="$PWD/.venv-sglang/lib/python3.12/site-packages/nvidia/cu13"
```

This machine uses the pip CUDA toolkit and has no `/usr/local/cuda`. SGLang's
JIT linker requires `$CUDA_HOME/lib64/libcudart.so`, so two previously missing
symlinks were added in the isolated environment: `lib64 → lib` and
`lib/libcudart.so → libcudart.so.13`. The venv was also activated so that the
installed `ninja` could be found on PATH. Neither the system CUDA nor the
SGLang source was modified.

```bash
HF_HUB_OFFLINE=1 python scripts/gpu_smoke.py sglang Qwen/Qwen3-0.6B \
  --mode eager --port 18977 --artifacts "$PWD/evidence/sglang-qwen3-eager-toolkit" -- \
  --host 127.0.0.1 --mem-fraction-static 0.45 --context-length 1024 \
  --max-running-requests 4 --disable-radix-cache \
  --attention-backend triton --sampling-backend pytorch \
  --revision c1899de289a04d12100db370d81485cdf75e47ca
```

## Issues found and fixed in this run

- `ldb attach -e` used to print two JSON documents in a row (the handshake and
  the command result), causing `json.loads` in the acceptance script to fail
  with `Extra data`. The handshake is still performed, but its content is now
  printed only in interactive mode.
- Without DP enabled, SGLang reports `dp_rank=None`; both startup attachment
  and hot attachment must normalize it to rank 0. The fix covers both paths and
  preserves non-zero DP ranks.

## Raw evidence

Raw evidence was archived outside the repository. It includes service logs,
HTTP responses, traces, patch summaries, dependency freezes, and source hashes.
The code had uncommitted changes, so `ldb_commit=unknown` in the reports is not
proof of the source version; `source-manifest.json` and the hashes of the files
actually executed are authoritative.

The two successful reports are `evidence/vllm-qwen3-eager-fixed/result.json`
and `evidence/sglang-qwen3-eager-toolkit/result.json`; the corresponding
`eager-evidence.json` files record the full request results. Logs from failed
attempts were also kept. The evidence archive SHA-256 matched on both ends:
`6d0c08c6353e4e84213cc5d5e62c64ea34248cef71e3befc241a00422ac44f84`.
After testing, both test ports were confirmed released, with no test service
processes left running.

For installation, see [vLLM GPU installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
and [SGLang installation](https://docs.sglang.io/docs/get-started/install). The
pass verdicts in this record are based on evidence from actual runs on the
machine described above.
