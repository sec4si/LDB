# DeepSeek V4.1 GPU validation record (2026-11-01)

## Environment

- Model: DeepSeek V4.1 Flash (40 decoder layers, 384 routed experts with top-6 routing,
  multi-stream hyper-connection residual, hidden size 5120).
- Hardware: 8× NVIDIA A800 80GB PCIe (Ampere).
- Runtime: a vLLM 0.31.0 development build with the `vllm.v1.worker.gpu` model runner,
  tensor parallel 8 with expert parallelism, eager execution, speculative decoding off,
  prefix caching on. This version is outside LDB's audited set, so the plugin ran with
  `LDB_ALLOW_UNVERIFIED_RUNTIME=1`.
- Attachment: prepared (`VLLM_PLUGINS=ldb`), one agent per tensor-parallel rank.

## Procedure

1. Greedy baseline: one 24-token completion with token log-probabilities, nothing armed.
2. Arm `block.*.residual.post` on all eight ranks, so every rank requests the decoder's
   aux capture and takes the same kernel path; capture tensors on rank 0 only. Repeat
   the completion.
3. Disarm and repeat the completion.
4. Try to install a `scale` patch on `block.10.residual.post`.

## Results

| Check | Result |
|---|---|
| Agents | 8 |
| Observe-only residual hooks | `block.0.residual.post` … `block.38.residual.post` (39) |
| Fused-MoE router hooks | 120 (`topk_ids`, `topk_weights`, `logits` per layer) |
| Armed vs. baseline | Identical tokens; maximum log-probability difference 0.0 |
| Disarmed vs. baseline | Identical tokens; maximum log-probability difference 0.0 |
| Trace | 936 events = 39 hooks × 24 forward steps; prefill tensor `[12, 5120]` |
| Values | All finite; mean token norm grows with depth (4.2 at block 0, 178.7 at block 38) |
| Patch on an observe-only hook | Rejected: `Observe-only hooks cannot be patched: block.10.residual.post` |
| Request context (`vllm.v1.worker.gpu` runner) | Events carry the vLLM request ID, prefill/decode phase, and `batch_requests = 1` |

An earlier run on the same deployment (2026-10-28) compared `block.N.moe.router.topk_ids`
with vLLM's own `--enable-return-routed-experts` output for one request: the selected
expert sets agreed on all 40 layers for all 56 tokens.

## Not covered

- `routing_bias` patches on `block.N.moe.router.logits` were not exercised on GPU.
- Hardware that uses the deep_gemm mega-mHC kernel (requesting the aux capture disables
  it for that call, so observed runs may differ numerically there).
- Sequence parallelism, CUDA Graph replay, hot attachment, and other models.
