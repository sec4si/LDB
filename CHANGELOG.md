# Changelog

## Unreleased

- `ldb logit-lens`: per-layer next-token readings from the residual stream, with a check
  that the final layer reproduces the model output.
- `ldb probe`: linear probes per layer with a shuffled-label control and selectivity.
- Observe fused-MoE routing (`block.N.moe.router.topk_ids` / `topk_weights`) and steer it
  with `routing_bias` patches on the selection-only correction bias; DeepSeek V4.1
  decoder layers expose routing hooks.
- Derive request context from the `vllm.v1.worker.gpu` runner's `InputBatch`, so
  request-scoped traces and patches work on that runner.
- Released under the MIT License; documentation translated to English.
