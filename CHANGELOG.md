# Changelog

## Unreleased

- Observe fused-MoE routing (`block.N.moe.router.topk_ids` / `topk_weights`) and steer it
  with `routing_bias` patches on the selection-only correction bias; DeepSeek V4.1
  decoder layers expose routing hooks.
- Derive request context from the `vllm.v1.worker.gpu` runner's `InputBatch`, so
  request-scoped traces and patches work on that runner.
- Released under the MIT License; documentation translated to English.
