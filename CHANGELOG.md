# Changelog

## Unreleased

- `ldb evaluate-generation`: native Transformers free-generation transcripts and
  literal output rules, requiring attack and benign controls. Reports retain
  baseline/patch outputs, token boundaries, stop reasons, incomplete/error outcomes,
  grouped metrics, and paired fixes/regressions. Existing log-probability verification
  remains unchanged; token-scoped patches are rejected by the new workflow.
- Generation input/report schemas and a synthetic canary/control suite; see
  [generation evaluation](docs/generation-evaluation.md) for semantics and limitations.
- Reject invalid paired attribution when prompts/templates or requested generation
  settings differ; record external patch resource hashes and detect resource drift.
  Negative rules retain raw payload evidence while excluding the terminal EOS token.
- Offline intervention installation rolls back on interruption, and cleanup attempts
  all handles before reporting failures.

- `ldb logit-lens --tuned-lens`: apply pretrained per-layer translators before the
  unembedding.
- `ldb transcoder`: transcoder features on an MLP input and the normalized error of their
  reconstruction of the real MLP output; dictionaries gain an optional `input_bias`.
- `ldb crosscoder-diff`: classify crosscoder features as exclusive to one model or shared
  from per-model decoder norms.
- DeepSeek V4.1: observe-only `block.N.residual.post` hooks read through the decoder's aux
  capture; patches on observe-only hooks are rejected.
- `mlp.input` hooks on the Transformers engine.
- `ldb logit-lens`: per-layer next-token readings from the residual stream, with a check
  that the final layer reproduces the model output.
- `ldb probe`: linear probes per layer with a shuffled-label control and selectivity.
- Observe fused-MoE routing (`block.N.moe.router.topk_ids` / `topk_weights`) and steer it
  with `routing_bias` patches on the selection-only correction bias; DeepSeek V4.1
  decoder layers expose routing hooks.
- Derive request context from the `vllm.v1.worker.gpu` runner's `InputBatch`, so
  request-scoped traces and patches work on that runner.
- Released under the MIT License; documentation translated to English.
