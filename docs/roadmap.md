# Roadmap and known gaps

These are known limitations, not commitments. Each item states what works today.

## Scheduler-isolated breakpoints

A request-scoped breakpoint pauses the worker while it runs a batch that may contain
other requests. Batching can change numerics and MoE routing, so an observation made in
a mixed batch may not reproduce alone. Isolating one request needs cooperation from the
engine scheduler, which runs in a different process from the worker LDB attaches to.

Today: every eager trace event records `batch_requests`, so analyses can keep only
unbatched observations; serving with a maximum of one running sequence gives isolated
batches at the cost of throughput.

## CUDA Graph capture

During CUDA Graph replay LDB can record scalar summaries and apply global scalar
patches through fixed-address slots. Full tensor capture, breakpoints, tensor patches,
and request-scoped control need tensors and control flow that a captured graph does not
expose.

Today: run with eager execution (`--enforce-eager` for vLLM, `--disable-cuda-graph` for
SGLang) whenever those features are needed.

## DeepSeek V4.1 residual stream

V4.1 layers expose fused-MoE routing hooks and observe-only `residual.post` readings
built from the model's own aux capture, which average the hyper-connection streams. Both
passed GPU validation on 8× A800 with TP8 and expert parallelism, with bit-identical
outputs while observed ([record](deepseek-v41-gpu-validation.md)). Not covered yet: the
individual streams, the final layer's output, interventions on the residual stream,
routing-bias patches on GPU, sequence-parallel deployments, and deep_gemm mega-mHC
hardware.

## Generation-level research evaluation

Today: `ldb evaluate-generation` runs complete greedy generation attempts with the
native Transformers adapter, records exact inputs and newly generated output, and
applies explicit literal rules. Attack and benign controls are mandatory. An optional
Neural Patch is checked against runtime identity, then compared with an unpatched
baseline. Incomplete or failed attempts remain visible and cannot count as passes.
See [generation evaluation](generation-evaluation.md).

Not implemented in this workflow: semantic safety judges; actual tool execution or
agent rollouts; dynamic multi-turn environments; randomized or interleaved arm order;
sampling repetitions and confidence intervals; per-phase/per-step intervention control;
stable prompt-span token selection across prefill/decode; TransformerLens or live-server
integration; quantized/MoE production-backend acceptance. Fixed chat history is input to
one generation, not a multi-turn evaluation. Literal-rule rates are not general safety
or attack-success rates. Actual model and GPU acceptance remain separate from contract tests.

## Training interpretability artifacts

LDB loads SAE, transcoder, tuned-lens, and crosscoder dictionaries in canonical layouts
but does not train them; train them with dedicated tooling and convert to the documented
layout.
