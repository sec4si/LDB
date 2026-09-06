# Runtime Compatibility

LDB runtime adapters intentionally depend on a small set of serving-runtime
boundaries. The adapter must fail closed when these boundaries disappear instead of
silently tracing the wrong tensors.

The plugins refuse startup if their audited worker/model-forward boundaries are
missing. Continuing would otherwise advertise request-scoped traces and patches
without authoritative batch context. This explicit check is necessary for SGLang
because its hook registry logs target-resolution failures instead of propagating
them to the plugin loader.

## Source-audited snapshots

| Runtime | Upstream commit | Required boundaries |
|---|---|---|
| SGLang | `2e8c03e2c71732b98ac29882ca1ee88bfa3fb1de` | `sglang.srt.plugins`, `TpModelWorker.__init__`, `ModelRunner.forward`, `ForwardBatch.rids`, `ForwardBatch.extend_seq_lens_cpu` |
| vLLM | `6865e67f0be02d53694517f6f71d7fb96492792d` | `vllm.general_plugins`, V1 `Worker.load_model`, `GPUModelRunner.execute_model`, `InputBatch.req_ids`, `SchedulerOutput.num_scheduled_tokens` |

These snapshots are source audits, not GPU integration-test claims. GPU acceptance
must cover each supported runtime release, model family, execution mode, and
parallel topology before that combination is listed as verified.

## Runtime matrix

| Capability | Transformers eager | SGLang eager | vLLM eager | CUDA Graph replay |
|---|---:|---:|---:|---:|
| Normalized module trace | Yes | Implemented | Implemented | Experimental summaries |
| Selective tensor capture | Yes | Implemented | Implemented | Not yet |
| Global scalar activation patch | N/A | Implemented | Implemented | Experimental |
| Global tensor activation patch | N/A | Implemented | Implemented | Not yet |
| Request-scoped activation patch | N/A | Implemented | Implemented | Not yet |
| Batch safe-point breakpoint | N/A | Implemented | Implemented | Not yet |
| Visible unfused expert output | Implemented | Implemented | Implemented | Not yet |
| Single-request scheduler suspension | N/A | Not yet | Not yet | Not yet |

"Implemented" means the adapter and protocol path exist and have dependency-free
contract tests. It does not mean every model architecture or kernel backend has
passed a real GPU acceptance run.

"Experimental" means fixed-address graph control/trace slots and CPU contract tests
exist, but the path has not passed real CUDA Graph replay acceptance. Enable it
explicitly with `LDB_CUDA_GRAPH=1`. Patches support global scalar `scale`, `add`, and
`ablate` only. Traces expose the latest scalar summary and replay occurrence count;
they do not expose request IDs or full tensors.

Expert-output hooks are available only when experts remain visible as individual
PyTorch modules. LDB does not claim visibility inside an opaque fused-MoE kernel.
