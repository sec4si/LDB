# Runtime Compatibility

LDB runtime adapters intentionally depend on a small set of serving-runtime
boundaries. The adapter must fail closed when these boundaries disappear instead of
silently tracing the wrong tensors.

## Source-audited snapshots

| Runtime | Upstream commit | Required boundaries |
|---|---|---|
| SGLang | `2c05ed4e7776c876478f4b2db61acb12b9a27d01` | `sglang.srt.plugins`, `TpModelWorker.__init__`, `ModelRunner.forward`, `ForwardBatch.rids`, `ForwardBatch.extend_seq_lens_cpu` |
| vLLM | `6865e67f0be02d53694517f6f71d7fb96492792d` | `vllm.general_plugins`, V1 `Worker.load_model`, `GPUModelRunner.execute_model`, `InputBatch.req_ids`, `SchedulerOutput.num_scheduled_tokens` |

These snapshots are source audits, not GPU integration-test claims. GPU acceptance
must cover each supported runtime release, model family, execution mode, and
parallel topology before that combination is listed as verified.

## Runtime matrix

| Capability | Transformers eager | SGLang eager | vLLM eager | CUDA Graph replay |
|---|---:|---:|---:|---:|
| Normalized module trace | Yes | Implemented | Implemented | Not yet |
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

"Experimental" means fixed-address graph control slots and CPU contract tests exist,
but the path has not passed real CUDA Graph replay acceptance. Enable it explicitly
with `LDB_CUDA_GRAPH=1`. It supports global scalar `scale`, `add`, and `ablate` only.

Expert-output hooks are available only when experts remain visible as individual
PyTorch modules. LDB does not claim visibility inside an opaque fused-MoE kernel.
