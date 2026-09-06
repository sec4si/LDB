# LDB Architecture v0.1

## Scope

LDB is a CLI-first runtime debugger for model internals. Version 0.1 focuses on a
single-process Hugging Face eager-mode vertical slice. The public domain model is
designed so that the CLI and target runtime can later be split into `ldb` and
`ldb-agent` processes without changing artifact formats.

Out of scope for v0.1:

- web interfaces and application-level LLM observability;
- `torch.compile`, fused-kernel internals, and mid-kernel suspension;
- tensor parallelism and distributed inference;
- vLLM and SGLang runtime adapters;
- permanent weight editing;
- automatic sparse autoencoder training.

## Component map

```mermaid
flowchart LR
    CLI[CLI and REPL] --> SESSION[Session Controller]
    SESSION --> ADAPTER[Runtime Adapter]
    ADAPTER <--> MODEL[Model Runtime]
    ADAPTER --> TRACE[Trace Recorder]
    TRACE --> BUNDLE[RunBundle]
    BUNDLE --> INSPECT[Inspectors]
    INSPECT --> CAUSAL[Causal Executor]
    CAUSAL --> SEARCH[Patch Search]
    SEARCH --> VERIFY[Regression Verifier]
    VERIFY --> PATCH[Neural Patch]
```

## Runtime boundary

The core uses normalized hook names and must not depend on architecture-specific
classes. An adapter maps backend modules onto names such as:

```text
block.17.residual.pre
block.17.attention.output
block.17.mlp.output
block.17.residual.post
block.17.moe.router.logits
logits
```

A breakpoint may suspend execution only at a declared safe point. In eager mode,
safe points are module pre-hooks, module post-hooks, and token boundaries. LDB must
never imply that it can suspend execution inside an opaque CUDA kernel.

## Control and data planes

The future multi-process design uses a versioned Protobuf control protocol over a
Unix domain socket. Control messages carry commands, events, metadata, and tensor
references. Bulk tensors are stored outside the control channel.

Version 0.1 now supports a prepared in-process agent controlled through a
user-private Unix domain socket and keeps these concepts separate:

- control: session commands and typed events;
- data: tensor files in a RunBundle;
- index: SQLite once cross-run querying is implemented.

See [runtime attachment](runtime-attachment.md) for the live protocol, runtime
plugin boundaries, and current eager-execution requirement.

## Evidence levels

Every finding must be classified as one of:

- `observed`
- `correlated`
- `attributed`
- `causally_tested`
- `replicated`

Only a successful intervention may promote a finding to `causally_tested`.

## Implementation order

1. RunBundle, normalized hook topology, real inference trace, and patch validation.
2. Interactive target thread with safe-point breakpoints and watches.
3. Clean/corrupt replay, ablation, and activation patching.
4. Regression suites and minimal patch search.
5. Out-of-process agent protocol.
6. vLLM, SGLang, SAE, and MoE plugins.
