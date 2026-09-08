# TransformerLens Integration

TransformerLens is LDB's primary offline instrumentation engine. LDB uses the 3.x
`TransformerBridge` API instead of the deprecated legacy `HookedTransformer` loading
path. The Bridge owns model-family adaptation and hook execution; LDB owns stable
artifact schemas, causal experiments, patch search, regression, and deployment to
prepared serving runtimes.

Install the integration and select it explicitly with:

```console
python -m pip install 'latent-debugger[interpretability]'
ldb run <model> --engine transformerlens --prompt '...'
```

It is the default offline engine. `--engine transformers` retains the smaller native
adapter for compatibility and adapter debugging.

## Numeric contract

GPT-2/MPS IOI parity was verified against the native adapter on 2026-09-08:
70 paired comparisons passed a predeclared `1e-3` absolute tolerance and matched
all top-1 tokens. See the [experiment record](../../expert-fingerprint-research/docs/transformer-lens-ioi-parity.md),
including the explicit BOS settings and differing tokenizer fingerprints.

LDB uses TransformerBridge's default raw Hugging Face weight semantics. It passes
`prepend_bos=False` for text inference so prompts align with normal Hugging Face,
vLLM, and SGLang serving inputs rather than receiving an implicit analysis-only BOS
token. Model revision, tokenizer hash, TransformerLens version, and LDB adapter class
are recorded in generated artifacts.

## Normalized hooks

Each normalized hook carries semantic axis metadata. This allows token and head
selection on four-dimensional tensors without guessing from shape:

| LDB hook | TransformerLens Bridge hook |
|---|---|
| `block.N.residual.pre` | `blocks.N.hook_in` |
| `block.N.attention.q` | `blocks.N.attn.hook_q` |
| `block.N.attention.k` | `blocks.N.attn.hook_k` |
| `block.N.attention.v` | `blocks.N.attn.hook_v` |
| `block.N.attention.scores` | `blocks.N.attn.hook_attn_scores` |
| `block.N.attention.pattern` | `blocks.N.attn.hook_pattern` |
| `block.N.attention.z` | `blocks.N.attn.hook_z` |
| `block.N.attention.result` | `blocks.N.attn.hook_result` |
| `block.N.attention.output` | `blocks.N.attn.hook_out` |
| `block.N.residual.mid` | `blocks.N.hook_resid_mid` |
| `block.N.mlp.input` | `blocks.N.mlp.hook_in` |
| `block.N.mlp.pre` | `blocks.N.mlp.hook_pre` |
| `block.N.mlp.activation` | `blocks.N.mlp.hook_post` |
| `block.N.mlp.output` | `blocks.N.mlp.hook_out` |
| `block.N.residual.post` | `blocks.N.hook_out` |

Bridge aliases are used where the underlying architecture exposes an equivalent
canonical hook under a different module path.

Q/K/V/Z/result use `[batch, token, head, channel]`; scores and patterns use
`[batch, head, query_token, key_token]`. `ldb attention` reports Q/K/V/Z/result
magnitudes, pattern entropy, and the strongest query-to-key edge for every exposed
head. The JSON report retains the configurable top-K edges per head. Use repeated
`--layer` options and `--max-capture-mib` to bound full Q/K/V/pattern/result capture
on large models. `ldb ablate --hook 'block.*.attention.z' --head N` measures a single head's
causal next-token effect.

## Runtime separation

TransformerLens is not loaded inside vLLM or SGLang workers. Serving adapters remain
small and runtime-native. A production trace can be replayed against the same model
revision through TransformerLens for deeper offline analysis, and the resulting
normalized Neural Patch can then be validated and loaded through LDB's live agent.
