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

LDB uses TransformerBridge's default raw Hugging Face weight semantics. It passes
`prepend_bos=False` for text inference so prompts align with normal Hugging Face,
vLLM, and SGLang serving inputs rather than receiving an implicit analysis-only BOS
token. Model revision, tokenizer hash, TransformerLens version, and LDB adapter class
are recorded in generated artifacts.

## Normalized hooks

The first integration maps only Bridge activations with unambiguous batch, token,
and feature axes:

| LDB hook | TransformerLens Bridge hook |
|---|---|
| `block.N.residual.pre` | `blocks.N.hook_in` |
| `block.N.attention.output` | `blocks.N.attn.hook_out` |
| `block.N.residual.mid` | `blocks.N.hook_resid_mid` |
| `block.N.mlp.input` | `blocks.N.mlp.hook_in` |
| `block.N.mlp.pre` | `blocks.N.mlp.hook_pre` |
| `block.N.mlp.activation` | `blocks.N.mlp.hook_post` |
| `block.N.mlp.output` | `blocks.N.mlp.hook_out` |
| `block.N.residual.post` | `blocks.N.hook_out` |

Bridge aliases are used where the underlying architecture exposes an equivalent
canonical hook under a different module path.

Q/K/V, per-head results, attention scores, and attention patterns are intentionally
not normalized yet. They use architecture-dependent four-dimensional layouts and
need explicit semantic axis metadata before LDB can safely apply token- or
head-scoped operations. Silently treating those tensors as residual-shaped would
produce incorrect interventions.

## Runtime separation

TransformerLens is not loaded inside vLLM or SGLang workers. Serving adapters remain
small and runtime-native. A production trace can be replayed against the same model
revision through TransformerLens for deeper offline analysis, and the resulting
normalized Neural Patch can then be validated and loaded through LDB's live agent.
