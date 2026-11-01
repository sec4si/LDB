# Crosscoder model diffing

A crosscoder learns one dictionary over the same layer of two models, for example a base
model and its chat-tuned version. LDB loads a pretrained crosscoder and diffs the two
models from its weights alone; it does not train crosscoders or load the models.

## Canonical layout

| Tensor | Shape | Required |
|---|---|---|
| `decoder` | `[feature_count, 2, model_width]` | Yes |

Decoder slot `0` belongs to model A and slot `1` to model B.

```console
ldb crosscoder-diff dictionaries/base-vs-chat.safetensors \
  --model-a base --model-b chat \
  --exclusive-threshold 0.1 --top-k 20 \
  --output crosscoder-diff.json
```

## Reading the report

For each feature, `relative_norm = |d_B| / (|d_A| + |d_B|)` measures how much of its
decoder norm model B owns:

| Group | Rule |
|---|---|
| A only | `relative_norm < exclusive_threshold` |
| B only | `relative_norm > 1 - exclusive_threshold` |
| Shared | everything in between |
| Dead | both decoder norms are effectively zero; excluded from the other groups |

`relative_norm_histogram` counts live features in ten equal bins from 0 to 1.
`shared_cosine_mean` is the mean cosine between the two decoder directions of shared
features; a shared feature whose directions disagree writes to different places in the
two models. `top_a_only` and `top_b_only` list the exclusive features with the largest
decoder norms.

A crosscoder's sparsity objective can mark concepts both models share as exclusive to
one of them, so check exclusive features individually before calling them new.
