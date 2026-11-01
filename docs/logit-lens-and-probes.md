# Logit lens and linear probes

Both tools read the residual stream (`block.N.residual.post`) on the offline engines.
They answer "is this information present at this layer?", not "does the model use it?".
Use an intervention (`ldb causal`, `ldb ablate`, a patch) before making a causal claim.

## Logit lens

```console
ldb logit-lens sshleifer/tiny-gpt2 \
  --prompt "The Eiffel Tower is in" \
  --target-token-id 6342 --top-k 5 \
  --output logit-lens.json
```

For every layer, LDB takes the residual stream at `--position` (default `-1`, the last
token), applies the model's final norm and unembedding, and reports the top tokens and,
with `--target-token-id`, the target's probability and rank. `--layer` restricts the
layers that are read.

The report carries `final_layer_matches_model`. When the last position is read, the final
layer's lens must reproduce the model's own next-token log-probability (within `1e-3`);
`false` means the unembedding path does not match the model and the readings should not
be trusted. It is `null` for other positions, where no reference exists.

Limits: middle layers are not trained to be read by the unembedding, so early-layer
readings can be meaningless, and on some model families the plain logit lens is known to
be unreliable. A tuned lens is not implemented.

The transformers engine supports models whose final norm is `transformer.ln_f`,
`model.norm`, or `model.model.norm`; the TransformerLens engine uses the bridge's
`ln_final` and `unembed`.

## Linear probes

```console
ldb probe sshleifer/tiny-gpt2 \
  --dataset examples.jsonl \
  --position last --controls 5 --seed 0 \
  --output probe-report.json
```

The dataset is JSON Lines, one `{"text": ..., "label": ...}` object per line, with at
least two labels. For each `residual.post` layer, LDB collects the activation at the last
token (`--position last`) or the mean over the prompt (`--position mean`), splits the
examples into train and test sets (`--test-fraction`, default `0.3`, seeded by `--seed`),
standardizes features with training statistics, and fits an L2-regularized multinomial
logistic regression.

| Field | Meaning |
|---|---|
| `accuracy` | Held-out accuracy of the probe |
| `control_accuracy` | Mean held-out accuracy of identical probes trained on shuffled training labels (`--controls` repetitions), scored against the true test labels |
| `selectivity` | `accuracy - control_accuracy`: what the representation contributes beyond the probe's own capacity |
| `majority_baseline` | Accuracy of always predicting the most common training label |

Read a probe result through its control: high accuracy with high control accuracy means
the probe, not the layer, is doing the work. Decodable is not the same as used, and a
probe only finds the properties you labeled.
