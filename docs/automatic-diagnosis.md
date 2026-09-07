# Automatic Diagnosis

`ldb diagnose` combines two bounded experiments:

1. clean-to-failing activation patching ranks hook points by causal recovery;
2. bounded per-head counterfactual patching ranks individual attention heads;
3. regression-aware scale search tests only the highest-ranked hooks and exports a
   Neural Patch when a tested intervention beats the no-op.

```console
ldb diagnose <model> \
  --clean-prompt 'known-good counterfactual' \
  --failing-prompt 'reproduced failure' \
  --target-token-id <desired-token-id> \
  --max-hooks 8 \
  --output diagnosis.json \
  --patch-output candidate-patch.yaml
```

When an SAE dictionary is available, the same command can discover active features
at a normalized hook and include feature-level interventions in the patch search:

```console
ldb diagnose <model> \
  --clean-prompt 'known-good counterfactual' \
  --failing-prompt 'reproduced failure' \
  --target-token-id <desired-token-id> \
  --sae-hook block.20.residual.post \
  --sae-dictionary block-20-sae.safetensors \
  --max-sae-features 8 \
  --feature-scale 0 --feature-scale 0.5 --feature-scale 1.5
```

LDB ranks the failing token's active SAE features, tests each feature scale against
the target and clean regression control, and compares the best feature patch with
the best whole-activation patch under the same objective.
When `--patch-output` selects an SAE patch, LDB copies the dictionary into a
content-addressed `<patch-name>.resources/` directory and rewrites the patch to a
relative path, so the patch and its required dictionary can be moved together.

Head discovery captures only the lightweight per-head Z shape by default. Add
`--attention-report` when the diagnosis artifact should also embed the full bounded
Q/K/V/pattern report.

The failing prompt is the target case. The clean prompt is also scored as a locality
regression control, so an intervention is penalized when it reduces the desired
token's probability on the known-good case. The complexity term penalizes distance
from the no-op scale of `1.0`.

This command searches a deliberately bounded one-hook scalar patch space. A report
is causal evidence for the supplied prompts and target token, not a global
explanation of the model. Broader regression suites and heterogeneous interventions
remain available through `ldb search` and `ldb search-combinations`.
