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

The failing prompt is the target case. The clean prompt is also scored as a locality
regression control, so an intervention is penalized when it reduces the desired
token's probability on the known-good case. The complexity term penalizes distance
from the no-op scale of `1.0`.

This command searches a deliberately bounded one-hook scalar patch space. A report
is causal evidence for the supplied prompts and target token, not a global
explanation of the model. Broader regression suites and heterogeneous interventions
remain available through `ldb search` and `ldb search-combinations`.
