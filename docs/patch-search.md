# Patch Search

LDB searches candidate interventions against an explicit target and regression
suite. A next-token case contains a prompt, a desired token ID, and an optional
weight. Case IDs must be unique.

```yaml
schema_version: ldb.suite/v1alpha1

target:
  - case_id: failure-1
    prompt: The Eiffel Tower is in
    target_token_id: 16046
    category: locality

related:
  - case_id: related-1
    prompt: Paris is home to the Eiffel Tower, located in
    target_token_id: 220
    category: generalization

regression:
  - case_id: control-1
    prompt: Water freezes at
    target_token_id: 362
    category: benchmark
```

The first search implementation evaluates one activation scale at one normalized
hook at a time. It maximizes:

```text
target_gain - regression_weight * regression_loss - complexity_weight * abs(scale - 1)
```

`target_gain` is the weighted mean change in desired-token log probability.
Related cases are reported separately as generalization evidence, while
`regression_loss` counts only decreases on control cases. Scale `1.0` is included
in every grid as a no-op baseline, so a search can correctly report that none of the
tested patches is worthwhile.

```console
ldb search <model> \
  --suite suite.yaml \
  --hook 'block.*.residual.post' \
  --output patch-search.json \
  --patch-output candidate-patch.yaml
```

Patch export fails when the no-op wins. Exported patches bind the resolved model
revision and tokenizer hash and default to request-scoped deployment.

The JSON report contains every per-case baseline and patched score. A high ranking
is evidence from the supplied suite, not proof of general safety or causal
uniqueness.

Verify a candidate against explicit target, related-generalization, and unrelated
regression thresholds before deployment:

```console
ldb patch verify candidate-patch.yaml \
  --suite suite.yaml \
  --target-min-gain 0.01 \
  --related-min-gain 0.0 \
  --regression-max-drop 0.01 \
  --output verification.json \
  --verified-patch-output verified-patch.yaml
```

Verification rejects model revision and tokenizer hash mismatches. A failed check
exits non-zero and never writes a verified patch.
Every case may declare a category such as `locality`, `generalization`,
`compositionality`, `multi-hop`, or `benchmark`. Verification reports case count,
success rate, mean delta, and worst delta independently for every category.
Cases specify exactly one `target_token_id` or `target_text`. Text continuations may
contain multiple tokens and are scored by mean token log-probability, so longer
answers do not receive a larger loss merely because of their length. Include any
required leading whitespace in `target_text`.

## Heterogeneous combination search

Define bounded candidates with explicit complexity costs. Candidates may use any
offline-supported Neural Patch operation, including head ablation and SAE feature
scaling.

```console
ldb search-combinations <model> \
  --suite suite.yaml \
  --candidates candidates.yaml \
  --max-interventions 2 \
  --max-trials 10000 \
  --output combination-search.json \
  --patch-output candidate-patch.yaml
```

LDB evaluates the exact combination set through `max-interventions`, including an
explicit no-op. It refuses a search whose combinatorial count exceeds `max-trials`.
Combination complexity is the sum of the candidates' declared costs and uses the
same target-gain, regression-loss, and complexity-weight objective.
Relative tensor and dictionary resources remain relative to the candidate file, so
the exported patch must be written beside that file.
