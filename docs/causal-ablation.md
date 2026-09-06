# Causal Ablation

Ablation measures whether a selected internal state supports or suppresses one
next-token behavior. LDB independently zeros one activation token at every matched
hook and compares the target token's log probability with an unmodified baseline.

```console
ldb ablate <model> \
  --prompt "The Eiffel Tower is in" \
  --target-token-id 16046 \
  --token -1 \
  --hook 'block.*.residual.post' \
  --output ablation-report.json
```

Trials are ranked by absolute effect:

```text
effect = ablated_log_probability - baseline_log_probability
```

A negative effect means the removed state supported the target token. A positive
effect means it suppressed the target token. Negative token indices count from the
end of the activation sequence.

An ablation result is causal evidence for the tested intervention, prompt, token,
and hook. It does not establish that the hook is a unique mechanism, that every
component of the zeroed vector is relevant, or that the result generalizes to other
prompts. Use clean/corrupt patching to test counterfactual recovery and a regression
suite before exporting or deploying a Neural Patch.
