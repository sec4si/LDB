# Causal Propagation Graph

Correlation does not establish a causal edge. LDB builds a directed propagation
graph with clean/corrupt intervention experiments:

1. Capture selected activations for clean and corrupt prompts.
2. Replace one source activation in the corrupt run with its clean value.
3. Re-run the model and capture every downstream selected activation.
4. Measure how far each downstream activation moves toward its clean state.

For a source patch and downstream target, activation recovery is:

```text
1 - norm(patched_target - clean_target) / norm(corrupt_target - clean_target)
```

```console
ldb causal-graph <model> \
  --clean-prompt "The Eiffel Tower is in" \
  --corrupt-prompt "The Eiffel Tower is on" \
  --target-token-id 16046 \
  --hook 'block.*.residual.post' \
  --minimum-recovery 0.01 \
  --output causal-graph.json
```

Hook direction follows the execution order observed in both runs, not lexical hook
order. Clean and corrupt executions must visit the same selected hooks. An edge
shows causal propagation from a source intervention to a downstream state; it does
not prove that the route is exclusive or that the target is a human-interpretable
mechanism.
