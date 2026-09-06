# Causal Propagation Graph

Correlation does not establish a causal edge. LDB builds a directed propagation
graph with clean/corrupt intervention experiments:

1. Capture selected activations for clean and corrupt prompts.
2. Replace one source activation token in the corrupt run with its clean value.
3. Re-run the model and capture every downstream selected activation.
4. Measure how far each downstream activation moves toward its clean state.
5. Transplant that moved downstream state into an otherwise corrupt run.
6. Measure how much target-token behavior is mediated through the downstream state.

For a source patch and downstream target, activation recovery is:

```text
1 - norm(patched_target - clean_target) / norm(corrupt_target - clean_target)
```

Behavioral mediation recovery is:

```text
(mediated_log_probability - corrupt_log_probability)
/ (clean_log_probability - corrupt_log_probability)
```

```console
ldb causal-graph <model> \
  --clean-prompt "The Eiffel Tower is in" \
  --corrupt-prompt "The Eiffel Tower is on" \
  --target-token-id 16046 \
  --token -1 \
  --hook 'block.*.residual.post' \
  --minimum-recovery 0.01 \
  --max-path-trials 256 \
  --output causal-graph.json
```

Hook direction follows the execution order observed in both runs, not lexical hook
order. Clean and corrupt executions must visit the same selected hooks, but their
sequence lengths may differ because activation recovery is measured at `--token`.
Negative indices address tokens from the end. Candidate paths below the activation
recovery threshold are not mediated. LDB refuses to exceed `--max-path-trials`
instead of silently truncating the experiment.

An edge reports both state recovery and the output effect obtained by transplanting
that recovered target state. This establishes mediation through the tested target
state, but it does not prove that the route is exclusive, that the source-to-target
connection is direct, or that either state is a human-interpretable mechanism.
