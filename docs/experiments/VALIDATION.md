# Integration verification (2026-09-11)

Reused the existing CPU research environment and CUDA environment. No model weights,
CUDA libraries, or Torch installations were downloaded or rebuilt.

- Full LDB test suite: 226 passed, 3 skipped (36.54 seconds).
- `ruff check .` and `git diff --check`: passed.
- 46 experiment tests: 45 passed, 1 macOS-only watchdog test skipped on Linux.
  Includes IR tensor integrity, eager-bank restoration after exceptions, donor
  controls, budget rejection, atomic delivery failure cleanup, fingerprint
  permutation invariance, CLI backend isolation and precision allocation versus
  exhaustive search.
- All 15 registered tools accepted `--help` using the appropriate existing CPU or
  CUDA environment. This checks imports and argument parsing, not CUDA execution.
- Built the wheel with hatchling. Extracted it to a temporary directory outside
  all repositories and checked CLI listing, IR arguments and an actual NPZ
  response-selection run. No original research directory was on PYTHONPATH.
- All 61 migrated Python source hashes match `provenance.json`.
- `mypy --python-version 3.12 --ignore-missing-imports src`: passed for 105 files.
  The unmodified default command is not reproducible in this environment:
  TransformerLens is absent and installed NumPy stubs use Python 3.12 syntax,
  while the repository defaults to 3.11. Dynamic research adapters are explicitly
  excluded from strict typing; the CLI remains checked.

CUDA numeric correctness, throughput, large-model quality and macOS execution
were not rerun for this migration. Historical results are not new acceptance
results. The native source snapshot is archival and still needs its original
MxMoE build dependencies.
