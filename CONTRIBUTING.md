# Contributing to LDB

Thanks for helping. Bug reports, runtime compatibility evidence, and focused pull
requests are all welcome.

## Development setup

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[transformers,experiments,dev,interpretability]'
```

## Before opening a pull request

Run the same checks as CI:

```sh
.venv/bin/ruff check .
.venv/bin/mypy src
.venv/bin/pytest
.venv/bin/python scripts/generate_schemas.py && git diff --exit-code -- schemas
```

- Keep changes focused; one concern per pull request.
- Add or update tests with every behavior change. Test doubles for serving runtimes
  should mirror the real interfaces they replace.
- Match the existing code style; `ruff` is the arbiter.

## Runtime compatibility claims

A vLLM or SGLang version is only marked verified after the GPU smoke workflow passes
on that version, model family, mode, and topology (see
[docs/gpu-validation.md](docs/gpu-validation.md)). If you have run it on hardware we
do not cover, please attach the generated evidence to an issue.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
