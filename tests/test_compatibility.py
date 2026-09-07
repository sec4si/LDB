import pytest

from ldb.compatibility import require_audited_runtime


@pytest.mark.parametrize(
    ("backend", "version"),
    (("vllm", "0.28.0"), ("sglang", "0.5.19")),
)
def test_audited_runtime_versions_are_accepted(backend: str, version: str) -> None:
    require_audited_runtime(backend, version)


def test_unaudited_runtime_version_fails_closed(monkeypatch) -> None:
    monkeypatch.delenv("LDB_ALLOW_UNVERIFIED_RUNTIME", raising=False)
    with pytest.raises(RuntimeError, match="has not audited vllm '99.0.0'"):
        require_audited_runtime("vllm", "99.0.0")


def test_unaudited_runtime_requires_explicit_override(monkeypatch) -> None:
    monkeypatch.setenv("LDB_ALLOW_UNVERIFIED_RUNTIME", "1")
    require_audited_runtime("sglang", "99.0.0")


def test_unknown_runtime_is_rejected_even_with_override(monkeypatch) -> None:
    monkeypatch.setenv("LDB_ALLOW_UNVERIFIED_RUNTIME", "1")
    with pytest.raises(ValueError, match="Unknown serving runtime"):
        require_audited_runtime("unknown", "1")
