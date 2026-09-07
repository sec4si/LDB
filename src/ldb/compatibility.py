"""Fail-closed compatibility policy for serving-runtime adapters."""

from __future__ import annotations

import os

AUDITED_RUNTIME_VERSIONS: dict[str, frozenset[str]] = {
    "sglang": frozenset({"0.5.19"}),
    "vllm": frozenset({"0.28.0"}),
}


def require_audited_runtime(backend: str, version: str) -> None:
    """Reject runtime versions whose adapter boundaries have not been audited."""

    audited = AUDITED_RUNTIME_VERSIONS.get(backend)
    if audited is None:
        raise ValueError(f"Unknown serving runtime: {backend}")
    if version in audited or os.environ.get("LDB_ALLOW_UNVERIFIED_RUNTIME") == "1":
        return
    supported = ", ".join(sorted(audited))
    raise RuntimeError(
        f"LDB has not audited {backend} {version!r}; audited versions: {supported}. "
        "Set LDB_ALLOW_UNVERIFIED_RUNTIME=1 only for explicit compatibility testing."
    )
