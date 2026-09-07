import platform

import pytest

from ldb.injector import inject_process


def test_hot_injection_is_linux_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    with pytest.raises(RuntimeError, match="only on Linux"):
        inject_process(123, backend="vllm")
