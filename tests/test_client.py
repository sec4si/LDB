from pathlib import Path
from typing import Any, cast

import pytest

from ldb.client import AgentClient, AgentGroupClient


class FakeClient:
    def __init__(self, name: str, *, fail_set: bool = False, fail_remove: bool = False) -> None:
        self.socket_path = Path(name)
        self.fail_set = fail_set
        self.fail_remove = fail_remove
        self.calls: list[str] = []

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, bool]:
        self.calls.append(method)
        if method == "patch.set" and self.fail_set:
            raise RuntimeError("set failed")
        if method == "patch.remove" and self.fail_remove:
            raise RuntimeError("remove failed")
        return {"ok": True}


def test_group_reports_failed_rollback() -> None:
    first = FakeClient("first.sock", fail_remove=True)
    second = FakeClient("second.sock", fail_set=True)
    group = AgentGroupClient((cast(AgentClient, first), cast(AgentClient, second)))

    with pytest.raises(RuntimeError, match="rollback failed"):
        group.call("patch.set", {"patch_id": "patch-1"})
    assert first.calls == ["patch.set", "patch.remove"]
    assert second.calls == ["patch.set"]
