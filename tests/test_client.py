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


class StatefulPatchClient:
    def __init__(self, name: str, patch: dict[str, Any], *, fail_remove: bool = False) -> None:
        self.socket_path = Path(name)
        self.patch = patch
        self.fail_remove = fail_remove
        self.calls: list[str] = []

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append(method)
        if method == "patch.list":
            return [self.patch] if self.patch else []
        if method == "patch.remove":
            if self.fail_remove:
                raise RuntimeError("remove failed")
            self.patch = {}
            return {"removed": True}
        if method == "patch.set":
            self.patch = dict(params or {})
            return self.patch
        raise AssertionError(f"Unexpected method: {method}")


class StatefulArmClient:
    def __init__(self, name: str, arm: dict[str, Any], *, fail_arm: bool = False) -> None:
        self.socket_path = Path(name)
        self.arm = arm
        self.fail_arm = fail_arm
        self.calls: list[str] = []

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append(method)
        if method == "hooks":
            return []
        if method == "status":
            return {"arm": self.arm}
        if method == "arm":
            if self.fail_arm:
                raise RuntimeError("arm failed")
            self.arm = dict(params or {})
            return self.arm
        if method == "disarm":
            self.arm = {"hooks": [], "request_ids": [], "capture_tensors": False}
            return {"armed": False}
        raise AssertionError(f"Unexpected method: {method}")


def test_group_reports_failed_rollback() -> None:
    first = FakeClient("first.sock", fail_remove=True)
    second = FakeClient("second.sock", fail_set=True)
    group = AgentGroupClient((cast(AgentClient, first), cast(AgentClient, second)))

    with pytest.raises(RuntimeError, match="rollback failed"):
        group.call("patch.set", {"patch_id": "patch-1"})
    assert first.calls == ["patch.set", "patch.remove"]
    assert second.calls == ["patch.set"]


def test_group_restores_removed_patch_after_rank_failure() -> None:
    patch = {
        "patch_id": "patch-1",
        "hook": "block.0.residual.post",
        "operation": "scale",
        "value": 0.5,
    }
    first = StatefulPatchClient("first.sock", patch.copy())
    second = StatefulPatchClient("second.sock", patch.copy(), fail_remove=True)
    group = AgentGroupClient((cast(AgentClient, first), cast(AgentClient, second)))

    with pytest.raises(RuntimeError, match="Agent group command failed"):
        group.call("patch.remove", {"patch_id": "patch-1"})

    assert first.patch == patch
    assert second.patch == patch
    assert first.calls == ["patch.list", "patch.remove", "patch.set"]
    assert second.calls == ["patch.list", "patch.remove"]


def test_group_restores_arm_configuration_after_rank_failure() -> None:
    previous = {"hooks": ["block.0.*"], "request_ids": ["request-a"], "capture_tensors": False}
    first = StatefulArmClient("first.sock", previous.copy())
    second = StatefulArmClient("second.sock", previous.copy(), fail_arm=True)
    group = AgentGroupClient((cast(AgentClient, first), cast(AgentClient, second)))

    with pytest.raises(RuntimeError, match="Agent group command failed"):
        group.call("arm", {"hooks": ["block.1.*"]})

    assert first.arm == previous
    assert second.arm == previous
    assert first.calls == ["hooks", "status", "arm", "arm"]
    assert second.calls == ["hooks", "status", "arm"]
