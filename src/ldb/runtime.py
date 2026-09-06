"""Runtime adapter contracts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from ldb.models import HookPoint

ProbeCallback = Callable[[HookPoint, Any], None]


class ProbeHandle(Protocol):
    def remove(self) -> None: ...


class RuntimeAdapter(Protocol):
    """The minimum runtime surface required by the debugger core."""

    @property
    def model_id(self) -> str: ...

    @property
    def backend_version(self) -> str: ...

    def hook_points(self) -> Sequence[HookPoint]: ...

    def install_probe(self, hook: HookPoint, callback: ProbeCallback) -> ProbeHandle: ...

    def tokenize(self, prompt: str) -> Sequence[int]: ...

    def generate(self, prompt: str, *, max_new_tokens: int) -> str: ...
