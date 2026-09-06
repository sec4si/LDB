"""Client for live LDB agents."""

from __future__ import annotations

import fnmatch
import json
import socket
import uuid
from pathlib import Path
from typing import Any

from ldb.live import AgentDescriptor, AgentRequest, AgentResponse

Rollback = tuple[str, dict[str, Any]] | None


class AgentClient:
    def __init__(self, socket_path: Path, *, timeout: float = 10.0) -> None:
        self.socket_path = socket_path
        self.timeout = timeout

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        request = AgentRequest(
            request_id=uuid.uuid4().hex,
            method=method,
            params=params or {},
        )
        payload = request.model_dump_json().encode() + b"\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(str(self.socket_path))
            connection.sendall(payload)
            response_data = _read_line(connection)

        response = AgentResponse.model_validate_json(response_data)
        if response.request_id != request.request_id:
            raise RuntimeError("Agent response request_id does not match the request")
        if not response.ok:
            raise RuntimeError(response.error or "Agent command failed")
        return response.result


class AgentGroupClient:
    def __init__(self, clients: tuple[AgentClient, ...]) -> None:
        if not clients:
            raise ValueError("At least one agent is required")
        self.clients = clients

    @classmethod
    def connect(cls, target: Path) -> AgentGroupClient:
        if target.is_socket():
            return cls((AgentClient(target),))
        if not target.is_dir():
            raise ValueError(f"Agent target is not a socket or directory: {target}")

        connected: list[tuple[AgentDescriptor, AgentClient]] = []
        for path in sorted(target.glob("*.json")):
            try:
                advertised = AgentDescriptor.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            socket_path = Path(advertised.socket_path)
            if not socket_path.is_socket():
                continue
            client = AgentClient(socket_path)
            try:
                descriptor = AgentDescriptor.model_validate(client.call("hello"))
            except Exception as exc:
                raise RuntimeError(f"Agent handshake failed for {socket_path}: {exc}") from exc
            if descriptor.socket_path != advertised.socket_path:
                raise RuntimeError(f"Agent descriptor does not match its live socket: {path}")
            connected.append((descriptor, client))
        if not connected:
            raise ValueError(f"No live LDB agents found in {target}")
        _validate_agent_group(tuple(descriptor for descriptor, _ in connected))
        connected.sort(key=lambda item: item[0].rank)
        return cls(tuple(client for _, client in connected))

    def call(self, method: str, params: dict[str, Any] | None = None) -> list[Any]:
        values = params or {}
        self._validate_hook_patterns(method, values)
        rollbacks = self._prepare_rollbacks(method, values)
        results: list[Any] = []
        errors: list[str] = []
        successful: list[tuple[AgentClient, Rollback]] = []
        for client, rollback in zip(self.clients, rollbacks, strict=True):
            try:
                results.append(client.call(method, values))
                successful.append((client, rollback))
            except Exception as exc:
                errors.append(f"{client.socket_path}: {exc}")
        if errors:
            rollback_errors = self._rollback(successful)
            detail = "Agent group command failed: " + "; ".join(errors)
            if rollback_errors:
                detail += "; rollback failed: " + "; ".join(rollback_errors)
            raise RuntimeError(detail)
        return results

    def _validate_hook_patterns(self, method: str, params: dict[str, Any]) -> None:
        if method == "arm":
            patterns = tuple(str(value) for value in params.get("hooks", ()))
        elif method in {"patch.set", "break.set"} and "hook" in params:
            patterns = (str(params["hook"]),)
        else:
            return
        try:
            available = {str(hook) for client in self.clients for hook in client.call("hooks")}
        except Exception as exc:
            raise RuntimeError(f"Agent group hook preflight failed: {exc}") from exc
        if not available:
            return
        missing = [
            pattern
            for pattern in patterns
            if not any(fnmatch.fnmatchcase(hook, pattern) for hook in available)
        ]
        if missing:
            raise ValueError(
                "Hook pattern matches no installed hooks across attached ranks: "
                + ", ".join(missing)
            )

    def _prepare_rollbacks(self, method: str, params: dict[str, Any]) -> list[Rollback]:
        if method == "patch.set":
            return [("patch.remove", {"patch_id": params.get("patch_id")})] * len(self.clients)
        if method == "break.set":
            return [("break.remove", {"breakpoint_id": params.get("breakpoint_id")})] * len(
                self.clients
            )
        if method in {"patch.remove", "break.remove", "arm", "disarm"}:
            try:
                return [self._snapshot(client, method, params) for client in self.clients]
            except Exception as exc:
                raise RuntimeError(f"Agent group preflight failed: {exc}") from exc
        return [None] * len(self.clients)

    @staticmethod
    def _snapshot(client: AgentClient, method: str, params: dict[str, Any]) -> Rollback:
        if method in {"arm", "disarm"}:
            arm = client.call("status")["arm"]
            return ("arm", arm) if arm["hooks"] else ("disarm", {})

        listing_method = "patch.list" if method == "patch.remove" else "break.list"
        identifier = "patch_id" if method == "patch.remove" else "breakpoint_id"
        previous = next(
            (
                item
                for item in client.call(listing_method)
                if item[identifier] == params.get(identifier)
            ),
            None,
        )
        restore_method = "patch.set" if method == "patch.remove" else "break.set"
        return (restore_method, previous) if previous is not None else None

    @staticmethod
    def _rollback(successful: list[tuple[AgentClient, Rollback]]) -> list[str]:
        errors: list[str] = []
        for client, rollback in successful:
            if rollback is None:
                continue
            try:
                client.call(*rollback)
            except Exception as exc:
                errors.append(f"{client.socket_path}: {exc}")
        return errors


def _read_line(connection: socket.socket, *, limit: int = 16 * 1024 * 1024) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = connection.recv(min(65536, limit - size + 1))
        if not chunk:
            raise ConnectionError("Agent closed the connection before sending a response")
        newline = chunk.find(b"\n")
        if newline >= 0:
            chunks.append(chunk[:newline])
            return b"".join(chunks)
        chunks.append(chunk)
        size += len(chunk)
        if size > limit:
            raise ValueError("Agent response exceeds the protocol size limit")


def _validate_agent_group(descriptors: tuple[AgentDescriptor, ...]) -> None:
    identity_fields = (
        "backend",
        "backend_version",
        "model_id",
        "model_revision",
        "tokenizer_hash",
        "world_size",
    )
    mismatched = [
        field
        for field in identity_fields
        if len({getattr(descriptor, field) for descriptor in descriptors}) != 1
    ]
    if mismatched:
        raise ValueError(
            "Agent directory contains incompatible runtimes: " + ", ".join(mismatched)
        )

    ranks = tuple(descriptor.rank for descriptor in descriptors)
    if len(set(ranks)) != len(ranks):
        raise ValueError("Agent directory contains duplicate distributed ranks")
    world_size = descriptors[0].world_size
    expected = set(range(world_size))
    actual = set(ranks)
    if actual != expected:
        missing = ", ".join(str(rank) for rank in sorted(expected - actual)) or "none"
        unexpected = ", ".join(str(rank) for rank in sorted(actual - expected)) or "none"
        raise ValueError(
            "Agent directory is not a complete distributed runtime: "
            f"missing ranks {missing}; unexpected ranks {unexpected}"
        )


def format_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)
