"""Client for live LDB agents."""

from __future__ import annotations

import json
import socket
import uuid
from pathlib import Path
from typing import Any

from ldb.live import AgentDescriptor, AgentRequest, AgentResponse


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

        descriptors: list[AgentDescriptor] = []
        for path in sorted(target.glob("*.json")):
            try:
                descriptor = AgentDescriptor.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if Path(descriptor.socket_path).is_socket():
                descriptors.append(descriptor)
        if not descriptors:
            raise ValueError(f"No live LDB agents found in {target}")
        return cls(tuple(AgentClient(Path(item.socket_path)) for item in descriptors))

    def call(self, method: str, params: dict[str, Any] | None = None) -> list[Any]:
        results: list[Any] = []
        errors: list[str] = []
        successful: list[AgentClient] = []
        for client in self.clients:
            try:
                results.append(client.call(method, params))
                successful.append(client)
            except Exception as exc:
                errors.append(f"{client.socket_path}: {exc}")
        if errors:
            rollback_errors = self._rollback(method, params or {}, successful)
            detail = "Agent group command failed: " + "; ".join(errors)
            if rollback_errors:
                detail += "; rollback failed: " + "; ".join(rollback_errors)
            raise RuntimeError(detail)
        return results

    @staticmethod
    def _rollback(method: str, params: dict[str, Any], clients: list[AgentClient]) -> list[str]:
        if method == "patch.set":
            rollback = ("patch.remove", {"patch_id": params.get("patch_id")})
        elif method == "break.set":
            rollback = ("break.remove", {"breakpoint_id": params.get("breakpoint_id")})
        elif method == "arm":
            rollback = ("disarm", {})
        else:
            return []
        errors: list[str] = []
        for client in clients:
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


def format_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)
