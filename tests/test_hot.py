import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ldb import hot


def test_bootstrap_payload_reports_installed_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status = tmp_path / "status.json"
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps(
            {
                "backend": "vllm",
                "model_id": "model",
                "config": {"instance_id": "live"},
                "status_path": str(status),
            }
        ),
        encoding="utf-8",
    )
    descriptor = SimpleNamespace(model_dump=lambda **_: {"socket_path": "/tmp/agent.sock"})
    monkeypatch.setattr(
        hot,
        "install_into_running_process",
        lambda **_: [SimpleNamespace(descriptor=descriptor)],
    )

    hot.bootstrap_payload(str(payload))

    result = json.loads(status.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert result["agents"] == [{"socket_path": "/tmp/agent.sock"}]


def test_install_rejects_unknown_hot_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported hot-attach backend"):
        hot.install_into_running_process(backend="other", model_id=None, config={})
