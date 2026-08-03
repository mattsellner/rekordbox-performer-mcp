import pytest

from rekordbox_performer import server
from rekordbox_performer.supervisor import (
    SUPERVISED_ENV,
    _is_restart_request,
    _message,
    _response_failed,
)


def test_restart_request_detection_is_exact() -> None:
    assert _is_restart_request(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "restart_performer", "arguments": {}},
        }
    )
    assert not _is_restart_request(
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {"name": "control_status", "arguments": {}},
        }
    )


def test_json_line_helpers_fail_closed() -> None:
    assert _message(b"not json\n") is None
    assert _response_failed({"id": 1, "error": {"code": -1}}) is True
    assert _response_failed({"id": 1, "result": {"isError": True}}) is True
    assert _response_failed({"id": 1, "result": {"isError": False}}) is False


def test_restart_tool_requires_supervisor(monkeypatch) -> None:
    monkeypatch.delenv(SUPERVISED_ENV, raising=False)
    with pytest.raises(RuntimeError, match="restart supervisor"):
        server.restart_performer()


def test_restart_tool_refuses_active_transition(monkeypatch) -> None:
    monkeypatch.setenv(SUPERVISED_ENV, "1")
    monkeypatch.setattr(server.scheduler, "metrics", lambda: {"active_jobs": 1})
    with pytest.raises(RuntimeError, match="transition job"):
        server.restart_performer()
