"""Stable stdio supervisor for hot-restarting the Performer MCP child."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from typing import Any, BinaryIO


RESTART_EXIT_CODE = 75
SUPERVISED_ENV = "REKORDBOX_PERFORMER_SUPERVISED"
GENERATION_ENV = "REKORDBOX_PERFORMER_GENERATION"


def _message(line: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _is_restart_request(message: dict[str, Any]) -> bool:
    return (
        message.get("method") == "tools/call"
        and isinstance(message.get("params"), dict)
        and message["params"].get("name") == "restart_performer"
    )


def _response_failed(message: dict[str, Any]) -> bool:
    if "error" in message:
        return True
    result = message.get("result")
    return isinstance(result, dict) and result.get("isError") is True


class PerformerSupervisor:
    """Proxy one MCP stdio session across replaceable server children."""

    def __init__(
        self,
        stdin: BinaryIO | None = None,
        stdout: BinaryIO | None = None,
    ) -> None:
        self.stdin = stdin or sys.stdin.buffer
        self.stdout = stdout or sys.stdout.buffer
        self.child: subprocess.Popen[bytes] | None = None
        self.generation = 0
        self.initialize_line: bytes | None = None
        self.initialized_line: bytes | None = None
        self.initialize_id: str | int | None = None
        self.restart_request_ids: set[str | int] = set()
        self.suppress_response_ids: set[str | int] = set()
        self.suppressed_response = threading.Event()
        self.ready = threading.Event()
        self.stopping = threading.Event()
        self.child_lock = threading.RLock()
        self.restart_lock = threading.Lock()

    def _spawn(self) -> subprocess.Popen[bytes]:
        self.generation += 1
        environment = os.environ.copy()
        environment[SUPERVISED_ENV] = "1"
        environment[GENERATION_ENV] = str(self.generation)
        child = subprocess.Popen(
            [sys.executable, "-m", "rekordbox_performer.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=environment,
            bufsize=0,
        )
        if child.stdin is None or child.stdout is None:
            child.kill()
            raise RuntimeError("Performer child did not expose stdio pipes")
        self.child = child
        threading.Thread(
            target=self._pump_child_output,
            args=(child,),
            name=f"rekordbox-performer-output-{self.generation}",
            daemon=True,
        ).start()
        return child

    def _write_child(self, line: bytes) -> None:
        with self.child_lock:
            child = self.child
            if child is None or child.stdin is None or child.poll() is not None:
                raise RuntimeError("Performer child is unavailable")
            child.stdin.write(line)
            child.stdin.flush()

    def _write_client(self, line: bytes) -> None:
        self.stdout.write(line)
        self.stdout.flush()

    def _pump_child_output(self, child: subprocess.Popen[bytes]) -> None:
        assert child.stdout is not None
        for line in iter(child.stdout.readline, b""):
            message = _message(line)
            response_id = message.get("id") if message else None
            if response_id in self.suppress_response_ids:
                self.suppress_response_ids.discard(response_id)
                self.suppressed_response.set()
                continue
            self._write_client(line)
            if response_id in self.restart_request_ids:
                self.restart_request_ids.discard(response_id)
                if message is not None and not _response_failed(message):
                    # Block the next client request until the replacement child
                    # has replayed the original MCP initialization handshake.
                    self.ready.clear()
        exit_code = child.wait()
        if self.stopping.is_set():
            return
        if exit_code == RESTART_EXIT_CODE:
            self._restart(child)
            return
        self.stopping.set()
        self.ready.set()

    def _restart(self, exited_child: subprocess.Popen[bytes]) -> None:
        with self.restart_lock:
            if self.stopping.is_set():
                return
            with self.child_lock:
                if self.child is not exited_child:
                    return
                self.ready.clear()
                self._spawn()
            if self.initialize_line is None or self.initialize_id is None:
                self.stopping.set()
                self.ready.set()
                return
            self.suppressed_response.clear()
            self.suppress_response_ids.add(self.initialize_id)
            self._write_child(self.initialize_line)
            if not self.suppressed_response.wait(timeout=30.0):
                self.stopping.set()
                self.ready.set()
                return
            if self.initialized_line is not None:
                self._write_child(self.initialized_line)
            self.ready.set()
            self._write_client(
                b'{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}\n'
            )

    def run(self) -> None:
        self._spawn()
        self.ready.set()
        try:
            for line in iter(self.stdin.readline, b""):
                message = _message(line)
                if message is not None:
                    method = message.get("method")
                    if method == "initialize":
                        self.initialize_line = line
                        self.initialize_id = message.get("id")
                    elif method == "notifications/initialized":
                        self.initialized_line = line
                    if _is_restart_request(message) and message.get("id") is not None:
                        self.restart_request_ids.add(message["id"])
                self.ready.wait()
                if self.stopping.is_set():
                    break
                self._write_child(line)
        finally:
            self.stopping.set()
            self.ready.set()
            with self.child_lock:
                child = self.child
                if child is not None and child.poll() is None:
                    if child.stdin is not None:
                        child.stdin.close()
                    try:
                        child.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        child.terminate()


def main() -> None:
    PerformerSupervisor().run()


if __name__ == "__main__":
    main()
