"""Shared Rekordbox UI observations with an explicitly bounded lifecycle."""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol

from .runtime import ControlLease


class StatusAdapter(Protocol):
    def status(self) -> dict[str, Any]: ...


class SharedDeckObserver:
    """Let one MCP process scan Rekordbox while all clients share its snapshot."""

    def __init__(
        self,
        adapter: StatusAdapter,
        data_dir: Path,
        *,
        interval_seconds: float = 0.2,
        idle_seconds: float = 0.75,
    ) -> None:
        self.adapter = adapter
        self.data_dir = data_dir
        self.interval_seconds = interval_seconds
        self.idle_seconds = idle_seconds
        self.snapshot_path = data_dir / "deck-observer.json"
        self.lease = ControlLease(data_dir / "deck-observer.lock")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()
        self._activity_lock = threading.Lock()
        self._adapter_lock = threading.Lock()
        self._active_until = 0.0

    def activate(self, seconds: float | None = None) -> None:
        """Keep observation active for a bounded demand or performance window."""
        duration = self.idle_seconds if seconds is None else max(0.0, seconds)
        with self._activity_lock:
            self._active_until = max(
                self._active_until,
                time.monotonic() + duration,
            )
        self._ensure_started()

    def deactivate(self) -> None:
        """Stop background observation after the current UI scan completes."""
        with self._activity_lock:
            self._active_until = 0.0

    def _is_active(self) -> bool:
        with self._activity_lock:
            return time.monotonic() < self._active_until

    @contextmanager
    def exclusive_adapter(self) -> Iterator[None]:
        """Pause observation while another operation owns Rekordbox UIA."""
        self.deactivate()
        with self._adapter_lock:
            yield

    def _read(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict) or "observed_at" not in value:
            return None
        value["age_ms"] = round(
            max(0.0, (time.time() - float(value["observed_at"])) * 1000),
            1,
        )
        return value

    def _write(self, snapshot: dict[str, Any]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.snapshot_path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(snapshot, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, self.snapshot_path)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                # Coordinate the idle-exit decision with activate(). Without
                # this recheck, activate() can see the old worker as alive
                # while that worker is already exiting, leaving no observer
                # to service the new request.
                with self._thread_lock:
                    if not self._is_active():
                        self.lease.release()
                        if self._thread is threading.current_thread():
                            self._thread = None
                        return
                started = time.monotonic()
                try:
                    with self._adapter_lock:
                        snapshot = {
                            **self.adapter.status(),
                            "observed_at": time.time(),
                            "observer_pid": os.getpid(),
                            "observation_source": "rekordbox_uia",
                        }
                    self._write(snapshot)
                except Exception as exc:
                    previous = self._read() or {}
                    self._write(
                        {
                            **previous,
                            "observer_pid": os.getpid(),
                            "observer_error": str(exc),
                            "error_observed_at": time.time(),
                        }
                    )
                elapsed = time.monotonic() - started
                self._stop.wait(
                    max(0.01, self.interval_seconds - elapsed)
                )
        finally:
            with self._thread_lock:
                self.lease.release()
                if self._thread is threading.current_thread():
                    self._thread = None

    def _ensure_started(self) -> None:
        with self._thread_lock:
            if self._thread and self._thread.is_alive():
                return
            try:
                self.lease.acquire()
            except RuntimeError:
                return
            self._thread = threading.Thread(
                target=self._run,
                name="rekordbox-deck-observer",
                daemon=True,
            )
            self._thread.start()

    def status(
        self,
        *,
        max_age_ms: float = 350.0,
        wait_seconds: float = 15.0,
    ) -> dict[str, Any]:
        # UI Automation can take several seconds on a large Rekordbox tree.
        # The worker remains demand-bounded by idle_seconds; this longer wait
        # merely lets the caller receive the one scan already in progress.
        # Give a newly restarted worker enough scheduling runway to complete
        # at least one scan.  A very short idle window can otherwise expire
        # while the caller is waiting and leave only the stale pre-idle file.
        self.activate(
            max(
                self.idle_seconds,
                self.interval_seconds + 0.1,
            )
        )
        snapshot = self._read()
        if snapshot and snapshot["age_ms"] <= max_age_ms:
            return snapshot
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            snapshot = self._read()
            if snapshot and snapshot["age_ms"] <= max_age_ms:
                return snapshot
            time.sleep(0.025)
        if snapshot:
            raise RuntimeError(
                "Rekordbox deck observation is stale "
                f"({snapshot['age_ms']} ms; maximum {max_age_ms} ms)"
            )
        raise RuntimeError("No Rekordbox deck observation is available")

    def invalidate(self) -> None:
        try:
            self.snapshot_path.unlink()
        except FileNotFoundError:
            pass

    def close(self) -> None:
        self.deactivate()
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.lease.release()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
