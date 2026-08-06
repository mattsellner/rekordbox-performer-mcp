"""Cross-process ownership primitives for the live control plane."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import BinaryIO


class ControlLease:
    """Hold an OS-backed, process-scoped lease for one live MIDI controller."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: BinaryIO | None = None
        self.acquired_at: float | None = None
        self._owner: dict[str, object] | None = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> None:
        if self.held:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = self.path.open("r+b")
        except FileNotFoundError:
            handle = self.path.open("w+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b" ")
                handle.flush()
            handle.seek(0)
            self._lock(handle)
            acquired_at = time.time()
            owner_data = {
                "pid": os.getpid(),
                "acquired_at": acquired_at,
            }
            owner = json.dumps(
                owner_data,
                separators=(",", ":"),
            ).encode("utf-8")
            handle.seek(1)
            handle.write(owner)
            handle.truncate()
            handle.flush()
        except Exception:
            handle.close()
            raise
        self._handle = handle
        self.acquired_at = acquired_at
        self._owner = owner_data

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        self.acquired_at = None
        self._owner = None
        try:
            handle.seek(0)
            self._unlock(handle)
        finally:
            handle.close()

    def status(self) -> dict[str, object]:
        owner = self._owner if self.held else self._read_owner()
        return {
            "held_by_this_process": self.held,
            "owner": owner,
            "path": str(self.path),
        }

    def _read_owner(self) -> dict[str, object] | None:
        try:
            raw = self.path.read_bytes()[1:]
        except (FileNotFoundError, PermissionError):
            return None
        if not raw:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _lock(handle: BinaryIO) -> None:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    "Another RekordBot or Performer process owns live MIDI control"
                ) from exc
            return

        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                "Another RekordBot or Performer process owns live MIDI control"
            ) from exc

    @staticmethod
    def _unlock(handle: BinaryIO) -> None:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass
