"""Structured standalone-DJ status shared by the engine and corner UI."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .intelligence import default_data_dir


class RuntimePhase(StrEnum):
    IDLE = "idle"
    PREFLIGHT = "preflight"
    ANALYZING = "analyzing"
    SELECTING = "selecting"
    STAGING = "staging"
    PLAYING = "playing"
    WAITING = "waiting"
    ESTABLISHING = "establishing"
    BASS_HANDOFF = "bass_handoff"
    RETIRING = "retiring"
    TEMPO_RAMP = "tempo_ramp"
    REPLACING = "replacing"
    HOLDING_LOOP = "holding_loop"
    RECOVERING = "recovering"
    STOPPING = "stopping"
    COMPLETE = "complete"
    FAILED = "failed"


class TrackRole(BaseModel):
    track_id: str | None = None
    title: str | None = None
    artist: str | None = None
    deck: int | None = Field(default=None, ge=1, le=2)
    bpm: float | None = None
    state: str = "unknown"


class RuntimeHealth(BaseModel):
    midi: Literal["ok", "degraded", "down", "unknown"] = "unknown"
    rekordbox: Literal["ok", "degraded", "down", "unknown"] = "unknown"
    audio_route: Literal["flx4", "pc", "unknown"] = "unknown"
    deck_state: Literal["fresh", "projected", "stale", "unknown"] = "unknown"
    rescue: Literal["armed", "active", "unavailable", "unknown"] = "unknown"


class RuntimeSnapshot(BaseModel):
    phase: RuntimePhase = RuntimePhase.IDLE
    severity: Literal["info", "success", "warning", "error"] = "info"
    headline: str = "Ready"
    detail: str = "Choose an opening track and start a set."
    current: TrackRole = Field(default_factory=TrackRole)
    staged: TrackRole = Field(default_factory=TrackRole)
    next: TrackRole = Field(default_factory=TrackRole)
    rescue: TrackRole = Field(default_factory=TrackRole)
    bpm: float | None = None
    bar: int | None = None
    beat: int | None = None
    action_in_bars: float | None = None
    action_in_seconds: float | None = None
    transition_name: str | None = None
    queue: list[TrackRole] = Field(default_factory=list)
    health: RuntimeHealth = Field(default_factory=RuntimeHealth)
    failures: list[str] = Field(default_factory=list)
    updated_at: float = Field(default_factory=time.time)


class StatusEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    at: float = Field(default_factory=time.time)
    phase: RuntimePhase
    severity: Literal["info", "success", "warning", "error"] = "info"
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class StandaloneStatusStore:
    """Atomic snapshot plus append-only event journal for a restartable UI."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = self.data_dir / "standalone-status.json"
        self.events_path = self.data_dir / "standalone-events.jsonl"
        self._lock = threading.RLock()

    def read(self) -> RuntimeSnapshot:
        try:
            return RuntimeSnapshot.model_validate_json(
                self.snapshot_path.read_text(encoding="utf-8")
            )
        except (FileNotFoundError, PermissionError, ValueError):
            return RuntimeSnapshot()

    def publish(
        self,
        *,
        phase: RuntimePhase,
        headline: str,
        detail: str,
        severity: Literal["info", "success", "warning", "error"] = "info",
        event_data: dict[str, Any] | None = None,
        **updates: Any,
    ) -> RuntimeSnapshot:
        with self._lock:
            current = self.read()
            snapshot = current.model_copy(
                update={
                    "phase": phase,
                    "headline": headline,
                    "detail": detail,
                    "severity": severity,
                    "updated_at": time.time(),
                    **updates,
                }
            )
            self._write_snapshot(snapshot)
            event = StatusEvent(
                phase=phase,
                severity=severity,
                message=headline,
                data={"detail": detail, **(event_data or {})},
            )
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")
            return snapshot

    def recent_events(self, limit: int = 100) -> list[StatusEvent]:
        if limit <= 0:
            return []
        try:
            lines = self.events_path.read_text(encoding="utf-8").splitlines()
        except (FileNotFoundError, PermissionError):
            return []
        return [
            StatusEvent.model_validate_json(line)
            for line in lines[-limit:]
            if line.strip()
        ]

    def _write_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        fd, raw = tempfile.mkstemp(dir=self.data_dir, suffix=".tmp")
        os.close(fd)
        temporary = Path(raw)
        try:
            temporary.write_text(
                json.dumps(snapshot.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, self.snapshot_path)
        finally:
            temporary.unlink(missing_ok=True)
