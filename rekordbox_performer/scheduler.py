"""Local monotonic-clock transition scheduler."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .engine import MidiEngine
from .protocol import encode_action


MAX_EVENTS = 500
MAX_DURATION_MS = 15 * 60 * 1000


@dataclass
class TransitionJob:
    id: str
    name: str
    events: list[dict[str, Any]]
    status: str = "scheduled"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    completed_events: int = 0
    error: str | None = None
    task: asyncio.Task | None = field(default=None, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "event_count": len(self.events),
            "completed_events": self.completed_events,
            "error": self.error,
        }


def validate_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not events:
        raise ValueError("A transition requires at least one event")
    if len(events) > MAX_EVENTS:
        raise ValueError(f"A transition may contain at most {MAX_EVENTS} events")
    normalized = []
    previous = -1
    for index, event in enumerate(events):
        at_ms = int(event["at_ms"])
        if at_ms < 0:
            raise ValueError(f"event {index}: at_ms must be non-negative")
        if at_ms < previous:
            raise ValueError("events must be ordered by at_ms")
        if at_ms > MAX_DURATION_MS:
            raise ValueError(
                f"transition duration may not exceed {MAX_DURATION_MS} ms"
            )
        action = str(event["action"])
        parameters = dict(event.get("parameters") or {})
        encode_action(action, parameters)
        normalized.append(
            {"at_ms": at_ms, "action": action, "parameters": parameters}
        )
        previous = at_ms
    return normalized


class TransitionScheduler:
    def __init__(self, engine: MidiEngine) -> None:
        self.engine = engine
        self.jobs: dict[str, TransitionJob] = {}

    def preview(
        self, name: str, events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        normalized = validate_events(events)
        return {
            "name": name,
            "event_count": len(normalized),
            "duration_ms": normalized[-1]["at_ms"],
            "events": normalized,
            "live_effect": False,
        }

    def start(
        self, name: str, events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not self.engine.is_armed():
            raise RuntimeError("Live MIDI control must be armed before scheduling")
        normalized = validate_events(events)
        job = TransitionJob(
            id=uuid.uuid4().hex,
            name=name,
            events=normalized,
        )
        self.jobs[job.id] = job
        job.task = asyncio.create_task(self._run(job))
        return job.public()

    async def _run(self, job: TransitionJob) -> None:
        job.status = "running"
        job.started_at = time.time()
        started = time.monotonic()
        try:
            for event in job.events:
                target = started + event["at_ms"] / 1000
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                if not self.engine.is_armed():
                    raise RuntimeError("Live MIDI control became disarmed")
                await self.engine.send_action(
                    event["action"], event["parameters"]
                )
                job.completed_events += 1
            job.status = "completed"
        except asyncio.CancelledError:
            job.status = "cancelled"
            raise
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
        finally:
            job.finished_at = time.time()

    def get(self, job_id: str) -> dict[str, Any]:
        if job_id not in self.jobs:
            raise KeyError(f"Unknown transition job: {job_id}")
        return self.jobs[job_id].public()

    def cancel(self, job_id: str) -> dict[str, Any]:
        if job_id not in self.jobs:
            raise KeyError(f"Unknown transition job: {job_id}")
        job = self.jobs[job_id]
        if job.task and not job.task.done():
            job.task.cancel()
        return job.public()

    def cancel_all(self) -> list[dict[str, Any]]:
        cancelled = []
        for job in self.jobs.values():
            if job.task and not job.task.done():
                job.task.cancel()
                cancelled.append(job.public())
        return cancelled

