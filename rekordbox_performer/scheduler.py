"""Local monotonic-clock transition scheduler."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .engine import MidiEngine
from .protocol import encode_action


MAX_EVENTS = 500
MAX_DURATION_MS = 15 * 60 * 1000


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, round((len(ordered) - 1) * percent)),
    )
    return ordered[index]


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
    event_lateness_ms: list[float] = field(default_factory=list, repr=False)
    task: asyncio.Task | None = field(default=None, repr=False)
    completion_verifier: Callable[[], dict[str, Any]] | None = field(
        default=None,
        repr=False,
    )
    event_observer: Callable[[dict[str, Any], float], None] | None = field(
        default=None,
        repr=False,
    )
    verification: dict[str, Any] | None = None
    execution_id: str | None = None
    control_reserved_until: float | None = None

    def public(self) -> dict[str, Any]:
        average_lateness = (
            sum(self.event_lateness_ms) / len(self.event_lateness_ms)
            if self.event_lateness_ms
            else 0.0
        )
        return {
            "id": self.id,
            "execution_id": self.execution_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "event_count": len(self.events),
            "completed_events": self.completed_events,
            "average_event_lateness_ms": round(average_lateness, 2),
            "max_event_lateness_ms": round(
                max(self.event_lateness_ms, default=0.0),
                2,
            ),
            "p95_event_lateness_ms": round(
                percentile(self.event_lateness_ms, 0.95),
                2,
            ),
            "p99_event_lateness_ms": round(
                percentile(self.event_lateness_ms, 0.99),
                2,
            ),
            "events_over_10ms_late": sum(
                value > 10.0 for value in self.event_lateness_ms
            ),
            "verification": self.verification,
            "control_reserved_until_monotonic": self.control_reserved_until,
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
        self.execution_jobs: dict[str, str] = {}

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
        self,
        name: str,
        events: list[dict[str, Any]],
        *,
        completion_verifier: Callable[[], dict[str, Any]] | None = None,
        event_observer: Callable[[dict[str, Any], float], None] | None = None,
        execution_id: str | None = None,
    ) -> dict[str, Any]:
        if not self.engine.is_armed():
            raise RuntimeError("Live MIDI control must be armed before scheduling")
        normalized = validate_events(events)
        if execution_id is not None:
            execution_id = execution_id.strip()
            if not execution_id:
                raise ValueError("execution_id must not be blank")
            existing_id = self.execution_jobs.get(execution_id)
            if existing_id is not None:
                existing = self.jobs[existing_id]
                if existing.name != name or existing.events != normalized:
                    raise RuntimeError(
                        "execution_id is already bound to a different transition"
                    )
                return {**existing.public(), "deduplicated": True}
        job = TransitionJob(
            id=uuid.uuid4().hex,
            name=name,
            events=normalized,
            completion_verifier=completion_verifier,
            event_observer=event_observer,
            execution_id=execution_id,
        )
        job.control_reserved_until = self.engine.reserve_job_control(
            normalized[-1]["at_ms"] / 1000.0
        )
        self.jobs[job.id] = job
        if execution_id is not None:
            self.execution_jobs[execution_id] = job.id
        job.task = asyncio.create_task(self._run(job))
        return {**job.public(), "deduplicated": False}

    async def _run(self, job: TransitionJob) -> None:
        job.status = "running"
        job.started_at = time.time()
        started = time.monotonic()
        try:
            index = 0
            while index < len(job.events):
                at_ms = job.events[index]["at_ms"]
                group = []
                while (
                    index < len(job.events)
                    and job.events[index]["at_ms"] == at_ms
                ):
                    group.append(job.events[index])
                    index += 1
                target = started + at_ms / 1000
                delay = target - time.monotonic()
                # Windows' default asyncio timer can overshoot a musical
                # boundary by 15-30 ms. Sleep most of the interval, then use a
                # short cooperative high-resolution tail so MIDI dispatch is
                # consistently inside the 20 ms QA gate.
                precision_tail = 0.025
                if delay > precision_tail:
                    await asyncio.sleep(delay - precision_tail)
                while time.monotonic() < target:
                    await asyncio.sleep(0)
                if not self.engine.is_armed():
                    raise RuntimeError("Live MIDI control became disarmed")

                async def dispatch(event: dict[str, Any]) -> None:
                    job.event_lateness_ms.append(
                        max(0.0, (time.monotonic() - target) * 1000)
                    )
                    dispatched_monotonic = time.monotonic()
                    await self.engine.send_action(
                        event["action"], event["parameters"]
                    )
                    if job.event_observer is not None:
                        job.event_observer(event, dispatched_monotonic)
                    job.completed_events += 1

                # Events sharing a musical timestamp must reach MIDI together.
                # Serial button holds otherwise skew simultaneous Hot Cues and
                # two-deck bass swaps by the note-hold duration.
                await asyncio.gather(*(dispatch(event) for event in group))
            if job.completion_verifier is not None:
                job.status = "verifying"
                job.verification = await asyncio.to_thread(
                    job.completion_verifier
                )
                if not job.verification.get("verified", False):
                    raise RuntimeError(
                        "Rekordbox did not satisfy transition postconditions: "
                        + "; ".join(job.verification.get("errors", []))
                    )
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

    def metrics(self) -> dict[str, Any]:
        jobs = list(self.jobs.values())
        lateness = [
            value
            for job in jobs
            for value in job.event_lateness_ms
        ]
        return {
            "job_count": len(jobs),
            "completed_jobs": sum(job.status == "completed" for job in jobs),
            "failed_jobs": sum(job.status == "failed" for job in jobs),
            "active_jobs": sum(
                job.status in {"scheduled", "running", "verifying"}
                for job in jobs
            ),
            "event_count": len(lateness),
            "p95_event_lateness_ms": round(percentile(lateness, 0.95), 2),
            "p99_event_lateness_ms": round(percentile(lateness, 0.99), 2),
            "max_event_lateness_ms": round(max(lateness, default=0.0), 2),
            "events_over_10ms_late": sum(value > 10.0 for value in lateness),
        }
