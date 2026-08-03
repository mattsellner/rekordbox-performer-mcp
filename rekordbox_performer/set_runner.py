"""Autonomous, deadline-aware continuous-set execution.

The language-model client chooses the musical graph. Once started, this local
runner owns the deck lifecycle so tool round trips cannot strand playback after
one handoff.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .intelligence import TransitionCard


class TrackLoadSpec(BaseModel):
    track_id: str
    title: str
    artist: str = ""
    source: str | None = None
    result_index: int | None = Field(default=None, ge=0)
    cue: int | None = Field(default=None, ge=1, le=8)
    cue_time_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def cue_pair(self) -> "TrackLoadSpec":
        if (self.cue is None) != (self.cue_time_ms is None):
            raise ValueError("cue and cue_time_ms must be supplied together")
        return self


class TempoPlan(BaseModel):
    """A gradual post-handoff tempo trajectory for the new master deck."""

    target_bpm: float = Field(gt=0)
    duration_bars: int = Field(default=32, ge=8, le=128)
    steps_per_bar: int = Field(default=1, ge=1, le=4)
    tempo_range_percent: float = Field(default=10.0, gt=0, le=100)
    max_stretch_percent: float = Field(default=4.0, gt=0, le=10)
    risk_accepted: bool = False

    def control_value(self, *, native_bpm: float, bpm: float) -> float:
        percent = (bpm / native_bpm - 1.0) * 100.0
        value = percent / self.tempo_range_percent
        if not -1.0 <= value <= 1.0:
            raise ValueError(
                f"target {bpm:.2f} BPM exceeds the configured "
                f"+/-{self.tempo_range_percent:.1f}% tempo range"
            )
        return value

    def validate_start(self, *, native_bpm: float, live_bpm: float) -> None:
        stretch = abs(live_bpm / native_bpm - 1.0) * 100.0
        if stretch > self.max_stretch_percent and not self.risk_accepted:
            raise ValueError(
                f"initial stretch {stretch:.2f}% exceeds "
                f"{self.max_stretch_percent:.2f}%"
            )
        self.control_value(native_bpm=native_bpm, bpm=self.target_bpm)


class TransitionOption(BaseModel):
    id: str
    card: TransitionCard
    incoming: TrackLoadSpec
    priority: int = Field(default=100, ge=0)
    tempo_after: TempoPlan | None = None

    @model_validator(mode="after")
    def identity_matches(self) -> "TransitionOption":
        if self.card.incoming_track_id != self.incoming.track_id:
            raise ValueError("incoming load spec does not match transition card")
        launch = [
            event
            for event in self.card.events
            if event.bar_offset == 0
            and event.beat_offset == 0
            and event.action in {"hot_cue", "play_pause"}
            and event.parameters.get("deck") == self.card.incoming_deck
        ]
        if len(launch) != 1:
            raise ValueError("option requires one bar-0 incoming launch")
        if launch[0].action == "hot_cue":
            if self.incoming.cue != int(launch[0].parameters["cue"]):
                raise ValueError("load-spec cue does not match card Hot Cue")
        elif self.incoming.cue is not None:
            raise ValueError("file-start launch cannot declare a Hot Cue")
        return self


class AutonomousSetPlan(BaseModel):
    name: str
    opening: TrackLoadSpec
    transitions: list[TransitionOption] = Field(min_length=1)
    target_track_count: int = Field(ge=2, le=100)
    stage_deadline_bars: int = Field(default=32, ge=16, le=128)
    reserve_deadline_bars: int = Field(default=16, ge=8, le=64)
    rescue_loop_trigger_bars: int = Field(default=8, ge=4, le=16)
    rescue_loop_beats: Literal[4, 8, 16] = 16
    retry_limit: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def reachable_primary_path(self) -> "AutonomousSetPlan":
        if not (
            self.stage_deadline_bars
            > self.reserve_deadline_bars
            > self.rescue_loop_trigger_bars
        ):
            raise ValueError(
                "deadlines must satisfy stage > reserve > rescue-loop bars"
            )
        current = self.opening.track_id
        visited = {current}
        for _ in range(self.target_track_count - 1):
            choices = sorted(
                (
                    option
                    for option in self.transitions
                    if option.card.outgoing_track_id == current
                ),
                key=lambda option: option.priority,
            )
            if not choices:
                raise ValueError(
                    f"no transition option leaves planned track {current}"
                )
            current = choices[0].incoming.track_id
            if current in visited:
                raise ValueError("primary transition path contains a cycle")
            visited.add(current)
        return self


class AutonomousSetState(BaseModel):
    plan: AutonomousSetPlan
    status: Literal[
        "prepared", "running", "recovering", "completed", "failed", "stopped"
    ] = "prepared"
    current_track_id: str
    current_deck: int = Field(ge=1, le=2)
    played_track_ids: list[str]
    active_job_id: str | None = None
    active_option_id: str | None = None
    attempted_options: dict[str, int] = Field(default_factory=dict)
    transition_jobs: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    rescue_loop_active: bool = False
    rescue_loop_deck: int | None = None
    last_remaining_bars: float | None = None
    deadline_phase: Literal["normal", "stage", "reserve", "rescue"] = "normal"
    updated_at: float = Field(default_factory=time.time)


ScheduleCallback = Callable[[TransitionOption, bool], Awaitable[dict[str, Any]]]
StatusCallback = Callable[[str], dict[str, Any]]
QaCallback = Callable[[str], dict[str, Any]]
RemainingCallback = Callable[[str, int], Awaitable[float]]
LoopCallback = Callable[[int, int], Awaitable[dict[str, Any]]]
TempoCallback = Callable[[TempoPlan, int, str], Awaitable[dict[str, Any]]]
AdvanceCallback = Callable[[str, bool, str | None], dict[str, Any]]
FinishCallback = Callable[[str], None]


class AutonomousSetRunner:
    """Local continuous-set state machine with deadline recovery."""

    def __init__(
        self,
        path: Path,
        *,
        schedule: ScheduleCallback,
        job_status: StatusCallback,
        job_qa: QaCallback,
        remaining_bars: RemainingCallback,
        engage_loop: LoopCallback,
        run_tempo: TempoCallback,
        advance: AdvanceCallback,
        finish: FinishCallback | None = None,
        poll_seconds: float = 0.25,
    ) -> None:
        self.path = path
        self.schedule = schedule
        self.job_status = job_status
        self.job_qa = job_qa
        self.remaining_bars = remaining_bars
        self.engage_loop = engage_loop
        self.run_tempo = run_tempo
        self.advance = advance
        self.finish = finish or (lambda _status: None)
        self.poll_seconds = poll_seconds
        self.task: asyncio.Task[None] | None = None
        self.state: AutonomousSetState | None = self._read()

    def _read(self) -> AutonomousSetState | None:
        if not self.path.exists():
            return None
        return AutonomousSetState.model_validate_json(
            self.path.read_text(encoding="utf-8")
        )

    def _write(self) -> None:
        if self.state is None:
            return
        self.state.updated_at = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        os.close(fd)
        temp = Path(raw)
        try:
            temp.write_text(self.state.model_dump_json(indent=2), encoding="utf-8")
            os.replace(temp, self.path)
        finally:
            temp.unlink(missing_ok=True)

    def prepare(
        self,
        plan: AutonomousSetPlan,
        *,
        opening_deck: int,
    ) -> dict[str, Any]:
        if self.task and not self.task.done():
            raise RuntimeError("an autonomous set is already running")
        self.state = AutonomousSetState(
            plan=plan,
            current_track_id=plan.opening.track_id,
            current_deck=opening_deck,
            played_track_ids=[plan.opening.track_id],
        )
        self._write()
        return self.public()

    def start_with_job(self, option_id: str, job_id: str) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("no autonomous set is prepared")
        if self.task and not self.task.done():
            raise RuntimeError("autonomous set is already running")
        option = self._option(option_id)
        self.state.status = "running"
        self.state.active_option_id = option.id
        self.state.active_job_id = job_id
        self.state.attempted_options[option.id] = (
            self.state.attempted_options.get(option.id, 0) + 1
        )
        self.state.transition_jobs.append(job_id)
        self._write()
        self.task = asyncio.create_task(self._run())
        return self.public()

    def resume(self) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("no autonomous set state exists")
        if self.state.status in {"completed", "failed", "stopped"}:
            raise RuntimeError(f"cannot resume a {self.state.status} set")
        if self.task and not self.task.done():
            return self.public()
        self.state.status = "running"
        self._write()
        self.task = asyncio.create_task(self._run())
        return self.public()

    def stop(self) -> dict[str, Any]:
        if self.task and not self.task.done():
            self.task.cancel()
        if self.state is not None:
            self.state.status = "stopped"
            self._write()
            self.finish("stopped")
        return self.public()

    def public(self) -> dict[str, Any]:
        if self.state is None:
            return {"active": False}
        return {
            "active": self.state.status in {"running", "recovering"},
            **self.state.model_dump(exclude={"plan"}),
            "plan_name": self.state.plan.name,
            "target_track_count": self.state.plan.target_track_count,
            "task_running": bool(self.task and not self.task.done()),
        }

    def _option(self, option_id: str) -> TransitionOption:
        assert self.state is not None
        return next(
            option
            for option in self.state.plan.transitions
            if option.id == option_id
        )

    def _choices(self) -> list[TransitionOption]:
        assert self.state is not None
        return sorted(
            (
                option
                for option in self.state.plan.transitions
                if option.card.outgoing_track_id == self.state.current_track_id
                and self.state.attempted_options.get(option.id, 0)
                < self.state.plan.retry_limit
            ),
            key=lambda option: (
                self.state.attempted_options.get(option.id, 0),
                option.priority,
            ),
        )

    @staticmethod
    def _is_transient_observer_failure(exc: Exception) -> bool:
        """Return whether a staging failure is only an observation outage.

        A stale/missing deck clock says nothing about the musical route.  It
        must not consume that route's finite retry budget, otherwise three
        short UIA stalls can terminate an otherwise safe set while the live
        track still has minutes of runway.
        """
        message = str(exc).casefold()
        return any(
            marker in message
            for marker in (
                "observation is stale",
                "deck observation is stale",
                "observation unavailable",
                "could not observe",
            )
        )

    async def _recover_if_late(self) -> None:
        assert self.state is not None
        remaining = await self.remaining_bars(
            self.state.current_track_id,
            self.state.current_deck,
        )
        self.state.last_remaining_bars = remaining
        if remaining <= self.state.plan.rescue_loop_trigger_bars:
            self.state.deadline_phase = "rescue"
        elif remaining <= self.state.plan.reserve_deadline_bars:
            self.state.deadline_phase = "reserve"
        elif remaining <= self.state.plan.stage_deadline_bars:
            self.state.deadline_phase = "stage"
        else:
            self.state.deadline_phase = "normal"
        self._write()
        if (
            remaining <= self.state.plan.rescue_loop_trigger_bars
            and not self.state.rescue_loop_active
        ):
            result = await self.engage_loop(
                self.state.current_deck,
                self.state.plan.rescue_loop_beats,
            )
            if result.get("verified") is not True:
                raise RuntimeError("rescue loop could not be verified")
            self.state.rescue_loop_active = True
            self.state.rescue_loop_deck = self.state.current_deck
            self.state.status = "recovering"
            self._write()

    async def _run(self) -> None:
        assert self.state is not None
        try:
            while True:
                if len(self.state.played_track_ids) >= self.state.plan.target_track_count:
                    self.state.status = "completed"
                    self._write()
                    self.finish("completed")
                    return

                if self.state.active_job_id:
                    job = self.job_status(self.state.active_job_id)
                    if job["status"] in {"scheduled", "running", "verifying"}:
                        await asyncio.sleep(self.poll_seconds)
                        continue
                    option = self._option(str(self.state.active_option_id))
                    qa = self.job_qa(self.state.active_job_id)
                    if job["status"] == "completed" and qa.get("passed"):
                        self.advance(self.state.active_job_id, True, None)
                        self.state.current_track_id = option.incoming.track_id
                        self.state.current_deck = option.card.incoming_deck
                        self.state.played_track_ids.append(option.incoming.track_id)
                        self.state.active_job_id = None
                        self.state.active_option_id = None
                        self.state.rescue_loop_active = False
                        self.state.rescue_loop_deck = None
                        self.state.status = "running"
                        self._write()
                        if option.tempo_after is not None:
                            await self.run_tempo(
                                option.tempo_after,
                                self.state.current_deck,
                                option.incoming.track_id,
                            )
                        continue
                    error = "; ".join(qa.get("faults", [])) or job.get("error")
                    self.advance(self.state.active_job_id, False, error)
                    self.state.failures.append(
                        f"{option.id}: {error or 'transition failed'}"
                    )
                    self.state.active_job_id = None
                    self.state.active_option_id = None
                    self.state.status = "recovering"
                    self._write()

                choices = self._choices()
                if not choices:
                    raise RuntimeError(
                        f"no safe transition remains from {self.state.current_track_id}"
                    )
                # Check the live runway before every staging attempt, not only
                # after an error.  This makes a verified rescue loop the
                # proactive deadline response when a prior handoff or UI scan
                # consumed more time than expected.
                await self._recover_if_late()
                option = choices[0]
                self.state.attempted_options[option.id] = (
                    self.state.attempted_options.get(option.id, 0) + 1
                )
                self._write()
                try:
                    scheduled = await self.schedule(
                        option,
                        self.state.rescue_loop_active,
                    )
                    if not scheduled.get("ready"):
                        raise RuntimeError(
                            "; ".join(scheduled.get("errors", []))
                            or "transition was not accepted"
                        )
                    job = scheduled.get("job") or {}
                    job_id = job.get("id")
                    if not job_id:
                        raise RuntimeError("scheduled transition has no job ID")
                    self.state.active_option_id = option.id
                    self.state.active_job_id = job_id
                    self.state.transition_jobs.append(job_id)
                    self.state.status = "running"
                    if self.state.rescue_loop_active:
                        # The scheduling callback inserts a phrase-boundary
                        # loop release into the accepted card.
                        self.state.rescue_loop_active = False
                        self.state.rescue_loop_deck = None
                    self._write()
                except Exception as exc:
                    if self._is_transient_observer_failure(exc):
                        # The option itself was never disproved. Preserve its
                        # retry budget and keep polling the deadline state; if
                        # the outage persists, the normal rescue-loop path
                        # protects the audible deck until observation returns.
                        attempts = self.state.attempted_options.get(option.id, 0)
                        if attempts <= 1:
                            self.state.attempted_options.pop(option.id, None)
                        else:
                            self.state.attempted_options[option.id] = attempts - 1
                    self.state.failures.append(f"{option.id}: {exc}")
                    self.state.status = "recovering"
                    self._write()
                    await self._recover_if_late()
                    await asyncio.sleep(self.poll_seconds)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state.status = "failed"
            self.state.failures.append(str(exc))
            self._write()
            self.finish("failed")
