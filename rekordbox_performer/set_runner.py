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
        initial_stretch = abs(live_bpm / native_bpm - 1.0) * 100.0
        target_stretch = abs(self.target_bpm / native_bpm - 1.0) * 100.0
        if (
            max(initial_stretch, target_stretch) > self.max_stretch_percent
            and not self.risk_accepted
        ):
            raise ValueError(
                "tempo trajectory stretch exceeds the safe limit "
                f"(initial {initial_stretch:.2f}%, target "
                f"{target_stretch:.2f}%, limit "
                f"{self.max_stretch_percent:.2f}%)"
            )
        self.control_value(native_bpm=native_bpm, bpm=self.target_bpm)


class TransitionOption(BaseModel):
    id: str
    card: TransitionCard
    incoming: TrackLoadSpec
    priority: int = Field(default=100, ge=0)
    tempo_after: TempoPlan | None = None
    technique: str = ""
    reason: str = ""
    alternatives: list[str] = Field(default_factory=list)
    fx_effect: Literal["echo", "reverb", "spiral", "vinyl_brake"] | None = None

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
    # Runtime work begins while there is still musical room to recover from a
    # slow Rekordbox UI scan.  The old 32/16/8 defaults left only ~15 seconds
    # at house tempos for a rescue loop and reproduced exactly the dead-air
    # failure this runner exists to prevent.
    stage_deadline_bars: int = Field(default=64, ge=16, le=128)
    reserve_deadline_bars: int = Field(default=32, ge=8, le=64)
    rescue_loop_trigger_bars: int = Field(default=16, ge=4, le=32)
    rescue_loop_beats: Literal[4, 8, 16] = 16
    retry_limit: int = Field(default=3, ge=1, le=10)
    tempo_strategy: Literal["auto", "manual", "hold"] = "auto"
    tempo_target_bpm: float | None = Field(default=None, gt=0)
    tempo_ramp_bars: int = Field(default=32, ge=8, le=128)
    tempo_steps_per_bar: int = Field(default=1, ge=1, le=4)
    tempo_range_percent: float = Field(default=10.0, gt=0, le=100)
    tempo_max_stretch_percent: float = Field(default=4.0, gt=0, le=10)
    tempo_arc_minimum_change_bpm: float = Field(default=2.0, ge=0, le=10)

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
                raise ValueError(f"no transition option leaves planned track {current}")
            current = choices[0].incoming.track_id
            if current in visited:
                raise ValueError("primary transition path contains a cycle")
            visited.add(current)
        return self

    def materialize_tempo_arc(
        self,
        native_bpms: dict[str, float],
    ) -> "AutonomousSetPlan":
        """Resolve an explicit gradual BPM target for every reachable depth."""
        required_ids = {self.opening.track_id}
        required_ids.update(option.incoming.track_id for option in self.transitions)
        missing = sorted(required_ids - native_bpms.keys())
        if missing:
            raise ValueError(f"tempo arc is missing native BPM for {missing}")

        current = self.opening.track_id
        primary: list[TransitionOption] = []
        depth_by_track = {current: 0}
        for depth in range(self.target_track_count - 1):
            choices = sorted(
                (
                    option
                    for option in self.transitions
                    if option.card.outgoing_track_id == current
                ),
                key=lambda option: option.priority,
            )
            option = choices[0]
            primary.append(option)
            depth_by_track.setdefault(option.incoming.track_id, depth + 1)
            current = option.incoming.track_id

        # Assign the same set-position depth to prepared fallback branches.
        # Do not overwrite an earlier depth: that safely ignores cycles such
        # as a preflight-only final-track route back to the opener.
        for depth in range(self.target_track_count - 1):
            for option in self.transitions:
                if depth_by_track.get(option.card.outgoing_track_id) != depth:
                    continue
                depth_by_track.setdefault(option.incoming.track_id, depth + 1)

        start_bpm = float(native_bpms[self.opening.track_id])
        target_bpm = float(
            self.tempo_target_bpm
            if self.tempo_target_bpm is not None
            else native_bpms[primary[-1].incoming.track_id]
        )
        total_change = target_bpm - start_bpm
        if self.tempo_strategy == "hold" or abs(total_change) < 0.05:
            return self
        if self.tempo_strategy == "manual":
            if abs(total_change) >= self.tempo_arc_minimum_change_bpm and not any(
                option.tempo_after is not None for option in primary
            ):
                raise ValueError(
                    "manual tempo strategy spans a material BPM change but "
                    "contains no TempoPlan"
                )
            return self

        steps = self.target_track_count - 1
        updated: list[TransitionOption] = []
        for option in self.transitions:
            depth = depth_by_track.get(option.card.outgoing_track_id)
            incoming_depth = depth_by_track.get(option.incoming.track_id)
            if (
                depth is None
                or depth >= steps
                or incoming_depth != depth + 1
                or option.tempo_after is not None
            ):
                updated.append(option)
                continue
            desired_bpm = start_bpm + total_change * incoming_depth / steps
            prior_bpm = start_bpm + total_change * depth / steps
            native_bpm = float(native_bpms[option.incoming.track_id])
            tempo = TempoPlan(
                target_bpm=round(desired_bpm, 3),
                duration_bars=self.tempo_ramp_bars,
                steps_per_bar=self.tempo_steps_per_bar,
                tempo_range_percent=self.tempo_range_percent,
                max_stretch_percent=self.tempo_max_stretch_percent,
            )
            tempo.validate_start(native_bpm=native_bpm, live_bpm=prior_bpm)
            updated.append(option.model_copy(update={"tempo_after": tempo}))
        return self.model_copy(update={"transitions": updated})

    def tempo_arc_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "option_id": option.id,
                "incoming_track_id": option.incoming.track_id,
                "target_bpm": option.tempo_after.target_bpm,
                "duration_bars": option.tempo_after.duration_bars,
            }
            for option in self.transitions
            if option.tempo_after is not None
        ]


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
    staged_option_id: str | None = None
    transition_start_at: float | None = None
    transition_critical_at: float | None = None
    pending_redirect: AutonomousSetPlan | None = None
    attempted_options: dict[str, int] = Field(default_factory=dict)
    transition_jobs: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recovery_attempts: int = 0
    last_recovery_error: str | None = None
    rescue_loop_active: bool = False
    rescue_loop_deck: int | None = None
    last_remaining_bars: float | None = None
    deadline_phase: Literal["normal", "stage", "reserve", "rescue"] = "normal"
    updated_at: float = Field(default_factory=time.time)


ScheduleCallback = Callable[[TransitionOption, bool], Awaitable[dict[str, Any]]]
PrestageCallback = Callable[[TransitionOption], Awaitable[dict[str, Any]]]
StatusCallback = Callable[[str], dict[str, Any]]
QaCallback = Callable[[str], dict[str, Any]]
RemainingCallback = Callable[[str, int], Awaitable[float]]
LoopCallback = Callable[[int, int], Awaitable[dict[str, Any]]]
TempoCallback = Callable[[TempoPlan, int, str], Awaitable[dict[str, Any]]]
AdvanceCallback = Callable[[str, bool, str | None], dict[str, Any]]
FinishCallback = Callable[[str], None]
RecoveryCallback = Callable[[dict[str, Any]], Awaitable[AutonomousSetPlan | None]]


class AutonomousSetRunner:
    """Local continuous-set state machine with deadline recovery."""

    def __init__(
        self,
        path: Path,
        *,
        schedule: ScheduleCallback,
        prestage: PrestageCallback | None = None,
        job_status: StatusCallback,
        job_qa: QaCallback,
        remaining_bars: RemainingCallback,
        engage_loop: LoopCallback,
        run_tempo: TempoCallback,
        advance: AdvanceCallback,
        finish: FinishCallback | None = None,
        recover_route: RecoveryCallback | None = None,
        poll_seconds: float = 0.25,
        recovery_retry_seconds: float = 2.0,
    ) -> None:
        self.path = path
        self.schedule = schedule
        self.prestage = prestage or self._noop_prestage
        self.job_status = job_status
        self.job_qa = job_qa
        self.remaining_bars = remaining_bars
        self.engage_loop = engage_loop
        self.run_tempo = run_tempo
        self.advance = advance
        self.finish = finish or (lambda _status: None)
        self.recover_route = recover_route
        self.poll_seconds = poll_seconds
        self.recovery_retry_seconds = recovery_retry_seconds
        self.task: asyncio.Task[None] | None = None
        self.state: AutonomousSetState | None = self._read()

    @staticmethod
    async def _noop_prestage(_option: TransitionOption) -> dict[str, Any]:
        return {"ready": True, "skipped": True}

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

    def start_with_job(
        self,
        option_id: str,
        job_id: str,
        schedule: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        self._set_transition_timing(option, schedule or {})
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
            **self.state.model_dump(exclude={"plan", "pending_redirect"}),
            "plan_name": self.state.plan.name,
            "target_track_count": self.state.plan.target_track_count,
            "retry_limit": self.state.plan.retry_limit,
            "exhausted_incoming_track_ids": self._exhausted_incoming_track_ids(),
            "steering_queued": self.state.pending_redirect is not None,
            "steering_target_track_id": (
                self._primary_path_ids(self.state.pending_redirect)[-1]
                if self.state.pending_redirect is not None
                else None
            ),
            "task_running": bool(self.task and not self.task.done()),
        }

    def _exhausted_incoming_track_ids(self) -> list[str]:
        if self.state is None:
            return []
        return list(
            dict.fromkeys(
                option.incoming.track_id
                for option in self.state.plan.transitions
                if option.card.outgoing_track_id == self.state.current_track_id
                and self.state.attempted_options.get(option.id, 0)
                >= self.state.plan.retry_limit
            )
        )

    @staticmethod
    def _primary_path_ids(plan: AutonomousSetPlan) -> list[str]:
        result = [plan.opening.track_id]
        current = plan.opening.track_id
        for _ in range(plan.target_track_count - 1):
            option = min(
                (
                    item
                    for item in plan.transitions
                    if item.card.outgoing_track_id == current
                ),
                key=lambda item: item.priority,
            )
            result.append(option.incoming.track_id)
            current = option.incoming.track_id
        return result

    def _project_redirect(self, future: AutonomousSetPlan) -> AutonomousSetPlan:
        assert self.state is not None
        prefix_ids = list(self.state.played_track_ids)
        if self.state.active_option_id is not None:
            active = self._option(self.state.active_option_id)
            if active.incoming.track_id != future.opening.track_id:
                raise ValueError(
                    "steering must begin after the already-armed incoming track"
                )
            if prefix_ids[-1] != active.incoming.track_id:
                prefix_ids.append(active.incoming.track_id)
        elif prefix_ids[-1] != future.opening.track_id:
            raise ValueError("steering plan does not begin at the current track")

        prefix: list[TransitionOption] = []
        for outgoing, incoming in zip(prefix_ids, prefix_ids[1:]):
            options = [
                item
                for item in self.state.plan.transitions
                if item.card.outgoing_track_id == outgoing
                and item.incoming.track_id == incoming
            ]
            if not options:
                raise ValueError(
                    f"existing set plan does not contain {outgoing} -> {incoming}"
                )
            prefix.append(min(options, key=lambda item: item.priority))

        payload = self.state.plan.model_dump()
        payload.update(
            {
                "name": f"{self.state.plan.name} -> {future.name}",
                "transitions": [
                    item.model_dump() for item in [*prefix, *future.transitions]
                ],
                "target_track_count": len(prefix_ids) + future.target_track_count - 1,
                "tempo_target_bpm": future.tempo_target_bpm,
            }
        )
        return AutonomousSetPlan.model_validate(payload)

    def queue_redirect(self, future: AutonomousSetPlan) -> dict[str, Any]:
        """Queue a new route after the handoff that is already armed."""
        if self.state is None or self.state.status not in {"running", "recovering"}:
            raise RuntimeError("an autonomous set must be running before steering")
        if self.state.active_option_id is None:
            raise RuntimeError(
                "wait until the next transition is armed before steering the set"
            )
        projected = self._project_redirect(future)
        self.state.pending_redirect = future
        self._write()
        return {
            "queued": True,
            "runner": self.public(),
            "projected_plan": projected.model_dump(),
        }

    def _apply_pending_redirect(self) -> None:
        assert self.state is not None
        future = self.state.pending_redirect
        if future is None:
            return
        if future.opening.track_id != self.state.current_track_id:
            return
        self.state.plan = self._project_redirect(future)
        self.state.pending_redirect = None
        self.state.staged_option_id = None
        valid_ids = {item.id for item in self.state.plan.transitions}
        self.state.attempted_options = {
            option_id: attempts
            for option_id, attempts in self.state.attempted_options.items()
            if option_id in valid_ids
        }
        self._write()

    def _option(self, option_id: str) -> TransitionOption:
        assert self.state is not None
        return next(
            option for option in self.state.plan.transitions if option.id == option_id
        )

    def _set_transition_timing(
        self,
        option: TransitionOption,
        scheduled: dict[str, Any],
    ) -> None:
        assert self.state is not None
        delay_ms = scheduled.get("start_delay_ms")
        bpm = scheduled.get("bpm")
        if delay_ms is None or bpm is None or float(bpm) <= 0:
            self.state.transition_start_at = None
            self.state.transition_critical_at = None
            return
        transition_start = time.time() + float(delay_ms) / 1000.0
        critical_seconds = option.card.critical_bar_offset * 4 * 60.0 / float(bpm)
        self.state.transition_start_at = transition_start
        self.state.transition_critical_at = transition_start + critical_seconds

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
                "deck title changed during transport observation",
                "result_index must be between",
            )
        )

    async def _recover_if_late(self) -> None:
        assert self.state is not None
        try:
            remaining = await self.remaining_bars(
                self.state.current_track_id,
                self.state.current_deck,
            )
        except Exception as exc:
            if not self._is_transient_observer_failure(exc):
                raise
            warning = f"deadline observation retrying: {exc}"
            if not self.state.warnings or self.state.warnings[-1] != warning:
                self.state.warnings.append(warning)
            self.state.status = "recovering"
            self._write()
            return
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

    async def _request_replacement_route(self) -> bool:
        assert self.state is not None
        if self.recover_route is None:
            return False
        self.state.status = "recovering"
        self.state.recovery_attempts += 1
        self._write()
        try:
            future = await self.recover_route(self.public())
            if future is None:
                raise RuntimeError("replacement planner returned no route")
            projected = self._project_redirect(future)
        except Exception as exc:
            message = str(exc)
            self.state.last_recovery_error = message
            warning = (
                f"replacement planning attempt {self.state.recovery_attempts}: "
                f"{message}"
            )
            if not self.state.warnings or self.state.warnings[-1] != warning:
                self.state.warnings.append(warning)
            self._write()
            return False

        self.state.plan = projected
        self.state.pending_redirect = None
        self.state.staged_option_id = None
        valid_ids = {item.id for item in projected.transitions}
        self.state.attempted_options = {
            option_id: attempts
            for option_id, attempts in self.state.attempted_options.items()
            if option_id in valid_ids
        }
        self.state.last_recovery_error = None
        first = min(
            (
                item
                for item in projected.transitions
                if item.card.outgoing_track_id == self.state.current_track_id
            ),
            key=lambda item: item.priority,
        )
        self.state.warnings.append(
            "replacement route selected: "
            f"{self.state.current_track_id} -> {first.incoming.track_id}"
        )
        self._write()
        return True

    async def _run(self) -> None:
        assert self.state is not None
        try:
            while True:
                if (
                    len(self.state.played_track_ids)
                    >= self.state.plan.target_track_count
                ):
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
                        self.state.transition_start_at = None
                        self.state.transition_critical_at = None
                        self.state.rescue_loop_active = False
                        self.state.rescue_loop_deck = None
                        self.state.status = "running"
                        self._write()
                        # A steering request never alters the handoff that was
                        # already armed.  Apply it only now, after that job has
                        # passed QA and its incoming track is authoritative.
                        self._apply_pending_redirect()
                        # The next deck must be loaded immediately after it is
                        # retired.  A tempo ramp may take 32 bars; awaiting it
                        # before staging violated the rolling two-track lead
                        # and could consume the entire following transition
                        # window.  Stage and ramp concurrently, then compile
                        # against the final tempo clock.
                        tempo_task = None
                        if option.tempo_after is not None:
                            tempo_task = asyncio.create_task(
                                self.run_tempo(
                                    option.tempo_after,
                                    self.state.current_deck,
                                    option.incoming.track_id,
                                )
                            )
                        if (
                            len(self.state.played_track_ids)
                            < self.state.plan.target_track_count
                        ):
                            next_choices = self._choices()
                            if next_choices:
                                next_option = next_choices[0]
                                try:
                                    staged = await self.prestage(next_option)
                                    if staged.get("ready") is not True:
                                        raise RuntimeError(
                                            "; ".join(staged.get("errors", []))
                                            or "next track was not staged"
                                        )
                                    self.state.staged_option_id = next_option.id
                                    self._write()
                                except Exception as exc:
                                    # Staging is retried by the normal atomic
                                    # schedule path.  Keep the audible track
                                    # and route alive; never burn the option's
                                    # retry budget for an early preload.
                                    self.state.failures.append(
                                        f"{next_option.id}: early staging: {exc}"
                                    )
                                    self._write()
                        if tempo_task is not None:
                            try:
                                await tempo_task
                            except Exception as exc:  # tempo is noncritical
                                # A rejected tempo CC must not strand a safe,
                                # audible deck or prevent the already-staged
                                # next transition. Preserve the musical route
                                # at its current BPM and surface the degraded
                                # energy arc separately from hard failures.
                                self.state.warnings.append(
                                    f"{option.id}: tempo ramp skipped: {exc}"
                                )
                                self._write()
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
                    # A failed browser route or transition candidate is not a
                    # reason to end an audible set. Ask the embedded local DJ
                    # planner for a replacement graph. If planning is briefly
                    # unavailable, keep the current track alive and move into
                    # the verified rescue loop as its deadline approaches.
                    await self._recover_if_late()
                    if await self._request_replacement_route():
                        continue
                    if self.recover_route is None:
                        raise RuntimeError(
                            "no safe transition remains from "
                            f"{self.state.current_track_id}"
                        )
                    await asyncio.sleep(self.recovery_retry_seconds)
                    continue
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
                    if (
                        self.state.pending_redirect is not None
                        and option.incoming.track_id
                        != self.state.pending_redirect.opening.track_id
                    ):
                        self.state.failures.append(
                            "queued steering was cancelled because recovery "
                            "selected a different incoming track"
                        )
                        self.state.pending_redirect = None
                    self.state.active_option_id = option.id
                    self.state.active_job_id = job_id
                    self.state.staged_option_id = option.id
                    self._set_transition_timing(option, scheduled)
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
