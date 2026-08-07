"""Standalone local DJ engine independent of Codex/MCP round trips."""

from __future__ import annotations

import asyncio
import threading
import time
import unicodedata
from typing import Any, Protocol

from .intelligence import ProfileStore, TrackProfile
from .playlist_catalog import PlaylistSummary, RekordboxPlaylistCatalog
from .rekordbox_ui import RekordboxUIAdapter
from .set_runner import AutonomousSetPlan
from .standalone_planner import DJBrief, LocalDJPlanner
from .standalone_status import (
    RuntimeHealth,
    RuntimePhase,
    StandaloneStatusStore,
    TrackRole,
)


class PerformerAdapter(Protocol):
    def connect(self) -> dict[str, Any]: ...
    def arm(self, seconds: int) -> dict[str, Any]: ...
    async def preflight(self, plan: AutonomousSetPlan) -> dict[str, Any]: ...
    async def start(self, plan: AutonomousSetPlan) -> dict[str, Any]: ...
    async def continue_set(self, plan: AutonomousSetPlan) -> dict[str, Any]: ...
    def runner_status(self) -> dict[str, Any]: ...
    def control_status(self) -> dict[str, Any]: ...
    def rekordbox_status(self) -> dict[str, Any]: ...
    def transport_status(self) -> dict[str, Any]: ...
    def queue_steering(self, plan: AutonomousSetPlan) -> dict[str, Any]: ...
    def register_recovery_planner(self, callback: Any) -> None: ...
    async def hold_loop(self, deck: int, beats: int) -> dict[str, Any]: ...
    def stop_automation(self) -> dict[str, Any]: ...
    def emergency_stop(self) -> dict[str, Any]: ...


class InProcessPerformerAdapter:
    """Use the existing Performer engine directly in the desktop process."""

    def __init__(self) -> None:
        self._passive_ui = RekordboxUIAdapter()
        self._passive_lock = threading.Lock()

    @staticmethod
    def _server():
        from . import server

        return server

    def connect(self) -> dict[str, Any]:
        return self._server().connect_midi()

    def arm(self, seconds: int) -> dict[str, Any]:
        return self._server().arm_control(seconds)

    async def preflight(self, plan: AutonomousSetPlan) -> dict[str, Any]:
        return await self._server().preflight_autonomous_set(plan)

    async def start(self, plan: AutonomousSetPlan) -> dict[str, Any]:
        return await self._server().start_autonomous_set(plan)

    async def continue_set(self, plan: AutonomousSetPlan) -> dict[str, Any]:
        return await self._server().continue_autonomous_set(plan)

    def runner_status(self) -> dict[str, Any]:
        return self._server().autonomous_set_status()

    def control_status(self) -> dict[str, Any]:
        return self._server().control_status()

    def rekordbox_status(self) -> dict[str, Any]:
        return self.transport_status()

    def transport_status(self) -> dict[str, Any]:
        # Idle observation must never load MCP, focus Rekordbox, capture its
        # pixels, or claim the live-control lease.
        with self._passive_lock:
            return self._passive_ui.transport_status()

    def queue_steering(self, plan: AutonomousSetPlan) -> dict[str, Any]:
        return self._server().queue_autonomous_steering(plan)

    def register_recovery_planner(self, callback: Any) -> None:
        self._server().register_autonomous_recovery_planner(callback)

    async def hold_loop(self, deck: int, beats: int) -> dict[str, Any]:
        return await self._server().engage_rescue_loop(deck, beats)

    def stop_automation(self) -> dict[str, Any]:
        return self._server().stop_autonomous_set()

    def emergency_stop(self) -> dict[str, Any]:
        return self._server().emergency_stop()


class StandaloneDJEngine:
    """Plan, launch, monitor, and recover a complete Rekordbox set locally."""

    def __init__(
        self,
        *,
        profile_store: ProfileStore | None = None,
        status_store: StandaloneStatusStore | None = None,
        adapter: PerformerAdapter | None = None,
        playlist_catalog: RekordboxPlaylistCatalog | None = None,
        monitor_seconds: float = 0.5,
    ) -> None:
        self.profile_store = profile_store or ProfileStore()
        self.status_store = status_store or StandaloneStatusStore(
            self.profile_store.data_dir
        )
        self.adapter = adapter or InProcessPerformerAdapter()
        self.playlist_catalog = playlist_catalog or RekordboxPlaylistCatalog()
        self.monitor_seconds = monitor_seconds
        self.plan: AutonomousSetPlan | None = None
        self.monitor_task: asyncio.Task[None] | None = None
        self._last_signature: tuple[Any, ...] | None = None
        self._automation_active = False
        self._manual_previous: dict[int, tuple[str, float | None]] = {}
        self._manual_playing_until: dict[int, float] = {}
        self._manual_current_deck: int | None = None
        self._last_manual_signature: tuple[Any, ...] | None = None
        self._endless = False
        self._endless_vibe = "maintain"
        self._candidate_track_ids: set[str] | None = None
        self._endless_extension_pending = False

    def playlists(self) -> list[PlaylistSummary]:
        ready = {
            profile.track_id
            for profile in self.profile_store.list_profiles(ready_only=True)
        }
        return self.playlist_catalog.list_playlists(ready)

    def find_tracks(self, query: str, limit: int = 20) -> list[TrackProfile]:
        normalized = self._search_key(query)
        profiles = self.profile_store.list_profiles(ready_only=True)
        matches = [
            profile
            for profile in profiles
            if not normalized
            or normalized in self._search_key(profile.title)
            or normalized in self._search_key(profile.artist)
            or normalized in self._search_key(self.track_label(profile))
        ]
        return matches[:limit]

    @staticmethod
    def _search_key(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        return " ".join(
            "".join(
                character
                for character in normalized
                if not unicodedata.combining(character)
            )
            .casefold()
            .replace("—", "-")
            .split()
        )

    @staticmethod
    def track_label(profile: TrackProfile) -> str:
        return f"{profile.title} — {profile.artist}"

    def resolve_track(self, query: str) -> TrackProfile:
        normalized = self._search_key(query)
        if not normalized:
            raise ValueError("choose a prepared track")
        profiles = self.profile_store.list_profiles(ready_only=True)
        exact = [
            profile
            for profile in profiles
            if normalized
            in {
                self._search_key(profile.track_id),
                self._search_key(profile.title),
                self._search_key(self.track_label(profile)),
            }
        ]
        if len(exact) == 1:
            return exact[0]
        matches = self.find_tracks(query, limit=20)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(
                "No automation-ready track matched. Choose one from the prepared list."
            )
        choices = ", ".join(self.track_label(item) for item in matches[:5])
        raise ValueError(
            f"Track search is ambiguous; choose one exact result: {choices}"
        )

    def loaded_track(self, deck: int = 1) -> TrackProfile:
        if deck not in {1, 2}:
            raise ValueError("deck must be 1 or 2")
        status = self.adapter.rekordbox_status()
        record = next(
            (item for item in status.get("decks", []) if item.get("deck") == deck),
            None,
        )
        if record is None or not str(record.get("title") or "").strip():
            raise RuntimeError(f"Rekordbox Deck {deck} has no observable loaded track")
        title_key = self._search_key(str(record["title"]))
        matches = [
            profile
            for profile in self.profile_store.list_profiles()
            if self._search_key(profile.title) == title_key
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Deck {deck} title {record['title']!r} does not uniquely "
                "match a prepared profile"
            )
        profile = matches[0]
        readiness = profile.readiness()
        if not readiness["ready"]:
            missing = ", ".join(
                name for name, passed in readiness["checks"].items() if not passed
            )
            raise RuntimeError(
                f"{profile.title} is loaded but not automation ready; missing {missing}"
            )
        return profile

    async def start_set(
        self,
        brief: DJBrief,
        *,
        endless: bool = False,
    ) -> dict[str, Any]:
        self._automation_active = True
        self._endless = endless
        self._endless_vibe = brief.vibe
        self._candidate_track_ids = (
            set(brief.allowed_track_ids) if brief.allowed_track_ids else None
        )
        self.status_store.publish(
            phase=RuntimePhase.SELECTING,
            headline="Choosing and validating the set",
            detail="Scoring prepared tracks and compiling every transition locally.",
            bpm=None,
            bar=None,
            beat=None,
            action_in_bars=None,
            action_in_seconds=None,
            transition_name=None,
            transition_family=None,
            transition_technique=None,
            transition_reason=None,
            transition_alternatives=[],
            critical_in_bars=None,
            failures=[],
        )
        planner = LocalDJPlanner(
            self.profile_store.list_profiles(ready_only=True),
            proficient_techniques=self.profile_store.proficient_techniques(),
        )
        try:
            plan = planner.build_plan(brief)
        except Exception as exc:
            self._automation_active = False
            self._fail("Set planning failed", str(exc))
            raise
        return await self._launch_plan(plan)

    async def start_playlist_set(
        self,
        playlist_id: str,
        *,
        opening_track_id: str | None = None,
        vibe: str = "maintain",
        endless: bool = False,
    ) -> dict[str, Any]:
        summary = next(
            (
                item
                for item in self.playlists()
                if item.playlist_id == str(playlist_id)
            ),
            None,
        )
        if summary is None:
            raise ValueError("Rekordbox playlist is no longer available")
        if summary.ready_count != summary.track_count:
            raise ValueError(
                f"{summary.name} has {summary.track_count} tracks but only "
                f"{summary.ready_count} are automation-ready; analyze the missing "
                "tracks before requesting an all-song set."
            )
        self._automation_active = True
        self._endless = endless
        self._endless_vibe = vibe
        self._candidate_track_ids = set(summary.track_ids)
        self.status_store.publish(
            phase=RuntimePhase.SELECTING,
            headline=f"Mapping playlist: {summary.name}",
            detail=(
                f"Optimizing one safe route through all {summary.track_count} tracks "
                "and compiling every handoff locally."
            ),
        )
        planner = LocalDJPlanner(
            self.profile_store.list_profiles(ready_only=True),
            proficient_techniques=self.profile_store.proficient_techniques(),
        )
        try:
            plan = planner.build_playlist_plan(
                summary.track_ids,
                opening_track_id=opening_track_id,
                vibe=vibe,
                name=f"RekordBot playlist — {summary.name}",
            )
        except Exception as exc:
            self._automation_active = False
            self._fail("Playlist mapping failed", str(exc))
            raise
        return await self._launch_plan(plan)

    async def _launch_plan(self, plan: AutonomousSetPlan) -> dict[str, Any]:
        self.plan = plan
        self._register_recovery_planner()
        queue = self._queue_roles(plan)
        self.status_store.publish(
            phase=RuntimePhase.PREFLIGHT,
            headline="Preflighting Rekordbox",
            detail=(
                f"Verifying {len(queue)} tracks, exact load routes, cues, "
                "audio output, and recovery controls."
            ),
            queue=queue,
            current=queue[0],
            staged=queue[1],
            next=queue[2] if len(queue) > 2 else TrackRole(),
        )
        try:
            connection = self.adapter.connect()
            if connection.get("connected") is not True:
                raise RuntimeError("MIDI output did not connect")
            # Manual arming is intentionally capped at 30 minutes.  Once the
            # explicitly requested set starts, Performer grants its separate
            # bounded set authorization from the analyzed track durations.
            self.adapter.arm(30 * 60)
            preflight = await self.adapter.preflight(plan)
            if preflight.get("ready") is not True:
                raise RuntimeError(
                    "; ".join(preflight.get("errors", []))
                    or "full-set preflight failed"
                )
            started = await self.adapter.start(plan)
            if started.get("ready") is not True:
                raise RuntimeError(
                    "; ".join(started.get("errors", [])) or "set launch failed"
                )
        except Exception as exc:
            self._automation_active = False
            self._fail("Rekordbox launch failed", str(exc))
            raise
        opening = queue[0]
        self.status_store.publish(
            phase=RuntimePhase.PLAYING,
            headline=f"Playing {opening.title}",
            detail=(
                f"Started at the opening track's native {opening.bpm:.1f} BPM. "
                "The first handoff is owned by the local scheduler."
            ),
            severity="success",
            current=opening,
            staged=queue[1],
            next=queue[2] if len(queue) > 2 else TrackRole(),
            queue=queue,
            bpm=opening.bpm,
            health=RuntimeHealth(
                midi="ok",
                rekordbox="ok",
                audio_route="flx4",
                deck_state="fresh",
                rescue="armed",
            ),
        )
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
        self.monitor_task = asyncio.create_task(self._monitor())
        return {"ready": True, "plan": plan.model_dump(), "launch": started}

    async def steer_set(
        self,
        *,
        target_query: str = "",
        vibe: str = "maintain",
        transition_count: int = 3,
    ) -> dict[str, Any]:
        """Queue a new direction after the transition already owned locally."""
        if not 2 <= transition_count <= 6:
            raise ValueError("steering transition count must be between 2 and 6")
        runner = self.adapter.runner_status()
        if runner.get("active") is not True or not runner.get("active_option_id"):
            return await self.continue_set(
                target_query=target_query,
                vibe=vibe,
                transition_count=transition_count,
            )
        active = self._option(str(runner["active_option_id"]))
        if active is None:
            raise RuntimeError("the armed transition is not present in the local plan")
        anchor_id = active.incoming.track_id
        anchor_deck = active.card.incoming_deck
        target = self.resolve_track(target_query) if target_query.strip() else None
        excluded = list(runner.get("played_track_ids") or [])
        future, actual_transitions = self._build_future_route(
            anchor_id=anchor_id,
            anchor_deck=anchor_deck,
            target=target,
            vibe=vibe,
            transition_count=transition_count,
            excluded=excluded,
        )
        result = self.adapter.queue_steering(future)
        if result.get("queued") is not True:
            raise RuntimeError(
                "; ".join(result.get("errors", [])) or "steering request was rejected"
            )
        self.plan = AutonomousSetPlan.model_validate(result["projected_plan"])
        destination = self._role(
            self._queue_roles(future)[-1].track_id,
            state="destination",
        )
        self.status_store.publish(
            phase=RuntimePhase.SELECTING,
            headline=f"Steering queued toward {destination.title}",
            detail=(
                "The transition already armed will remain unchanged. The new "
                f"{vibe} route begins after that track and arrives in "
                f"{actual_transitions} transitions."
            ),
            severity="success",
            current=self._role(
                runner.get("current_track_id"),
                deck=runner.get("current_deck"),
                state="playing",
            ),
            staged=self._role(anchor_id, deck=anchor_deck, state="armed"),
            next=destination,
            queue=self._queue_roles(self.plan),
        )
        return {
            "queued": True,
            "anchor_track_id": anchor_id,
            "destination_track_id": destination.track_id,
            "projected_plan": self.plan.model_dump(),
        }

    async def continue_set(
        self,
        *,
        target_query: str = "",
        vibe: str = "maintain",
        transition_count: int = 4,
        endless: bool | None = None,
    ) -> dict[str, Any]:
        """Attach a new plan to a final track that is still playing."""
        if not 2 <= transition_count <= 8:
            raise ValueError("continuation transition count must be between 2 and 8")
        runner = self.adapter.runner_status()
        anchor_id = runner.get("current_track_id")
        anchor_deck = runner.get("current_deck")
        if not anchor_id or anchor_deck not in {1, 2}:
            snapshot = self.status_store.read()
            anchor_id = snapshot.current.track_id
            anchor_deck = snapshot.current.deck
        if not anchor_id or anchor_deck not in {1, 2}:
            raise RuntimeError("no playing prepared track is available to continue")
        target = self.resolve_track(target_query) if target_query.strip() else None
        excluded = list(runner.get("played_track_ids") or [])
        future, actual_transitions = self._build_future_route(
            anchor_id=str(anchor_id),
            anchor_deck=int(anchor_deck),
            target=target,
            vibe=vibe,
            transition_count=transition_count,
            excluded=excluded,
        )
        self._automation_active = True
        self._register_recovery_planner()
        if endless is not None:
            self._endless = endless
        self._endless_vibe = vibe
        self.status_store.publish(
            phase=RuntimePhase.PREFLIGHT,
            headline="Extending the live set",
            detail=(
                f"Attaching {actual_transitions} transitions to the playing final "
                "track without restarting it."
            ),
        )
        try:
            connection = self.adapter.connect()
            if connection.get("connected") is not True:
                raise RuntimeError("MIDI output did not connect")
            self.adapter.arm(30 * 60)
            started = await self.adapter.continue_set(future)
            if started.get("ready") is not True:
                raise RuntimeError(
                    "; ".join(started.get("errors", []))
                    or "continuation launch failed"
                )
        except Exception as exc:
            self._automation_active = False
            self._fail("Set continuation failed", str(exc))
            raise
        self.plan = future
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
        self.monitor_task = asyncio.create_task(self._monitor())
        return {
            "ready": True,
            "continued": True,
            "actual_transitions": actual_transitions,
            "plan": future.model_dump(),
            "launch": started,
        }

    def _register_recovery_planner(self) -> None:
        register = getattr(self.adapter, "register_recovery_planner", None)
        if callable(register):
            register(self._build_replacement_route)

    async def _build_replacement_route(
        self,
        runner: dict[str, Any],
    ) -> AutonomousSetPlan | None:
        """Replace an exhausted handoff without interrupting the audible deck."""
        anchor_id = str(runner.get("current_track_id") or "")
        anchor_deck = runner.get("current_deck")
        if not anchor_id or anchor_deck not in {1, 2}:
            raise RuntimeError("replacement planner has no authoritative live deck")
        played = list(runner.get("played_track_ids") or [])
        rejected = list(runner.get("exhausted_incoming_track_ids") or [])
        remaining = max(
            1,
            int(runner.get("target_track_count") or len(played) + 1)
            - len(played),
        )
        excluded = list(dict.fromkeys([*played, *rejected]))
        rejected_titles = [
            self.profile_store.get(track_id).title
            for track_id in rejected
            if track_id in {item.track_id for item in self.profile_store.list_profiles()}
        ]
        self.status_store.publish(
            phase=RuntimePhase.SELECTING,
            headline="Selecting a replacement transition",
            detail=(
                "Keeping the current deck audible while replacing "
                + (", ".join(rejected_titles) if rejected_titles else "the failed route")
                + "."
            ),
            severity="warning",
            current=self._role(anchor_id, deck=int(anchor_deck), state="playing"),
        )

        restricted = self._candidate_track_ids
        try:
            future, _ = self._build_future_route(
                anchor_id=anchor_id,
                anchor_deck=int(anchor_deck),
                target=None,
                vibe=self._endless_vibe,
                transition_count=remaining,
                excluded=excluded,
            )
        except ValueError:
            if restricted is None:
                raise
            # Playlist/set constraints are preferred, but uninterrupted audio
            # has priority once every in-scope route is exhausted. Broaden to
            # the complete prepared library only for this emergency branch.
            self._candidate_track_ids = None
            try:
                future, _ = self._build_future_route(
                    anchor_id=anchor_id,
                    anchor_deck=int(anchor_deck),
                    target=None,
                    vibe=self._endless_vibe,
                    transition_count=remaining,
                    excluded=excluded,
                )
            finally:
                self._candidate_track_ids = restricted
        self.plan = future
        return future

    def _build_future_route(
        self,
        *,
        anchor_id: str,
        anchor_deck: int,
        target: TrackProfile | None,
        vibe: str,
        transition_count: int,
        excluded: list[str],
    ) -> tuple[AutonomousSetPlan, int]:
        planner = LocalDJPlanner(
            self.profile_store.list_profiles(ready_only=True),
            proficient_techniques=self.profile_store.proficient_techniques(),
        )
        allowed = set(self._candidate_track_ids or planner.profiles.keys())
        allowed.add(anchor_id)
        if target is not None:
            allowed.add(target.track_id)
        recent_excluded = list(dict.fromkeys(excluded))
        available = allowed - set(recent_excluded) - {anchor_id}
        if len(available) < transition_count:
            # Endless playback eventually consumes the whole candidate pool.
            # Retain only as much repeat-cooldown as still leaves enough unique
            # tracks to compile the requested horizon; never dead-end merely
            # because every song has appeared earlier in the same set.
            cooldown = max(0, len(allowed - {anchor_id}) - transition_count)
            recent_excluded = recent_excluded[-cooldown:] if cooldown else []
        if target is not None:
            recent_excluded = [
                track_id for track_id in recent_excluded if track_id != target.track_id
            ]
        last_error: Exception | None = None
        for transitions in range(transition_count, min(12, transition_count + 4) + 1):
            try:
                return (
                    planner.build_plan(
                        DJBrief(
                            start_track_id=anchor_id,
                            target_track_count=transitions + 1,
                            target_track_id=(
                                target.track_id if target is not None else None
                            ),
                            target_bpm=target.bpm if target is not None else None,
                            vibe=vibe,
                            excluded_track_ids=recent_excluded,
                            allowed_track_ids=sorted(allowed),
                            name=(
                                f"Steer to {target.title}"
                                if target is not None
                                else f"Continue {vibe}"
                            ),
                        ),
                        opening_deck=anchor_deck,
                    ),
                    transitions,
                )
            except ValueError as exc:
                last_error = exc
        raise ValueError(
            f"no safe route could reach the requested direction: {last_error}"
        )

    async def _monitor(self) -> None:
        while True:
            try:
                runner = self.adapter.runner_status()
                control = self.adapter.control_status()
                if self._endless and runner.get("active") is True:
                    await self._extend_endless_horizon(runner)
                    runner = self.adapter.runner_status()
                self._publish_runner_state(runner, control)
                if runner.get("status") in {"completed", "failed", "stopped"}:
                    self._automation_active = False
                    return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - adapter boundary
                self.status_store.publish(
                    phase=RuntimePhase.RECOVERING,
                    headline="Status observation degraded",
                    detail=str(exc),
                    severity="warning",
                )
            await asyncio.sleep(self.monitor_seconds)

    async def _extend_endless_horizon(self, runner: dict[str, Any]) -> None:
        """Keep at least three future decisions compiled in endless mode."""
        if self._endless_extension_pending or runner.get("steering_queued"):
            return
        active_id = runner.get("active_option_id")
        if not active_id:
            return
        remaining = int(runner.get("target_track_count") or 0) - len(
            runner.get("played_track_ids") or []
        )
        if remaining > 3:
            return
        active = self._option(str(active_id))
        if active is None:
            return
        self._endless_extension_pending = True
        try:
            future, _ = self._build_future_route(
                anchor_id=active.incoming.track_id,
                anchor_deck=active.card.incoming_deck,
                target=None,
                vibe=self._endless_vibe,
                transition_count=5,
                excluded=list(runner.get("played_track_ids") or []),
            )
            result = self.adapter.queue_steering(future)
            if result.get("queued") is not True:
                raise RuntimeError(
                    "; ".join(result.get("errors", []))
                    or "endless extension was rejected"
                )
            self.plan = AutonomousSetPlan.model_validate(result["projected_plan"])
            self.status_store.publish(
                phase=RuntimePhase.SELECTING,
                headline="Endless set extended",
                detail="Five more transitions are compiled and locally owned.",
                severity="success",
                queue=self._queue_roles(self.plan),
            )
        finally:
            self._endless_extension_pending = False

    async def monitor_manual_playback(self) -> None:
        """Reflect manual Rekordbox transport while automation is inactive."""
        if self._manual_monitor_can_publish():
            self._last_manual_signature = ("warming",)
            self.status_store.publish(
                phase=RuntimePhase.IDLE,
                headline="Connecting to Rekordbox monitor",
                detail=(
                    "Reading deck clocks without screenshots or MIDI control. "
                    "The first observation can take up to a minute."
                ),
                current=TrackRole(),
                staged=TrackRole(),
                bpm=None,
                bar=None,
                beat=None,
                health=RuntimeHealth(
                    midi="down",
                    rekordbox="degraded",
                    deck_state="unknown",
                    rescue="unavailable",
                ),
            )
        while True:
            try:
                status = await asyncio.to_thread(self.adapter.transport_status)
                if self._manual_monitor_can_publish():
                    self._publish_manual_transport(status)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - UI adapter boundary
                if self._manual_monitor_can_publish():
                    signature = ("error", str(exc))
                    if signature != self._last_manual_signature:
                        self._last_manual_signature = signature
                        self.status_store.publish(
                            phase=RuntimePhase.IDLE,
                            headline="Connecting to Rekordbox monitor",
                            detail=(
                                "The passive deck reader is retrying; MIDI remains "
                                f"disconnected. {exc}"
                            ),
                            severity="warning",
                            current=TrackRole(),
                            staged=TrackRole(),
                            bpm=None,
                            bar=None,
                            beat=None,
                            health=RuntimeHealth(
                                midi="down",
                                rekordbox="degraded",
                                deck_state="unknown",
                                rescue="unavailable",
                            ),
                        )
            await asyncio.sleep(max(0.5, self.monitor_seconds))

    def _manual_monitor_can_publish(self) -> bool:
        """Keep a launch failure visible until the next explicit set attempt."""
        return (
            not self._automation_active
            and self.status_store.read().phase != RuntimePhase.FAILED
        )

    def _publish_manual_transport(self, status: dict[str, Any]) -> None:
        now = time.monotonic()
        records = [
            record
            for record in status.get("decks", [])
            if int(record.get("deck", 0)) in {1, 2}
        ]
        for record in records:
            deck = int(record["deck"])
            title = str(record.get("title") or "").strip()
            elapsed_value = record.get("elapsed_seconds")
            elapsed = float(elapsed_value) if elapsed_value is not None else None
            previous = self._manual_previous.get(deck)
            if (
                title
                and previous is not None
                and self._search_key(previous[0]) == self._search_key(title)
                and elapsed is not None
                and previous[1] is not None
                and abs(elapsed - previous[1]) >= 0.5
            ):
                self._manual_playing_until[deck] = now + 2.5
            elif previous is not None and self._search_key(
                previous[0]
            ) != self._search_key(title):
                self._manual_playing_until.pop(deck, None)
            self._manual_previous[deck] = (title, elapsed)

        moving = [
            record
            for record in records
            if str(record.get("title") or "").strip()
            and self._manual_playing_until.get(int(record["deck"]), 0) > now
        ]
        moving_decks = {int(record["deck"]) for record in moving}
        if self._manual_current_deck not in moving_decks:
            self._manual_current_deck = int(moving[0]["deck"]) if moving else None

        if self._manual_current_deck is None:
            loaded = [
                f"Deck {record['deck']}: {record['title']}"
                for record in records
                if str(record.get("title") or "").strip()
            ]
            signature = ("stopped", tuple(loaded), status.get("mode"))
            if signature == self._last_manual_signature:
                return
            self._last_manual_signature = signature
            self.status_store.publish(
                phase=RuntimePhase.IDLE,
                headline="Ready",
                detail=(
                    "; ".join(loaded) + " loaded; transport is stopped."
                    if loaded
                    else "Choose an opening track and start a set."
                ),
                current=TrackRole(),
                staged=TrackRole(),
                bpm=None,
                bar=None,
                beat=None,
                health=RuntimeHealth(
                    midi="down",
                    rekordbox="ok",
                    deck_state="fresh",
                    rescue="unavailable",
                ),
            )
            return

        record = next(
            item for item in moving if int(item["deck"]) == self._manual_current_deck
        )
        role = self._manual_role(record)
        other = [int(item["deck"]) for item in moving if item is not record]
        signature = (
            "playing",
            role.deck,
            role.title,
            round(float(record.get("bpm") or 0), 2),
            tuple(other),
        )
        if signature == self._last_manual_signature:
            return
        self._last_manual_signature = signature
        self.status_store.publish(
            phase=RuntimePhase.IDLE,
            headline=f"Manual playback detected on Deck {role.deck}",
            detail=(
                "Both decks are moving; automation is not armed."
                if other
                else "The app is observing Rekordbox; automation is not armed."
            ),
            severity="success",
            current=role,
            staged=TrackRole(),
            bpm=record.get("bpm"),
            bar=None,
            beat=None,
            health=RuntimeHealth(
                midi="down",
                rekordbox="ok",
                deck_state="fresh",
                rescue="unavailable",
            ),
        )

    def _manual_role(self, record: dict[str, Any]) -> TrackRole:
        title = str(record.get("title") or "").strip()
        artist = str(record.get("artist") or "").strip()
        matches = [
            profile
            for profile in self.profile_store.list_profiles()
            if self._search_key(profile.title) == self._search_key(title)
        ]
        if artist:
            artist_matches = [
                profile
                for profile in matches
                if self._search_key(profile.artist) == self._search_key(artist)
            ]
            if artist_matches:
                matches = artist_matches
        if len(matches) == 1:
            return self._role(
                matches[0].track_id,
                deck=int(record["deck"]),
                state="manual",
            )
        return TrackRole(
            title=title,
            artist=artist or None,
            deck=int(record["deck"]),
            bpm=record.get("bpm"),
            state="manual",
        )

    def _publish_runner_state(
        self,
        runner: dict[str, Any],
        control: dict[str, Any],
    ) -> None:
        if self.plan is None:
            return
        now = time.time()
        start_at = runner.get("transition_start_at")
        critical_at = runner.get("transition_critical_at")
        current_id = runner.get("current_track_id")
        current = self._role(
            current_id, deck=runner.get("current_deck"), state="playing"
        )
        active_option = self._option(runner.get("active_option_id"))
        staged_option = self._option(
            runner.get("staged_option_id") or runner.get("active_option_id")
        )
        selected_option = staged_option or next(
            (
                item
                for item in sorted(
                    self.plan.transitions, key=lambda value: value.priority
                )
                if item.card.outgoing_track_id == current_id
            ),
            None,
        )
        staged = (
            self._role(
                selected_option.incoming.track_id,
                deck=selected_option.card.incoming_deck,
                state="staged" if staged_option is not None else "selected",
            )
            if selected_option is not None
            else TrackRole()
        )
        following = self._following(staged.track_id)
        action_seconds = (
            max(0.0, float(start_at) - now) if start_at is not None else None
        )
        live_bpm, bar, beat, freshness = self._live_clock(
            control, runner.get("current_deck")
        )
        action_bars = (
            action_seconds * live_bpm / 240.0
            if action_seconds is not None and live_bpm is not None
            else None
        )
        critical_seconds = (
            max(0.0, float(critical_at) - now) if critical_at is not None else None
        )
        critical_bars = (
            critical_seconds * live_bpm / 240.0
            if critical_seconds is not None and live_bpm is not None
            else None
        )
        status = runner.get("status")
        deadline = runner.get("deadline_phase")
        if status == "completed":
            phase = RuntimePhase.COMPLETE
            headline = "Set complete"
            detail = "All planned tracks and deck retirements completed."
            severity = "success"
        elif status == "failed":
            phase = RuntimePhase.FAILED
            headline = "Set stopped safely after a runtime failure"
            detail = (runner.get("failures") or ["Unknown runtime failure"])[-1]
            severity = "error"
        elif runner.get("rescue_loop_active"):
            phase = RuntimePhase.HOLDING_LOOP
            headline = "Holding the prepared rescue loop"
            detail = (
                "The next route is being recovered while the current deck "
                "remains audible."
            )
            severity = "warning"
        elif status == "recovering":
            phase = RuntimePhase.RECOVERING
            headline = "Recovering the next transition"
            detail = (
                runner.get("last_recovery_error")
                or "The current deck remains audible while a replacement route is selected."
            )
            severity = "warning"
        elif critical_at is not None and now >= float(critical_at):
            phase = RuntimePhase.RETIRING
            headline = f"Retiring {current.title}"
            detail = (
                f"{staged.title} owns the bass and energy; clearing the outgoing deck."
            )
            severity = "info"
        elif start_at is not None and now >= float(start_at):
            phase = RuntimePhase.ESTABLISHING
            headline = f"Establishing {staged.title}"
            detail = (
                "Incoming channel is rising while the outgoing channel remains full."
            )
            severity = "info"
        elif active_option is not None:
            phase = RuntimePhase.WAITING
            headline = f"Preparing to bring in {staged.title}"
            detail = self._countdown_detail(action_bars, action_seconds)
            severity = "info"
        elif deadline in {"reserve", "rescue"}:
            phase = RuntimePhase.RECOVERING
            headline = "Protecting transition runway"
            detail = "The prepared loop/fallback deadline is active."
            severity = "warning"
        else:
            phase = RuntimePhase.TEMPO_RAMP if staged_option else RuntimePhase.PLAYING
            headline = (
                f"Adjusting tempo on {current.title}"
                if phase == RuntimePhase.TEMPO_RAMP
                else f"Playing {current.title}"
            )
            detail = (
                "The following track is already staged while the tempo moves gradually."
                if phase == RuntimePhase.TEMPO_RAMP
                else "Maintaining the current track and preparing the next decision."
            )
            severity = "info"
        failures = list(runner.get("failures") or [])
        signature = (
            phase,
            current.track_id,
            staged.track_id,
            runner.get("active_job_id"),
            runner.get("rescue_loop_active"),
            failures[-1] if failures else None,
            round(action_bars or -1, 1),
        )
        if signature == self._last_signature:
            return
        self._last_signature = signature
        self.status_store.publish(
            phase=phase,
            headline=headline,
            detail=detail,
            severity=severity,
            current=current,
            staged=staged,
            next=following,
            queue=self._queue_roles(self.plan),
            bpm=live_bpm,
            bar=bar,
            beat=beat,
            action_in_bars=action_bars,
            action_in_seconds=action_seconds,
            transition_name=active_option.card.name if active_option else None,
            transition_family=(
                active_option.card.transition_family if active_option else None
            ),
            transition_technique=(
                active_option.technique or active_option.card.transition_family
                if active_option
                else None
            ),
            transition_reason=(active_option.reason if active_option else None),
            transition_alternatives=(
                active_option.alternatives if active_option else []
            ),
            critical_in_bars=critical_bars,
            failures=failures[-10:],
            health=RuntimeHealth(
                midi="ok" if control.get("connected") else "down",
                rekordbox="ok" if freshness != "stale" else "degraded",
                audio_route="flx4",
                deck_state=freshness,
                rescue="active" if runner.get("rescue_loop_active") else "armed",
            ),
        )

    async def hold_current(self) -> dict[str, Any]:
        runner = self.adapter.runner_status()
        deck = int(runner["current_deck"])
        result = await self.adapter.hold_loop(deck, 4)
        if result.get("verified") is not True:
            raise RuntimeError("prepared hold loop could not be verified")
        self.adapter.stop_automation()
        self.status_store.publish(
            phase=RuntimePhase.HOLDING_LOOP,
            headline="Holding the current track",
            detail="Autonomous transitions are paused and the verified loop is active.",
            severity="warning",
        )
        return result

    def stop_after_current(self) -> dict[str, Any]:
        self._endless = False
        result = self.adapter.stop_automation()
        self.status_store.publish(
            phase=RuntimePhase.STOPPING,
            headline="Stopping after the current track",
            detail="No additional track will be loaded or launched.",
        )
        return result

    def emergency_stop(self) -> dict[str, Any]:
        result = self.adapter.emergency_stop()
        self.status_store.publish(
            phase=RuntimePhase.FAILED,
            headline="Emergency stop",
            detail="Pending automation was cancelled and both decks were silenced.",
            severity="error",
        )
        return result

    def _option(self, option_id: str | None):
        if self.plan is None or option_id is None:
            return None
        return next(
            (item for item in self.plan.transitions if item.id == option_id), None
        )

    def _following(self, track_id: str | None) -> TrackRole:
        if self.plan is None or track_id is None:
            return TrackRole()
        option = next(
            (
                item
                for item in sorted(
                    self.plan.transitions, key=lambda value: value.priority
                )
                if item.card.outgoing_track_id == track_id
            ),
            None,
        )
        return (
            self._role(option.incoming.track_id, state="selected")
            if option
            else TrackRole()
        )

    def _role(
        self, track_id: str | None, *, deck: int | None = None, state: str = "unknown"
    ) -> TrackRole:
        if track_id is None:
            return TrackRole()
        profile = self.profile_store.get(track_id)
        return TrackRole(
            track_id=track_id,
            title=profile.title,
            artist=profile.artist,
            deck=deck,
            bpm=profile.bpm,
            state=state,
        )

    def _queue_roles(self, plan: AutonomousSetPlan) -> list[TrackRole]:
        result = [self._role(plan.opening.track_id, deck=1, state="current")]
        current = plan.opening.track_id
        deck = 2
        for _ in range(plan.target_track_count - 1):
            option = next(
                item
                for item in sorted(plan.transitions, key=lambda value: value.priority)
                if item.card.outgoing_track_id == current
            )
            result.append(
                self._role(option.incoming.track_id, deck=deck, state="planned")
            )
            current = option.incoming.track_id
            deck = 1 if deck == 2 else 2
        return result

    @staticmethod
    def _live_clock(
        control: dict[str, Any],
        deck: int | None,
    ) -> tuple[float | None, int | None, int | None, str]:
        live = (control.get("live_state") or {}).get("decks") or []
        record = next((item for item in live if item.get("deck") == deck), None)
        if record is None:
            return None, None, None, "unknown"
        age = float(record.get("observation_age_ms", 1e9))
        freshness = (
            "fresh"
            if age <= 1000
            else "projected"
            if record.get("playing")
            else "stale"
        )
        return record.get("bpm"), record.get("bar"), record.get("beat"), freshness

    @staticmethod
    def _countdown_detail(bars: float | None, seconds: float | None) -> str:
        if bars is None or seconds is None:
            return "The transition is armed on a verified phrase boundary."
        minutes, remainder = divmod(max(0, round(seconds)), 60)
        return f"Launch in {bars:.1f} bars · approximately {minutes}:{remainder:02d}."

    def _fail(self, headline: str, detail: str) -> None:
        self.status_store.publish(
            phase=RuntimePhase.FAILED,
            headline=headline,
            detail=detail,
            severity="error",
        )
