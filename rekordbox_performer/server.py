"""FastMCP server for live Rekordbox control through virtual MIDI."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from .audio import RehearsalCapture, analyze_rehearsal_audio
from .engine import MidiEngine
from .intelligence import (
    DeckObservation,
    LiveState,
    PlaylistTrackMetadata,
    ProfileStore,
    RekordboxAnalysisImport,
    RehearsalReview,
    SetPlan,
    TrackProfile,
    TransitionCard,
    audit_set_plan,
    camelot_compatibility,
    compile_transition_card,
)
from .observer import SharedDeckObserver
from .protocol import (
    ALL_ACTIONS,
    CONTINUOUS_ACTIONS,
    TRIGGER_ACTIONS,
    mapping_manifest,
)
from .rekordbox_ui import RekordboxUIAdapter, normalize_title
from .scheduler import TransitionScheduler


class TransitionEvent(BaseModel):
    at_ms: int = Field(ge=0)
    action: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class TransitionCandidate(BaseModel):
    track_id: str
    title: str
    artist: str = ""
    bpm: float = Field(gt=0)
    key: str | None = None


mcp = FastMCP("Rekordbox Performer")
engine = MidiEngine()
scheduler = TransitionScheduler(engine)
live_state = LiveState()
profile_store = ProfileStore()
rehearsal_capture = RehearsalCapture(profile_store.data_dir / "recordings")
rekordbox_ui = RekordboxUIAdapter()
deck_observer = SharedDeckObserver(rekordbox_ui, profile_store.data_dir)
verified_hot_cues: dict[tuple[int, str, int], dict[str, Any]] = {}
VERIFIED_CUE_TTL_SECONDS = 15 * 60


def _deck_record(status: dict[str, Any], deck: int) -> dict[str, Any]:
    for record in status.get("decks", []):
        if record.get("deck") == deck:
            return record
    raise RuntimeError(f"Rekordbox status omitted deck {deck}")


def _transport_changed(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    before = first.get("elapsed_seconds")
    after = second.get("elapsed_seconds")
    return before is not None and after is not None and before != after


def _hot_cue_position_window(
    observed_seconds: int,
    snapshot_started_after_seconds: float,
    snapshot_finished_after_seconds: float,
) -> tuple[float, float]:
    """Bound cue position despite slow UIA reads and whole-second clocks.

    The displayed second is sampled at an unknown instant during the UIA
    snapshot and represents a one-second bucket. Subtracting the snapshot's
    full duration from that coarse value can manufacture a negative cue
    position. Return the complete possible position interval instead.
    """
    lower = observed_seconds - snapshot_finished_after_seconds
    upper = observed_seconds + 1.0 - snapshot_started_after_seconds
    return lower, upper


def _card_hot_cue_launch(
    card: TransitionCard,
) -> tuple[int, str, int] | None:
    launches = [
        event
        for event in card.events
        if event.bar_offset == 0
        and event.beat_offset == 0
        and event.action == "hot_cue"
        and event.parameters.get("deck") == card.incoming_deck
    ]
    if not launches:
        return None
    if len(launches) != 1:
        raise RuntimeError("Transition card has multiple incoming Hot Cues")
    return (
        card.incoming_deck,
        card.incoming_track_id,
        int(launches[0].parameters["cue"]),
    )


def _require_verified_hot_cue(
    card: TransitionCard,
) -> list[dict[str, Any]] | None:
    keys = []
    for event in card.events:
        if not (
            event.bar_offset == 0
            and event.beat_offset == 0
            and event.action == "hot_cue"
        ):
            continue
        deck = int(event.parameters["deck"])
        track_id = (
            card.outgoing_track_id
            if deck == card.outgoing_deck
            else card.incoming_track_id
        )
        keys.append((deck, track_id, int(event.parameters["cue"])))
    if not keys:
        return None
    results = []
    for key in keys:
        record = verified_hot_cues.get(key)
        if record is None:
            raise RuntimeError(
                f"Hot Cue {key[2]} on deck {key[0]} has not been verified "
                "in this Rekordbox session. Call verify_hot_cue before "
                "committing the transition."
            )
        age = time.monotonic() - float(record["verified_monotonic"])
        if age > VERIFIED_CUE_TTL_SECONDS:
            verified_hot_cues.pop(key, None)
            raise RuntimeError(
                f"Hot Cue {key[2]} on deck {key[0]} expired; verify it "
                "again before committing the transition."
            )
        results.append({**record, "age_seconds": round(age, 3)})
    return results


def _require_staged_incoming_position(
    incoming: dict[str, Any],
    *,
    deck: int,
    track_id: str,
    hot_cue: int | None,
) -> None:
    elapsed = incoming.get("elapsed_seconds")
    if elapsed is None:
        raise RuntimeError("Incoming deck position is unavailable")
    if hot_cue is None:
        if elapsed > 1:
            raise RuntimeError("Incoming deck is not staged at its file start")
        return
    key = (deck, track_id, hot_cue)
    record = verified_hot_cues.get(key)
    if record is None:
        raise RuntimeError(
            f"Hot Cue {hot_cue} on deck {deck} has not been verified in this "
            "Rekordbox session"
        )
    age = time.monotonic() - float(record["verified_monotonic"])
    if age > VERIFIED_CUE_TTL_SECONDS:
        verified_hot_cues.pop(key, None)
        raise RuntimeError(f"Hot Cue {hot_cue} on deck {deck} expired")


def _verify_transition_postconditions(
    card: TransitionCard,
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    outgoing_first = _deck_record(first, card.outgoing_deck)
    outgoing_second = _deck_record(second, card.outgoing_deck)
    incoming_first = _deck_record(first, card.incoming_deck)
    incoming_second = _deck_record(second, card.incoming_deck)
    errors: list[str] = []
    if normalize_title(outgoing_second.get("title", "")) != normalize_title(
        profile_store.get(card.outgoing_track_id).title
    ):
        errors.append("outgoing deck title does not match the card")
    if normalize_title(incoming_second.get("title", "")) != normalize_title(
        profile_store.get(card.incoming_track_id).title
    ):
        errors.append("incoming deck title does not match the card")
    if _transport_changed(outgoing_first, outgoing_second):
        errors.append("outgoing deck is still playing after retirement")
    if not _transport_changed(incoming_first, incoming_second):
        errors.append("incoming deck did not start playing")
    return {
        "verified": not errors,
        "errors": errors,
        "outgoing": {
            "first": outgoing_first,
            "second": outgoing_second,
        },
        "incoming": {
            "first": incoming_first,
            "second": incoming_second,
        },
    }


def _card_completion_verifier(
    card: TransitionCard,
    launch_clock: dict[str, float] | None = None,
) -> dict[str, Any]:
    try:
        with deck_observer.exclusive_adapter():
            first = rekordbox_ui.status()
            time.sleep(1.1)
            second = rekordbox_ui.status()
        result = _verify_transition_postconditions(card, first, second)
        launched_monotonic = (launch_clock or {}).get("incoming_monotonic")
        if result["verified"] and launched_monotonic is not None:
            profile = profile_store.get(card.incoming_track_id)
            launch = next(
                event
                for event in card.events
                if event.bar_offset == 0
                and event.beat_offset == 0
                and event.action == "hot_cue"
                and event.parameters.get("deck") == card.incoming_deck
            )
            cue = int(launch.parameters["cue"])
            landmark = next(
                item
                for item in profile.landmarks
                if item.cue == cue
                and item.kind in {"mix_in", "phrase_start"}
                and item.confidence in {"verified", "high"}
            )
            elapsed_beats = (
                time.monotonic() - launched_monotonic
            ) * profile.bpm / 60.0
            start_track_beat = (
                (landmark.bar - 1) * profile.time_signature
                + (landmark.beat or 1)
            )
            total = start_track_beat - 1 + elapsed_beats
            within_bar = total % profile.time_signature
            observation = DeckObservation(
                deck=card.incoming_deck,
                track_id=card.incoming_track_id,
                title=profile.title,
                bpm=profile.bpm,
                playing=True,
                bar=int(total // profile.time_signature) + 1,
                beat=int(within_bar) + 1,
                track_beat=int(total) + 1,
                beat_phase=within_bar - int(within_bar),
                sync_enabled=result["incoming"]["second"].get(
                    "beat_sync_enabled"
                ),
                quantize_enabled=result["incoming"]["second"].get(
                    "quantize_enabled"
                ),
                source="native",
                confidence="high",
            )
            result["promoted_live_clock"] = live_state.update(observation)
        return result
    except Exception as exc:
        return {
            "verified": False,
            "errors": [f"post-transition observation failed: {exc}"],
        }


@mcp.tool(annotations={"readOnlyHint": True})
def list_midi_outputs() -> dict[str, Any]:
    """List MIDI outputs visible to the server."""
    return {"outputs": engine.list_output_ports()}


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def connect_midi(port_name: str | None = None) -> dict[str, Any]:
    """Connect to the loopMIDI output. Does not send control messages."""
    return engine.connect(port_name)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def disconnect_midi() -> dict[str, Any]:
    """Disconnect MIDI and disarm live control."""
    deck_observer.deactivate()
    engine.disconnect()
    return engine.status()


@mcp.tool(annotations={"readOnlyHint": True})
def control_status() -> dict[str, Any]:
    """Return connection, arming, and scheduler status."""
    return {
        **engine.status(),
        "active_jobs": [
            job.public()
            for job in scheduler.jobs.values()
            if job.status in {"scheduled", "running"}
        ],
        "live_state": live_state.snapshot(),
        "scheduler_metrics": scheduler.metrics(),
    }


@mcp.tool(annotations={"readOnlyHint": True})
def scheduler_metrics() -> dict[str, Any]:
    """Return aggregate transition timing and deadline metrics."""
    return scheduler.metrics()


@mcp.tool(annotations={"readOnlyHint": True})
def rank_transition_candidates(
    outgoing_bpm: float,
    outgoing_key: str,
    candidates: list[TransitionCandidate],
    max_bpm_delta: float = 4.0,
    allow_incompatible: bool = False,
) -> dict[str, Any]:
    """
    Rank candidates by Camelot compatibility and tempo proximity.

    Vocal density is intentionally not considered. Incompatible harmonic moves
    are excluded by default and require an explicit override.
    """
    if outgoing_bpm <= 0:
        raise ValueError("outgoing_bpm must be positive")
    if not 0 <= max_bpm_delta <= 20:
        raise ValueError("max_bpm_delta must be between 0 and 20")
    ranked = []
    excluded = []
    relationship_score = {
        "same_key": 4,
        "relative_major_minor": 3,
        "wheel_neighbor": 2,
        "incompatible": 0,
        "unknown": -1,
    }
    for candidate in candidates:
        harmonic = camelot_compatibility(outgoing_key, candidate.key)
        bpm_delta = abs(candidate.bpm - outgoing_bpm)
        record = {
            **candidate.model_dump(),
            "bpm_delta": round(bpm_delta, 3),
            "harmonic": harmonic,
        }
        reasons = []
        if bpm_delta > max_bpm_delta:
            reasons.append(
                f"BPM delta {bpm_delta:.2f} exceeds {max_bpm_delta:.2f}"
            )
        if not harmonic["verified"]:
            reasons.append("key could not be normalized to Camelot")
        elif not harmonic["compatible"] and not allow_incompatible:
            reasons.append(
                f"incompatible Camelot move "
                f"{harmonic['outgoing']} -> {harmonic['incoming']}"
            )
        if reasons:
            excluded.append({**record, "reasons": reasons})
            continue
        record["score"] = round(
            relationship_score[harmonic["relationship"]] * 10
            + max(0.0, max_bpm_delta - bpm_delta),
            3,
        )
        ranked.append(record)
    ranked.sort(key=lambda item: (-item["score"], item["bpm_delta"], item["title"]))
    return {
        "outgoing_bpm": outgoing_bpm,
        "outgoing_key": outgoing_key,
        "ranked": ranked,
        "excluded": excluded,
        "vocal_clash_scoring": "disabled",
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def observe_deck_state(observation: DeckObservation) -> dict[str, Any]:
    """Ingest a fresh native, MIDI, vision, or manual deck-clock observation."""
    if (
        observation.source == "midi"
        and observation.confidence in {"high", "verified"}
    ):
        raise ValueError(
            "This server has outbound MIDI only. A sent command is not a deck "
            "observation; use launch_verified_hot_cue or native feedback."
        )
    return live_state.update(observation)


@mcp.tool(annotations={"readOnlyHint": True})
def live_state_status() -> dict[str, Any]:
    """Return extrapolated deck clocks and observation freshness."""
    return live_state.snapshot()


@mcp.tool(annotations={"readOnlyHint": True})
def rekordbox_ui_status() -> dict[str, Any]:
    """Observe Performance mode, loaded titles, BPM, Sync, and Quantize."""
    return deck_observer.status()


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def upsert_track_profile(profile: TrackProfile) -> dict[str, Any]:
    """Store verified beatgrid, phrase, vocal, bass, and cue preparation data."""
    return profile_store.upsert(profile)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def ingest_rekordbox_analysis(
    analysis: RekordboxAnalysisImport,
    display_title: str | None = None,
    display_artist: str | None = None,
) -> dict[str, Any]:
    """
    Merge beat-grid and phrase truth returned by rekordbox-mcp.

    Spotify records with encrypted database metadata require display_title and
    display_artist from the visible playlist/service metadata.
    """
    return profile_store.ingest_analysis(
        analysis,
        display_title=display_title,
        display_artist=display_artist,
    )


@mcp.tool(annotations={"readOnlyHint": True})
def get_track_profile(track_id: str) -> dict[str, Any]:
    """Return one prepared track profile and its live-readiness audit."""
    profile = profile_store.get(track_id)
    return {"profile": profile.model_dump(), "readiness": profile.readiness()}


@mcp.tool(annotations={"readOnlyHint": True})
def audit_track_profiles(track_ids: list[str] | None = None) -> dict[str, Any]:
    """Audit whether tracks have enough verified structure for live automation."""
    return profile_store.audit(track_ids)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def preflight_playlist(
    tracks: list[PlaylistTrackMetadata],
) -> dict[str, Any]:
    """Seed a whole playlist for preparation without overwriting verified profiles."""
    return profile_store.seed_playlist(tracks)


@mcp.tool(annotations={"readOnlyHint": True})
def preview_set_plan(plan: SetPlan) -> dict[str, Any]:
    """Verify every profile and adjacent transition before the first track plays."""
    return audit_set_plan(plan, profile_store)


def _compile_card(card: TransitionCard) -> dict[str, Any]:
    outgoing = profile_store.get(card.outgoing_track_id)
    incoming = profile_store.get(card.incoming_track_id)
    return compile_transition_card(card, outgoing, incoming, live_state)


@mcp.tool(annotations={"readOnlyHint": True})
def preview_transition_card(card: TransitionCard) -> dict[str, Any]:
    """Compile a verified musical transition card against the fresh deck clock."""
    compiled = _compile_card(card)
    if not compiled["ready"]:
        return compiled
    preview = scheduler.preview(card.name, compiled["events"])
    return {**compiled, "preview": preview, "live_effect": False}


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def perform_transition_card(
    card: TransitionCard,
    commit: bool = False,
    execution_id: str | None = None,
) -> dict[str, Any]:
    """Run a verified, bar-quantized card. Dry-run unless commit is true."""
    compiled = _compile_card(card)
    if not compiled["ready"]:
        return compiled
    if not commit:
        return {
            **compiled,
            "preview": scheduler.preview(card.name, compiled["events"]),
            "live_effect": False,
            "message": "Dry run only. Set commit=true after review.",
        }
    if execution_id is None or not execution_id.strip():
        raise ValueError(
            "execution_id is required for committed transitions so retries "
            "cannot schedule duplicate performances"
        )
    cue_verification = _require_verified_hot_cue(card)
    launch_clock: dict[str, float] = {}

    def observe_dispatch(event: dict[str, Any], dispatched: float) -> None:
        if (
            event["action"] == "hot_cue"
            and event["parameters"].get("deck") == card.incoming_deck
        ):
            launch_clock["incoming_monotonic"] = dispatched

    return {
        **compiled,
        "cue_verification": cue_verification,
        "job": scheduler.start(
            card.name,
            compiled["events"],
            completion_verifier=lambda: _card_completion_verifier(
                card,
                launch_clock,
            ),
            event_observer=observe_dispatch,
            execution_id=execution_id,
        ),
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def record_rehearsal(review: RehearsalReview) -> dict[str, Any]:
    """Persist one graded transition rehearsal for library-specific learning."""
    return profile_store.record_rehearsal(review)


@mcp.tool(annotations={"readOnlyHint": True})
def rehearsal_summary(
    outgoing_track_id: str | None = None,
    incoming_track_id: str | None = None,
) -> dict[str, Any]:
    """Summarize rehearsal quality for a transition pair."""
    return profile_store.rehearsal_summary(
        outgoing_track_id,
        incoming_track_id,
    )


@mcp.tool(annotations={"readOnlyHint": True})
def rehearsal_capture_status() -> dict[str, Any]:
    """Return whether an explicitly started system-audio rehearsal is recording."""
    return rehearsal_capture.status()


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def start_rehearsal_capture(
    name: str,
    sample_rate: int = 48_000,
) -> dict[str, Any]:
    """Start explicit Windows system-audio capture for a rehearsal."""
    return rehearsal_capture.start(name, sample_rate)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def stop_rehearsal_capture() -> dict[str, Any]:
    """Stop the active rehearsal capture and save it as a WAV file."""
    return rehearsal_capture.stop()


@mcp.tool(annotations={"readOnlyHint": True})
def analyze_rehearsal(
    path: str,
    bpm: float,
    expected_downbeats_ms: list[float] | None = None,
) -> dict[str, Any]:
    """Analyze levels, onset timing, and expected-downbeat errors in a WAV."""
    return analyze_rehearsal_audio(path, bpm, expected_downbeats_ms)


@mcp.tool(annotations={"readOnlyHint": True})
def get_mapping_manifest() -> dict[str, Any]:
    """Return the complete stable MIDI mapping for Rekordbox MIDI Learn."""
    return {
        "trigger_actions": sorted(TRIGGER_ACTIONS),
        "continuous_actions": sorted(CONTINUOUS_ACTIONS),
        "assignments": mapping_manifest(),
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def send_midi_learn_signal(
    action: str,
    deck: int | None = None,
    cue: int | None = None,
) -> dict[str, Any]:
    """Send a setup-only signal while Rekordbox MIDI Learn is listening."""
    parameters: dict[str, Any] = {}
    if deck is not None:
        parameters["deck"] = deck
    if cue is not None:
        parameters["cue"] = cue
    return await engine.send_learn_signal(action, parameters)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def arm_control(seconds: int = 300) -> dict[str, Any]:
    """Arm live control for a bounded period after the user requests performance."""
    return engine.arm(seconds)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def disarm_control() -> dict[str, Any]:
    """Disarm live controls without changing current Rekordbox mixer state."""
    deck_observer.deactivate()
    return engine.disarm()


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def trigger_control(
    action: str,
    deck: int | None = None,
    cue: int | None = None,
) -> dict[str, Any]:
    """Trigger a mapped button action such as play, cue, sync, loop, or hot cue."""
    if action not in TRIGGER_ACTIONS:
        raise ValueError(
            f"action must be a trigger action: {', '.join(sorted(TRIGGER_ACTIONS))}"
        )
    if action in {"sync", "quantize"}:
        raise ValueError(
            f"{action} is a stateful toggle. Use ensure_deck_modes with a fresh "
            "high-confidence observation."
        )
    parameters: dict[str, Any] = {}
    if deck is not None:
        parameters["deck"] = deck
    if cue is not None:
        parameters["cue"] = cue
    messages = await engine.send_action(action, parameters)
    return {
        "action": action,
        "parameters": parameters,
        "messages": messages,
        "effect_verified": False,
        "message": (
            "MIDI was dispatched. This result does not prove Rekordbox applied it."
        ),
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def verify_hot_cue(
    deck: int,
    track_id: str,
    title: str,
    cue: int,
    expected_time_ms: int,
    tolerance_seconds: float = 2.0,
) -> dict[str, Any]:
    """
    Recall a muted Hot Cue and prove its title, position, and transport response.

    If the requested pad is empty, Rekordbox may create a cue on the first
    press. That is not treated as verified; call this tool again only after
    confirming the intended cue location.
    """
    if deck not in (1, 2):
        raise ValueError("deck must be 1 or 2")
    if not 1 <= cue <= 8:
        raise ValueError("cue must be between 1 and 8")
    if expected_time_ms < 0:
        raise ValueError("expected_time_ms must be non-negative")
    if not 0.5 <= tolerance_seconds <= 5.0:
        raise ValueError("tolerance_seconds must be between 0.5 and 5.0")
    profile = profile_store.get(track_id)
    if normalize_title(profile.title) != normalize_title(title):
        raise ValueError("profile title does not match requested title")

    result: dict[str, Any] = {
        "deck": deck,
        "track_id": track_id,
        "title": title,
        "cue": cue,
        "expected_time_ms": expected_time_ms,
        "verified": False,
    }
    with deck_observer.exclusive_adapter():
        initial = rekordbox_ui.deck_snapshot(deck)
        if normalize_title(initial.title) != normalize_title(title):
            raise RuntimeError(
                f"Deck {deck} loaded {initial.title!r}, expected {title!r}"
            )
        if rekordbox_ui.deck_is_playing(deck, initial=initial):
            raise RuntimeError(
                f"Deck {deck} must be stopped before Hot Cue verification"
            )
        result["mute_messages"] = await engine.send_action(
            "channel_fader",
            {"deck": deck, "value": 0},
        )
        try:
            launched_at = time.monotonic()
            result["launch_messages"] = await engine.send_action(
                "hot_cue",
                {"deck": deck, "cue": cue},
            )
            first_snapshot_started_after_seconds = (
                time.monotonic() - launched_at
            )
            first = rekordbox_ui.deck_snapshot(deck)
            first_observed_after_seconds = time.monotonic() - launched_at
            await asyncio.sleep(1.1)
            second = rekordbox_ui.deck_snapshot(deck)
            result["observed"] = {
                "first": first.public(),
                "second": second.public(),
            }
            moved = (
                first.elapsed_seconds is not None
                and second.elapsed_seconds is not None
                and first.elapsed_seconds != second.elapsed_seconds
            )
            observed_seconds = first.elapsed_seconds
            position_window = None
            latency_corrected_seconds = None
            position_ok = False
            if observed_seconds is not None:
                position_window = _hot_cue_position_window(
                    observed_seconds,
                    first_snapshot_started_after_seconds,
                    first_observed_after_seconds,
                )
                latency_corrected_seconds = sum(position_window) / 2.0
                expected_seconds = expected_time_ms / 1000.0
                position_ok = (
                    position_window[0] - tolerance_seconds
                    <= expected_seconds
                    <= position_window[1] + tolerance_seconds
                )
            result["first_snapshot_started_after_seconds"] = round(
                first_snapshot_started_after_seconds,
                3,
            )
            result["first_observed_after_seconds"] = round(
                first_observed_after_seconds, 3
            )
            result["latency_corrected_range_seconds"] = (
                None
                if position_window is None
                else [round(value, 3) for value in position_window]
            )
            result["latency_corrected_seconds"] = (
                None
                if latency_corrected_seconds is None
                else round(latency_corrected_seconds, 3)
            )
            result["transport_started"] = moved
            result["position_matches"] = position_ok
            result["verified"] = moved and position_ok
        finally:
            result["stop_messages"] = await engine.send_action(
                "cue",
                {"deck": deck},
            )

    if result["verified"]:
        key = (deck, track_id, cue)
        record = {
            "deck": deck,
            "track_id": track_id,
            "title": title,
            "cue": cue,
            "expected_time_ms": expected_time_ms,
            "verified_monotonic": time.monotonic(),
        }
        verified_hot_cues[key] = record
        result["session_verification"] = {
            **record,
            "verified_monotonic": "current_session",
        }
    else:
        result["message"] = (
            "Hot Cue was not proven. It may have been empty, mapped incorrectly, "
            "or stored at a different position; do not use it in a live card."
        )
    return result


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def launch_verified_hot_cue(
    deck: int,
    track_id: str,
    title: str,
    cue: int,
) -> dict[str, Any]:
    """Launch a session-verified Hot Cue and create an honest native deck clock."""
    key = (deck, track_id, cue)
    record = verified_hot_cues.get(key)
    if record is None:
        raise RuntimeError("Hot Cue is not verified in this Rekordbox session")
    age = time.monotonic() - float(record["verified_monotonic"])
    if age > VERIFIED_CUE_TTL_SECONDS:
        verified_hot_cues.pop(key, None)
        raise RuntimeError("Hot Cue verification expired")
    profile = profile_store.get(track_id)
    landmark = next(
        (
            item
            for item in profile.landmarks
            if item.cue == cue
            and item.kind in {"mix_in", "phrase_start"}
            and item.confidence in {"verified", "high"}
        ),
        None,
    )
    if landmark is None:
        raise RuntimeError(
            "Verified Hot Cue is not a high-confidence profile landmark"
        )
    with deck_observer.exclusive_adapter():
        initial = rekordbox_ui.deck_snapshot(deck)
        if normalize_title(initial.title) != normalize_title(title):
            raise RuntimeError(
                f"Deck {deck} loaded {initial.title!r}, expected {title!r}"
            )
        if rekordbox_ui.deck_is_playing(deck, initial=initial):
            raise RuntimeError(f"Deck {deck} is already playing")
        launched_at = time.monotonic()
        messages = await engine.send_action(
            "hot_cue",
            {"deck": deck, "cue": cue},
        )
        first = rekordbox_ui.deck_snapshot(deck)
        await asyncio.sleep(1.1)
        second = rekordbox_ui.deck_snapshot(deck)
    if (
        first.elapsed_seconds is None
        or second.elapsed_seconds is None
        or first.elapsed_seconds == second.elapsed_seconds
    ):
        await engine.send_action("cue", {"deck": deck})
        raise RuntimeError("Rekordbox did not start the verified Hot Cue")

    elapsed_beats = (time.monotonic() - launched_at) * profile.bpm / 60.0
    landmark_track_beat = (
        (landmark.bar - 1) * profile.time_signature + (landmark.beat or 1)
    )
    total = landmark_track_beat - 1 + elapsed_beats
    bar = int(total // profile.time_signature) + 1
    within_bar = total % profile.time_signature
    observation = DeckObservation(
        deck=deck,
        track_id=track_id,
        title=title,
        bpm=profile.bpm,
        playing=True,
        bar=bar,
        beat=int(within_bar) + 1,
        track_beat=int(total) + 1,
        beat_phase=within_bar - int(within_bar),
        sync_enabled=second.beat_sync_enabled,
        quantize_enabled=second.quantize_enabled,
        source="native",
        confidence="high",
    )
    return {
        "messages": messages,
        "observed": {
            "first": first.public(),
            "second": second.public(),
        },
        "live_state": live_state.update(observation),
        "effect_verified": True,
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def launch_staged_track(
    deck: int,
    track_id: str,
    title: str,
) -> dict[str, Any]:
    """
    Launch a stopped track from its verified file-start landmark.

    Unlike trigger_control(play_pause), this proves that Rekordbox started and
    creates the native live clock required by transition-card compilation.
    """
    if deck not in (1, 2):
        raise ValueError("deck must be 1 or 2")
    profile = profile_store.get(track_id)
    entry = next(
        (
            item
            for item in profile.landmarks
            if item.kind in {"mix_in", "phrase_start"}
            and item.bar == 1
            and (item.beat or 1) == 1
            and item.confidence in {"verified", "high"}
        ),
        None,
    )
    if entry is None:
        raise RuntimeError(
            "Track has no high-confidence file-start phrase landmark"
        )
    with deck_observer.exclusive_adapter():
        initial = rekordbox_ui.deck_snapshot(deck)
        if normalize_title(initial.title) != normalize_title(title):
            raise RuntimeError(
                f"Deck {deck} loaded {initial.title!r}, expected {title!r}"
            )
        if rekordbox_ui.deck_is_playing(deck, initial=initial):
            raise RuntimeError(f"Deck {deck} is already playing")
        if (
            initial.elapsed_seconds is None
            or initial.elapsed_seconds > 1
        ):
            raise RuntimeError(
                f"Deck {deck} is not staged at the file start"
            )
        launched_at = time.monotonic()
        messages = await engine.send_action(
            "play_pause",
            {"deck": deck},
        )
        first = rekordbox_ui.deck_snapshot(deck)
        await asyncio.sleep(1.1)
        second = rekordbox_ui.deck_snapshot(deck)
    if (
        first.elapsed_seconds is None
        or second.elapsed_seconds is None
        or first.elapsed_seconds == second.elapsed_seconds
    ):
        await engine.send_action("cue", {"deck": deck})
        raise RuntimeError("Rekordbox did not start the staged track")

    elapsed_beats = (time.monotonic() - launched_at) * profile.bpm / 60.0
    total = elapsed_beats
    bar = int(total // profile.time_signature) + 1
    within_bar = total % profile.time_signature
    observation = DeckObservation(
        deck=deck,
        track_id=track_id,
        title=title,
        bpm=profile.bpm,
        playing=True,
        bar=bar,
        beat=int(within_bar) + 1,
        track_beat=int(total) + 1,
        beat_phase=within_bar - int(within_bar),
        sync_enabled=second.beat_sync_enabled,
        quantize_enabled=second.quantize_enabled,
        source="native",
        confidence="high",
    )
    return {
        "messages": messages,
        "observed": {
            "first": first.public(),
            "second": second.public(),
        },
        "live_state": live_state.update(observation),
        "effect_verified": True,
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def refresh_transition_state(
    outgoing_deck: int,
    outgoing_track_id: str,
    outgoing_title: str,
    incoming_deck: int,
    incoming_track_id: str,
    incoming_title: str,
    incoming_hot_cue: int | None = None,
) -> dict[str, Any]:
    """
    Verify the live/staged transport pair and refresh both native deck clocks.

    The live deck clock must originate from launch_staged_track or
    launch_verified_hot_cue. This tool confirms that its elapsed display still
    advances while the incoming deck remains stopped.
    """
    if {outgoing_deck, incoming_deck} != {1, 2}:
        raise ValueError("outgoing_deck and incoming_deck must be decks 1 and 2")
    prior = live_state.get(outgoing_deck)
    if prior["track_id"] != outgoing_track_id:
        raise RuntimeError("Existing live clock does not match outgoing track")
    if prior["source"] != "native" or prior["confidence"] not in {
        "verified",
        "high",
    }:
        raise RuntimeError("Outgoing deck lacks a verified native launch clock")

    with deck_observer.exclusive_adapter():
        first_status = rekordbox_ui.status()
        await asyncio.sleep(1.1)
        second_status = rekordbox_ui.status()
    outgoing_first = _deck_record(first_status, outgoing_deck)
    outgoing_second = _deck_record(second_status, outgoing_deck)
    incoming_first = _deck_record(first_status, incoming_deck)
    incoming_second = _deck_record(second_status, incoming_deck)
    if normalize_title(outgoing_second.get("title", "")) != normalize_title(
        outgoing_title
    ):
        raise RuntimeError("Outgoing deck title changed")
    if normalize_title(incoming_second.get("title", "")) != normalize_title(
        incoming_title
    ):
        raise RuntimeError("Incoming deck title changed")
    if not _transport_changed(outgoing_first, outgoing_second):
        raise RuntimeError("Outgoing deck is not playing")
    if _transport_changed(incoming_first, incoming_second):
        raise RuntimeError("Incoming deck is already playing")
    _require_staged_incoming_position(
        incoming_second,
        deck=incoming_deck,
        track_id=incoming_track_id,
        hot_cue=incoming_hot_cue,
    )

    current = live_state.get(outgoing_deck)
    outgoing_observation = DeckObservation(
        deck=outgoing_deck,
        track_id=outgoing_track_id,
        title=outgoing_title,
        bpm=current["bpm"],
        playing=True,
        bar=current["bar"],
        beat=current["beat"],
        track_beat=current["track_beat"],
        beat_phase=current["beat_phase"],
        sync_enabled=outgoing_second.get("beat_sync_enabled"),
        quantize_enabled=outgoing_second.get("quantize_enabled"),
        source="native",
        confidence="high",
    )
    incoming_profile = profile_store.get(incoming_track_id)
    incoming_observation = DeckObservation(
        deck=incoming_deck,
        track_id=incoming_track_id,
        title=incoming_title,
        bpm=incoming_profile.bpm,
        playing=False,
        bar=1,
        beat=1,
        track_beat=1,
        beat_phase=0,
        sync_enabled=incoming_second.get("beat_sync_enabled"),
        quantize_enabled=incoming_second.get("quantize_enabled"),
        source="native",
        confidence="high",
    )
    return {
        "outgoing": live_state.update(outgoing_observation),
        "incoming": live_state.update(incoming_observation),
        "observed": {
            "first": first_status,
            "second": second_status,
        },
        "effect_verified": True,
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def ensure_deck_modes(
    deck: int,
    beat_sync: bool | None = None,
    quantize: bool | None = None,
) -> dict[str, Any]:
    """
    Toggle Beat Sync/Quantize only when a fresh observation proves it is needed.

    The resulting state must be observed again before compiling a transition.
    """
    if deck not in (1, 2):
        raise ValueError("deck must be 1 or 2")
    if beat_sync is None and quantize is None:
        raise ValueError("Request beat_sync, quantize, or both")
    state = live_state.get(deck)
    if state["observation_age_ms"] > 1000:
        raise ValueError(
            f"deck observation is stale ({state['observation_age_ms']} ms)"
        )
    if state["confidence"] not in {"verified", "high"}:
        raise ValueError("deck observation confidence is below high")

    requested = (
        ("sync", "sync_enabled", beat_sync),
        ("quantize", "quantize_enabled", quantize),
    )
    actions = []
    for action, field, desired in requested:
        if desired is None:
            continue
        actual = state[field]
        if actual is None:
            raise ValueError(f"deck {deck} {field} state is unobserved")
        if actual != desired:
            messages = await engine.send_action(action, {"deck": deck})
            actions.append(
                {
                    "action": action,
                    "from": actual,
                    "expected": desired,
                    "messages": messages,
                }
            )
    return {
        "deck": deck,
        "actions": actions,
        "changed": bool(actions),
        "verification_required": (
            "Re-observe the deck's Beat Sync and Quantize indicators before "
            "previewing or performing a transition card."
        ),
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def browse_tracks(steps: int) -> dict[str, Any]:
    """Move Rekordbox's browser selection like the FLX4 browse encoder."""
    if not -100 <= steps <= 100:
        raise ValueError("steps must be between -100 and 100")
    action = "browse_down" if steps > 0 else "browse_up"
    messages: list[str] = []
    for _ in range(abs(steps)):
        messages.extend(await engine.send_action(action))
    return {
        "action": action,
        "steps": steps,
        "messages": messages,
        "verification_required": (
            "Confirm the highlighted browser row before loading a deck."
        ),
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def load_selected_track(deck: int) -> dict[str, Any]:
    """Press FLX4-style LOAD after browse_tracks has focused the track list."""
    if deck not in (1, 2):
        raise ValueError("deck must be 1 or 2")
    action = f"load_deck_{deck}"
    messages = await engine.send_action(action)
    return {
        "action": action,
        "deck": deck,
        "messages": messages,
        "verification_required": (
            "Call browse_tracks first so the track list owns focus, then confirm "
            "the expected track is resident and stopped before arming cues."
        ),
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def select_track_exact(
    title: str,
    result_index: int | None = None,
) -> dict[str, Any]:
    """Search Rekordbox and focus exactly one full-title match."""
    return rekordbox_ui.select_exact_track(
        title,
        result_index=result_index,
    )


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def load_track_exact(
    deck: int,
    title: str,
    artist: str | None = None,
) -> dict[str, Any]:
    """Select one exact title, load it, and verify the stopped deck title."""
    return await _stage_track(
        deck=deck,
        track_id=f"title:{normalize_title(title)}",
        title=title,
        artist=artist,
    )


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def stage_track(
    deck: int,
    track_id: str,
    title: str,
    artist: str | None = None,
    source: str | None = None,
    result_index: int | None = None,
) -> dict[str, Any]:
    """Load and verify one stable catalog identity on a stopped deck."""
    return await _stage_track(
        deck=deck,
        track_id=track_id,
        title=title,
        artist=artist,
        source=source,
        result_index=result_index,
    )


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def stage_and_schedule_transition_card(
    card: TransitionCard,
    incoming_title: str,
    incoming_artist: str,
    incoming_cue: int,
    incoming_cue_time_ms: int,
    execution_id: str,
    source: str | None = None,
    result_index: int | None = None,
) -> dict[str, Any]:
    """Stage, verify, mode-check, and schedule one handoff without agent delay.

    This is the live-path primitive for a rolling two-deck set.  All expensive
    UI work happens inside one MCP request, so model/tool round trips cannot
    consume the outgoing track's remaining phrase runway.
    """
    profile = profile_store.get(card.incoming_track_id)
    if normalize_title(profile.title) != normalize_title(incoming_title):
        raise ValueError("incoming_title does not match the card profile")
    launch = _card_hot_cue_launch(card)
    expected_launch = (
        card.incoming_deck,
        card.incoming_track_id,
        incoming_cue,
    )
    if launch != expected_launch:
        raise ValueError(
            "incoming_cue must match the card's bar-0 incoming Hot Cue"
        )
    if not execution_id.strip():
        raise ValueError("execution_id is required")

    started = time.monotonic()
    staged = await _stage_track(
        deck=card.incoming_deck,
        track_id=card.incoming_track_id,
        title=incoming_title,
        artist=incoming_artist,
        source=source,
        result_index=result_index,
    )
    cue_verification = await verify_hot_cue(
        deck=card.incoming_deck,
        track_id=card.incoming_track_id,
        title=incoming_title,
        cue=incoming_cue,
        expected_time_ms=incoming_cue_time_ms,
    )
    if not cue_verification.get("verified"):
        raise RuntimeError("Incoming Hot Cue verification failed")

    outgoing_title = profile_store.get(card.outgoing_track_id).title
    refresh_before = await refresh_transition_state(
        outgoing_deck=card.outgoing_deck,
        outgoing_track_id=card.outgoing_track_id,
        outgoing_title=outgoing_title,
        incoming_deck=card.incoming_deck,
        incoming_track_id=card.incoming_track_id,
        incoming_title=incoming_title,
        incoming_hot_cue=incoming_cue,
    )
    mode_result = await ensure_deck_modes(
        deck=card.incoming_deck,
        beat_sync=True if card.beat_sync_required else None,
        quantize=True if card.quantize_required else None,
    )
    if mode_result["changed"]:
        # Rekordbox's indicator paint trails the MIDI command.  Re-observe only
        # inside this transaction, then compile immediately from that proof.
        await asyncio.sleep(1.2)
    refresh_after = await refresh_transition_state(
        outgoing_deck=card.outgoing_deck,
        outgoing_track_id=card.outgoing_track_id,
        outgoing_title=outgoing_title,
        incoming_deck=card.incoming_deck,
        incoming_track_id=card.incoming_track_id,
        incoming_title=incoming_title,
        incoming_hot_cue=incoming_cue,
    )
    scheduled = await perform_transition_card(
        card=card,
        commit=True,
        execution_id=execution_id,
    )
    if not scheduled.get("ready"):
        return {
            "ready": False,
            "stage": staged,
            "cue_verification": cue_verification,
            "refresh_before": refresh_before,
            "mode_result": mode_result,
            "refresh_after": refresh_after,
            "schedule": scheduled,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }
    return {
        "ready": True,
        "stage": staged,
        "cue_verification": cue_verification,
        "mode_result": mode_result,
        "refresh": refresh_after,
        "schedule": scheduled,
        "elapsed_ms": round((time.monotonic() - started) * 1000),
    }


async def _stage_track(
    *,
    deck: int,
    track_id: str,
    title: str,
    artist: str | None = None,
    source: str | None = None,
    result_index: int | None = None,
) -> dict[str, Any]:
    with deck_observer.exclusive_adapter():
        return await _stage_track_exclusive(
            deck=deck,
            track_id=track_id,
            title=title,
            artist=artist,
            source=source,
            result_index=result_index,
        )


async def _stage_track_exclusive(
    *,
    deck: int,
    track_id: str,
    title: str,
    artist: str | None = None,
    source: str | None = None,
    result_index: int | None = None,
) -> dict[str, Any]:
    if deck not in (1, 2):
        raise ValueError("deck must be 1 or 2")
    if rekordbox_ui.deck_is_playing(deck):
        raise RuntimeError(f"Refusing to load playing deck {deck}")
    preload_messages = []
    preload_messages.extend(
        await engine.send_action(
            "channel_fader",
            {"deck": deck, "value": 0},
        )
    )
    preload_messages.extend(await engine.send_action("cue", {"deck": deck}))
    await asyncio.sleep(0.1)
    if rekordbox_ui.deck_is_playing(deck):
        raise RuntimeError(
            f"Deck {deck} did not enter a paused state before loading"
        )
    search_title = title.translate(
        str.maketrans({"\u2018": "'", "\u2019": "'", "\uff07": "'"})
    )
    route_cache = rekordbox_ui.stage_route_cache
    cached_route = route_cache.get(track_id)
    search_queries = []
    for query in (
        cached_route.get("search_query") if cached_route else None,
        f"{title} {artist}" if artist else title,
        title,
        f"{search_title} {artist}" if artist else search_title,
        search_title,
        re.sub(r"\s*\([^)]*\)\s*$", "", title).strip(),
    ):
        if query and query not in search_queries:
            search_queries.append(query)
    first = None
    empty_searches = []
    for search_query in search_queries:
        try:
            first = rekordbox_ui.select_exact_track(
                title,
                search_query=search_query,
                result_index=0,
            )
            break
        except RuntimeError as exc:
            if "returned no visible rows" not in str(exc):
                raise
            empty_searches.append(search_query)
    if first is None:
        raise RuntimeError(
            f"No visible Rekordbox rows for {title!r}; "
            f"tried {empty_searches!r}"
        )
    search_query = first["search_query"]
    result_count = first["result_count"]
    if result_index is not None and not 0 <= result_index < result_count:
        raise ValueError(
            f"result_index must be between 0 and {result_count - 1}"
        )
    if result_count > 1 and not artist and result_index is None:
        raise RuntimeError(
            f"{result_count} rows match {title!r}; provide artist or result_index"
        )
    candidate_indices = (
        [result_index]
        if result_index is not None
        else list(range(result_count))
    )
    if (
        result_index is None
        and cached_route
        and cached_route.get("search_query") == search_query
        and cached_route.get("result_index") in candidate_indices
    ):
        cached_index = cached_route["result_index"]
        candidate_indices.remove(cached_index)
        candidate_indices.insert(0, cached_index)
    attempts = []
    for candidate_index in candidate_indices:
        selection = (
            first
            if candidate_index == 0
            else rekordbox_ui.select_exact_track(
                title,
                search_query=search_query,
                result_index=candidate_index,
            )
        )
        messages = []
        cached_method = (
            cached_route.get("method")
            if cached_route
            and cached_route.get("search_query") == search_query
            and cached_route.get("result_index") == candidate_index
            else None
        )
        method = cached_method or "midi_load"
        observed = None
        if method != "drag_to_deck":
            if result_count > 1 and candidate_index == result_count - 1:
                browse_actions = ("browse_up", "browse_down")
            else:
                browse_actions = ("browse_down", "browse_up")
            for action in browse_actions:
                messages.extend(await engine.send_action(action))
            messages.extend(await engine.send_action(f"load_deck_{deck}"))
            await asyncio.sleep(0.2)
            try:
                observed = rekordbox_ui.wait_for_deck_title(
                    deck,
                    title,
                    timeout_seconds=1.25,
                )
            except RuntimeError:
                method = "drag_to_deck"
        if method == "drag_to_deck" and observed is None:
            drag = rekordbox_ui.drag_selected_track_to_deck(
                selected_row_top=selection["selected_row_top"],
                deck=deck,
                selected_row_point=selection.get("selected_row_point"),
                deck_drop_point=selection.get("deck_drop_points", {}).get(deck),
            )
            method = drag["method"]
            try:
                observed = rekordbox_ui.wait_for_deck_title(
                    deck,
                    title,
                    timeout_seconds=4.0,
                )
            except RuntimeError as exc:
                # A broadened query can legitimately return another remix as
                # its first row.  Record that rejected candidate and continue
                # to the next visible row instead of aborting the exact-load
                # search after the drag fallback.
                attempts.append(
                    {
                        "result_index": candidate_index,
                        "method": method,
                        "observed": None,
                        "artist_matches": False,
                        "error": str(exc),
                    }
                )
                continue
        artist_matches = (
            artist is None
            or normalize_title(observed.artist) == normalize_title(artist)
        )
        attempts.append(
            {
                "result_index": candidate_index,
                "method": method,
                "observed": observed.public(),
                "artist_matches": artist_matches,
            }
        )
        if artist_matches:
            if rekordbox_ui.deck_is_playing(deck, initial=observed):
                await engine.send_action(
                    "channel_fader",
                    {"deck": deck, "value": 0},
                )
                await engine.send_action("cue", {"deck": deck})
                raise RuntimeError(
                    f"Loaded deck {deck} auto-started; it was muted and stopped. "
                    "Stage it again before verification."
                )
            deck_observer.invalidate()
            route_cache[track_id] = {
                "search_query": search_query,
                "result_index": candidate_index,
                "method": method,
            }
            return {
                "selection": selection,
                "deck": deck,
                "track_id": track_id,
                "title": title,
                "artist": artist,
                "source": source,
                "messages": messages,
                "preload_stop_messages": preload_messages,
                "observed": observed.public(),
                "transport_verified_stopped": True,
                "attempts": attempts,
                "route_cache_hit": cached_method is not None,
                "verified": True,
            }
    raise RuntimeError(
        f"No visible {title!r} row loaded expected artist {artist!r}; "
        f"observed attempts: {attempts}"
    )


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def set_continuous_control(
    action: str,
    value: float,
    deck: int | None = None,
) -> dict[str, Any]:
    """Set a mapped fader, EQ, filter, tempo, gain, or FX control."""
    if action not in CONTINUOUS_ACTIONS:
        raise ValueError(
            "action must be a continuous action: "
            + ", ".join(sorted(CONTINUOUS_ACTIONS))
        )
    parameters: dict[str, Any] = {"value": value}
    if deck is not None:
        parameters["deck"] = deck
    messages = await engine.send_action(action, parameters)
    return {"action": action, "parameters": parameters, "messages": messages}


@mcp.tool(annotations={"readOnlyHint": True})
def preview_transition(
    name: str, events: list[TransitionEvent]
) -> dict[str, Any]:
    """Validate and preview a transition without sending MIDI."""
    return scheduler.preview(name, [event.model_dump() for event in events])


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def perform_transition(
    name: str,
    events: list[TransitionEvent],
    commit: bool = False,
) -> dict[str, Any]:
    """Preview low-level timing tests; live musical handoffs require a card."""
    raw_events = [event.model_dump() for event in events]
    if not commit:
        return {
            **scheduler.preview(name, raw_events),
            "message": (
                "Dry-run timing preview only. Use perform_transition_card for "
                "live musical handoffs."
            ),
        }
    raise RuntimeError(
        "Wall-clock perform_transition is disabled for live commits. Import "
        "Rekordbox analysis, observe both decks, and use perform_transition_card."
    )


@mcp.tool(annotations={"readOnlyHint": True})
def transition_status(job_id: str) -> dict[str, Any]:
    """Get a scheduled transition's progress."""
    return scheduler.get(job_id)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def cancel_transition(job_id: str) -> dict[str, Any]:
    """Cancel future events in one transition without moving any controls."""
    return scheduler.cancel(job_id)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def emergency_stop() -> dict[str, Any]:
    """Cancel all automation and disarm; leave current Rekordbox state untouched."""
    cancelled = scheduler.cancel_all()
    deck_observer.deactivate()
    status = engine.disarm()
    return {
        "cancelled_jobs": cancelled,
        "control_status": status,
        "message": "Automation cancelled. Physical FLX4 control remains available.",
    }


def main() -> None:
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
