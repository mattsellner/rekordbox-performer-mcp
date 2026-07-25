"""FastMCP server for live Rekordbox control through virtual MIDI."""

from __future__ import annotations

import asyncio
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
    compile_transition_card,
)
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


mcp = FastMCP("Rekordbox Performer")
engine = MidiEngine()
scheduler = TransitionScheduler(engine)
live_state = LiveState()
profile_store = ProfileStore()
rehearsal_capture = RehearsalCapture(profile_store.data_dir / "recordings")
rekordbox_ui = RekordboxUIAdapter()


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
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def observe_deck_state(observation: DeckObservation) -> dict[str, Any]:
    """Ingest a fresh native, MIDI, vision, or manual deck-clock observation."""
    return live_state.update(observation)


@mcp.tool(annotations={"readOnlyHint": True})
def live_state_status() -> dict[str, Any]:
    """Return extrapolated deck clocks and observation freshness."""
    return live_state.snapshot()


@mcp.tool(annotations={"readOnlyHint": True})
def rekordbox_ui_status() -> dict[str, Any]:
    """Observe Performance mode, loaded titles, BPM, Sync, and Quantize."""
    return rekordbox_ui.status()


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
    return {
        **compiled,
        "job": scheduler.start(card.name, compiled["events"]),
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
    return {"action": action, "parameters": parameters, "messages": messages}


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
    if deck not in (1, 2):
        raise ValueError("deck must be 1 or 2")
    if rekordbox_ui.deck_is_playing(deck):
        raise RuntimeError(f"Refusing to load playing deck {deck}")
    first = rekordbox_ui.select_exact_track(title, result_index=0)
    result_count = first["result_count"]
    if result_count > 1 and not artist:
        raise RuntimeError(
            f"{result_count} rows match {title!r}; provide the expected artist"
        )
    attempts = []
    for result_index in range(result_count):
        selection = (
            first
            if result_index == 0
            else rekordbox_ui.select_exact_track(
                title,
                result_index=result_index,
            )
        )
        messages = await engine.send_action(f"load_deck_{deck}")
        await asyncio.sleep(0.75)
        observed = rekordbox_ui.wait_for_deck_title(deck, title)
        artist_matches = (
            artist is None
            or normalize_title(observed.artist) == normalize_title(artist)
        )
        attempts.append(
            {
                "result_index": result_index,
                "observed": observed.public(),
                "artist_matches": artist_matches,
            }
        )
        if artist_matches:
            return {
                "selection": selection,
                "deck": deck,
                "title": title,
                "artist": artist,
                "messages": messages,
                "observed": observed.public(),
                "attempts": attempts,
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
