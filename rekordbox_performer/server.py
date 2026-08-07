"""FastMCP server for live Rekordbox control through virtual MIDI."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from .audio import RehearsalCapture, analyze_rehearsal_audio
from .engine import MidiEngine
from .intelligence import (
    DeckObservation,
    LiveState,
    MusicalEvent,
    PlaylistTrackMetadata,
    ProfileStore,
    RekordboxAnalysisImport,
    RehearsalReview,
    SetPlan,
    TrackProfile,
    TransitionCard,
    audit_set_plan,
    bass_phrase_evidence,
    camelot_compatibility,
    compile_transition_card,
    validate_transition_card,
)
from .observer import SharedDeckObserver
from .performance import (
    SetSessionManager,
    cue_preparation_plan,
    fx_recipe,
    observation_from_elapsed,
    rescue_loop_target,
    sync_report,
    transition_qa,
    vocal_handoff,
)
from .protocol import (
    ALL_ACTIONS,
    CONTINUOUS_ACTIONS,
    TRIGGER_ACTIONS,
    STEM_ACTIONS,
    mapping_manifest,
)
from .rekordbox_ui import RekordboxUIAdapter, normalize_title
from .scheduler import TransitionScheduler
from .set_runner import (
    AutonomousSetPlan,
    AutonomousSetRunner,
    TempoPlan,
    TrackLoadSpec,
    TransitionOption,
)
from .supervisor import GENERATION_ENV, RESTART_EXIT_CODE, SUPERVISED_ENV


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


mcp = FastMCP("RekordBot Performer")
engine = MidiEngine()
scheduler = TransitionScheduler(engine)
live_state = LiveState()
profile_store = ProfileStore()
rehearsal_capture = RehearsalCapture(profile_store.data_dir / "recordings")
rekordbox_ui = RekordboxUIAdapter()
deck_observer = SharedDeckObserver(rekordbox_ui, profile_store.data_dir)
set_sessions = SetSessionManager(profile_store.data_dir / "active-set.json")
verified_hot_cues: dict[tuple[int, str, int], dict[str, Any]] = {}
sync_guard_tasks: set[asyncio.Task[Any]] = set()
sync_guard_status: dict[str, dict[str, Any]] = {}
VERIFIED_CUE_TTL_SECONDS = 15 * 60
verified_rescue_loops: dict[int, dict[str, Any]] = {}


def _set_sync_guard_status(job_id: str, payload: dict[str, Any]) -> None:
    sync_guard_status[job_id] = payload
    diagnostic = {
        "job_id": job_id,
        "updated_at": time.time(),
        **payload,
    }
    (profile_store.data_dir / "last-sync-guard.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _cancel_for_sync_guard(job_id: str, errors: list[str]) -> None:
    job = getattr(scheduler, "jobs", {}).get(job_id)
    if job is not None:
        job.error = "; ".join(errors)
    scheduler.cancel(job_id)
autonomous_runner: AutonomousSetRunner | None = None
autonomous_recovery_planner: Any = None
track_title_aliases: dict[str, str] = {}
preflighted_set_fingerprints: set[str] = set()


def _set_fingerprint(plan: AutonomousSetPlan) -> str:
    return hashlib.sha256(plan.model_dump_json().encode("utf-8")).hexdigest()


def _materialize_autonomous_plan(
    plan: AutonomousSetPlan,
) -> AutonomousSetPlan:
    track_ids = {plan.opening.track_id}
    track_ids.update(option.incoming.track_id for option in plan.transitions)
    native_bpms = {track_id: profile_store.get(track_id).bpm for track_id in track_ids}
    return plan.materialize_tempo_arc(native_bpms)


def _expected_track_title(track_id: str) -> str:
    return track_title_aliases.get(track_id) or profile_store.get(track_id).title


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


async def _observe_live_deck(
    *,
    deck: int,
    track_id: str,
    title: str,
) -> dict[str, Any]:
    """Refresh one authoritative deck clock without requiring a staged pair."""
    profile = profile_store.get(track_id)
    with deck_observer.exclusive_adapter():
        rekordbox_ui.invalidate_status_cache()
        first_status = rekordbox_ui.status()
        await asyncio.sleep(1.1)
        second_status = rekordbox_ui.status()
    first = _deck_record(first_status, deck)
    second = _deck_record(second_status, deck)
    if normalize_title(second.get("title", "")) != normalize_title(title):
        raise RuntimeError(f"Deck {deck} loaded title does not match {title!r}")
    playing = _transport_changed(first, second)
    observation = observation_from_elapsed(
        deck=deck,
        profile=profile,
        elapsed_seconds=float(second.get("elapsed_seconds") or 0),
        playing=playing,
        sync_enabled=second.get("beat_sync_enabled"),
        quantize_enabled=second.get("quantize_enabled"),
        title=title,
        playback_bpm=second.get("bpm"),
    )
    return live_state.update(observation)


def _verify_rescue_loop(
    *,
    deck: int,
    title: str,
    beats: int,
    bpm: float,
) -> dict[str, Any]:
    """Prove loop repetition from Rekordbox's transport instead of MIDI intent."""
    loop_seconds = beats * 60.0 / bpm
    with deck_observer.exclusive_adapter():
        rekordbox_ui.invalidate_status_cache()
        first_status = rekordbox_ui.status()
        time.sleep(loop_seconds + 0.6)
        rekordbox_ui.invalidate_status_cache()
        second_status = rekordbox_ui.status()
        time.sleep(1.1)
        rekordbox_ui.invalidate_status_cache()
        third_status = rekordbox_ui.status()
    first = _deck_record(first_status, deck)
    second = _deck_record(second_status, deck)
    third = _deck_record(third_status, deck)
    errors: list[str] = []
    if normalize_title(third.get("title", "")) != normalize_title(title):
        errors.append("deck title changed while verifying rescue loop")
    if not _transport_changed(second, third):
        errors.append("deck transport stopped while verifying rescue loop")
    first_elapsed = first.get("elapsed_seconds")
    second_elapsed = second.get("elapsed_seconds")
    if first_elapsed is None or second_elapsed is None:
        errors.append("loop transport position is unavailable")
    elif float(second_elapsed) - float(first_elapsed) > max(3.0, loop_seconds * 0.6):
        errors.append("transport did not repeat inside the requested loop")
    result = {
        "verified": not errors,
        "errors": errors,
        "deck": deck,
        "title": title,
        "beats": beats,
        "loop_seconds": round(loop_seconds, 3),
        "observed": {
            "first": first,
            "second": second,
            "third": third,
        },
    }
    if result["verified"]:
        verified_rescue_loops[deck] = {
            **result,
            "verified_monotonic": time.monotonic(),
        }
    return result


async def _engage_rescue_loop(deck: int, beats: int) -> dict[str, Any]:
    if beats not in {4, 8, 16}:
        raise ValueError("rescue loop must be 4, 8, or 16 beats")
    try:
        state = live_state.get(deck)
    except KeyError as exc:
        raise RuntimeError("live deck clock is unavailable") from exc
    if not state.get("playing"):
        raise RuntimeError("cannot rescue-loop a stopped deck")
    title = str(state["title"])
    # Refresh before calculating the next bar boundary. The resulting local
    # schedule is short and remains independent of the MCP client.
    state = await _observe_live_deck(
        deck=deck,
        track_id=str(state["track_id"]),
        title=title,
    )
    if not state.get("playing"):
        raise RuntimeError("audible deck stopped before rescue loop")
    position = (float(state["beat"]) - 1.0) + float(state["beat_phase"])
    beats_to_bar = (-position) % 4.0
    if beats_to_bar < 0.08:
        beats_to_bar = 0.0
    delay_ms = round(beats_to_bar * 60_000.0 / float(state["bpm"]))
    start_bar = int(state["bar"]) if beats_to_bar == 0.0 else int(state["bar"]) + 1
    window = rescue_loop_target(
        profile_store.get(str(state["track_id"])),
        earliest_start_bar=start_bar,
        # Only the direct four-beat AutoLoop mapping is deterministic in the
        # installed profile. Never synthesize longer rescue loops through the
        # repeatable Loop Double control.
        requested_beats=4,
    )
    target_bar = int(window.get("start_bar", start_bar))
    delay_ms += round(
        max(0, target_bar - start_bar)
        * 4
        * 60_000.0
        / float(state["bpm"])
    )
    if window.get("verified") is not True:
        return {
            "verified": False,
            "errors": [str(window["error"])],
            "deck": deck,
            "title": title,
            "beats": 0,
            "requested_beats": beats,
            "loop_window": window,
            "scheduled_at_bar_boundary_ms": delay_ms,
        }
    effective_beats = int(window["beats"])
    events = [
        {
            "at_ms": delay_ms,
            "action": "loop_4",
            "parameters": {"deck": deck},
        }
    ]
    doubles = {4: 0, 8: 1, 16: 2}[effective_beats]
    for index in range(doubles):
        events.append(
            {
                "at_ms": delay_ms + 90 * (index + 1),
                "action": "loop_double",
                "parameters": {"deck": deck},
            }
        )
    job = scheduler.start(
        f"verified {effective_beats}-beat rescue loop on deck {deck}",
        events,
        completion_verifier=lambda: _verify_rescue_loop(
            deck=deck,
            title=title,
            beats=effective_beats,
            bpm=float(state["bpm"]),
        ),
        execution_id=(
            f"rescue-loop-{deck}-{state['track_id']}-{int(time.time() * 1000)}"
        ),
    )
    task = scheduler.jobs[job["id"]].task
    if task is not None:
        await task
    final = scheduler.get(job["id"])
    verification = final.get("verification") or {}
    # Observation can occasionally miss the wrap even when Rekordbox accepted
    # the command. Replace the uncertain loop with a direct 4-beat AutoLoop
    # and verify again; never leave a huge or unknown loop armed.
    fallback = None
    if verification.get("verified") is not True and effective_beats != 4:
        fallback_job = scheduler.start(
            f"verified 4-beat rescue fallback on deck {deck}",
            [{"at_ms": 0, "action": "loop_4", "parameters": {"deck": deck}}],
            completion_verifier=lambda: _verify_rescue_loop(
                deck=deck,
                title=title,
                beats=4,
                bpm=float(state["bpm"]),
            ),
            execution_id=(
                f"rescue-loop-fallback-{deck}-{state['track_id']}-"
                f"{int(time.time() * 1000)}"
            ),
        )
        fallback_task = scheduler.jobs[fallback_job["id"]].task
        if fallback_task is not None:
            await fallback_task
        fallback = scheduler.get(fallback_job["id"])
        fallback_verification = fallback.get("verification") or {}
        if fallback_verification.get("verified") is True:
            verification = fallback_verification
    return {
        **verification,
        "job": final,
        "fallback_job": fallback,
        "requested_beats": beats,
        "effective_beats": verification.get("beats", effective_beats),
        "loop_window": window,
        "scheduled_at_bar_boundary_ms": delay_ms,
    }


def _with_rescue_loop_release(card: TransitionCard) -> TransitionCard:
    record = verified_rescue_loops.get(card.outgoing_deck)
    if record is None or record.get("verified") is not True:
        raise RuntimeError("outgoing rescue loop is not verified active")
    events = list(card.events)
    events.append(
        card.events[0].__class__(
            bar_offset=0,
            beat_offset=0,
            action="loop_toggle",
            parameters={"deck": card.outgoing_deck},
        )
    )
    events.sort(key=lambda event: (event.bar_offset, event.beat_offset))
    return card.model_copy(update={"events": events, "loop_plan_verified": True})


async def _execute_tempo_plan(
    plan: TempoPlan,
    deck: int,
    track_id: str,
) -> dict[str, Any]:
    profile = profile_store.get(track_id)
    state = await _observe_live_deck(
        deck=deck,
        track_id=track_id,
        title=profile.title,
    )
    if not state.get("playing"):
        raise RuntimeError("tempo ramp requires a playing deck")
    live_bpm = float(state["bpm"])
    plan.validate_start(native_bpm=profile.bpm, live_bpm=live_bpm)
    await ensure_master_deck(deck)
    tempo_takeover = await _acquire_tempo_soft_takeover(
        plan=plan,
        deck=deck,
        native_bpm=profile.bpm,
        live_bpm=live_bpm,
    )
    steps = plan.duration_bars * plan.steps_per_bar
    total_ms = round(plan.duration_bars * 4 * 60_000.0 / live_bpm)
    events = []
    for index in range(1, steps + 1):
        fraction = index / steps
        bpm = live_bpm + (plan.target_bpm - live_bpm) * fraction
        events.append(
            {
                "at_ms": round(total_ms * fraction),
                "action": "tempo",
                "parameters": {
                    "deck": deck,
                    "value": plan.control_value(
                        native_bpm=profile.bpm,
                        bpm=bpm,
                    ),
                },
            }
        )

    def verify() -> dict[str, Any]:
        with deck_observer.exclusive_adapter():
            rekordbox_ui.invalidate_status_cache()
            status = rekordbox_ui.status()
        observed = _deck_record(status, deck)
        observed_bpm = observed.get("bpm")
        errors = []
        if observed.get("master_enabled") is not True:
            errors.append("tempo-ramped deck is not Master")
        if observed_bpm is None:
            errors.append("live BPM is unavailable after tempo ramp")
        elif abs(float(observed_bpm) - plan.target_bpm) > 0.15:
            errors.append(
                f"tempo ramp ended at {float(observed_bpm):.2f}, "
                f"expected {plan.target_bpm:.2f}"
            )
        return {
            "verified": not errors,
            "errors": errors,
            "observed": observed,
            "target_bpm": plan.target_bpm,
        }

    job = scheduler.start(
        f"tempo ramp {profile.title} to {plan.target_bpm:.2f}",
        events,
        completion_verifier=verify,
        execution_id=f"tempo-{track_id}-{int(time.time() * 1000)}",
    )
    task = scheduler.jobs[job["id"]].task
    if task is not None:
        await task
    final = scheduler.get(job["id"])
    if final["status"] != "completed":
        raise RuntimeError(final.get("error") or "tempo ramp failed")
    return {**final, "tempo_takeover": tempo_takeover}


async def _acquire_tempo_soft_takeover(
    *,
    plan: TempoPlan,
    deck: int,
    native_bpm: float,
    live_bpm: float,
) -> dict[str, Any]:
    """Cross Rekordbox's tempo pickup point with a sub-half-BPM nudge."""
    current = plan.control_value(native_bpm=native_bpm, bpm=live_bpm)
    # One bipolar MIDI CC increment is about 0.01575. Use a slightly larger
    # bracket so both rounded CC values straddle the software pickup point,
    # while limiting the audible nudge to roughly 0.25 BPM at house tempos.
    step = 0.02
    values = [
        max(-1.0, current - step),
        min(1.0, current + step),
        current,
    ]
    messages = []
    for value in values:
        messages.extend(
            await engine.send_action(
                "tempo",
                {"deck": deck, "value": value},
            )
        )
        await asyncio.sleep(0.1)
    return {
        "deck": deck,
        "live_bpm": live_bpm,
        "control_value": current,
        "pickup_values": values,
        "messages": messages,
    }


def _guard_status_snapshot(*, invalidate: bool = False) -> dict[str, Any]:
    """Capture one guard frame without occupying the scheduler event loop."""
    with deck_observer.exclusive_adapter():
        if invalidate:
            rekordbox_ui.invalidate_status_cache()
        return rekordbox_ui.status()


async def _guard_status(*, invalidate: bool = False) -> dict[str, Any]:
    if isinstance(rekordbox_ui, RekordboxUIAdapter):
        if invalidate:
            deck_observer.invalidate()
        return await _guard_status_subprocess()
    return await asyncio.to_thread(
        _guard_status_snapshot,
        invalidate=invalidate,
    )


async def _guard_status_subprocess() -> dict[str, Any]:
    """Capture Rekordbox in another process so UIA cannot jitter MIDI."""
    handle, raw_path = tempfile.mkstemp(
        prefix="guard-status-",
        suffix=".json",
        dir=profile_store.data_dir,
    )
    os.close(handle)
    path = Path(raw_path)
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--guard-capture", str(path)]
    else:
        command = [
            sys.executable,
            "-m",
            "rekordbox_performer.guard_capture",
            str(path),
        ]
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await asyncio.wait_for(process.wait(), timeout=15.0)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("ok") is not True:
            raise RuntimeError(payload.get("error") or "guard capture failed")
        return dict(payload["status"])
    finally:
        path.unlink(missing_ok=True)


async def _guard_status_pair(
    *,
    capture_at_monotonic: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(rekordbox_ui, RekordboxUIAdapter):
        await asyncio.sleep(max(0.0, capture_at_monotonic - time.monotonic()))
        first = await _guard_status(invalidate=True)
        await asyncio.sleep(0.35)
        second = await _guard_status()
        return first, second

    handle, raw_path = tempfile.mkstemp(
        prefix="guard-pair-",
        suffix=".json",
        dir=profile_store.data_dir,
    )
    os.close(handle)
    path = Path(raw_path)
    if getattr(sys, "frozen", False):
        command = [
            sys.executable,
            "--guard-capture-pair",
            str(path),
            "--capture-at-monotonic",
            str(capture_at_monotonic),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "rekordbox_performer.guard_capture",
            str(path),
            "--pair",
            "--capture-at-monotonic",
            str(capture_at_monotonic),
        ]
    timeout = max(15.0, capture_at_monotonic - time.monotonic() + 15.0)
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await asyncio.wait_for(process.wait(), timeout=timeout)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("ok") is not True:
            raise RuntimeError(payload.get("error") or "guard pair failed")
        first, second = payload["statuses"]
        return dict(first), dict(second)
    finally:
        path.unlink(missing_ok=True)


async def _guard_incoming_sync(
    *,
    job_id: str,
    card: TransitionCard,
    launch_delay_ms: int,
    outgoing_title: str,
    incoming_title: str,
) -> None:
    """Abort an inaudible incoming deck if live Sync did not take effect."""
    _set_sync_guard_status(job_id, {
        "status": "waiting",
        "launch_delay_ms": launch_delay_ms,
    })
    try:
        # Pre-spawn one isolated helper and let it sleep until just after the
        # incoming launch. This removes process startup from the four-bar
        # pre-audible window and reuses one warm UIA cache for both consensus
        # frames, so both observations finish before outgoing retirement.
        capture_at = time.monotonic() + launch_delay_ms / 1000.0 + 0.25
        first_status, second_status = await _guard_status_pair(
            capture_at_monotonic=capture_at,
        )
        correction = None
        bar_alignment = _bar_alignment_consensus(first_status, second_status)
        if (
            bar_alignment.get("verified") is True
            and float(bar_alignment.get("error_beats", 0)) > 0.15
        ):
            signed_error = float(bar_alignment.get("signed_error_beats", 0))
            jump_beats = round(abs(signed_error))
            if jump_beats in {1, 2} and abs(abs(signed_error) - jump_beats) <= 0.15:
                direction = "forward" if signed_error > 0 else "back"
                action = f"beat_jump_{jump_beats}_{direction}"
                messages = await engine.send_action(
                    action,
                    {"deck": card.incoming_deck},
                )
                await asyncio.sleep(0.45)
                corrected_first_status, corrected_status = await _guard_status_pair(
                    capture_at_monotonic=time.monotonic() + 0.1,
                )
                corrected_alignment = _bar_alignment_consensus(
                    corrected_first_status,
                    corrected_status,
                )
                correction = {
                    "action": action,
                    "messages": messages,
                    "before": bar_alignment,
                    "after": corrected_alignment,
                }
                second_status = {
                    **corrected_status,
                    "bar_alignment": corrected_alignment,
                }
        else:
            second_status = {
                **second_status,
                "bar_alignment": bar_alignment,
            }
        outgoing = _deck_record(second_status, card.outgoing_deck)
        incoming_first = _deck_record(first_status, card.incoming_deck)
        incoming = _deck_record(second_status, card.incoming_deck)
        errors = []
        if normalize_title(outgoing.get("title", "")) != normalize_title(
            outgoing_title
        ):
            errors.append("outgoing title changed before sync guard")
        if normalize_title(incoming.get("title", "")) != normalize_title(
            incoming_title
        ):
            errors.append("incoming title changed before sync guard")
        if not _transport_changed(incoming_first, incoming):
            errors.append("incoming transport did not start")
        if incoming.get("beat_sync_enabled") is not True:
            errors.append("incoming Beat Sync is not confirmed on")
        outgoing_bpm = outgoing.get("bpm")
        incoming_bpm = incoming.get("bpm")
        if outgoing_bpm is None or incoming_bpm is None:
            errors.append("live deck BPM is unavailable")
        elif abs(float(outgoing_bpm) - float(incoming_bpm)) > 0.05:
            errors.append(
                "live deck BPMs do not match "
                f"({float(outgoing_bpm):.2f} vs {float(incoming_bpm):.2f})"
            )
        bar_alignment = second_status.get("bar_alignment") or {}
        if bar_alignment.get("verified") is not True:
            errors.append("visible deck bar alignment could not be verified")
        elif float(bar_alignment.get("error_beats", 4.0)) > 0.15:
            errors.append(
                "visible deck bar alignment is off by "
                f"{float(bar_alignment['error_beats']):.2f} beats"
            )
        if errors:
            _cancel_for_sync_guard(job_id, errors)
            await engine.send_action(
                "channel_fader", {"deck": card.incoming_deck, "value": 0}
            )
            await engine.send_action(
                "eq_low", {"deck": card.incoming_deck, "value": -1}
            )
            await engine.send_action("cue", {"deck": card.incoming_deck})
            _set_sync_guard_status(job_id, {
                "status": "failed_safe",
                "errors": errors,
                "observed": second_status,
            })
            return
        _set_sync_guard_status(job_id, {
            "status": "passed",
            "errors": [],
            "observed": second_status,
            "bar_alignment_correction": correction,
        })
    except asyncio.CancelledError:
        _set_sync_guard_status(job_id, {"status": "cancelled"})
        raise
    except Exception as exc:
        errors = [f"sync guard error: {exc}"]
        _cancel_for_sync_guard(job_id, errors)
        await engine.send_action(
            "channel_fader", {"deck": card.incoming_deck, "value": 0}
        )
        await engine.send_action("cue", {"deck": card.incoming_deck})
        _set_sync_guard_status(job_id, {
            "status": "failed_safe",
            "errors": errors,
        })


def _bar_alignment_consensus(*statuses: dict[str, Any]) -> dict[str, Any]:
    """Require repeated waveform captures to agree before trusting or fixing."""
    alignments = [status.get("bar_alignment") or {} for status in statuses]
    if len(alignments) < 2 or any(
        alignment.get("verified") is not True for alignment in alignments
    ):
        return {
            "verified": False,
            "error": "two verified waveform alignment samples are required",
            "samples": alignments,
        }
    errors = [float(item.get("error_beats", 4.0)) for item in alignments]
    signed = [float(item.get("signed_error_beats", 4.0)) for item in alignments]
    if max(errors) - min(errors) > 0.12:
        return {
            "verified": False,
            "error": "waveform alignment samples disagree",
            "samples": alignments,
        }
    near_two = all(abs(value - 2.0) <= 0.15 for value in errors)
    if not near_two and max(signed) - min(signed) > 0.12:
        return {
            "verified": False,
            "error": "waveform alignment direction is unstable",
            "samples": alignments,
        }
    signed_error = sum(signed) / len(signed)
    if near_two:
        signed_error = 2.0
    return {
        "verified": True,
        "error_beats": round(sum(errors) / len(errors), 3),
        "signed_error_beats": round(signed_error, 3),
        "sample_count": len(alignments),
        "samples": alignments,
    }


def _arm_sync_guard(
    *,
    job_id: str,
    card: TransitionCard,
    events: list[dict[str, Any]],
    outgoing_title: str,
    incoming_title: str,
    expected_phrase_boundary_ms: int | None = None,
) -> dict[str, Any]:
    launches = [
        event
        for event in events
        if event.get("action") in {"play_pause", "hot_cue"}
        and event.get("parameters", {}).get("deck") == card.incoming_deck
    ]
    if len(launches) != 1:
        raise RuntimeError("sync guard requires exactly one incoming launch event")
    launch_delay_ms = int(launches[0]["at_ms"])
    expected_phrase_boundary_ms = (
        launch_delay_ms
        if expected_phrase_boundary_ms is None
        else int(expected_phrase_boundary_ms)
    )
    if launch_delay_ms != expected_phrase_boundary_ms:
        raise RuntimeError(
            "incoming transport is not scheduled on the exact phrase beat-1 "
            f"boundary ({launch_delay_ms} ms vs {expected_phrase_boundary_ms} ms)"
        )
    task = asyncio.create_task(
        _guard_incoming_sync(
            job_id=job_id,
            card=card,
            launch_delay_ms=launch_delay_ms,
            outgoing_title=outgoing_title,
            incoming_title=incoming_title,
        )
    )
    sync_guard_tasks.add(task)
    task.add_done_callback(sync_guard_tasks.discard)
    return {
        "status": "armed",
        "launch_delay_ms": launch_delay_ms,
        "phrase_beat_one_gate": {
            "verified": True,
            "transport_at_ms": launch_delay_ms,
            "phrase_boundary_at_ms": expected_phrase_boundary_ms,
        },
    }


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


def _card_incoming_launch(card: TransitionCard):
    """Return the single bar-zero incoming transport event.

    Rolling performance used to assume every track had a Hot Cue. Simple
    cuts/resets are also allowed to launch a stopped deck from a verified
    file-start phrase, so the atomic staging path must understand both forms.
    """
    launches = [
        event
        for event in card.events
        if event.bar_offset == 0
        and event.beat_offset == 0
        and event.action in {"hot_cue", "play_pause"}
        and event.parameters.get("deck") == card.incoming_deck
    ]
    if len(launches) != 1:
        raise RuntimeError("Transition card requires exactly one bar-0 incoming launch")
    return launches[0]


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
        _expected_track_title(card.outgoing_track_id)
    ):
        errors.append("outgoing deck title does not match the card")
    if normalize_title(incoming_second.get("title", "")) != normalize_title(
        _expected_track_title(card.incoming_track_id)
    ):
        errors.append("incoming deck title does not match the card")
    if _transport_changed(outgoing_first, outgoing_second):
        errors.append("outgoing deck is still playing after retirement")
    if not _transport_changed(incoming_first, incoming_second):
        errors.append("incoming deck did not start playing")
    stem_fields = {
        "stem_vocal": "stem_vocal_enabled",
        "stem_instrumental": "stem_instrumental_enabled",
        "stem_drums": "stem_drums_enabled",
    }
    for action, field in stem_fields.items():
        for deck in {
            event.parameters.get("deck")
            for event in card.events
            if event.action == action
        }:
            record = outgoing_second if deck == card.outgoing_deck else incoming_second
            if record.get(field) is not True:
                errors.append(f"deck {deck} {action} was not restored after transition")
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
        if result["verified"]:
            # Musical alignment is measured while both decks are running by
            # the pre-audible sync guard. Comparing the retired deck's old
            # clock with the incoming deck at completion creates false BPM and
            # bar faults after master transfer.
            guard_job_id = str((launch_clock or {}).get("job_id"))
            guard = sync_guard_status.get(guard_job_id)
            # UIA captures run off-loop to protect MIDI timing, but can finish
            # slightly after the final mixer event on a busy Rekordbox tree.
            # Completion verification already runs in a worker thread, so wait
            # here for the guard's bounded terminal result without delaying
            # any scheduled event.
            guard_deadline = time.monotonic() + 15.0
            while (
                card.beat_sync_required
                and (guard is None or guard.get("status") == "waiting")
                and time.monotonic() < guard_deadline
            ):
                time.sleep(0.025)
                guard = sync_guard_status.get(guard_job_id)
            if guard and guard.get("status") == "passed":
                alignment = (guard.get("observed") or {}).get("bar_alignment", {})
                result["sync"] = {
                    "verified": True,
                    "errors": [],
                    "beat_phase_error_ms": 0.0,
                    "bar_phase_error_beats": alignment.get("error_beats", 0.0),
                    "source": "pre_audible_sync_guard",
                }
            elif card.beat_sync_required:
                result["sync"] = {
                    "verified": False,
                    "errors": ["pre-audible sync guard did not pass"],
                    "source": "pre_audible_sync_guard",
                }
        launched_monotonic = (launch_clock or {}).get("incoming_monotonic")
        if result["verified"] and launched_monotonic is not None:
            profile = profile_store.get(card.incoming_track_id)
            launch = next(
                event
                for event in card.events
                if event.bar_offset == 0
                and event.beat_offset == 0
                and event.action in {"hot_cue", "play_pause"}
                and event.parameters.get("deck") == card.incoming_deck
            )
            if launch.action == "hot_cue":
                cue = int(launch.parameters["cue"])
                landmark = next(
                    item
                    for item in profile.landmarks
                    if item.cue == cue
                    and item.kind in {"mix_in", "phrase_start"}
                    and item.confidence in {"verified", "high"}
                )
            else:
                landmark = next(
                    item
                    for item in profile.landmarks
                    if item.kind in {"mix_in", "phrase_start"}
                    and item.bar == 1
                    and (item.beat or 1) == 1
                    and item.confidence in {"verified", "high"}
                )
            elapsed_beats = (time.monotonic() - launched_monotonic) * profile.bpm / 60.0
            start_track_beat = (landmark.bar - 1) * profile.time_signature + (
                landmark.beat or 1
            )
            total = start_track_beat - 1 + elapsed_beats
            within_bar = total % profile.time_signature
            observation = DeckObservation(
                deck=card.incoming_deck,
                track_id=card.incoming_track_id,
                title=profile.title,
                bpm=result["incoming"]["second"].get("bpm") or profile.bpm,
                playing=True,
                bar=int(total // profile.time_signature) + 1,
                beat=int(within_bar) + 1,
                track_beat=int(total) + 1,
                beat_phase=within_bar - int(within_bar),
                sync_enabled=result["incoming"]["second"].get("beat_sync_enabled"),
                quantize_enabled=result["incoming"]["second"].get("quantize_enabled"),
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
    if autonomous_runner is not None:
        autonomous_runner.stop()
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
        "autonomous_set": (
            autonomous_runner.public()
            if autonomous_runner is not None
            else {"active": False}
        ),
        "verified_rescue_loops": {
            str(deck): record for deck, record in verified_rescue_loops.items()
        },
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
    max_stretch_percent: float = 4.0,
    allow_incompatible: bool = False,
    outgoing_track_id: str | None = None,
    outgoing_start_bar: int = 1,
    incoming_start_bar: int = 1,
    overlap_bars: int = 16,
) -> dict[str, Any]:
    """
    Rank by Camelot, tempo, and an optional soft vocal-overlap penalty.
    """
    if outgoing_bpm <= 0:
        raise ValueError("outgoing_bpm must be positive")
    if not 0 <= max_bpm_delta <= 20:
        raise ValueError("max_bpm_delta must be between 0 and 20")
    if not 0 < max_stretch_percent <= 10:
        raise ValueError("max_stretch_percent must be between 0 and 10")
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
        stretch_percent = bpm_delta / outgoing_bpm * 100.0
        record = {
            **candidate.model_dump(),
            "bpm_delta": round(bpm_delta, 3),
            "stretch_percent": round(stretch_percent, 3),
            "harmonic": harmonic,
        }
        vocal = None
        if outgoing_track_id is not None:
            try:
                vocal = vocal_handoff(
                    profile_store.get(outgoing_track_id),
                    profile_store.get(candidate.track_id),
                    outgoing_start_bar=outgoing_start_bar,
                    incoming_start_bar=incoming_start_bar,
                    overlap_bars=overlap_bars,
                )
            except KeyError:
                vocal = {"available": False, "soft_penalty": 0, "blocking": False}
            record["vocal"] = vocal
        reasons = []
        if bpm_delta > max_bpm_delta:
            reasons.append(f"BPM delta {bpm_delta:.2f} exceeds {max_bpm_delta:.2f}")
        if stretch_percent > max_stretch_percent:
            reasons.append(
                f"tempo stretch {stretch_percent:.2f}% exceeds "
                f"{max_stretch_percent:.2f}%"
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
            + max(0.0, max_bpm_delta - bpm_delta)
            - float((vocal or {}).get("soft_penalty", 0)),
            3,
        )
        ranked.append(record)
    ranked.sort(key=lambda item: (-item["score"], item["bpm_delta"], item["title"]))
    return {
        "outgoing_bpm": outgoing_bpm,
        "outgoing_key": outgoing_key,
        "max_stretch_percent": max_stretch_percent,
        "ranked": ranked,
        "excluded": excluded,
        "vocal_clash_scoring": (
            "soft_penalty" if outgoing_track_id is not None else "not_requested"
        ),
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def observe_deck_state(observation: DeckObservation) -> dict[str, Any]:
    """Ingest a fresh native, MIDI, vision, or manual deck-clock observation."""
    if observation.source == "midi" and observation.confidence in {"high", "verified"}:
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


@mcp.tool(annotations={"readOnlyHint": True})
def plan_vocal_handoff(
    outgoing_track_id: str,
    incoming_track_id: str,
    outgoing_start_bar: int,
    incoming_start_bar: int,
    overlap_bars: int = 16,
) -> dict[str, Any]:
    """Use Rekordbox vocal analysis as a soft transition-planning signal."""
    if overlap_bars < 1 or overlap_bars > 64:
        raise ValueError("overlap_bars must be between 1 and 64")
    return vocal_handoff(
        profile_store.get(outgoing_track_id),
        profile_store.get(incoming_track_id),
        outgoing_start_bar=outgoing_start_bar,
        incoming_start_bar=incoming_start_bar,
        overlap_bars=overlap_bars,
    )


@mcp.tool(annotations={"readOnlyHint": True})
def prepare_track_cues(track_id: str) -> dict[str, Any]:
    """Plan an 8/16-bar pre-drop cue workflow for later Rekordbox verification."""
    return cue_preparation_plan(profile_store.get(track_id))


@mcp.tool(annotations={"readOnlyHint": True})
def analyze_incoming_bass_phrase(
    track_id: str,
    start_bar: int,
    window_bars: int = 8,
) -> dict[str, Any]:
    """Measure whether a phrase has enough low-end to carry a bass swap."""
    return bass_phrase_evidence(
        profile_store.get(track_id),
        start_bar,
        window_bars,
    )


@mcp.tool(annotations={"readOnlyHint": True})
def recommend_transition_fx(card: TransitionCard) -> dict[str, Any]:
    """Recommend mapped outgoing-deck FX plus an explicit cleanup tail."""
    return fx_recipe(card)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def prepare_set_session(name: str, track_ids: list[str]) -> dict[str, Any]:
    """Persist a rolling current/next/following queue across MCP restarts."""
    audit = profile_store.audit(track_ids)
    session = set_sessions.create(name, track_ids)
    return {"session": session, "track_audit": audit}


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def start_set_session() -> dict[str, Any]:
    """Mark the prepared rolling set active."""
    return set_sessions.start()


@mcp.tool(annotations={"readOnlyHint": True})
def set_session_status() -> dict[str, Any]:
    """Return the persistent three-track execution horizon."""
    return set_sessions.status()


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def settle_set_transition(job_id: str) -> dict[str, Any]:
    """Advance the rolling queue only after a transition job passes QA."""
    job = scheduler.get(job_id)
    qa = transition_qa(job)
    if job["status"] not in {"completed", "failed", "cancelled"}:
        raise RuntimeError("Transition job is not finished")
    session = set_sessions.advance(
        job_id,
        qa["passed"],
        "; ".join(qa["faults"]) or job.get("error"),
    )
    return {"session": session, "qa": qa}


@mcp.tool(annotations={"readOnlyHint": True})
def transition_quality_report(job_id: str) -> dict[str, Any]:
    """Grade timing, phase, postconditions, and dispatch health for one job."""
    return transition_qa(scheduler.get(job_id))


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


def _require_active_stem_preconditions(card: TransitionCard) -> dict[str, Any] | None:
    requested = {
        (
            int(event.parameters["deck"]),
            f"{event.action}_enabled",
        )
        for event in card.events
        if event.action in STEM_ACTIONS
    }
    if not requested:
        return None
    with deck_observer.exclusive_adapter():
        rekordbox_ui.invalidate_status_cache()
        status = rekordbox_ui.status()
    errors = []
    for deck, field in sorted(requested):
        if _deck_record(status, deck).get(field) is not True:
            errors.append(f"deck {deck} {field} is not visually confirmed active")
    if errors:
        raise RuntimeError("; ".join(errors))
    return status


def _accepted_transition_job(
    scheduled: dict[str, Any],
    execution_id: str,
) -> tuple[bool, str | None]:
    """Require proof that the scheduler owns a non-empty, reserved job."""
    if not scheduled.get("ready"):
        return False, "; ".join(
            scheduled.get("errors", ["transition compilation was not ready"])
        )
    job = scheduled.get("job")
    if not isinstance(job, dict) or not job.get("id"):
        return False, "transition response did not contain a scheduler job ID"
    if job.get("execution_id") != execution_id:
        return False, "scheduler job execution_id does not match the request"
    if job.get("status") not in {"scheduled", "running"}:
        return False, f"scheduler job is not active: {job.get('status')!r}"
    if int(job.get("event_count", 0)) <= 0:
        return False, "scheduler job contains no events"
    if job.get("control_reserved_until_monotonic") is None:
        return False, "scheduler job does not hold the live-control reservation"
    return True, None


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
    stem_preconditions = _require_active_stem_preconditions(card)
    launch_clock: dict[str, float] = {}

    def observe_dispatch(event: dict[str, Any], dispatched: float) -> None:
        if (
            event["action"] in {"hot_cue", "play_pause"}
            and event["parameters"].get("deck") == card.incoming_deck
        ):
            launch_clock["incoming_monotonic"] = dispatched

    job = scheduler.start(
        card.name,
        compiled["events"],
        completion_verifier=lambda: _card_completion_verifier(
            card,
            launch_clock,
        ),
        event_observer=observe_dispatch,
        execution_id=execution_id,
    )
    launch_clock["job_id"] = job["id"]
    guard = None
    if card.beat_sync_required:
        guard = _arm_sync_guard(
            job_id=job["id"],
            card=card,
            events=compiled["events"],
            outgoing_title=_expected_track_title(card.outgoing_track_id),
            incoming_title=_expected_track_title(card.incoming_track_id),
            expected_phrase_boundary_ms=int(compiled["start_delay_ms"]),
        )
    return {
        **compiled,
        "cue_verification": cue_verification,
        "stem_preconditions": stem_preconditions,
        "job": job,
        "sync_guard": guard,
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
    if action in {"sync", "quantize", *STEM_ACTIONS}:
        raise ValueError(
            f"{action} is a stateful toggle. Use "
            + (
                "ensure_stem_state with a fresh visual observation."
                if action in STEM_ACTIONS
                else "ensure_deck_modes with a fresh high-confidence observation."
            )
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
            first_snapshot_started_after_seconds = time.monotonic() - launched_at
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
        raise RuntimeError("Verified Hot Cue is not a high-confidence profile landmark")
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
    landmark_track_beat = (landmark.bar - 1) * profile.time_signature + (
        landmark.beat or 1
    )
    total = landmark_track_beat - 1 + elapsed_beats
    bar = int(total // profile.time_signature) + 1
    within_bar = total % profile.time_signature
    observation = DeckObservation(
        deck=deck,
        track_id=track_id,
        title=title,
        bpm=second.bpm or profile.bpm,
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
    entry = _opening_file_start_landmark(profile)
    with deck_observer.exclusive_adapter():
        initial = rekordbox_ui.deck_snapshot(deck)
        if normalize_title(initial.title) != normalize_title(title):
            raise RuntimeError(
                f"Deck {deck} loaded {initial.title!r}, expected {title!r}"
            )
        if rekordbox_ui.deck_is_playing(deck, initial=initial):
            raise RuntimeError(f"Deck {deck} is already playing")
        if initial.elapsed_seconds is None or initial.elapsed_seconds > 1:
            raise RuntimeError(f"Deck {deck} is not staged at the file start")
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
    entry_offset_beats = (entry.time_ms or 0) * profile.bpm / 60_000.0
    track_total = max(0.0, elapsed_beats - entry_offset_beats)
    grid_total = max(
        0.0,
        (entry.bar - 1) * profile.time_signature
        + entry.beat
        - 1
        + elapsed_beats
        - entry_offset_beats,
    )
    grid_whole = int(grid_total)
    within_bar = grid_total % profile.time_signature
    observation = DeckObservation(
        deck=deck,
        track_id=track_id,
        title=title,
        bpm=second.bpm or profile.bpm,
        playing=True,
        bar=grid_whole // profile.time_signature + 1,
        beat=int(within_bar) + 1,
        track_beat=int(track_total) + 1,
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


def _opening_file_start_landmark(profile: TrackProfile):
    """Return a proven phrase start at the beginning of an opening track.

    An opener may contain a short pickup before its first downbeat. Incoming
    transition launches remain governed by the stricter beat-one contract in
    transition-card validation.
    """
    entry = next(
        (
            item
            for item in profile.landmarks
            if item.kind in {"mix_in", "phrase_start"}
            and item.bar == 1
            and item.confidence in {"verified", "high"}
            and (
                (item.time_ms is not None and item.time_ms <= 1_000)
                or (item.time_ms is None and item.beat == 1)
            )
        ),
        None,
    )
    if entry is None:
        raise RuntimeError("Track has no high-confidence file-start phrase landmark")
    return entry


async def _establish_opening_mixer_contract(*, audible_deck: int) -> dict[str, Any]:
    """Put every absolute MIDI mixer control in a known opening state.

    Rekordbox's virtual-MIDI route has no feedback channel for absolute CCs.
    Reasserting the complete contract immediately before launch makes the
    commanded mixer state deterministic; visual deck state is verified by the
    UI observer separately.
    """
    if audible_deck not in (1, 2):
        raise ValueError("audible_deck must be 1 or 2")
    # The user's crossfader is disabled by design. Never send it as part of
    # automated mixing; channel faders own all level handoffs.
    actions: list[dict[str, Any]] = []
    for deck in (1, 2):
        values = {
            "channel_fader": 1 if deck == audible_deck else 0,
            "gain": 0,
            "eq_high": 0,
            "eq_mid": 0,
            "eq_low": 0,
            "filter": 0,
            "fx_wet_dry": 0,
        }
        actions.extend(
            {
                "action": action,
                "parameters": {"deck": deck, "value": value},
            }
            for action, value in values.items()
        )
    dispatched = []
    for item in actions:
        dispatched.append(
            {
                **item,
                "messages": await engine.send_action(
                    item["action"], item["parameters"]
                ),
            }
        )
    return {
        "audible_deck": audible_deck,
        "controls": dispatched,
        "fx_toggle_safety": (
            "FX wet/dry is zero on both decks; toggle position cannot make "
            "an effect audible until a scheduled wet/dry event occurs."
        ),
    }


async def _precondition_incoming_bass_blend(card: TransitionCard) -> dict[str, Any] | None:
    """Close the staged channel and its low EQ before any transport launch.

    Events at the same musical timestamp are deliberately dispatched together,
    so a bar-zero EQ event alone cannot prove it reached Rekordbox before the
    play/Hot Cue trigger.  Applying the absolute controls during staging makes
    the incoming launch deterministic and prevents a full-bass transient.
    """
    requires_closed_low = any(
        event.bar_offset == 0
        and event.beat_offset == 0
        and event.action == "eq_low"
        and event.parameters.get("deck") == card.incoming_deck
        and float(event.parameters.get("value", 0)) <= -0.9
        for event in card.events
    )
    if not requires_closed_low:
        return None
    controls = []
    for action, value in (("channel_fader", 0), ("eq_low", -1)):
        controls.append(
            {
                "action": action,
                "value": value,
                "messages": await engine.send_action(
                    action,
                    {"deck": card.incoming_deck, "value": value},
                ),
            }
        )
    return {"deck": card.incoming_deck, "controls": controls}


async def _reset_stopped_deck_tempo(deck: int) -> dict[str, Any]:
    """Acquire Rekordbox tempo soft takeover and return a stopped deck native."""
    messages = []
    for value in (-1.0, 1.0, 0.0):
        messages.extend(
            await engine.send_action("tempo", {"deck": deck, "value": value})
        )
        await asyncio.sleep(0.05)
    return {
        "deck": deck,
        "pickup_values": [-1.0, 1.0, 0.0],
        "messages": messages,
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def launch_and_schedule_opening_transition(
    card: TransitionCard,
    outgoing_title: str,
    incoming_title: str,
    execution_id: str,
    incoming_artist: str | None = None,
    incoming_cue: int | None = None,
    incoming_cue_time_ms: int | None = None,
    source: str | None = None,
    result_index: int | None = None,
) -> dict[str, Any]:
    """Atomically launch track one and schedule the first handoff.

    The set is not considered started until the incoming track/cue is prepared,
    both stopped decks are verified, the opener is launched at its native BPM,
    the audible deck has been established as Master, Sync/Quantize are then
    armed, and the first transition job is accepted.  Any post-launch failure
    immediately silences and stops both decks instead of allowing an
    unscheduled outgoing track to run to its end.
    """
    if not execution_id.strip():
        raise ValueError("execution_id is required")
    if card.anchor_deck != card.outgoing_deck:
        raise ValueError("opening transition must use the outgoing deck as anchor")
    if card.start_phrase_index is None:
        raise ValueError(
            "opening transition requires an explicit analyzed start_phrase_index"
        )
    incoming_launches = [
        event
        for event in card.events
        if event.bar_offset == 0
        and event.beat_offset == 0
        and event.action in {"play_pause", "hot_cue"}
        and event.parameters.get("deck") == card.incoming_deck
    ]
    if len(incoming_launches) != 1:
        raise ValueError(
            "opening transition requires exactly one bar-0 incoming launch"
        )
    incoming_launch = incoming_launches[0]
    if incoming_launch.action == "hot_cue":
        expected_cue = incoming_launch.parameters.get("cue")
        if incoming_cue != expected_cue:
            raise ValueError(
                "incoming_cue must match the card's bar-0 incoming Hot Cue"
            )
        if incoming_cue_time_ms is None:
            raise ValueError("incoming_cue_time_ms is required for a Hot Cue opening")
    elif incoming_cue is not None or incoming_cue_time_ms is not None:
        raise ValueError("incoming cue arguments require a bar-0 Hot Cue launch")
    outgoing_profile = profile_store.get(card.outgoing_track_id)
    incoming_profile = profile_store.get(card.incoming_track_id)
    # Track IDs bind the prepared analysis. Supplied canonical titles are
    # verified against the loaded Rekordbox decks, allowing legacy profiles
    # that omitted a mix/version suffix to remain usable.

    # Finish every expensive incoming-deck operation before audible playback.
    # _stage_track bypasses the browser when this exact track is already loaded.
    staged_incoming = await _stage_track(
        deck=card.incoming_deck,
        track_id=card.incoming_track_id,
        title=incoming_title,
        artist=incoming_artist,
        source=source,
        result_index=result_index,
    )

    with deck_observer.exclusive_adapter():
        first_status = rekordbox_ui.status()
        await asyncio.sleep(1.1)
        second_status = rekordbox_ui.status()
    for deck, title in (
        (card.outgoing_deck, outgoing_title),
        (card.incoming_deck, incoming_title),
    ):
        first = _deck_record(first_status, deck)
        second = _deck_record(second_status, deck)
        if normalize_title(second.get("title", "")) != normalize_title(title):
            raise RuntimeError(f"Deck {deck} loaded the wrong opening track")
        if _transport_changed(first, second):
            raise RuntimeError(f"Deck {deck} is already playing")
        requires_file_start = not (
            deck == card.incoming_deck and incoming_launch.action == "hot_cue"
        )
        if requires_file_start and (
            second.get("elapsed_seconds") is None or second["elapsed_seconds"] > 1
        ):
            raise RuntimeError(f"Deck {deck} is not staged at file start")

    audio_route = rekordbox_ui.ensure_pc_master_out(False)
    mixer_contract = await _establish_opening_mixer_contract(
        audible_deck=card.outgoing_deck
    )
    incoming_mixer_precondition = await _precondition_incoming_bass_blend(card)

    for deck, profile, title in (
        (card.outgoing_deck, outgoing_profile, outgoing_title),
        (card.incoming_deck, incoming_profile, incoming_title),
    ):
        record = _deck_record(second_status, deck)
        live_state.update(
            observation_from_elapsed(
                deck=deck,
                profile=profile,
                elapsed_seconds=record["elapsed_seconds"],
                playing=False,
                sync_enabled=record.get("beat_sync_enabled"),
                quantize_enabled=record.get("quantize_enabled"),
                title=title,
                playback_bpm=record.get("bpm"),
            )
        )

    master_messages: list[Any] = []
    master_state: dict[str, Any] | None = None
    # Never launch the first record while another stopped deck can still own
    # Rekordbox's Sync tempo.  Master assignment is ignored on stopped decks,
    # so both decks must have Sync explicitly off until the opener is moving
    # at its analyzed/native BPM and has become Master.
    outgoing_modes = await ensure_deck_modes(
        deck=card.outgoing_deck,
        beat_sync=False if card.beat_sync_required else None,
        quantize=True if card.quantize_required else None,
    )
    incoming_modes = await ensure_deck_modes(
        deck=card.incoming_deck,
        beat_sync=False if card.beat_sync_required else None,
        quantize=True if card.quantize_required else None,
    )
    if outgoing_modes["changed"] or incoming_modes["changed"]:
        await asyncio.sleep(1.2)

    # Tempo pickup must happen after Sync is visibly off. A stopped Sync slave
    # ignores its pitch CCs and can otherwise inherit the other deck's BPM at
    # launch even though the control was commanded to center.
    tempo_resets = [
        await _reset_stopped_deck_tempo(card.outgoing_deck),
        await _reset_stopped_deck_tempo(card.incoming_deck),
    ]
    await asyncio.sleep(0.2)

    with deck_observer.exclusive_adapter():
        mode_status = rekordbox_ui.status()
    for deck, profile, title in (
        (card.outgoing_deck, outgoing_profile, outgoing_title),
        (card.incoming_deck, incoming_profile, incoming_title),
    ):
        record = _deck_record(mode_status, deck)
        live_state.update(
            observation_from_elapsed(
                deck=deck,
                profile=profile,
                elapsed_seconds=record["elapsed_seconds"],
                playing=False,
                sync_enabled=record.get("beat_sync_enabled"),
                quantize_enabled=record.get("quantize_enabled"),
                title=title,
                playback_bpm=record.get("bpm"),
            )
        )
    outgoing_state = live_state.get(card.outgoing_deck)
    incoming_state = live_state.get(card.incoming_deck)
    mode_errors = []
    if card.beat_sync_required:
        if outgoing_state["sync_enabled"] is not False:
            mode_errors.append("opening deck Beat Sync is not confirmed off")
        if incoming_state["sync_enabled"] is not False:
            mode_errors.append("staged deck Beat Sync is not confirmed off")
    if card.quantize_required:
        if outgoing_state["quantize_enabled"] is not True:
            mode_errors.append("outgoing Quantize is not confirmed on")
        if incoming_state["quantize_enabled"] is not True:
            mode_errors.append("incoming Quantize is not confirmed on")
    observed_opening_bpm = _deck_record(
        mode_status, card.outgoing_deck
    ).get("bpm")
    if observed_opening_bpm is None or abs(
        float(observed_opening_bpm) - float(outgoing_profile.bpm)
    ) > 0.05:
        mode_errors.append(
            f"opening deck stopped tempo is not native "
            f"({observed_opening_bpm!r} vs {float(outgoing_profile.bpm):.2f})"
        )
    if mode_errors:
        return {
            "ready": False,
            "started": False,
            "errors": mode_errors,
            "mixer_contract": mixer_contract,
            "incoming_mixer_precondition": incoming_mixer_precondition,
            "audio_route": audio_route,
            "master_messages": master_messages,
            "outgoing_modes": outgoing_modes,
            "incoming_modes": incoming_modes,
            "tempo_resets": tempo_resets,
            "observed": mode_status,
            "stage": staged_incoming,
        }

    cue_verification = None
    if incoming_launch.action == "hot_cue":
        cue_verification = await verify_hot_cue(
            deck=card.incoming_deck,
            track_id=card.incoming_track_id,
            title=incoming_title,
            cue=int(incoming_cue),
            expected_time_ms=int(incoming_cue_time_ms),
        )
        if not cue_verification.get("verified"):
            return {
                "ready": False,
                "started": False,
                "errors": ["incoming Hot Cue verification failed"],
                "stage": staged_incoming,
                "cue_verification": cue_verification,
                "mixer_contract": mixer_contract,
                "incoming_mixer_precondition": incoming_mixer_precondition,
                "audio_route": audio_route,
            }

    launch = await launch_staged_track(
        deck=card.outgoing_deck,
        track_id=card.outgoing_track_id,
        title=outgoing_title,
    )
    # Rekordbox ignores Master assignment on a stopped deck.  Establish the
    # reference only after transport is proven moving, then compile immediately.
    scheduled: dict[str, Any] | None = None
    try:
        master_state = await ensure_master_deck(card.outgoing_deck)
        master_messages = master_state.get("messages", [])
        native_refresh = await refresh_transition_state(
            outgoing_deck=card.outgoing_deck,
            outgoing_track_id=card.outgoing_track_id,
            outgoing_title=outgoing_title,
            incoming_deck=card.incoming_deck,
            incoming_track_id=card.incoming_track_id,
            incoming_title=incoming_title,
            incoming_hot_cue=(
                int(incoming_cue) if incoming_launch.action == "hot_cue" else None
            ),
        )
        native_bpm = float(native_refresh["outgoing"].get("bpm", outgoing_profile.bpm))
        if abs(native_bpm - float(outgoing_profile.bpm)) > 0.05:
            raise RuntimeError(
                "opening deck did not start at its native BPM "
                f"({native_bpm:.2f} vs {float(outgoing_profile.bpm):.2f})"
            )
        # With the opener now authoritative, enabling Sync cannot inherit the
        # incoming track's native tempo.  Refresh once more after any toggles
        # so the transition compiler receives a current, verified clock.
        outgoing_modes = await ensure_deck_modes(
            deck=card.outgoing_deck,
            beat_sync=True if card.beat_sync_required else None,
            quantize=True if card.quantize_required else None,
        )
        incoming_modes = await ensure_deck_modes(
            deck=card.incoming_deck,
            beat_sync=True if card.beat_sync_required else None,
            quantize=True if card.quantize_required else None,
        )
        if outgoing_modes["changed"] or incoming_modes["changed"]:
            await asyncio.sleep(1.2)
        refreshed = await refresh_transition_state(
            outgoing_deck=card.outgoing_deck,
            outgoing_track_id=card.outgoing_track_id,
            outgoing_title=outgoing_title,
            incoming_deck=card.incoming_deck,
            incoming_track_id=card.incoming_track_id,
            incoming_title=incoming_title,
            incoming_hot_cue=(
                int(incoming_cue) if incoming_launch.action == "hot_cue" else None
            ),
        )
        scheduled = await perform_transition_card(
            card=card,
            commit=True,
            execution_id=execution_id,
        )
        accepted, acceptance_error = _accepted_transition_job(
            scheduled,
            execution_id,
        )
        if not accepted:
            raise RuntimeError(
                acceptance_error or "first transition job was not accepted"
            )
    except Exception as exc:
        # No unowned playback: if the scheduler is not conclusively holding
        # the handoff, silence and stop both decks immediately.
        cancelled_job = None
        job_id = (
            scheduled.get("job", {}).get("id")
            if isinstance(scheduled, dict) and isinstance(scheduled.get("job"), dict)
            else None
        )
        if job_id:
            try:
                cancelled_job = scheduler.cancel(job_id)
            except KeyError:
                cancelled_job = {"id": job_id, "status": "not_found"}
        abort_messages = []
        for deck in (card.outgoing_deck, card.incoming_deck):
            abort_messages.extend(
                await engine.send_action("channel_fader", {"deck": deck, "value": 0})
            )
            abort_messages.extend(await engine.send_action("cue", {"deck": deck}))
        return {
            "ready": False,
            "started": False,
            "aborted_after_launch": True,
            "errors": [str(exc)],
            "stage": staged_incoming,
            "cue_verification": cue_verification,
            "mixer_contract": mixer_contract,
            "incoming_mixer_precondition": incoming_mixer_precondition,
            "audio_route": audio_route,
            "launch": launch,
            "cancelled_job": cancelled_job,
            "abort_messages": abort_messages,
        }
    return {
        "ready": True,
        "started": True,
        "session": set_sessions.start(),
        "mixer_contract": mixer_contract,
        "incoming_mixer_precondition": incoming_mixer_precondition,
        "audio_route": audio_route,
        "master_messages": master_messages,
        "master_state": master_state,
        "opening_native_bpm": float(outgoing_profile.bpm),
        "outgoing_modes": outgoing_modes,
        "incoming_modes": incoming_modes,
        "tempo_resets": tempo_resets,
        "stage": staged_incoming,
        "cue_verification": cue_verification,
        "launch": launch,
        "refresh": refreshed,
        "schedule": scheduled,
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def ensure_pc_master_out(enabled: bool = False) -> dict[str, Any]:
    """Observe and set Rekordbox PC MASTER OUT without changing deck audio."""
    return rekordbox_ui.ensure_pc_master_out(enabled)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def ensure_master_deck(deck: int) -> dict[str, Any]:
    """Make one playing deck the sole observable Rekordbox Master."""
    if deck not in (1, 2):
        raise ValueError("deck must be 1 or 2")
    other = 2 if deck == 1 else 1
    with deck_observer.exclusive_adapter():
        rekordbox_ui.invalidate_status_cache()
        before = rekordbox_ui.status()
    target_before = _deck_record(before, deck).get("master_enabled")
    other_before = _deck_record(before, other).get("master_enabled")
    messages: list[Any] = []
    if target_before is not True or other_before is not False:
        messages = await engine.send_action("master", {"deck": deck})
        await asyncio.sleep(0.45)
    with deck_observer.exclusive_adapter():
        rekordbox_ui.invalidate_status_cache()
        after = rekordbox_ui.status()
    target_after = _deck_record(after, deck).get("master_enabled")
    other_after = _deck_record(after, other).get("master_enabled")
    if target_after is not True or other_after is not False:
        raise RuntimeError(
            f"Deck {deck} is not the sole visually confirmed Rekordbox Master"
        )
    return {
        "deck": deck,
        "before": {"target": target_before, "other": other_before},
        "after": {"target": target_after, "other": other_after},
        "changed": bool(messages),
        "messages": messages,
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

    outgoing_elapsed = outgoing_second.get("elapsed_seconds")
    if outgoing_elapsed is None:
        raise RuntimeError("Outgoing deck elapsed time is unavailable")
    # The visible deck clock is integer-second resolution.  Rebuilding the
    # running musical clock from it can throw away almost one second of phase
    # (two beats at 120 BPM), which then makes an otherwise correct phrase
    # target launch late.  The outgoing deck already owns a native monotonic
    # launch clock; preserve its extrapolated beat/phase and refresh only the
    # visually observed mode/BPM fields.
    native_clock = live_state.get(outgoing_deck)
    outgoing_observation = DeckObservation(
        deck=outgoing_deck,
        track_id=outgoing_track_id,
        title=outgoing_title,
        bpm=outgoing_second.get("bpm") or native_clock["bpm"],
        playing=True,
        bar=native_clock["bar"],
        beat=native_clock["beat"],
        track_beat=native_clock["track_beat"],
        beat_phase=native_clock["beat_phase"],
        sync_enabled=outgoing_second.get("beat_sync_enabled"),
        quantize_enabled=outgoing_second.get("quantize_enabled"),
        source="native",
        confidence="high",
    )
    incoming_profile = profile_store.get(incoming_track_id)
    incoming_elapsed = incoming_second.get("elapsed_seconds")
    if incoming_elapsed is None:
        raise RuntimeError("Incoming deck elapsed time is unavailable")
    incoming_observation = observation_from_elapsed(
        deck=incoming_deck,
        profile=incoming_profile,
        elapsed_seconds=incoming_elapsed,
        playing=False,
        sync_enabled=incoming_second.get("beat_sync_enabled"),
        quantize_enabled=incoming_second.get("quantize_enabled"),
        title=incoming_title,
        playback_bpm=incoming_second.get("bpm"),
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
async def ensure_stem_state(
    deck: int,
    vocal: bool | None = None,
    instrumental: bool | None = None,
    drums: bool | None = None,
) -> dict[str, Any]:
    """Set Rekordbox stem-part state only after observing the current buttons."""
    if deck not in (1, 2):
        raise ValueError("deck must be 1 or 2")
    desired = {
        "stem_vocal": vocal,
        "stem_instrumental": instrumental,
        "stem_drums": drums,
    }
    if all(value is None for value in desired.values()):
        raise ValueError("Request vocal, instrumental, drums, or a combination")
    with deck_observer.exclusive_adapter():
        rekordbox_ui.invalidate_status_cache()
        before_status = rekordbox_ui.status()
    before = _deck_record(before_status, deck)
    actions = []
    for action, target in desired.items():
        if target is None:
            continue
        field = f"{action}_enabled"
        actual = before.get(field)
        if actual is None:
            raise RuntimeError(f"deck {deck} {field} is not visually observable")
        if actual != target:
            actions.append(
                {
                    "action": action,
                    "from": actual,
                    "expected": target,
                    "messages": await engine.send_action(action, {"deck": deck}),
                }
            )
    if actions:
        await asyncio.sleep(0.35)
    with deck_observer.exclusive_adapter():
        after_status = rekordbox_ui.status()
    after = _deck_record(after_status, deck)
    errors = []
    for action, target in desired.items():
        if target is None:
            continue
        field = f"{action}_enabled"
        if after.get(field) is not target:
            errors.append(f"deck {deck} {field} did not reach requested state {target}")
    if errors:
        raise RuntimeError("; ".join(errors))
    return {
        "deck": deck,
        "actions": actions,
        "changed": bool(actions),
        "observed_before": before,
        "observed_after": after,
        "verified": True,
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
    execution_id: str,
    incoming_cue: int | None = None,
    incoming_cue_time_ms: int | None = None,
    source: str | None = None,
    result_index: int | None = None,
) -> dict[str, Any]:
    """Stage, verify, mode-check, and schedule one handoff without agent delay.

    This is the live-path primitive for a rolling two-deck set.  All expensive
    UI work happens inside one MCP request, so model/tool round trips cannot
    consume the outgoing track's remaining phrase runway.
    """
    # Stable track_id is authoritative. Rekordbox's canonical display title
    # may include a mix suffix omitted by an older prepared profile; staging
    # verifies the supplied title/artist against the actual loaded deck.
    profile_store.get(card.incoming_track_id)
    launch = _card_incoming_launch(card)
    if launch.action == "hot_cue":
        if incoming_cue != int(launch.parameters["cue"]):
            raise ValueError(
                "incoming_cue must match the card's bar-0 incoming Hot Cue"
            )
        if incoming_cue_time_ms is None:
            raise ValueError("incoming_cue_time_ms is required for a Hot Cue launch")
    elif incoming_cue is not None or incoming_cue_time_ms is not None:
        raise ValueError("incoming cue arguments require a bar-0 Hot Cue launch")
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
    incoming_mixer_precondition = await _precondition_incoming_bass_blend(card)
    cue_verification = None
    if launch.action == "hot_cue":
        cue_verification = await verify_hot_cue(
            deck=card.incoming_deck,
            track_id=card.incoming_track_id,
            title=incoming_title,
            cue=int(incoming_cue),
            expected_time_ms=int(incoming_cue_time_ms),
        )
        if not cue_verification.get("verified"):
            raise RuntimeError("Incoming Hot Cue verification failed")

    outgoing_title = _expected_track_title(card.outgoing_track_id)
    master_result = None
    outgoing_mode_result = None
    if card.beat_sync_required:
        # Establish the audible deck as Rekordbox's tempo/phase authority before
        # enabling Sync on the staged deck.  A lit incoming Sync button alone is
        # insufficient: Rekordbox can leave the stopped deck at its native BPM.
        # Never blindly toggle Master: if the audible deck is already Master,
        # a toggle would turn it off and invalidate every compiled timestamp.
        master_result = await ensure_master_deck(card.outgoing_deck)
    # Master verification is an expensive UIA operation.  Capture the native
    # transport clock *after* it, otherwise the observation is already several
    # seconds stale by the time ensure_deck_modes runs.
    refresh_before = await refresh_transition_state(
        outgoing_deck=card.outgoing_deck,
        outgoing_track_id=card.outgoing_track_id,
        outgoing_title=outgoing_title,
        incoming_deck=card.incoming_deck,
        incoming_track_id=card.incoming_track_id,
        incoming_title=incoming_title,
        incoming_hot_cue=(int(incoming_cue) if launch.action == "hot_cue" else None),
    )
    if card.beat_sync_required:
        outgoing_mode_result = await ensure_deck_modes(
            deck=card.outgoing_deck,
            beat_sync=True,
            quantize=True if card.quantize_required else None,
        )
    mode_result = await ensure_deck_modes(
        deck=card.incoming_deck,
        beat_sync=True if card.beat_sync_required else None,
        quantize=True if card.quantize_required else None,
    )
    if mode_result["changed"] or (
        outgoing_mode_result is not None and outgoing_mode_result["changed"]
    ):
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
        incoming_hot_cue=(int(incoming_cue) if launch.action == "hot_cue" else None),
    )
    scheduled = await perform_transition_card(
        card=card,
        commit=True,
        execution_id=execution_id,
    )
    accepted, acceptance_error = _accepted_transition_job(
        scheduled,
        execution_id,
    )
    if not accepted:
        cancelled_job = None
        job_id = (
            scheduled.get("job", {}).get("id")
            if isinstance(scheduled.get("job"), dict)
            else None
        )
        if job_id:
            try:
                cancelled_job = scheduler.cancel(job_id)
            except KeyError:
                cancelled_job = {"id": job_id, "status": "not_found"}
        return {
            "ready": False,
            "errors": [acceptance_error],
            "stage": staged,
            "incoming_mixer_precondition": incoming_mixer_precondition,
            "cue_verification": cue_verification,
            "refresh_before": refresh_before,
            "mode_result": mode_result,
            "outgoing_mode_result": outgoing_mode_result,
            "master_result": master_result,
            "refresh_after": refresh_after,
            "schedule": scheduled,
            "cancelled_job": cancelled_job,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }
    return {
        "ready": True,
        "stage": staged,
        "incoming_mixer_precondition": incoming_mixer_precondition,
        "cue_verification": cue_verification,
        "mode_result": mode_result,
        "outgoing_mode_result": outgoing_mode_result,
        "master_result": master_result,
        "refresh": refresh_after,
        "schedule": scheduled,
        "elapsed_ms": round((time.monotonic() - started) * 1000),
    }


async def _runner_schedule(
    option: TransitionOption,
    release_rescue_loop: bool,
) -> dict[str, Any]:
    card = (
        _with_rescue_loop_release(option.card) if release_rescue_loop else option.card
    )
    fx_preparation = None
    if option.fx_effect is not None:
        try:
            card, fx_preparation = await _prepare_option_fx(card, option.fx_effect)
        except Exception as exc:  # noqa: BLE001 - optional live decoration
            # Beat FX are never allowed to own transport continuity.  Any UI
            # observation, selector, or MIDI-preparation failure preserves the
            # already validated dry card and proceeds with the handoff.
            fx_preparation = {
                "verified": False,
                "desired": option.fx_effect,
                "error": str(exc),
                "fallback": "dry transition card",
            }
    result = await stage_and_schedule_transition_card(
        card=card,
        incoming_title=option.incoming.title,
        incoming_artist=option.incoming.artist,
        execution_id=(f"autonomous-{option.id}-{int(time.time() * 1000)}"),
        incoming_cue=option.incoming.cue,
        incoming_cue_time_ms=option.incoming.cue_time_ms,
        source=option.incoming.source,
        result_index=option.incoming.result_index,
    )
    if not result.get("ready"):
        return result
    if release_rescue_loop:
        verified_rescue_loops.pop(card.outgoing_deck, None)
    scheduled = dict(result.get("schedule") or {})
    scheduled["ready"] = True
    scheduled["atomic_stage"] = result
    scheduled["fx_preparation"] = fx_preparation
    return scheduled


async def _prepare_option_fx(
    card: TransitionCard,
    desired_effect: str,
) -> tuple[TransitionCard, dict[str, Any]]:
    """Select and verify one outgoing Beat FX, or preserve the dry card."""
    deck = card.outgoing_deck
    await engine.send_action("fx_wet_dry", {"deck": deck, "value": 0})
    try:
        with deck_observer.exclusive_adapter():
            observed = rekordbox_ui.fx_effect(deck)
    except Exception as exc:  # noqa: BLE001 - optional effect observation
        return card, {
            "verified": False,
            "desired": desired_effect,
            "error": str(exc),
            "fallback": "dry transition card",
        }
    visited = [observed]
    for _ in range(24):
        if observed == desired_effect:
            break
        await engine.send_action("fx_select_next", {"deck": deck})
        await asyncio.sleep(0.12)
        try:
            with deck_observer.exclusive_adapter():
                observed = rekordbox_ui.fx_effect(deck)
        except Exception as exc:  # noqa: BLE001 - optional effect observation
            return card, {
                "verified": False,
                "desired": desired_effect,
                "error": str(exc),
                "visited": visited,
                "fallback": "dry transition card",
            }
        if observed in visited:
            break
        visited.append(observed)
    if observed != desired_effect:
        return card, {
            "verified": False,
            "desired": desired_effect,
            "observed": observed,
            "visited": visited,
            "fallback": "dry transition card",
        }
    recipe = fx_recipe(card)
    if recipe["effect"] != desired_effect:
        return card, {
            "verified": False,
            "desired": desired_effect,
            "observed": observed,
            "fallback": "recipe mismatch; dry transition card",
        }
    fx_events = [MusicalEvent.model_validate(item) for item in recipe["events"]]
    updated = sorted(
        [*card.events, *fx_events],
        key=lambda event: (event.bar_offset, event.beat_offset),
    )
    prepared = card.model_copy(update={"events": updated})
    outgoing = profile_store.get(card.outgoing_track_id)
    incoming = profile_store.get(card.incoming_track_id)
    errors = validate_transition_card(prepared, outgoing, incoming)
    if errors:
        return card, {
            "verified": False,
            "desired": desired_effect,
            "observed": observed,
            "errors": errors,
            "fallback": "FX card validation failed; dry transition card",
        }
    return prepared, {
        "verified": True,
        "desired": desired_effect,
        "observed": observed,
        "visited": visited,
    }


async def _runner_prestage(option: TransitionOption) -> dict[str, Any]:
    """Load the following track as soon as its deck is retired.

    This intentionally stops before mode/cue verification and card compilation.
    Those checks remain atomic in ``stage_and_schedule_transition_card`` after
    any post-handoff tempo ramp has completed.  The early pass removes browser
    work from the live transition window without compiling against a changing
    BPM clock.
    """
    staged = await _stage_track_isolated(
        deck=option.card.incoming_deck,
        track_id=option.incoming.track_id,
        title=option.incoming.title,
        artist=option.incoming.artist,
        source=option.incoming.source,
        result_index=option.incoming.result_index,
    )
    return {
        "ready": staged.get("verified") is True,
        "option_id": option.id,
        "stage": staged,
        "errors": (
            []
            if staged.get("verified") is True
            else ["following track could not be verified after early staging"]
        ),
    }


async def _stage_track_isolated(
    *,
    deck: int,
    track_id: str,
    title: str,
    artist: str | None = None,
    source: str | None = None,
    result_index: int | None = None,
    timeout_seconds: float = 50.0,
) -> dict[str, Any]:
    """Prestage through a killable helper so UIA cannot freeze the runner."""
    if result_index is not None:
        # The standalone planner currently uses exact local-library metadata
        # and never needs an index. Preserve the explicit-index route in the
        # atomic in-process staging path.
        return await _stage_track(
            deck=deck,
            track_id=track_id,
            title=title,
            artist=artist,
            source=source,
            result_index=result_index,
        )
    preload_messages = []
    preload_messages.extend(
        await engine.send_action("channel_fader", {"deck": deck, "value": 0})
    )
    preload_messages.extend(await engine.send_action("cue", {"deck": deck}))
    await asyncio.sleep(0.1)
    handle, raw_path = tempfile.mkstemp(
        prefix="stage-track-",
        suffix=".json",
        dir=profile_store.data_dir,
    )
    os.close(handle)
    path = Path(raw_path)
    if getattr(sys, "frozen", False):
        command = [
            sys.executable,
            "--stage-capture",
            str(path),
            "--stage-deck",
            str(deck),
            "--stage-title",
            title,
        ]
        if artist:
            command.extend(["--stage-artist", artist])
    else:
        command = [
            sys.executable,
            "-m",
            "rekordbox_performer.stage_capture",
            str(path),
            "--deck",
            str(deck),
            "--title",
            title,
        ]
        if artist:
            command.extend(["--artist", artist])
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise RuntimeError(
                f"bounded staging timed out after {timeout_seconds:.0f}s"
            ) from exc
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("ok") is not True:
            if payload.get("auto_started"):
                await engine.send_action(
                    "channel_fader", {"deck": deck, "value": 0}
                )
                await engine.send_action("cue", {"deck": deck})
            raise RuntimeError(payload.get("error") or "isolated staging failed")
        observed = dict(payload.get("observed") or {})
        track_title_aliases[track_id] = title
        deck_observer.invalidate()
        return {
            "selection": payload.get("selection"),
            "deck": deck,
            "track_id": track_id,
            "title": title,
            "artist": artist,
            "source": source,
            "messages": [],
            "preload_stop_messages": preload_messages,
            "observed": observed,
            "transport_verified_stopped": True,
            "attempts": payload.get("attempts", []),
            "route_cache_hit": False,
            "already_loaded": bool(payload.get("already_loaded")),
            "browser_search_skipped": bool(
                payload.get("browser_search_skipped")
            ),
            "isolated": True,
            "verified": True,
        }
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        path.unlink(missing_ok=True)


async def _runner_remaining_bars(track_id: str, deck: int) -> float:
    profile = profile_store.get(track_id)
    try:
        state = live_state.get(deck)
    except KeyError:
        state = await _observe_live_deck(
            deck=deck,
            track_id=track_id,
            title=_expected_track_title(track_id),
        )
    if state.get("track_id") != track_id:
        raise RuntimeError("live deck does not match autonomous-set state")
    end_beat = profile.beat_count or max(
        (point.index for point in profile.beat_grid),
        default=0,
    )
    if end_beat <= 0:
        raise RuntimeError("track has no analyzed end position")
    return max(
        0.0,
        (float(end_beat) - float(state["track_beat"])) / profile.time_signature,
    )


def _runner_advance(
    job_id: str,
    succeeded: bool,
    error: str | None,
) -> dict[str, Any]:
    return set_sessions.advance(job_id, succeeded, error)


def _get_autonomous_runner() -> AutonomousSetRunner:
    global autonomous_runner
    if autonomous_runner is None:
        autonomous_runner = AutonomousSetRunner(
            profile_store.data_dir / "autonomous-set.json",
            schedule=_runner_schedule,
            prestage=_runner_prestage,
            job_status=scheduler.get,
            job_qa=lambda job_id: transition_qa(scheduler.get(job_id)),
            remaining_bars=_runner_remaining_bars,
            engage_loop=_engage_rescue_loop,
            run_tempo=_execute_tempo_plan,
            advance=_runner_advance,
            finish=lambda _status: engine.release_set_control(),
            recover_route=autonomous_recovery_planner,
        )
    return autonomous_runner


def register_autonomous_recovery_planner(callback: Any) -> None:
    """Install the standalone app's local replacement-route planner."""
    global autonomous_recovery_planner
    autonomous_recovery_planner = callback
    if autonomous_runner is not None:
        autonomous_runner.recover_route = callback


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def engage_rescue_loop(deck: int, beats: int = 16) -> dict[str, Any]:
    """Engage and transport-verify a phrase-aligned emergency loop."""
    return await _engage_rescue_loop(deck, beats)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def release_rescue_loop(deck: int) -> dict[str, Any]:
    """Release only a rescue loop that this process verified as active."""
    record = verified_rescue_loops.get(deck)
    if record is None or record.get("verified") is not True:
        raise RuntimeError("no verified rescue loop is active on this deck")
    messages = await engine.send_action("loop_toggle", {"deck": deck})
    verified_rescue_loops.pop(deck, None)
    return {
        "verified": True,
        "released": True,
        "deck": deck,
        "messages": messages,
    }


@mcp.tool(annotations={"readOnlyHint": True})
def rescue_loop_status() -> dict[str, Any]:
    """Return loops proven active by transport repetition in this process."""
    return {
        "active": bool(verified_rescue_loops),
        "decks": {str(deck): record for deck, record in verified_rescue_loops.items()},
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def execute_tempo_plan(
    plan: TempoPlan,
    deck: int,
    track_id: str,
) -> dict[str, Any]:
    """Run a verified gradual BPM trajectory on the active Master deck."""
    return await _execute_tempo_plan(plan, deck, track_id)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def preflight_autonomous_set(plan: AutonomousSetPlan) -> dict[str, Any]:
    """Validate every branch and warm exact Rekordbox load/cue routes."""
    errors: list[str] = []
    try:
        resolved_plan = _materialize_autonomous_plan(plan)
    except (KeyError, ValueError) as exc:
        return {"ready": False, "errors": [f"tempo arc: {exc}"]}
    unique_track_ids = {resolved_plan.opening.track_id}
    try:
        _opening_file_start_landmark(
            profile_store.get(resolved_plan.opening.track_id)
        )
    except (KeyError, RuntimeError) as exc:
        errors.append(f"opening: {exc}")
    for option in resolved_plan.transitions:
        unique_track_ids.add(option.incoming.track_id)
        try:
            outgoing = profile_store.get(option.card.outgoing_track_id)
            incoming = profile_store.get(option.card.incoming_track_id)
        except KeyError as exc:
            errors.append(f"{option.id}: {exc}")
            continue
        errors.extend(
            f"{option.id}: {error}"
            for error in validate_transition_card(
                option.card,
                outgoing,
                incoming,
            )
        )
    audit = profile_store.audit(sorted(unique_track_ids))
    if errors:
        return {"ready": False, "errors": errors, "audit": audit}

    warmed = []
    seen: set[tuple[str, int]] = set()
    for option in sorted(resolved_plan.transitions, key=lambda item: item.priority):
        key = (option.incoming.track_id, option.card.incoming_deck)
        if key in seen:
            continue
        seen.add(key)
        try:
            staged = await _stage_track(
                deck=option.card.incoming_deck,
                track_id=option.incoming.track_id,
                title=option.incoming.title,
                artist=option.incoming.artist,
                source=option.incoming.source,
                result_index=option.incoming.result_index,
            )
        except Exception as exc:
            raise RuntimeError(f"{option.id}: staging failed: {exc}") from exc
        cue = None
        if option.incoming.cue is not None:
            cue = await verify_hot_cue(
                deck=option.card.incoming_deck,
                track_id=option.incoming.track_id,
                title=option.incoming.title,
                cue=option.incoming.cue,
                expected_time_ms=int(option.incoming.cue_time_ms),
            )
            if cue.get("verified") is not True:
                raise RuntimeError(f"{option.id}: incoming Hot Cue verification failed")
        warmed.append({"option_id": option.id, "stage": staged, "cue": cue})

    primary = min(
        (
            option
            for option in resolved_plan.transitions
            if option.card.outgoing_track_id == resolved_plan.opening.track_id
        ),
        key=lambda option: option.priority,
    )
    opening_stage = await _stage_track(
        deck=primary.card.outgoing_deck,
        track_id=resolved_plan.opening.track_id,
        title=resolved_plan.opening.title,
        artist=resolved_plan.opening.artist,
        source=resolved_plan.opening.source,
        result_index=resolved_plan.opening.result_index,
    )
    incoming_stage = await _stage_track(
        deck=primary.card.incoming_deck,
        track_id=primary.incoming.track_id,
        title=primary.incoming.title,
        artist=primary.incoming.artist,
        source=primary.incoming.source,
        result_index=primary.incoming.result_index,
    )
    if primary.incoming.cue is not None:
        await verify_hot_cue(
            deck=primary.card.incoming_deck,
            track_id=primary.incoming.track_id,
            title=primary.incoming.title,
            cue=primary.incoming.cue,
            expected_time_ms=int(primary.incoming.cue_time_ms),
        )
    fingerprint = _set_fingerprint(resolved_plan)
    preflighted_set_fingerprints.add(fingerprint)
    return {
        "ready": True,
        "fingerprint": fingerprint,
        "audit": audit,
        "tempo_arc": resolved_plan.tempo_arc_summary(),
        "warmed_routes": warmed,
        "opening_stage": opening_stage,
        "incoming_stage": incoming_stage,
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def start_autonomous_set(plan: AutonomousSetPlan) -> dict[str, Any]:
    """Start track one and keep every later handoff local to Performer."""
    resolved_plan = _materialize_autonomous_plan(plan)
    fingerprint = _set_fingerprint(resolved_plan)
    if fingerprint not in preflighted_set_fingerprints:
        raise RuntimeError(
            "autonomous set has not passed preflight in this Performer session"
        )
    primary = min(
        (
            option
            for option in resolved_plan.transitions
            if option.card.outgoing_track_id == resolved_plan.opening.track_id
        ),
        key=lambda option: option.priority,
    )
    runner = _get_autonomous_runner()
    runner.prepare(resolved_plan, opening_deck=primary.card.outgoing_deck)
    set_sessions.create(
        resolved_plan.name,
        [resolved_plan.opening.track_id]
        + [option.incoming.track_id for option in _primary_path(resolved_plan)],
    )
    estimated_seconds = (
        sum(
            (profile_store.get(track_id).duration_ms or 6 * 60_000) / 1000.0
            for track_id in [resolved_plan.opening.track_id]
            + [option.incoming.track_id for option in _primary_path(resolved_plan)]
        )
        + 20 * 60
    )
    engine.authorize_set(min(4 * 60 * 60, estimated_seconds))
    await _stage_track(
        deck=primary.card.outgoing_deck,
        track_id=resolved_plan.opening.track_id,
        title=resolved_plan.opening.title,
        artist=resolved_plan.opening.artist,
        source=resolved_plan.opening.source,
        result_index=resolved_plan.opening.result_index,
    )
    opening = await launch_and_schedule_opening_transition(
        card=primary.card,
        outgoing_title=resolved_plan.opening.title,
        incoming_title=primary.incoming.title,
        execution_id=f"autonomous-opening-{fingerprint[:16]}",
        incoming_artist=primary.incoming.artist,
        incoming_cue=primary.incoming.cue,
        incoming_cue_time_ms=primary.incoming.cue_time_ms,
        source=primary.incoming.source,
        result_index=primary.incoming.result_index,
    )
    if not opening.get("ready"):
        engine.release_set_control()
        return opening
    job_id = opening["schedule"]["job"]["id"]
    runner_state = runner.start_with_job(
        primary.id,
        job_id,
        opening.get("schedule") or {},
    )
    return {
        "ready": True,
        "opening": opening,
        "runner": runner_state,
        "tempo_arc": resolved_plan.tempo_arc_summary(),
        "set_fingerprint": fingerprint,
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def continue_autonomous_set(plan: AutonomousSetPlan) -> dict[str, Any]:
    """Attach a new locally scheduled route to the currently playing final track."""
    resolved_plan = _materialize_autonomous_plan(plan)
    primary = min(
        (
            option
            for option in resolved_plan.transitions
            if option.card.outgoing_track_id == resolved_plan.opening.track_id
        ),
        key=lambda option: option.priority,
    )
    for option in resolved_plan.transitions:
        errors = validate_transition_card(
            option.card,
            profile_store.get(option.card.outgoing_track_id),
            profile_store.get(option.card.incoming_track_id),
        )
        if errors:
            raise RuntimeError(f"{option.id}: {'; '.join(errors)}")
    observed = await _observe_live_deck(
        deck=primary.card.outgoing_deck,
        track_id=resolved_plan.opening.track_id,
        title=resolved_plan.opening.title,
    )
    if observed.get("playing") is not True:
        raise RuntimeError("the continuation anchor deck is not playing")
    runner = _get_autonomous_runner()
    runner.prepare(resolved_plan, opening_deck=primary.card.outgoing_deck)
    set_sessions.create(
        resolved_plan.name,
        [resolved_plan.opening.track_id]
        + [option.incoming.track_id for option in _primary_path(resolved_plan)],
    )
    estimated_seconds = sum(
        (profile_store.get(track_id).duration_ms or 6 * 60_000) / 1000.0
        for track_id in [resolved_plan.opening.track_id]
        + [option.incoming.track_id for option in _primary_path(resolved_plan)]
    ) + 20 * 60
    engine.authorize_set(min(4 * 60 * 60, estimated_seconds))
    scheduled = await _runner_schedule(primary, False)
    if scheduled.get("ready") is not True:
        engine.release_set_control()
        return scheduled
    runner_state = runner.start_with_job(
        primary.id,
        scheduled["job"]["id"],
        scheduled,
    )
    return {
        "ready": True,
        "continuation": scheduled,
        "runner": runner_state,
        "tempo_arc": resolved_plan.tempo_arc_summary(),
    }


def _primary_path(plan: AutonomousSetPlan) -> list[TransitionOption]:
    current = plan.opening.track_id
    result = []
    for _ in range(plan.target_track_count - 1):
        option = min(
            (
                item
                for item in plan.transitions
                if item.card.outgoing_track_id == current
            ),
            key=lambda item: item.priority,
        )
        result.append(option)
        current = option.incoming.track_id
    return result


@mcp.tool(annotations={"readOnlyHint": True})
def autonomous_set_status() -> dict[str, Any]:
    """Return the local runner state, deadlines, failures, and current job."""
    return _get_autonomous_runner().public()


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def queue_autonomous_steering(
    future_plan: AutonomousSetPlan,
) -> dict[str, Any]:
    """Safely redirect future selections after the already-armed handoff."""
    errors: list[str] = []
    for option in future_plan.transitions:
        try:
            outgoing = profile_store.get(option.card.outgoing_track_id)
            incoming = profile_store.get(option.card.incoming_track_id)
        except KeyError as exc:
            errors.append(f"{option.id}: {exc}")
            continue
        errors.extend(
            f"{option.id}: {error}"
            for error in validate_transition_card(
                option.card,
                outgoing,
                incoming,
            )
        )
    if errors:
        return {"queued": False, "errors": errors}
    return _get_autonomous_runner().queue_redirect(future_plan)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def resume_autonomous_set() -> dict[str, Any]:
    """Resume persisted runner state after a Performer process restart."""
    return _get_autonomous_runner().resume()


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def stop_autonomous_set() -> dict[str, Any]:
    """Stop the runner and release its long-lived control authorization."""
    state = _get_autonomous_runner().stop()
    engine.release_set_control()
    return state


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
        result = await _stage_track_exclusive(
            deck=deck,
            track_id=track_id,
            title=title,
            artist=artist,
            source=source,
            result_index=result_index,
        )
    if result.get("verified"):
        track_title_aliases[track_id] = title
    return result


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
        raise RuntimeError(f"Deck {deck} did not enter a paused state before loading")
    # A correctly loaded, stopped deck is already staged.  Searching again is
    # both unnecessary and dangerous during a live set: a hidden/filtered
    # browser can consume the outgoing runway even though the exact track is
    # resident on the deck.  Prove title, artist, and stopped transport from a
    # fresh deck snapshot, then bypass the browser entirely.
    snapshot_reader = getattr(rekordbox_ui, "deck_snapshot", None)
    if callable(snapshot_reader):
        resident = snapshot_reader(deck)
        resident_title_matches = normalize_title(resident.title) == normalize_title(
            title
        )
        resident_artist_matches = artist is None or normalize_title(
            resident.artist
        ) == normalize_title(artist)
        resident_at_file_start = (
            resident.elapsed_seconds is not None
            and resident.elapsed_seconds <= 1
        )
        if (
            resident_title_matches
            and resident_artist_matches
            and resident_at_file_start
        ):
            if rekordbox_ui.deck_is_playing(deck, initial=resident):
                raise RuntimeError(f"Resident track on deck {deck} is not stopped")
            deck_observer.invalidate()
            return {
                "selection": None,
                "deck": deck,
                "track_id": track_id,
                "title": title,
                "artist": artist,
                "source": source,
                "messages": [],
                "preload_stop_messages": preload_messages,
                "observed": resident.public(),
                "transport_verified_stopped": True,
                "attempts": [],
                "route_cache_hit": False,
                "already_loaded": True,
                "browser_search_skipped": True,
                "verified": True,
            }
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
    for search_pass in range(2):
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
        if first is not None:
            break
        if search_pass == 0:
            # Rekordbox can briefly expose an empty Collection after a prior
            # streaming/load repaint. Reacquire the UI and repeat the bounded
            # exact-query sequence once before declaring the route absent.
            await asyncio.sleep(0.6)
    if first is None:
        raise RuntimeError(
            f"No visible Rekordbox rows for {title!r}; tried {empty_searches!r}"
        )
    search_query = first["search_query"]
    result_count = first["result_count"]
    if result_index is not None and not 0 <= result_index < result_count:
        raise ValueError(f"result_index must be between 0 and {result_count - 1}")
    if result_count > 1 and not artist and result_index is None:
        raise RuntimeError(
            f"{result_count} rows match {title!r}; provide artist or result_index"
        )
    candidate_indices = (
        [result_index] if result_index is not None else list(range(result_count))
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
        if candidate_index == 0:
            selection = first
        else:
            try:
                selection = rekordbox_ui.select_exact_track(
                    title,
                    search_query=search_query,
                    result_index=candidate_index,
                )
            except ValueError as exc:
                if "result_index must be between" not in str(exc):
                    raise
                # Rekordbox can repaint the browser between the initial row
                # count and selection (especially while a prior streaming
                # result is still resolving). Re-read the now-single result
                # at row zero instead of failing on the stale duplicate index.
                selection = rekordbox_ui.select_exact_track(
                    title,
                    search_query=search_query,
                    result_index=0,
                )
                candidate_index = 0
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
        artist_matches = artist is None or normalize_title(
            observed.artist
        ) == normalize_title(artist)
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
                "already_loaded": False,
                "browser_search_skipped": False,
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
def preview_transition(name: str, events: list[TransitionEvent]) -> dict[str, Any]:
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
    return {
        **scheduler.get(job_id),
        "sync_guard": sync_guard_status.get(job_id),
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def cancel_transition(job_id: str) -> dict[str, Any]:
    """Cancel future events in one transition without moving any controls."""
    return scheduler.cancel(job_id)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def emergency_stop() -> dict[str, Any]:
    """Cancel all automation and disarm; leave current Rekordbox state untouched."""
    cancelled = scheduler.cancel_all()
    runner_state = (
        autonomous_runner.stop() if autonomous_runner is not None else {"active": False}
    )
    verified_rescue_loops.clear()
    deck_observer.deactivate()
    status = engine.disarm()
    return {
        "cancelled_jobs": cancelled,
        "autonomous_set": runner_state,
        "control_status": status,
        "message": "Automation cancelled. Physical FLX4 control remains available.",
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def restart_performer() -> dict[str, Any]:
    """Safely reload Performer code without restarting Codex or Rekordbox."""
    if os.environ.get(SUPERVISED_ENV) != "1":
        raise RuntimeError(
            "Performer is not running under its restart supervisor. Restart "
            "Codex once after installing this update; later Performer reloads "
            "will not require a Codex restart."
        )
    active_jobs = scheduler.metrics()["active_jobs"]
    if active_jobs:
        raise RuntimeError(
            f"Cannot restart Performer while {active_jobs} transition job(s) are active"
        )
    if autonomous_runner is not None and autonomous_runner.public().get("active"):
        raise RuntimeError("Cannot restart Performer while an autonomous set is active")
    for task in tuple(sync_guard_tasks):
        task.cancel()
    sync_guard_tasks.clear()
    deck_observer.close()
    engine.disconnect()
    generation = int(os.environ.get(GENERATION_ENV, "1"))
    timer = threading.Timer(0.75, lambda: os._exit(RESTART_EXIT_CODE))
    timer.daemon = True
    timer.start()
    return {
        "restarting": True,
        "current_generation": generation,
        "next_generation": generation + 1,
        "midi_disconnected": True,
        "control_armed": False,
        "resume_after_ms": 1500,
        "message": (
            "Performer is safely disarmed and will reload in place. Rekordbox "
            "stays open; reconnect MIDI and re-arm before live control."
        ),
    }


def main() -> None:
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
