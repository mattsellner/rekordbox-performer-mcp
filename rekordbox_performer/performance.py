"""High-reliability set orchestration, clocking, planning, and QA."""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .intelligence import (
    DeckObservation,
    TrackProfile,
    TransitionCard,
    bass_phrase_evidence,
)


def observation_from_elapsed(
    *,
    deck: int,
    profile: TrackProfile,
    elapsed_seconds: float,
    playing: bool,
    sync_enabled: bool | None,
    quantize_enabled: bool | None,
    title: str | None = None,
    playback_bpm: float | None = None,
) -> DeckObservation:
    """Convert Rekordbox's transport time into an analyzed-track clock."""
    elapsed_ms = max(0.0, elapsed_seconds * 1000.0)
    grid = sorted(profile.beat_grid, key=lambda point: point.time_ms)
    if grid:
        anchor = min(grid, key=lambda point: abs(point.time_ms - elapsed_ms))
        bpm = anchor.bpm
        beat_delta = (elapsed_ms - anchor.time_ms) * bpm / 60_000.0
        track_total = max(0.0, anchor.index - 1 + beat_delta)
        grid_total = max(
            0.0,
            (anchor.bar - 1) * profile.time_signature
            + (anchor.beat - 1)
            + beat_delta,
        )
    else:
        bpm = profile.bpm
        track_total = elapsed_ms * bpm / 60_000.0
        grid_total = track_total
    track_whole = int(track_total)
    grid_whole = int(grid_total)
    phase = min(0.9999, max(0.0, grid_total - grid_whole))
    beat_in_bar = grid_whole % profile.time_signature + 1
    return DeckObservation(
        deck=deck,
        track_id=profile.track_id,
        title=title or profile.title,
        # The analyzed grid BPM locates the track beat.  The displayed deck BPM
        # is the actual scheduler clock after tempo/master-sync changes.
        bpm=playback_bpm if playback_bpm is not None else bpm,
        playing=playing,
        bar=grid_whole // profile.time_signature + 1,
        beat=beat_in_bar,
        track_beat=track_whole + 1,
        beat_phase=phase,
        sync_enabled=sync_enabled,
        quantize_enabled=quantize_enabled,
        source="native",
        confidence="high" if grid else "medium",
    )


def vocal_handoff(
    outgoing: TrackProfile,
    incoming: TrackProfile,
    *,
    outgoing_start_bar: int,
    incoming_start_bar: int,
    overlap_bars: int,
) -> dict[str, Any]:
    """Score vocal overlap as a soft planning signal, never a hard blocker."""
    def vocal_at(profile: TrackProfile, bar: int) -> bool:
        return any(
            segment.kind == "vocal" and segment.start_bar <= bar < segment.end_bar
            for segment in profile.segments
        )

    clashes: list[int] = []
    outgoing_vocal_bars: list[int] = []
    incoming_vocal_bars: list[int] = []
    for offset in range(overlap_bars):
        out_vocal = vocal_at(outgoing, outgoing_start_bar + offset)
        in_vocal = vocal_at(incoming, incoming_start_bar + offset)
        if out_vocal:
            outgoing_vocal_bars.append(offset)
        if in_vocal:
            incoming_vocal_bars.append(offset)
        if out_vocal and in_vocal:
            clashes.append(offset)
    owner = (
        "outgoing" if outgoing_vocal_bars and not incoming_vocal_bars
        else "incoming" if incoming_vocal_bars and not outgoing_vocal_bars
        else "none" if not outgoing_vocal_bars and not incoming_vocal_bars
        else "intentional_overlap"
    )
    return {
        "outgoing_vocal_offsets": outgoing_vocal_bars,
        "incoming_vocal_offsets": incoming_vocal_bars,
        "overlap_offsets": clashes,
        "overlap_bars": len(clashes),
        "recommended_owner": owner,
        "soft_penalty": min(12, len(clashes) * 2),
        "blocking": False,
    }


def cue_preparation_plan(profile: TrackProfile) -> dict[str, Any]:
    """Suggest phrase-safe entry cues before drops without mutating Rekordbox."""
    phrase_starts = {
        (phrase.start_bar, phrase.beat_in_bar)
        for phrase in profile.phrase_boundaries
        if phrase.confidence in {"verified", "high"}
    }
    target_records = []
    seen_bars = set()
    candidates = [
        item for item in profile.landmarks
        if item.kind in {"drop", "bass_in"}
        and item.confidence in {"verified", "high"}
        and (item.bar, item.beat) in phrase_starts
    ]
    candidates.sort(
        key=lambda item: (
            -float(bass_phrase_evidence(profile, item.bar).get("score", 0)),
            0 if item.kind == "drop" else 1,
            item.bar,
        )
    )
    for item in candidates:
        evidence = bass_phrase_evidence(profile, item.bar)
        if not evidence["verified"] or item.bar in seen_bars:
            continue
        seen_bars.add(item.bar)
        target_records.append((item, evidence))
    existing = {item.cue for item in profile.landmarks if item.cue is not None}
    # Reserve the user's A/B/C workflow. Automation-owned preparation uses
    # only G/H (pads 7/8), as explicitly requested.
    free = [cue for cue in (7, 8) if cue not in existing]
    suggestions = []
    for target, bass_evidence in target_records:
        for lead in (16, 8):
            cue_bar = target.bar - lead
            if cue_bar < 1:
                continue
            if (cue_bar, target.beat) not in phrase_starts:
                continue
            beat_index = (cue_bar - 1) * profile.time_signature + target.beat
            point = next((p for p in profile.beat_grid if p.index == beat_index), None)
            suggestions.append({
                "cue": free.pop(0) if free else None,
                "role": f"{lead}_bars_before_{target.kind}",
                "bar": cue_bar,
                "beat": target.beat,
                "time_ms": point.time_ms if point else round((beat_index - 1) * 60_000 / profile.bpm),
                "target": target.model_dump(),
                "verified_bass_phrase_start": bass_evidence["verified"],
                "bass_waveform_evidence": bass_evidence,
                "requires_rekordbox_verification": True,
            })
            break
    return {
        "track_id": profile.track_id,
        "title": profile.title,
        "suggestions": suggestions,
        "ready": bool(suggestions),
        "mutated_rekordbox": False,
        "cue_policy": "automation uses Hot Cue G/H only",
    }


def sync_report(
    outgoing: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, Any]:
    bpm_error = abs(float(outgoing["bpm"]) - float(incoming["bpm"]))
    phase = abs(float(outgoing.get("beat_phase", 0)) - float(incoming.get("beat_phase", 0)))
    phase = min(phase, 1.0 - phase)
    bpm = max(float(outgoing["bpm"]), float(incoming["bpm"]))
    phase_ms = phase * 60_000.0 / bpm
    errors = []
    if incoming.get("sync_enabled") is not True:
        errors.append("incoming Beat Sync is not confirmed on")
    if bpm_error > 0.05:
        errors.append(f"deck BPM mismatch is {bpm_error:.2f}")
    if phase_ms > 35:
        errors.append(f"beat phase error is {phase_ms:.1f} ms")
    bar_phase_error_beats = None
    if outgoing.get("beat") is not None and incoming.get("beat") is not None:
        outgoing_bar_phase = (
            float(outgoing["beat"]) - 1.0
            + float(outgoing.get("beat_phase", 0))
        ) % 4.0
        incoming_bar_phase = (
            float(incoming["beat"]) - 1.0
            + float(incoming.get("beat_phase", 0))
        ) % 4.0
        bar_delta = abs(outgoing_bar_phase - incoming_bar_phase)
        bar_phase_error_beats = min(bar_delta, 4.0 - bar_delta)
        if bar_phase_error_beats > 0.15:
            errors.append(
                "beat-in-bar alignment error is "
                f"{bar_phase_error_beats:.2f} beats"
            )
    return {
        "verified": not errors,
        "errors": errors,
        "bpm_error": round(bpm_error, 3),
        "beat_phase_error_ms": round(phase_ms, 2),
        "bar_phase_error_beats": (
            None
            if bar_phase_error_beats is None
            else round(bar_phase_error_beats, 3)
        ),
        "correction": "re-enable Beat Sync and relaunch on the next phrase" if errors else "none",
    }


def fx_recipe(card: TransitionCard) -> dict[str, Any]:
    recipes = {
        "echo_exit": ("echo", 0.5),
        "breakdown_handoff": ("reverb", 0.4),
        "loop_bridge": ("spiral", 0.35),
        "phrase_cut": ("vinyl_brake", 0.65),
        "bass_swap": ("echo", 0.3),
        "long_blend": ("reverb", 0.25),
        "double_drop": ("echo", 0.2),
    }
    effect, wet = recipes[card.transition_family]
    return {
        "deck": card.outgoing_deck,
        "effect": effect,
        "selection": {
            "actions": ["fx_select_next", "fx_select_back"],
            "requires_observed_current_effect": True,
            "note": "Cycle from the observed current FX; Rekordbox MIDI Learn has no fixed-effect selector.",
        },
        "events": [
            {"bar_offset": max(0, card.critical_bar_offset - 1), "beat_offset": 0, "action": "fx_wet_dry", "parameters": {"deck": card.outgoing_deck, "value": wet}},
            {"bar_offset": max(0, card.critical_bar_offset - 1), "beat_offset": 0, "action": "fx_toggle", "parameters": {"deck": card.outgoing_deck}},
            {"bar_offset": card.critical_bar_offset + 1, "beat_offset": 0, "action": "fx_wet_dry", "parameters": {"deck": card.outgoing_deck, "value": 0}},
            {"bar_offset": card.critical_bar_offset + 1, "beat_offset": 0, "action": "fx_toggle", "parameters": {"deck": card.outgoing_deck}},
        ],
        "reset_verified_by_plan": True,
    }


def transition_qa(job: dict[str, Any]) -> dict[str, Any]:
    verification = job.get("verification") or {}
    sync = verification.get("sync") or {}
    phase_ms = float(sync.get("beat_phase_error_ms", 0))
    late = float(job.get("p99_event_lateness_ms", 0))
    faults = list(verification.get("errors", []))
    # Transport and mixer postconditions can pass even when the musical
    # handoff is audibly off-grid. Sync verification is therefore a hard QA
    # gate, not just a score penalty. Without this, an 83 ms phase error (or
    # even a one-beat cue mistake) can be reported as a successful transition
    # and advance the rolling set.
    if sync and sync.get("verified") is not True:
        sync_errors = sync.get("errors") or ["beat sync was not verified"]
        faults.extend(error for error in sync_errors if error not in faults)
    if late > 20:
        faults.append(f"p99 MIDI dispatch lateness was {late:.1f} ms")
    score = 100 - min(30, late) - min(35, phase_ms / 2) - 20 * len(faults)
    return {
        "job_id": job.get("id"),
        "passed": job.get("status") == "completed" and not faults,
        "score": round(max(0.0, score), 1),
        "faults": faults,
        "metrics": {
            "p99_event_lateness_ms": late,
            "beat_phase_error_ms": phase_ms,
            "completed_events": job.get("completed_events", 0),
        },
    }


class SetSession(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str
    track_ids: list[str] = Field(min_length=2)
    current_index: int = Field(default=0, ge=0)
    status: Literal["prepared", "running", "completed", "failed"] = "prepared"
    transition_jobs: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    updated_at: float = Field(default_factory=time.time)

    @model_validator(mode="after")
    def valid_index(self) -> "SetSession":
        if self.current_index >= len(self.track_ids):
            raise ValueError("current_index is outside the track list")
        return self


class SetSessionManager:
    """Atomic, restart-safe rolling-set state with a three-track horizon."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _read(self) -> SetSession | None:
        if not self.path.exists():
            return None
        return SetSession.model_validate_json(self.path.read_text(encoding="utf-8"))

    def _write(self, session: SetSession) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        session.updated_at = time.time()
        fd, raw_path = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        os.close(fd)
        temp = Path(raw_path)
        try:
            temp.write_text(session.model_dump_json(indent=2), encoding="utf-8")
            os.replace(temp, self.path)
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def public(session: SetSession) -> dict[str, Any]:
        index = session.current_index
        tracks = session.track_ids
        return {
            **session.model_dump(),
            "current": tracks[index],
            "next": tracks[index + 1] if index + 1 < len(tracks) else None,
            "following": tracks[index + 2] if index + 2 < len(tracks) else None,
            "remaining_transitions": max(0, len(tracks) - index - 1),
        }

    def create(self, name: str, track_ids: list[str]) -> dict[str, Any]:
        session = SetSession(name=name, track_ids=track_ids)
        self._write(session)
        return self.public(session)

    def status(self) -> dict[str, Any]:
        session = self._read()
        return {"active": False} if session is None else {"active": True, **self.public(session)}

    def start(self) -> dict[str, Any]:
        session = self._read()
        if session is None:
            raise RuntimeError("No prepared set session")
        session.status = "running"
        self._write(session)
        return self.public(session)

    def advance(self, job_id: str, succeeded: bool, error: str | None = None) -> dict[str, Any]:
        session = self._read()
        if session is None:
            raise RuntimeError("No active set session")
        session.transition_jobs.append(job_id)
        if succeeded:
            session.current_index = min(session.current_index + 1, len(session.track_ids) - 1)
            session.status = "completed" if session.current_index == len(session.track_ids) - 1 else "running"
        else:
            session.status = "failed"
            session.failures.append(error or "transition failed")
        self._write(session)
        return self.public(session)
