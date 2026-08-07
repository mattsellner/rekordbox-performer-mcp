"""Musical state, transition-card validation, and rehearsal persistence."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .protocol import encode_action


Confidence = Literal["verified", "high", "medium", "low", "unknown"]
LandmarkKind = Literal[
    "phrase_start",
    "mix_in",
    "mix_out",
    "drop",
    "breakdown",
    "bass_in",
    "bass_out",
]
SegmentKind = Literal["vocal", "instrumental", "bass", "breakdown", "build", "drop"]

_CAMELOT_BY_KEY = {
    "g#m": "1A",
    "abm": "1A",
    "d#m": "2A",
    "ebm": "2A",
    "a#m": "3A",
    "bbm": "3A",
    "fm": "4A",
    "cm": "5A",
    "gm": "6A",
    "dm": "7A",
    "am": "8A",
    "em": "9A",
    "bm": "10A",
    "f#m": "11A",
    "gbm": "11A",
    "c#m": "12A",
    "dbm": "12A",
    "b": "1B",
    "f#": "2B",
    "gb": "2B",
    "c#": "3B",
    "db": "3B",
    "g#": "4B",
    "ab": "4B",
    "d#": "5B",
    "eb": "5B",
    "a#": "6B",
    "bb": "6B",
    "f": "7B",
    "c": "8B",
    "g": "9B",
    "d": "10B",
    "a": "11B",
    "e": "12B",
}


def normalize_camelot(key: str | None) -> str | None:
    """Normalize Rekordbox Camelot or conventional key labels."""
    if not key:
        return None
    value = key.strip().replace("♭", "b").replace("♯", "#")
    camelot = re.fullmatch(r"(1[0-2]|[1-9])([AaBb])", value)
    if camelot:
        return f"{int(camelot.group(1))}{camelot.group(2).upper()}"
    compact = value.casefold().replace("minor", "m").replace("major", "")
    compact = re.sub(r"\s+", "", compact)
    return _CAMELOT_BY_KEY.get(compact)


def camelot_compatibility(
    outgoing_key: str | None,
    incoming_key: str | None,
) -> dict[str, Any]:
    """Return a conservative harmonic-mixing compatibility report."""
    outgoing = normalize_camelot(outgoing_key)
    incoming = normalize_camelot(incoming_key)
    if outgoing is None or incoming is None:
        return {
            "verified": False,
            "compatible": False,
            "outgoing": outgoing,
            "incoming": incoming,
            "wheel_distance": None,
            "relationship": "unknown",
        }
    outgoing_number, outgoing_mode = int(outgoing[:-1]), outgoing[-1]
    incoming_number, incoming_mode = int(incoming[:-1]), incoming[-1]
    raw_distance = abs(outgoing_number - incoming_number)
    wheel_distance = min(raw_distance, 12 - raw_distance)
    same_mode_neighbor = outgoing_mode == incoming_mode and wheel_distance <= 1
    relative_mode = outgoing_number == incoming_number
    compatible = same_mode_neighbor or relative_mode
    relationship = (
        "same_key"
        if outgoing == incoming
        else "relative_major_minor"
        if relative_mode
        else "wheel_neighbor"
        if same_mode_neighbor
        else "incompatible"
    )
    return {
        "verified": True,
        "compatible": compatible,
        "outgoing": outgoing,
        "incoming": incoming,
        "wheel_distance": wheel_distance,
        "relationship": relationship,
    }


class TrackLandmark(BaseModel):
    name: str
    kind: LandmarkKind
    bar: int = Field(ge=1)
    beat: int = Field(default=1, ge=1, le=4)
    cue: int | None = Field(default=None, ge=1, le=8)
    time_ms: int | None = Field(default=None, ge=0)
    confidence: Confidence = "unknown"


class TrackSegment(BaseModel):
    kind: SegmentKind
    start_bar: int = Field(ge=1)
    end_bar: int = Field(gt=1)
    confidence: Confidence = "unknown"
    notes: str = ""

    @model_validator(mode="after")
    def ordered(self) -> "TrackSegment":
        if self.end_bar <= self.start_bar:
            raise ValueError("segment end_bar must be greater than start_bar")
        return self


class PhraseBoundary(BaseModel):
    index: int = Field(ge=1)
    start_beat: int = Field(ge=1)
    end_beat: int = Field(ge=1)
    start_bar: int = Field(ge=1)
    beat_in_bar: int = Field(ge=1, le=4)
    length_beats: int = Field(ge=1)
    length_bars: int | None = Field(default=None, ge=1)
    kind_code: int = Field(ge=0)
    label: str
    fill: bool = False
    confidence: Confidence = "high"

    @model_validator(mode="after")
    def ordered(self) -> "PhraseBoundary":
        if self.end_beat < self.start_beat:
            raise ValueError("phrase end_beat must not precede start_beat")
        return self


class AnalysisBeatGridPoint(BaseModel):
    index: int = Field(ge=1)
    bar: int = Field(ge=1)
    beat: int = Field(ge=1, le=4)
    bpm: float = Field(gt=0)
    time_ms: int = Field(ge=0)


class BassEnergyBar(BaseModel):
    """Rekordbox PWV7 low-band energy summarized for one analyzed bar."""

    bar: int = Field(ge=1)
    median: float = Field(ge=0)
    mean: float = Field(ge=0)
    peak: float = Field(ge=0)


class RekordboxAnalysisImport(BaseModel):
    track_id: str
    title: str = ""
    artist: str = ""
    metadata_encrypted: bool = False
    source: str = "local"
    service_uri: str | None = None
    bpm: float | None = Field(default=None, gt=0)
    key: str | None = None
    beatgrid_available: bool = False
    phrase_analysis_available: bool = False
    beat_count: int = Field(default=0, ge=0)
    phrase_count: int = Field(default=0, ge=0)
    beat_grid: list[AnalysisBeatGridPoint] = Field(default_factory=list)
    phrases: list[PhraseBoundary] = Field(default_factory=list)
    vocal_analysis_available: bool = False
    vocal_segments: list[TrackSegment] = Field(default_factory=list)
    bass_analysis_available: bool = False
    bass_energy_by_bar: list[BassEnergyBar] = Field(default_factory=list)
    bass_segments: list[TrackSegment] = Field(default_factory=list)


class TrackProfile(BaseModel):
    track_id: str
    title: str
    artist: str
    version: str = ""
    bpm: float = Field(gt=0)
    key: str | None = None
    time_signature: int = Field(default=4, ge=3, le=8)
    duration_ms: int | None = Field(default=None, gt=0)
    source: str = "local"
    service_uri: str | None = None
    beat_count: int = Field(default=0, ge=0)
    beat_grid: list[AnalysisBeatGridPoint] = Field(default_factory=list)
    beatgrid_confidence: Confidence = "unknown"
    phrase_confidence: Confidence = "unknown"
    vocal_confidence: Confidence = "unknown"
    phrase_boundaries: list[PhraseBoundary] = Field(default_factory=list)
    landmarks: list[TrackLandmark] = Field(default_factory=list)
    segments: list[TrackSegment] = Field(default_factory=list)
    bass_energy_by_bar: list[BassEnergyBar] = Field(default_factory=list)
    notes: str = ""

    def readiness(self) -> dict[str, Any]:
        live_confidence = {"verified", "high"}
        kinds = {landmark.kind for landmark in self.landmarks}
        vocal_mapped = any(segment.kind == "vocal" for segment in self.segments)
        checks = {
            "beatgrid": self.beatgrid_confidence in live_confidence,
            "phrases": (
                self.phrase_confidence in live_confidence
                and bool(self.phrase_boundaries)
            ),
            "vocals": self.vocal_confidence in live_confidence and vocal_mapped,
            "mix_in": "mix_in" in kinds or "phrase_start" in kinds,
            "mix_out": "mix_out" in kinds,
            "bass": {"bass_in", "bass_out"} & kinds != set()
            or any(segment.kind == "bass" for segment in self.segments),
        }
        ready = all(checks.values())
        analysis_ready = all(
            checks[name] for name in ("beatgrid", "phrases", "mix_in", "mix_out")
        )
        tier = "A" if ready else "B" if analysis_ready else "C"
        return {"ready": ready, "tier": tier, "checks": checks}

    def analysis_readiness(self) -> dict[str, Any]:
        live_confidence = {"verified", "high"}
        kinds = {landmark.kind for landmark in self.landmarks}
        checks = {
            "beatgrid": (
                self.beatgrid_confidence in live_confidence and bool(self.beat_grid)
            ),
            "phrases": (
                self.phrase_confidence in live_confidence
                and bool(self.phrase_boundaries)
            ),
            "mix_in": "mix_in" in kinds or "phrase_start" in kinds,
            "mix_out": "mix_out" in kinds,
        }
        return {"ready": all(checks.values()), "checks": checks}


class PlaylistTrackMetadata(BaseModel):
    track_id: str
    title: str
    artist: str
    bpm: float = Field(ge=0)
    key: str | None = None
    duration_ms: int | None = Field(default=None, gt=0)
    version: str = ""


class DeckObservation(BaseModel):
    deck: int = Field(ge=1, le=2)
    track_id: str
    title: str
    bpm: float = Field(gt=0)
    playing: bool
    bar: int = Field(ge=1)
    beat: int = Field(ge=1, le=4)
    track_beat: int | None = Field(
        default=None,
        ge=1,
        description="One-based absolute beat index from the analyzed track grid",
    )
    beat_phase: float = Field(default=0.0, ge=0.0, lt=1.0)
    sync_enabled: bool | None = None
    quantize_enabled: bool | None = None
    source: Literal["native", "midi", "vision", "manual"] = "manual"
    confidence: Confidence = "unknown"


@dataclass
class ObservedDeck:
    observation: DeckObservation
    observed_monotonic: float


class LiveState:
    """Fresh, extrapolated deck state supplied by a native/MIDI/vision adapter."""

    def __init__(self) -> None:
        self.decks: dict[int, ObservedDeck] = {}

    def update(self, observation: DeckObservation) -> dict[str, Any]:
        self.decks[observation.deck] = ObservedDeck(observation, time.monotonic())
        return self.get(observation.deck)

    def get(self, deck: int) -> dict[str, Any]:
        if deck not in self.decks:
            raise KeyError(f"No live observation for deck {deck}")
        item = self.decks[deck]
        age = time.monotonic() - item.observed_monotonic
        observation = item.observation
        base_track_beat = observation.track_beat or (
            (observation.bar - 1) * 4 + observation.beat
        )
        elapsed_beats = 0.0
        if observation.playing:
            elapsed_beats = age * observation.bpm / 60.0
        track_total = (
            base_track_beat - 1 + observation.beat_phase + elapsed_beats
        )
        grid_total = (
            (observation.bar - 1) * 4
            + observation.beat
            - 1
            + observation.beat_phase
            + elapsed_beats
        )
        # Normalize before deriving beat/bar fields.  Rounding only the final
        # fractional phase can produce the impossible value 1.0 immediately
        # before a beat rollover, which then fails DeckObservation validation
        # during a live refresh.
        track_total = round(track_total, 4)
        grid_total = round(grid_total, 4)
        bar = int(grid_total // 4) + 1
        within_bar = grid_total % 4
        beat = int(within_bar) + 1
        beat_phase = within_bar - int(within_bar)
        track_beat = int(track_total) + 1
        return {
            **observation.model_dump(),
            "bar": bar,
            "beat": beat,
            "track_beat": track_beat,
            "beat_phase": beat_phase,
            "observation_age_ms": round(age * 1000),
            "fresh": age <= 1.0,
        }

    def snapshot(self) -> dict[str, Any]:
        return {"decks": [self.get(deck) for deck in sorted(self.decks)]}


class MusicalEvent(BaseModel):
    bar_offset: int = Field(ge=0)
    beat_offset: int = Field(default=0, ge=0, le=3)
    action: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_action(self) -> "MusicalEvent":
        encode_action(self.action, self.parameters)
        return self


class TransitionCard(BaseModel):
    name: str
    outgoing_track_id: str
    incoming_track_id: str
    anchor_deck: int = Field(ge=1, le=2)
    outgoing_deck: int = Field(ge=1, le=2)
    incoming_deck: int = Field(ge=1, le=2)
    transition_family: Literal[
        "long_blend",
        "bass_swap",
        "phrase_cut",
        "echo_exit",
        "loop_bridge",
        "breakdown_handoff",
        "double_drop",
    ] = "bass_swap"
    start_quantum_bars: Literal[1, 4, 8, 16, 32] = 16
    minimum_lead_bars: int = Field(default=1, ge=1, le=16)
    start_phrase_index: int | None = Field(default=None, ge=1)
    start_phrase_label: str | None = None
    beat_sync_required: bool = True
    quantize_required: bool = True
    phrase_alignment_verified: bool = False
    vocal_plan_verified: bool = False
    vocal_risk_accepted: bool = False
    harmonic_risk_accepted: bool = False
    bass_plan_verified: bool = False
    intentional_energy_drop: bool = False
    loop_plan_verified: bool = False
    incoming_loaded_verified: bool = False
    max_tempo_stretch_percent: float = Field(default=4.0, gt=0, le=10)
    tempo_risk_accepted: bool = False
    intended_vocal_owner: Literal["outgoing", "incoming", "none", "intentional_overlap"]
    critical_bar_offset: int = Field(ge=0)
    events: list[MusicalEvent]
    abort_plan: str = Field(min_length=5)
    notes: str = ""

    @model_validator(mode="after")
    def different_decks(self) -> "TransitionCard":
        if self.outgoing_deck == self.incoming_deck:
            raise ValueError("incoming and outgoing decks must differ")
        if not self.events:
            raise ValueError("transition card requires musical events")
        return self


class SetPlan(BaseModel):
    name: str
    track_ids: list[str] = Field(min_length=2)
    cards: list[TransitionCard]
    preload_lead_bars: int = Field(default=32, ge=16, le=128)
    rescue_loop_bars: Literal[4, 8, 16] = 8

    @model_validator(mode="after")
    def card_count(self) -> "SetPlan":
        if len(self.cards) != len(self.track_ids) - 1:
            raise ValueError("a set plan requires exactly one card per adjacent pair")
        return self


class RehearsalReview(BaseModel):
    transition_name: str
    outgoing_track_id: str
    incoming_track_id: str
    beat_phase_error_ms: float = Field(ge=0)
    bar_error: int = Field(ge=0)
    bass_swap_error_beats: float = Field(ge=0)
    vocal_clash: bool = False
    energy_continuity: int = Field(ge=1, le=10)
    cleanliness: int = Field(ge=1, le=10)
    user_rating: int = Field(ge=1, le=10)
    notes: str = ""

    def score(self) -> float:
        score = (
            self.energy_continuity * 0.25
            + self.cleanliness * 0.25
            + self.user_rating * 0.5
        )
        score -= min(3.0, self.bar_error * 2.0)
        score -= min(1.5, self.beat_phase_error_ms / 100.0)
        score -= min(1.5, self.bass_swap_error_beats * 0.5)
        if self.vocal_clash:
            score -= 2.0
        return round(max(0.0, min(10.0, score)), 2)


def default_data_dir() -> Path:
    base = (
        os.environ.get("REKORDBOX_PERFORMER_DATA")
        or os.environ.get("LOCALAPPDATA")
        or str(Path.home())
    )
    return Path(base) / "rekordbox-performer"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bass_phrase_evidence(
    profile: TrackProfile,
    start_bar: int,
    window_bars: int = 8,
) -> dict[str, Any]:
    """Measure whether an incoming phrase can carry a full low-EQ handoff.

    A segment-level "bass present" flag is not enough for a musical swap. This
    uses Rekordbox's PWV7 low-band curve to require useful energy on the first
    bar and sustained energy throughout the next phrase. Scores are relative
    to the track, so quiet masters are not penalized merely for being quiet.
    """
    if start_bar < 1:
        raise ValueError("start_bar must be positive")
    if window_bars not in {4, 8, 16}:
        raise ValueError("window_bars must be 4, 8, or 16")
    curve = {item.bar: item for item in profile.bass_energy_by_bar}
    if not curve:
        return {
            "verified": False,
            "available": False,
            "reason": "per-bar Rekordbox PWV7 low-band energy is unavailable",
            "start_bar": start_bar,
            "window_bars": window_bars,
        }
    track_values = [
        float(item.median) for item in profile.bass_energy_by_bar if item.median > 0
    ]
    reference = _percentile(track_values, 0.60)
    bars = [curve.get(bar) for bar in range(start_bar, start_bar + window_bars)]
    available = [item for item in bars if item is not None]
    coverage_ratio = len(available) / window_bars
    values = [float(item.median) for item in available]
    phrase_median = _percentile(values, 0.50)
    phrase_mean = sum(values) / len(values) if values else 0.0
    downbeat_energy = float(bars[0].median) if bars[0] is not None else 0.0
    first_two = [float(item.median) for item in bars[:2] if item is not None]
    attack_energy = sum(first_two) / len(first_two) if first_two else 0.0
    strong_floor = reference * 0.90
    sustained_ratio = (
        sum(value >= strong_floor for value in values) / len(values)
        if values and reference > 0
        else 0.0
    )
    previous = [
        float(curve[bar].median)
        for bar in range(max(1, start_bar - window_bars), start_bar)
        if bar in curve
    ]
    previous_median = _percentile(previous, 0.50)
    lift_ratio = phrase_median / previous_median if previous_median > 0 else None
    attack_ratio = attack_energy / reference if reference > 0 else 0.0
    downbeat_ratio = downbeat_energy / reference if reference > 0 else 0.0
    phrase_ratio = phrase_median / reference if reference > 0 else 0.0
    verified = (
        coverage_ratio >= 0.75
        and reference > 0
        and downbeat_ratio >= 0.75
        and attack_ratio >= 0.85
        and phrase_ratio >= 0.95
        and sustained_ratio >= 0.625
    )
    reasons = []
    if coverage_ratio < 0.75:
        reasons.append("fewer than 75% of phrase bars have waveform evidence")
    if reference <= 0:
        reasons.append("track has no positive low-band reference energy")
    if downbeat_ratio < 0.75:
        reasons.append("phrase downbeat low end is too weak")
    if attack_ratio < 0.85:
        reasons.append("first two phrase bars do not establish enough low end")
    if phrase_ratio < 0.95:
        reasons.append("phrase median low end is below the track reference")
    if sustained_ratio < 0.625:
        reasons.append("low end is not sustained across enough phrase bars")
    score = max(
        0.0,
        min(
            100.0,
            20.0 * min(1.25, downbeat_ratio)
            + 25.0 * min(1.25, attack_ratio)
            + 30.0 * min(1.25, phrase_ratio)
            + 25.0 * sustained_ratio,
        ),
    )
    return {
        "verified": verified,
        "available": True,
        "reason": None if verified else "; ".join(reasons),
        "start_bar": start_bar,
        "window_bars": window_bars,
        "bars_observed": len(available),
        "coverage_ratio": round(coverage_ratio, 3),
        "track_reference_median": round(reference, 3),
        "downbeat_energy": round(downbeat_energy, 3),
        "downbeat_ratio": round(downbeat_ratio, 3),
        "attack_energy": round(attack_energy, 3),
        "attack_ratio": round(attack_ratio, 3),
        "phrase_median": round(phrase_median, 3),
        "phrase_mean": round(phrase_mean, 3),
        "phrase_ratio": round(phrase_ratio, 3),
        "sustained_strong_bar_ratio": round(sustained_ratio, 3),
        "previous_phrase_median": round(previous_median, 3),
        "lift_ratio": None if lift_ratio is None else round(lift_ratio, 3),
        "score": round(score, 1),
    }


def energy_phrase_evidence(
    profile: TrackProfile,
    start_bar: int,
    window_bars: int = 8,
) -> dict[str, Any]:
    """Measure whether a phrase can take ownership of the room's energy.

    Sustained sub energy is only one kind of useful handoff.  A chorus can have
    a lighter continuous low-band median while still landing with much stronger
    kick/transient energy.  Rekordbox PWV7 exposes median, mean, and peak curves;
    this combines all three and only promotes transient-led evidence on a
    verified chorus/drop phrase boundary.
    """
    bass = bass_phrase_evidence(profile, start_bar, window_bars=window_bars)
    curve = {item.bar: item for item in profile.bass_energy_by_bar}
    phrase = next(
        (
            item
            for item in profile.phrase_boundaries
            if item.start_bar == start_bar
            and item.beat_in_bar == 1
            and item.confidence in {"verified", "high"}
        ),
        None,
    )
    if not curve or phrase is None:
        return {
            **bass,
            "verified": bool(bass.get("verified")),
            "energy_verified": False,
            "energy_source": "sustained_bass" if bass.get("verified") else None,
            "phrase_label": phrase.label if phrase is not None else None,
        }

    bars = [curve.get(bar) for bar in range(start_bar, start_bar + window_bars)]
    available = [item for item in bars if item is not None]
    coverage_ratio = len(available) / window_bars
    track_means = [float(item.mean) for item in curve.values() if item.mean > 0]
    track_peaks = [float(item.peak) for item in curve.values() if item.peak > 0]
    mean_reference = _percentile(track_means, 0.60)
    peak_reference = _percentile(track_peaks, 0.60)
    phrase_means = [float(item.mean) for item in available]
    phrase_peaks = [float(item.peak) for item in available]
    downbeat_mean = float(bars[0].mean) if bars[0] is not None else 0.0
    downbeat_peak = float(bars[0].peak) if bars[0] is not None else 0.0
    phrase_mean = sum(phrase_means) / len(phrase_means) if phrase_means else 0.0
    phrase_peak = _percentile(phrase_peaks, 0.50)
    mean_ratio = phrase_mean / mean_reference if mean_reference > 0 else 0.0
    peak_ratio = phrase_peak / peak_reference if peak_reference > 0 else 0.0
    downbeat_mean_ratio = downbeat_mean / mean_reference if mean_reference > 0 else 0.0
    downbeat_peak_ratio = downbeat_peak / peak_reference if peak_reference > 0 else 0.0
    transient_phrase = phrase.label.casefold() in {"chorus", "drop"}
    transient_verified = (
        coverage_ratio >= 0.75
        and transient_phrase
        and downbeat_mean_ratio >= 0.90
        and downbeat_peak_ratio >= 0.95
        and mean_ratio >= 0.85
        and peak_ratio >= 0.95
    )
    verified = bool(bass.get("verified")) or transient_verified
    transient_score = max(
        0.0,
        min(
            100.0,
            25.0 * min(1.2, downbeat_mean_ratio)
            + 25.0 * min(1.2, downbeat_peak_ratio)
            + 25.0 * min(1.2, mean_ratio)
            + 25.0 * min(1.2, peak_ratio),
        ),
    )
    return {
        **bass,
        "verified": verified,
        "energy_verified": transient_verified,
        "energy_source": (
            "sustained_bass"
            if bass.get("verified")
            else "chorus_transient_energy"
            if transient_verified
            else None
        ),
        "phrase_label": phrase.label,
        "mean_reference": round(mean_reference, 3),
        "peak_reference": round(peak_reference, 3),
        "downbeat_mean_ratio": round(downbeat_mean_ratio, 3),
        "downbeat_peak_ratio": round(downbeat_peak_ratio, 3),
        "phrase_mean_ratio": round(mean_ratio, 3),
        "phrase_peak_ratio": round(peak_ratio, 3),
        "transient_score": round(transient_score, 1),
        "score": round(max(float(bass.get("score", 0)), transient_score), 1),
        "reason": None if verified else bass.get("reason"),
    }


def incoming_bass_handoff_evidence(
    card: TransitionCard,
    incoming: TrackProfile,
) -> dict[str, Any] | None:
    """Resolve a card's critical swap bar and return its waveform evidence."""
    has_handoff = any(
        event.action == "eq_low"
        and event.parameters.get("deck") == card.outgoing_deck
        and float(event.parameters.get("value", 0)) <= -0.9
        for event in card.events
    ) and any(
        event.action == "eq_low"
        and event.parameters.get("deck") == card.incoming_deck
        and float(event.parameters.get("value", -1)) >= -0.1
        for event in card.events
    )
    if not has_handoff:
        return None
    launch = next(
        (
            event
            for event in card.events
            if event.bar_offset == 0
            and event.beat_offset == 0
            and event.action in {"hot_cue", "play_pause"}
            and event.parameters.get("deck") == card.incoming_deck
        ),
        None,
    )
    if launch is None:
        return {
            "verified": False,
            "available": bool(incoming.bass_energy_by_bar),
            "reason": "incoming phrase anchor is unavailable",
        }
    if launch.action == "hot_cue":
        entry = next(
            (
                landmark
                for landmark in incoming.landmarks
                if landmark.cue == launch.parameters.get("cue")
                and landmark.kind in {"mix_in", "phrase_start"}
                and landmark.confidence in {"verified", "high"}
            ),
            None,
        )
    else:
        entry = next(
            (
                landmark
                for landmark in incoming.landmarks
                if landmark.kind in {"mix_in", "phrase_start"}
                and landmark.bar == 1
                and (landmark.beat or 1) == 1
                and landmark.confidence in {"verified", "high"}
            ),
            None,
        )
    if entry is None:
        return {
            "verified": False,
            "available": bool(incoming.bass_energy_by_bar),
            "reason": "incoming launch has no verified phrase landmark",
        }
    target_bar = entry.bar + card.critical_bar_offset
    evidence = energy_phrase_evidence(
        incoming,
        target_bar,
        window_bars=8,
    )
    user_verified_drop = next(
        (
            landmark
            for landmark in incoming.landmarks
            if landmark.kind == "drop"
            and landmark.bar == target_bar
            and (landmark.beat or 1) == 1
            and landmark.confidence == "verified"
            and landmark.name.casefold().startswith("user verified")
        ),
        None,
    )
    if user_verified_drop is not None:
        return {
            **evidence,
            "verified": True,
            "reason": None,
            "source": "user_verified_drop",
            "landmark": user_verified_drop.model_dump(),
        }
    return evidence


def _grid_aligned_phrases(
    phrases: list[PhraseBoundary],
    beat_grid: list[AnalysisBeatGridPoint],
) -> list[PhraseBoundary]:
    """Map PSSI phrase indices onto PQTZ bar/beat labels.

    PSSI's start_beat is an absolute grid index. Reducing it modulo four
    silently shifts every phrase on tracks with pickup beats—the recurring
    one- and two-beat live error this performer must never accept.
    """
    grid_by_index = {point.index: point for point in beat_grid}
    return [
        phrase.model_copy(
            update={
                "start_bar": grid_by_index[phrase.start_beat].bar,
                "beat_in_bar": grid_by_index[phrase.start_beat].beat,
            }
        )
        if phrase.start_beat in grid_by_index
        else phrase
        for phrase in phrases
    ]


class ProfileStore:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.data_dir / "performance-state.sqlite3"
        self.profile_path = self.data_dir / "track-profiles.json"
        self.rehearsal_path = self.data_dir / "rehearsals.jsonl"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    track_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rehearsals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    outgoing_track_id TEXT NOT NULL,
                    incoming_track_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    payload TEXT NOT NULL,
                    recorded_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS rehearsals_pair
                    ON rehearsals(outgoing_track_id, incoming_track_id);
                """
            )
            profile_count = connection.execute(
                "SELECT COUNT(*) FROM profiles"
            ).fetchone()[0]
            if profile_count == 0 and self.profile_path.exists():
                profiles = json.loads(self.profile_path.read_text(encoding="utf-8"))
                now = time.time()
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO profiles(track_id, payload, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (
                            track_id,
                            json.dumps(payload, ensure_ascii=False),
                            now,
                        )
                        for track_id, payload in profiles.items()
                    ],
                )
            rehearsal_count = connection.execute(
                "SELECT COUNT(*) FROM rehearsals"
            ).fetchone()[0]
            if rehearsal_count == 0 and self.rehearsal_path.exists():
                records = [
                    json.loads(line)
                    for line in self.rehearsal_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                ]
                connection.executemany(
                    """
                    INSERT INTO rehearsals(
                        outgoing_track_id,
                        incoming_track_id,
                        score,
                        payload,
                        recorded_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            record["outgoing_track_id"],
                            record["incoming_track_id"],
                            float(record["score"]),
                            json.dumps(record, ensure_ascii=False),
                            float(record.get("recorded_at", time.time())),
                        )
                        for record in records
                    ],
                )

    def _load_profiles(self) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT track_id, payload FROM profiles"
            ).fetchall()
        return {row["track_id"]: json.loads(row["payload"]) for row in rows}

    def _write_profiles(
        self,
        profiles: dict[str, dict[str, Any]],
    ) -> None:
        if not profiles:
            return
        now = time.time()
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO profiles(track_id, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(track_id) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                [
                    (
                        track_id,
                        json.dumps(payload, ensure_ascii=False),
                        now,
                    )
                    for track_id, payload in profiles.items()
                ],
            )

    def upsert(self, profile: TrackProfile) -> dict[str, Any]:
        self._write_profiles({profile.track_id: profile.model_dump()})
        return {
            "profile": profile.model_dump(),
            "readiness": profile.readiness(),
        }

    def ingest_analysis(
        self,
        analysis: RekordboxAnalysisImport,
        *,
        display_title: str | None = None,
        display_artist: str | None = None,
    ) -> dict[str, Any]:
        """Merge native Rekordbox beat/phrase truth into a performance profile."""
        profiles = self._load_profiles()
        existing_raw = profiles.get(analysis.track_id)
        phrases = _grid_aligned_phrases(
            analysis.phrases,
            analysis.beat_grid,
        )
        title = display_title or analysis.title
        artist = display_artist or analysis.artist
        if existing_raw:
            existing = TrackProfile.model_validate(existing_raw)
            title = title or existing.title
            artist = artist or existing.artist
            bpm = analysis.bpm or existing.bpm
            first_phrase_landmarks = [
                landmark
                for landmark in existing.landmarks
                if landmark.kind not in {"phrase_start"}
            ]
            profile = existing.model_copy(
                update={
                    "title": title,
                    "artist": artist,
                    "bpm": bpm,
                    "key": analysis.key or existing.key,
                    "source": analysis.source,
                    "service_uri": analysis.service_uri,
                    "beat_count": analysis.beat_count or len(analysis.beat_grid),
                    "beat_grid": analysis.beat_grid,
                    "beatgrid_confidence": (
                        "high" if analysis.beat_grid else "unknown"
                    ),
                    "phrase_confidence": ("high" if phrases else "unknown"),
                    "phrase_boundaries": phrases,
                    "bass_energy_by_bar": (
                        analysis.bass_energy_by_bar or existing.bass_energy_by_bar
                    ),
                    "landmarks": first_phrase_landmarks,
                    "vocal_confidence": (
                        "high"
                        if analysis.vocal_analysis_available
                        else existing.vocal_confidence
                    ),
                    "segments": [
                        segment
                        for segment in existing.segments
                        if segment.kind not in {"vocal", "bass"}
                    ]
                    + analysis.vocal_segments
                    + analysis.bass_segments,
                }
            )
        else:
            if not title:
                raise ValueError(
                    "display_title is required when streaming metadata is encrypted"
                )
            if not artist:
                raise ValueError(
                    "display_artist is required when streaming metadata is encrypted"
                )
            if not analysis.bpm:
                raise ValueError("analysis must include BPM for a new profile")
            profile = TrackProfile(
                track_id=analysis.track_id,
                title=title,
                artist=artist,
                bpm=analysis.bpm,
                key=analysis.key,
                source=analysis.source,
                service_uri=analysis.service_uri,
                beat_count=analysis.beat_count or len(analysis.beat_grid),
                beat_grid=analysis.beat_grid,
                beatgrid_confidence=("high" if analysis.beat_grid else "unknown"),
                phrase_confidence=("high" if phrases else "unknown"),
                phrase_boundaries=phrases,
                bass_energy_by_bar=analysis.bass_energy_by_bar,
                vocal_confidence=(
                    "high" if analysis.vocal_analysis_available else "unknown"
                ),
                segments=analysis.vocal_segments + analysis.bass_segments,
            )
        if phrases:
            first = phrases[0]
            first_time_ms = next(
                (
                    point.time_ms
                    for point in analysis.beat_grid
                    if point.index == first.start_beat
                ),
                None,
            )
            phrase_landmark = TrackLandmark(
                name=f"Rekordbox phrase 1: {first.label}",
                kind="phrase_start",
                bar=first.start_bar,
                beat=first.beat_in_bar,
                time_ms=first_time_ms,
                confidence="high",
            )
            profile.landmarks = [
                landmark
                for landmark in profile.landmarks
                if landmark.kind != "phrase_start"
            ] + [phrase_landmark]
            outro = next(
                (
                    phrase
                    for phrase in reversed(phrases)
                    if phrase.label.casefold() == "outro"
                ),
                None,
            )
            if outro and not any(
                landmark.kind == "mix_out" for landmark in profile.landmarks
            ):
                outro_time_ms = next(
                    (
                        point.time_ms
                        for point in analysis.beat_grid
                        if point.index == outro.start_beat
                    ),
                    None,
                )
                profile.landmarks.append(
                    TrackLandmark(
                        name="Rekordbox analyzed outro phrase start",
                        kind="mix_out",
                        bar=outro.start_bar,
                        beat=outro.beat_in_bar,
                        time_ms=outro_time_ms,
                        confidence="high",
                    )
                )
        if analysis.bass_energy_by_bar and phrases:
            profile.landmarks = [
                landmark
                for landmark in profile.landmarks
                if landmark.kind not in {"bass_in", "bass_out"}
            ]
            strong_phrases = []
            for phrase in phrases:
                if phrase.beat_in_bar != 1 or phrase.confidence not in {
                    "verified",
                    "high",
                }:
                    continue
                evidence = bass_phrase_evidence(profile, phrase.start_bar)
                if not evidence["verified"]:
                    continue
                point = next(
                    (
                        item
                        for item in analysis.beat_grid
                        if item.index == phrase.start_beat
                    ),
                    None,
                )
                strong_phrases.append(
                    TrackLandmark(
                        name=(
                            "Rekordbox PWV7 strong low-energy phrase "
                            f"({evidence['score']:.1f})"
                        ),
                        kind="bass_in",
                        bar=phrase.start_bar,
                        beat=1,
                        time_ms=point.time_ms if point else None,
                        confidence="high",
                    )
                )
            profile.landmarks.extend(strong_phrases)
            if analysis.bass_segments:
                last_bass = max(
                    analysis.bass_segments,
                    key=lambda segment: segment.end_bar,
                )
                profile.landmarks.append(
                    TrackLandmark(
                        name="Rekordbox 3-band low-energy exit",
                        kind="bass_out",
                        bar=last_bass.end_bar,
                        confidence="high",
                    )
                )
        elif analysis.bass_segments:
            first_bass = min(
                analysis.bass_segments,
                key=lambda segment: segment.start_bar,
            )
            last_bass = max(
                analysis.bass_segments,
                key=lambda segment: segment.end_bar,
            )
            profile.landmarks = [
                landmark
                for landmark in profile.landmarks
                if landmark.kind not in {"bass_in", "bass_out"}
            ]
            profile.landmarks.extend(
                [
                    TrackLandmark(
                        name="Rekordbox 3-band low-energy entry",
                        kind="bass_in",
                        bar=first_bass.start_bar,
                        confidence="high",
                    ),
                    TrackLandmark(
                        name="Rekordbox 3-band low-energy exit",
                        kind="bass_out",
                        bar=max(first_bass.start_bar + 1, last_bass.end_bar),
                        confidence="high",
                    ),
                ]
            )
        return self.upsert(profile)

    def seed_playlist(
        self,
        tracks: list[PlaylistTrackMetadata],
    ) -> dict[str, Any]:
        profiles = self._load_profiles()
        created = 0
        preserved = 0
        skipped = []
        for track in tracks:
            if track.bpm <= 0:
                skipped.append(
                    {
                        "track_id": track.track_id,
                        "title": track.title,
                        "reason": "BPM must be greater than zero",
                    }
                )
                continue
            if track.track_id in profiles:
                preserved += 1
                continue
            profile = TrackProfile(
                **track.model_dump(),
                beatgrid_confidence="unknown",
                phrase_confidence="unknown",
                vocal_confidence="unknown",
            )
            profiles[track.track_id] = profile.model_dump()
            created += 1
        eligible_ids = [track.track_id for track in tracks if track.bpm > 0]
        self._write_profiles(
            {
                track_id: profiles[track_id]
                for track_id in eligible_ids
                if track_id in profiles
            }
        )
        audit = self.audit(eligible_ids)
        return {
            "track_count": len(tracks),
            "created": created,
            "preserved": preserved,
            "skipped": skipped,
            "audit": audit,
        }

    def get(self, track_id: str) -> TrackProfile:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM profiles WHERE track_id = ?",
                (track_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"No performance profile for track {track_id}")
        return TrackProfile.model_validate(json.loads(row["payload"]))

    def list_profiles(self, *, ready_only: bool = False) -> list[TrackProfile]:
        """Return prepared profiles for the standalone local DJ planner."""
        profiles = [
            TrackProfile.model_validate(payload)
            for payload in self._load_profiles().values()
        ]
        if ready_only:
            profiles = [profile for profile in profiles if profile.readiness()["ready"]]
        return sorted(
            profiles, key=lambda item: (item.artist.casefold(), item.title.casefold())
        )

    def audit(self, track_ids: list[str] | None = None) -> dict[str, Any]:
        profiles = self._load_profiles()
        wanted = track_ids or list(profiles)
        results = []
        for track_id in wanted:
            if track_id not in profiles:
                results.append(
                    {"track_id": track_id, "ready": False, "missing_profile": True}
                )
                continue
            profile = TrackProfile.model_validate(profiles[track_id])
            results.append(
                {
                    "track_id": track_id,
                    "title": profile.title,
                    **profile.readiness(),
                    "analysis_readiness": profile.analysis_readiness(),
                }
            )
        return {
            "ready": bool(results) and all(item["ready"] for item in results),
            "tracks": results,
        }

    def record_rehearsal(self, review: RehearsalReview) -> dict[str, Any]:
        record = {
            **review.model_dump(),
            "score": review.score(),
            "recorded_at": time.time(),
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rehearsals(
                    outgoing_track_id,
                    incoming_track_id,
                    score,
                    payload,
                    recorded_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    review.outgoing_track_id,
                    review.incoming_track_id,
                    float(record["score"]),
                    json.dumps(record, ensure_ascii=False),
                    float(record["recorded_at"]),
                ),
            )
        return record

    def rehearsal_summary(
        self,
        outgoing_track_id: str | None = None,
        incoming_track_id: str | None = None,
    ) -> dict[str, Any]:
        clauses = []
        parameters: list[str] = []
        if outgoing_track_id:
            clauses.append("outgoing_track_id = ?")
            parameters.append(outgoing_track_id)
        if incoming_track_id:
            clauses.append("incoming_track_id = ?")
            parameters.append(incoming_track_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT payload FROM rehearsals{where} ORDER BY id",
                parameters,
            ).fetchall()
        records = [json.loads(row["payload"]) for row in rows]
        average = (
            round(sum(item["score"] for item in records) / len(records), 2)
            if records
            else None
        )
        return {"count": len(records), "average_score": average, "reviews": records}

    def proficient_techniques(self) -> set[str]:
        """Return advanced techniques promoted by clean rehearsal evidence.

        Promotion requires three clean passes across two materially different
        track pairs.  Foundational dry blends/cuts are handled by the compiler;
        this registry gates techniques whose failure depends on extra mutable
        state such as stems or double-drop ownership.
        """
        records = self.rehearsal_summary()["reviews"]
        aliases = {
            "stem_vocal_blend": ("stem", "acapella"),
            "double_drop": ("double-drop", "double_drop", "double drop"),
        }
        proficient: set[str] = set()
        for technique, markers in aliases.items():
            clean = [
                item
                for item in records
                if any(
                    marker in str(item.get("transition_name", "")).casefold()
                    for marker in markers
                )
                and int(item.get("bar_error", 1)) == 0
                and float(item.get("bass_swap_error_beats", 9)) <= 0.15
                and not bool(item.get("vocal_clash"))
                and int(item.get("energy_continuity", 0)) >= 7
                and int(item.get("cleanliness", 0)) >= 7
                and int(item.get("user_rating", 0)) >= 7
            ]
            pairs = {
                (item.get("outgoing_track_id"), item.get("incoming_track_id"))
                for item in clean
            }
            if len(clean) >= 3 and len(pairs) >= 2:
                proficient.add(technique)
        return proficient


def _profile_live_ready(
    profile: TrackProfile,
    role: str,
    *,
    vocal_risk_accepted: bool = False,
) -> list[str]:
    readiness = profile.readiness()["checks"]
    required = ["beatgrid", "phrases", "vocals", "bass"]
    if vocal_risk_accepted:
        required.remove("vocals")
    required.append("mix_out" if role == "outgoing" else "mix_in")
    return [name for name in required if not readiness[name]]


def _card_accepts_vocal_risk(card: TransitionCard) -> bool:
    return card.vocal_risk_accepted or (
        card.vocal_plan_verified and card.intended_vocal_owner == "intentional_overlap"
    )


def validate_transition_card(
    card: TransitionCard,
    outgoing: TrackProfile,
    incoming: TrackProfile,
) -> list[str]:
    errors: list[str] = []
    vocal_risk_accepted = _card_accepts_vocal_risk(card)
    harmonic = camelot_compatibility(outgoing.key, incoming.key)
    if card.outgoing_track_id != outgoing.track_id:
        errors.append("outgoing profile does not match card")
    if card.incoming_track_id != incoming.track_id:
        errors.append("incoming profile does not match card")
    bpm_delta = abs(outgoing.bpm - incoming.bpm)
    stretch_percent = bpm_delta / outgoing.bpm * 100.0
    if bpm_delta > 0.05 and not card.beat_sync_required:
        errors.append("tempo-mismatched tracks require verified Beat Sync")
    if (
        stretch_percent > card.max_tempo_stretch_percent
        and not card.tempo_risk_accepted
    ):
        errors.append(
            "native BPM move requires a verified tempo ramp or compatible "
            "bridge track "
            f"({outgoing.bpm:.2f} -> {incoming.bpm:.2f}, "
            f"{stretch_percent:.2f}% exceeds "
            f"{card.max_tempo_stretch_percent:.2f}%)"
        )
    for label, value in (
        ("phrase alignment", card.phrase_alignment_verified),
        ("bass plan", card.bass_plan_verified),
        ("incoming load", card.incoming_loaded_verified),
    ):
        if not value:
            errors.append(f"{label} is not verified")
    if not (card.vocal_plan_verified or vocal_risk_accepted):
        errors.append("vocal plan is not verified and vocal risk is not accepted")
    if not card.harmonic_risk_accepted:
        if not harmonic["verified"]:
            errors.append(
                "harmonic relationship is unknown; verify both keys or "
                "explicitly accept harmonic risk"
            )
        elif not harmonic["compatible"]:
            errors.append(
                "harmonic move is outside same/relative/adjacent Camelot keys "
                f"({harmonic['outgoing']} -> {harmonic['incoming']}); "
                "choose a compatible track or explicitly accept harmonic risk"
            )
    for missing in _profile_live_ready(
        outgoing,
        "outgoing",
        vocal_risk_accepted=vocal_risk_accepted,
    ):
        errors.append(f"outgoing profile missing {missing}")
    for missing in _profile_live_ready(
        incoming,
        "incoming",
        vocal_risk_accepted=vocal_risk_accepted,
    ):
        errors.append(f"incoming profile missing {missing}")

    unsafe_mode_toggles = [
        event.action for event in card.events if event.action in {"sync", "quantize"}
    ]
    if unsafe_mode_toggles:
        errors.append(
            "transition cards may not contain blind Sync/Quantize toggles; "
            "use ensure_deck_modes and re-observe first"
        )

    ordered = sorted(
        card.events, key=lambda event: (event.bar_offset, event.beat_offset)
    )
    if ordered != card.events:
        errors.append("musical events must be ordered by bar_offset and beat_offset")

    drop_anchored = card.transition_family in {
        "long_blend",
        "bass_swap",
        "double_drop",
    }
    critical = [
        event
        for event in card.events
        if event.bar_offset == card.critical_bar_offset and event.beat_offset == 0
    ]
    outgoing_low_cut = any(
        event.action == "eq_low"
        and event.parameters.get("deck") == card.outgoing_deck
        and float(event.parameters.get("value", 0)) <= -0.9
        for event in critical
    )
    incoming_low_open = any(
        event.action == "eq_low"
        and event.parameters.get("deck") == card.incoming_deck
        and float(event.parameters.get("value", -1)) >= -0.1
        for event in critical
    )
    any_outgoing_low_cut = any(
        event.action == "eq_low"
        and event.parameters.get("deck") == card.outgoing_deck
        and float(event.parameters.get("value", 0)) <= -0.9
        for event in card.events
    )
    any_incoming_low_open = any(
        event.action == "eq_low"
        and event.parameters.get("deck") == card.incoming_deck
        and float(event.parameters.get("value", -1)) >= -0.1
        for event in card.events
    )
    planned_bass_handoff = any_outgoing_low_cut and any_incoming_low_open
    critical_bass_handoff = outgoing_low_cut and incoming_low_open
    verified_bass_anchor_required = drop_anchored or planned_bass_handoff
    if drop_anchored and not (outgoing_low_cut and incoming_low_open):
        errors.append("critical downbeat lacks a simultaneous two-deck bass swap")
    if planned_bass_handoff and not critical_bass_handoff:
        errors.append(
            "bass handoff must swap both low EQs together on the critical downbeat"
        )
    incoming_audible_events = [
        event
        for event in card.events
        if event.action == "channel_fader"
        and event.parameters.get("deck") == card.incoming_deck
        and float(event.parameters.get("value", 0)) > 0
    ]
    audible_entry_position = min(
        ((event.bar_offset, event.beat_offset) for event in incoming_audible_events),
        default=None,
    )
    audible_lead_bars = (
        None
        if audible_entry_position is None
        else card.critical_bar_offset - audible_entry_position[0]
    )
    energy_preserving_house = card.transition_family in {
        "long_blend",
        "bass_swap",
    }
    if energy_preserving_house:
        incoming_low_closed = any(
            event.action == "eq_low"
            and event.parameters.get("deck") == card.incoming_deck
            and float(event.parameters.get("value", 0)) <= -0.9
            and event.bar_offset == 0
            and event.beat_offset == 0
            for event in card.events
        )
        if not incoming_low_closed:
            errors.append("house handoff must launch with the incoming low EQ closed")
        outgoing_reductions_before_swap = [
            event
            for event in card.events
            if event.action == "channel_fader"
            and event.parameters.get("deck") == card.outgoing_deck
            and float(event.parameters.get("value", 1)) < 0.9
            and (event.bar_offset, event.beat_offset) <= (card.critical_bar_offset, 0)
        ]
        if outgoing_reductions_before_swap:
            errors.append(
                "outgoing channel fader must stay at or above 0.9 until "
                "after the bass handoff"
            )
        establish_lead = min(4, max(2, int((audible_lead_bars or 4) / 2)))
        establish_deadline = max(0, card.critical_bar_offset - establish_lead)
        incoming_established = any(
            event.action == "channel_fader"
            and event.parameters.get("deck") == card.incoming_deck
            and float(event.parameters.get("value", 0)) >= 0.7
            and (event.bar_offset, event.beat_offset) <= (establish_deadline, 0)
            for event in card.events
        )
        if not incoming_established:
            errors.append(
                "incoming channel must reach at least 0.7 before the final "
                f"{establish_lead}-bar approach to the bass handoff"
            )
        incoming_reductions_after_established = []
        established = False
        for event in card.events:
            if (
                event.action == "channel_fader"
                and event.parameters.get("deck") == card.incoming_deck
            ):
                value = float(event.parameters.get("value", 0))
                if value >= 0.7:
                    established = True
                elif established:
                    incoming_reductions_after_established.append(event)
        if incoming_reductions_after_established:
            errors.append("incoming channel may not be lowered after it is established")
    if card.transition_family == "phrase_cut" and card.critical_bar_offset > 4:
        errors.append(
            "phrase_cut must complete within four bars; use a verified blend, "
            "bass swap, breakdown, or loop card for a longer handoff"
        )
    bass_evidence = incoming_bass_handoff_evidence(card, incoming)
    if (
        planned_bass_handoff
        and bass_evidence is not None
        and not bass_evidence.get("verified")
        and not card.intentional_energy_drop
    ):
        errors.append(
            "incoming bass phrase cannot carry the low-EQ handoff: "
            f"{bass_evidence.get('reason') or 'waveform evidence failed'}; "
            "choose a stronger phrase or explicitly mark an intentional "
            "energy drop"
        )

    launch_events = [
        event
        for event in card.events
        if event.bar_offset == 0
        and event.beat_offset == 0
        and event.action in {"hot_cue", "play_pause"}
        and event.parameters.get("deck") == card.incoming_deck
    ]
    verified_entry_cues = {
        landmark.cue
        for landmark in incoming.landmarks
        if landmark.kind in {"mix_in", "phrase_start"}
        and landmark.cue is not None
        and landmark.confidence in {"verified", "high"}
    }
    if len(launch_events) != 1:
        errors.append(
            "incoming deck must launch exactly once on transition bar 0 beat 1"
        )
    else:
        launch = launch_events[0]
        file_start_entries: list[TrackLandmark] = []
        if (
            launch.action == "hot_cue"
            and launch.parameters.get("cue") not in verified_entry_cues
        ):
            errors.append(
                "incoming launch cue is not a high-confidence mix-in or "
                "phrase-start landmark"
            )
        elif launch.action == "play_pause":
            file_start_entries = [
                landmark
                for landmark in incoming.landmarks
                if landmark.kind in {"mix_in", "phrase_start"}
                and landmark.bar == 1
                and (landmark.beat or 1) == 1
                and landmark.confidence in {"verified", "high"}
            ]
            if not file_start_entries:
                errors.append(
                    "play_pause launch requires a high-confidence file-start "
                    "mix-in or phrase-start landmark"
                )
        verified_phrase_starts = {
            (phrase.start_bar, phrase.beat_in_bar)
            for phrase in incoming.phrase_boundaries
            if phrase.confidence in {"verified", "high"}
        }
        if verified_bass_anchor_required and launch.action == "hot_cue":
            if card.critical_bar_offset not in {8, 16}:
                errors.append(
                    "bass handoff must land 8 or 16 bars after the incoming cue"
                )
            entry = next(
                (
                    landmark
                    for landmark in incoming.landmarks
                    if landmark.cue == launch.parameters.get("cue")
                    and landmark.kind in {"mix_in", "phrase_start"}
                    and landmark.confidence in {"verified", "high"}
                ),
                None,
            )
            if entry is not None:
                if (entry.beat or 1) != 1:
                    errors.append("incoming launch cue must mark beat 1 of its phrase")
                if (entry.bar, entry.beat or 1) not in verified_phrase_starts:
                    errors.append(
                        "incoming launch cue does not land on a verified "
                        "phrase boundary"
                    )
                entry_beat = (entry.bar - 1) * incoming.time_signature + (
                    entry.beat or 1
                )
                critical_beat = (
                    entry_beat + card.critical_bar_offset * incoming.time_signature
                )
                verified_bass_phrase_starts = {
                    (landmark.bar - 1) * incoming.time_signature + (landmark.beat or 1)
                    for landmark in incoming.landmarks
                    if landmark.kind in {"drop", "bass_in"}
                    and landmark.confidence in {"verified", "high"}
                    and (landmark.bar, landmark.beat or 1) in verified_phrase_starts
                }
                if any(
                    landmark.kind in {"drop", "bass_in", "bass_out"}
                    for landmark in incoming.landmarks
                ) or any(segment.kind == "bass" for segment in incoming.segments):
                    verified_bass_phrase_starts.update(
                        (phrase.start_bar - 1) * incoming.time_signature
                        + phrase.beat_in_bar
                        for phrase in incoming.phrase_boundaries
                        if phrase.confidence in {"verified", "high"}
                        and energy_phrase_evidence(incoming, phrase.start_bar).get(
                            "verified"
                        )
                    )
                if critical_beat not in verified_bass_phrase_starts:
                    errors.append(
                        "critical bass swap does not land on the downbeat of "
                        "a verified incoming drop or bass phrase"
                    )
        elif (
            verified_bass_anchor_required
            and launch.action == "play_pause"
            and file_start_entries
        ):
            entry = file_start_entries[0]
            if audible_entry_position is None:
                errors.append(
                    "file-start bass handoff never opens the incoming channel"
                )
            else:
                reveal_bar_offset, reveal_beat_offset = audible_entry_position
                audible_bar = entry.bar + reveal_bar_offset
                if (
                    reveal_beat_offset != 0
                    or (audible_bar, 1) not in verified_phrase_starts
                ):
                    errors.append(
                        "file-start pre-roll must become audible on beat 1 of a "
                        "verified incoming phrase"
                    )
                if audible_lead_bars not in {4, 8, 16}:
                    errors.append(
                        "file-start pre-roll must become audible 4, 8, or 16 "
                        "bars before the bass handoff"
                    )
                if reveal_bar_offset == 0 and card.critical_bar_offset == 4:
                    short_drop = any(
                        landmark.kind == "drop"
                        and landmark.bar == entry.bar + card.critical_bar_offset
                        and (landmark.beat or 1) == 1
                        and landmark.confidence == "verified"
                        for landmark in incoming.landmarks
                    )
                    if not short_drop:
                        errors.append(
                            "four-bar file-start blend requires an explicitly "
                            "verified incoming drop"
                        )

    outgoing_fader_zero: tuple[int, int] | None = None
    outgoing_stop: tuple[int, int] | None = None
    for event in card.events:
        position = (event.bar_offset, event.beat_offset)
        if (
            event.action == "channel_fader"
            and event.parameters.get("deck") == card.outgoing_deck
            and float(event.parameters.get("value", 1)) == 0.0
        ):
            outgoing_fader_zero = position
        if event.action == "cue" and event.parameters.get("deck") == card.outgoing_deck:
            outgoing_stop = position
    if outgoing_fader_zero is None:
        errors.append("outgoing deck never reaches channel fader zero")
    if outgoing_stop is None:
        errors.append("outgoing deck is not stopped with cue")
    if (
        outgoing_fader_zero is not None
        and outgoing_stop is not None
        and outgoing_stop < outgoing_fader_zero
    ):
        errors.append("outgoing deck is stopped before it is silent")
    if card.transition_family == "long_blend":
        progressive_retirement = [
            event
            for event in card.events
            if event.action == "channel_fader"
            and event.parameters.get("deck") == card.outgoing_deck
            and 0 < float(event.parameters.get("value", 1)) < 0.9
            and (event.bar_offset, event.beat_offset) > (card.critical_bar_offset, 0)
        ]
        if len(progressive_retirement) < 3:
            errors.append(
                "long blend must retire the outgoing channel through at least "
                "three progressive post-swap fader steps"
            )
        if (
            outgoing_fader_zero is None
            or outgoing_fader_zero[0] < card.critical_bar_offset + 4
        ):
            errors.append(
                "long blend must keep the outgoing tail alive for at least "
                "four bars after the bass swap"
            )
        outgoing_filter_moves = [
            event
            for event in card.events
            if event.action == "filter"
            and event.parameters.get("deck") == card.outgoing_deck
            and float(event.parameters.get("value", 0)) != 0
            and (event.bar_offset, event.beat_offset) > (card.critical_bar_offset, 0)
        ]
        filter_reset = any(
            event.action == "filter"
            and event.parameters.get("deck") == card.outgoing_deck
            and float(event.parameters.get("value", 1)) == 0
            and outgoing_fader_zero is not None
            and (event.bar_offset, event.beat_offset) >= outgoing_fader_zero
            for event in card.events
        )
        if not outgoing_filter_moves or not filter_reset:
            errors.append(
                "long blend requires a post-swap outgoing CFX move and a "
                "neutral filter reset at retirement"
            )
    outgoing_fx_on = [
        index
        for index, event in enumerate(card.events)
        if event.action == "fx_toggle"
        and event.parameters.get("deck") == card.outgoing_deck
    ]
    if outgoing_fx_on:
        final_toggle = outgoing_fx_on[-1]
        wet_reset = any(
            event.action == "fx_wet_dry"
            and event.parameters.get("deck") == card.outgoing_deck
            and float(event.parameters.get("value", 1)) == 0.0
            for event in card.events[final_toggle - 1 :]
        )
        if len(outgoing_fx_on) % 2 or not wet_reset:
            errors.append("outgoing FX plan must toggle off and reset wet/dry to zero")
    stem_events: dict[tuple[int, str], int] = {}
    for event in card.events:
        if event.action not in {
            "stem_vocal",
            "stem_instrumental",
            "stem_drums",
        }:
            continue
        key = (int(event.parameters.get("deck", 0)), event.action)
        stem_events[key] = stem_events.get(key, 0) + 1
    for (deck, action), count in stem_events.items():
        if count % 2:
            errors.append(f"deck {deck} {action} must be restored with a paired toggle")

    for index, event in enumerate(card.events):
        if event.action not in {"loop_4", "loop_8", "loop_16"}:
            continue
        if not card.loop_plan_verified:
            errors.append("transition loop requires verified observable loop state")
        deck = event.parameters.get("deck")
        if not any(
            later.action == "loop_toggle" and later.parameters.get("deck") == deck
            for later in card.events[index + 1 :]
        ):
            errors.append(f"deck {deck} transition loop is not explicitly released")
    return errors


def compile_transition_card(
    card: TransitionCard,
    outgoing: TrackProfile,
    incoming: TrackProfile,
    live_state: LiveState,
    *,
    max_observation_age_ms: int = 1000,
) -> dict[str, Any]:
    errors = validate_transition_card(card, outgoing, incoming)
    try:
        state = live_state.get(card.anchor_deck)
    except KeyError as exc:
        return {"ready": False, "errors": [str(exc)], "events": []}
    try:
        incoming_state = live_state.get(card.incoming_deck)
    except KeyError as exc:
        errors.append(str(exc))
        incoming_state = None
    if state["track_id"] != (
        outgoing.track_id
        if card.anchor_deck == card.outgoing_deck
        else incoming.track_id
    ):
        errors.append("anchor deck track does not match transition card")
    anchor_launches = [
        event
        for event in card.events
        if event.bar_offset == 0
        and event.beat_offset == 0
        and event.action == "hot_cue"
        and event.parameters.get("deck") == card.anchor_deck
    ]
    stopped_dual_launch = not state["playing"] and len(anchor_launches) == 1
    if not state["playing"] and not stopped_dual_launch:
        errors.append("anchor deck is not playing")
    if state["observation_age_ms"] > max_observation_age_ms:
        errors.append(f"anchor observation is stale ({state['observation_age_ms']} ms)")
    if state["confidence"] not in {"verified", "high"}:
        errors.append("anchor observation confidence is below high")
    if state["source"] == "manual":
        errors.append("manual observations are not authoritative live clocks")
    if state["source"] == "midi":
        errors.append(
            "outbound MIDI commands are not authoritative deck observations; "
            "use verified native transport feedback"
        )
    if state["source"] == "vision" and state["confidence"] != "verified":
        errors.append("vision observations require verified adapter confidence")
    if incoming_state is not None:
        if incoming_state["track_id"] != incoming.track_id:
            errors.append("incoming deck track does not match transition card")
        if incoming_state["observation_age_ms"] > max_observation_age_ms:
            errors.append(
                "incoming observation is stale "
                f"({incoming_state['observation_age_ms']} ms)"
            )
        if incoming_state["confidence"] not in {"verified", "high"}:
            errors.append("incoming observation confidence is below high")
        if incoming_state["source"] == "manual":
            errors.append(
                "manual incoming observations are not authoritative live state"
            )
        if incoming_state["source"] == "midi":
            errors.append(
                "outbound MIDI commands are not authoritative incoming-deck "
                "state; use verified native transport feedback"
            )
        if (
            incoming_state["source"] == "vision"
            and incoming_state["confidence"] != "verified"
        ):
            errors.append(
                "incoming vision observations require verified adapter confidence"
            )
        if incoming_state["playing"]:
            errors.append("incoming deck must be stopped before its scheduled launch")
        if card.beat_sync_required:
            if state["sync_enabled"] is None:
                errors.append("outgoing Beat Sync state is unobserved")
            elif not state["sync_enabled"]:
                errors.append("outgoing Beat Sync is off")
            if incoming_state["sync_enabled"] is None:
                errors.append("incoming Beat Sync state is unobserved")
            elif not incoming_state["sync_enabled"]:
                errors.append("incoming Beat Sync is off")
            # A stopped Rekordbox deck continues to display its analyzed/native
            # BPM even with Sync armed.  Actual playback BPM is verified by the
            # post-launch sync guard before the incoming fader may rise.
        if card.quantize_required:
            if incoming_state["quantize_enabled"] is None:
                errors.append("incoming Quantize state is unobserved")
            elif not incoming_state["quantize_enabled"]:
                errors.append("incoming Quantize is off")
    anchor_profile = outgoing if card.anchor_deck == card.outgoing_deck else incoming
    phrases = anchor_profile.phrase_boundaries
    if not phrases:
        errors.append("anchor profile has no Rekordbox phrase boundaries")
    if errors:
        return {"ready": False, "errors": errors, "events": []}

    bpm = state["bpm"]
    beat_ms = 60_000.0 / bpm
    beats_per_bar = anchor_profile.time_signature
    current_beat = state["track_beat"] + state["beat_phase"]
    if stopped_dual_launch:
        anchor_launch = anchor_launches[0]
        cue = anchor_launch.parameters.get("cue")
        landmark = next(
            (
                item
                for item in anchor_profile.landmarks
                if item.cue == cue
                and item.kind in {"mix_in", "phrase_start"}
                and item.confidence in {"verified", "high"}
            ),
            None,
        )
        if landmark is None:
            return {
                "ready": False,
                "errors": [
                    "stopped anchor launch cue is not a high-confidence "
                    "mix-in or phrase-start landmark"
                ],
                "events": [],
            }
        if (landmark.beat or 1) != 1:
            return {
                "ready": False,
                "errors": ["stopped anchor launch cue must mark phrase beat 1"],
                "events": [],
            }
        start_delay_ms = 250
        events = []
        for event in card.events:
            offset_beats = (
                event.bar_offset * anchor_profile.time_signature + event.beat_offset
            )
            events.append(
                {
                    "at_ms": start_delay_ms + round(offset_beats * beat_ms),
                    "action": event.action,
                    "parameters": event.parameters,
                }
            )
        return {
            "ready": True,
            "errors": [],
            "harmonic_compatibility": camelot_compatibility(
                outgoing.key,
                incoming.key,
            ),
            "bpm": state["bpm"],
            "anchor": state,
            "incoming": incoming_state,
            "start_basis": "verified_dual_hot_cue",
            "start_delay_ms": start_delay_ms,
            "start_track_beat": (
                (landmark.bar - 1) * anchor_profile.time_signature
                + (landmark.beat or 1)
            ),
            "start_bar": landmark.bar,
            "start_beat_in_bar": landmark.beat or 1,
            "start_phrase": None,
            "incoming_bass_handoff": incoming_bass_handoff_evidence(card, incoming),
            "events": events,
        }
    minimum_lead_beats = card.minimum_lead_bars * beats_per_bar
    earliest = current_beat + minimum_lead_beats
    candidates = list(phrases)
    if card.start_phrase_index is not None:
        candidates = [
            phrase for phrase in candidates if phrase.index == card.start_phrase_index
        ]
    if card.start_phrase_label:
        label = card.start_phrase_label.casefold()
        candidates = [
            phrase for phrase in candidates if phrase.label.casefold() == label
        ]
    candidates = [phrase for phrase in candidates if phrase.start_beat >= earliest]
    if not candidates:
        return {
            "ready": False,
            "errors": [
                "no analyzed phrase boundary satisfies the requested start "
                "and minimum lead"
            ],
            "events": [],
        }
    chosen_phrase = min(candidates, key=lambda phrase: phrase.start_beat)
    if chosen_phrase.beat_in_bar != 1:
        return {
            "ready": False,
            "errors": [
                "selected outgoing phrase must start on beat 1; "
                f"analysis reports beat {chosen_phrase.beat_in_bar}"
            ],
            "events": [],
        }
    start_beat = chosen_phrase.start_beat
    start_delay_ms = round((start_beat - current_beat) * beat_ms)

    outgoing_stops = [
        event
        for event in card.events
        if event.action == "cue" and event.parameters.get("deck") == card.outgoing_deck
    ]
    if anchor_profile.beat_count and outgoing_stops:
        final_stop = max(
            outgoing_stops,
            key=lambda event: (event.bar_offset, event.beat_offset),
        )
        retirement_beat = (
            start_beat + final_stop.bar_offset * beats_per_bar + final_stop.beat_offset
        )
        if retirement_beat > anchor_profile.beat_count:
            return {
                "ready": False,
                "errors": [
                    "outgoing deck would reach the end of its analyzed beat "
                    "grid before the scheduled silent cue retirement"
                ],
                "events": [],
                "start_track_beat": start_beat,
                "outgoing_beat_count": anchor_profile.beat_count,
                "scheduled_retirement_beat": retirement_beat,
            }

    # A verified Hot Cue on a stopped Rekordbox deck starts immediately even
    # when Quantize is enabled.  It does not wait for the next master-deck beat.
    # Pre-arming that trigger therefore moves the incoming phrase off beat 1
    # by the lead interval itself (347 ms was measured as a ~one-beat miss in
    # the Jump -> The Rapture rehearsal).  Dispatch stopped-deck transports at
    # the compiled phrase boundary; Quantize remains a phase-safety net rather
    # than part of our scheduling calculation.
    quantized_transport_lead_ms = 0
    events = []
    for event in card.events:
        offset_beats = event.bar_offset * beats_per_bar + event.beat_offset
        at_ms = start_delay_ms + round(offset_beats * beat_ms)
        if (
            quantized_transport_lead_ms
            and event.bar_offset == 0
            and event.beat_offset == 0
            and event.action in {"play_pause", "hot_cue"}
            and event.parameters.get("deck") == card.incoming_deck
        ):
            at_ms = max(0, at_ms - quantized_transport_lead_ms)
        events.append(
            {
                "at_ms": at_ms,
                "action": event.action,
                "parameters": event.parameters,
            }
        )
    events.sort(key=lambda item: item["at_ms"])
    return {
        "ready": True,
        "errors": [],
        "harmonic_compatibility": camelot_compatibility(
            outgoing.key,
            incoming.key,
        ),
        "bpm": bpm,
        "anchor": state,
        "incoming": incoming_state,
        "start_basis": "rekordbox_phrase_analysis",
        "start_delay_ms": start_delay_ms,
        "start_track_beat": start_beat,
        "start_bar": chosen_phrase.start_bar,
        "start_beat_in_bar": chosen_phrase.beat_in_bar,
        "start_phrase": chosen_phrase.model_dump(),
        "incoming_bass_handoff": incoming_bass_handoff_evidence(card, incoming),
        "quantized_transport_lead_ms": quantized_transport_lead_ms,
        "events": events,
    }


def audit_set_plan(plan: SetPlan, store: ProfileStore) -> dict[str, Any]:
    errors: list[str] = []
    track_audit = store.audit(plan.track_ids)
    for index, card in enumerate(plan.cards):
        expected_outgoing = plan.track_ids[index]
        expected_incoming = plan.track_ids[index + 1]
        if card.outgoing_track_id != expected_outgoing:
            errors.append(
                f"card {index + 1} outgoing track is not set-plan track {index + 1}"
            )
        if card.incoming_track_id != expected_incoming:
            errors.append(
                f"card {index + 1} incoming track is not set-plan track {index + 2}"
            )
        try:
            outgoing = store.get(expected_outgoing)
            incoming = store.get(expected_incoming)
        except KeyError as exc:
            errors.append(str(exc))
            continue
        errors.extend(
            f"card {index + 1}: {error}"
            for error in validate_transition_card(card, outgoing, incoming)
        )
    if plan.preload_lead_bars < 32:
        errors.append("preload lead must be at least 32 bars for this workflow")
    waived_vocal_track_ids = {
        track_id
        for card in plan.cards
        if _card_accepts_vocal_risk(card)
        for track_id in (card.outgoing_track_id, card.incoming_track_id)
    }
    plan_track_ready = all(
        track["ready"]
        or (
            track["track_id"] in waived_vocal_track_ids
            and set(name for name, passed in track["checks"].items() if not passed)
            <= {"vocals"}
        )
        for track in track_audit["tracks"]
    )
    return {
        "ready": plan_track_ready and not errors,
        "errors": errors,
        "track_audit": track_audit,
        "preload_lead_bars": plan.preload_lead_bars,
        "rescue_loop_bars": plan.rescue_loop_bars,
        "cards": len(plan.cards),
    }
