"""Deterministic local DJ selection and TransitionKing card compilation.

The live application never asks an LLM to choose a track or author MIDI events.
Creative preferences become scored inputs; safety-critical transition rules are
ordinary Python with validation and repeatable tests.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, Field

from .intelligence import (
    MusicalEvent,
    PhraseBoundary,
    TrackLandmark,
    TrackProfile,
    TransitionCard,
    bass_phrase_evidence,
    camelot_compatibility,
    energy_phrase_evidence,
    validate_transition_card,
)
from .set_runner import AutonomousSetPlan, TrackLoadSpec, TransitionOption


class DJBrief(BaseModel):
    start_track_id: str
    target_track_count: int = Field(default=6, ge=2, le=100)
    name: str = "RekordBot set"
    target_bpm: float | None = Field(default=None, gt=0)
    target_track_id: str | None = None
    vibe: Literal[
        "maintain",
        "downtempo",
        "energetic",
        "deeper",
        "vocal",
        "instrumental",
    ] = "maintain"
    excluded_track_ids: list[str] = Field(default_factory=list)
    allowed_track_ids: list[str] = Field(default_factory=list)
    max_native_bpm_delta: float = Field(default=4.0, gt=0, le=12)
    allow_safe_cuts: bool = True


@dataclass(frozen=True)
class CompiledHandoff:
    card: TransitionCard
    load: TrackLoadSpec
    technique: str
    cue_landmark: TrackLandmark | None
    bass_landmark: TrackLandmark | None
    reason: str = ""
    alternatives: tuple[str, ...] = ()
    suitability_score: float = 0.0


@dataclass(frozen=True)
class EnergyHandoff:
    """One phrase-exact outgoing/incoming energy ownership pairing."""

    outgoing_start: PhraseBoundary
    outgoing_critical: PhraseBoundary
    incoming_entry: TrackLandmark | None
    incoming_audible: PhraseBoundary
    incoming_critical: PhraseBoundary
    bass_landmark: TrackLandmark | None
    critical_offset: int
    audible_entry_offset: int
    score: float
    reason: str


def _confidence_ready(value: str) -> bool:
    return value in {"high", "verified"}


def _phrase_at(profile: TrackProfile, bar: int) -> PhraseBoundary | None:
    return next(
        (
            phrase
            for phrase in profile.phrase_boundaries
            if phrase.start_bar == bar
            and phrase.beat_in_bar == 1
            and _confidence_ready(phrase.confidence)
        ),
        None,
    )


def _automation_entries(profile: TrackProfile) -> list[TrackLandmark]:
    entries = [
        landmark
        for landmark in profile.landmarks
        if landmark.kind in {"mix_in", "phrase_start"}
        and landmark.cue is not None
        and landmark.time_ms is not None
        and (landmark.beat or 1) == 1
        and _confidence_ready(landmark.confidence)
        and _phrase_at(profile, landmark.bar) is not None
    ]
    # G/H are owned by automation.  Existing user cues remain usable when
    # already verified, but never outrank an automation-owned cue.
    return sorted(
        entries,
        key=lambda item: (
            0 if item.cue in {7, 8} else 1,
            item.bar,
            item.cue or 9,
        ),
    )


def _verified_bass_handoff(
    profile: TrackProfile,
) -> tuple[TrackLandmark, TrackLandmark, int] | None:
    bass_landmarks = [
        landmark
        for landmark in profile.landmarks
        if landmark.kind in {"drop", "bass_in"}
        and (landmark.beat or 1) == 1
        and _confidence_ready(landmark.confidence)
        and _phrase_at(profile, landmark.bar) is not None
    ]
    for entry in _automation_entries(profile):
        for lead in (16, 8):
            bass = next(
                (
                    landmark
                    for landmark in bass_landmarks
                    if landmark.bar == entry.bar + lead
                ),
                None,
            )
            if bass is None:
                continue
            evidence = bass_phrase_evidence(profile, bass.bar, window_bars=8)
            if evidence.get("verified"):
                return entry, bass, lead
    return None


def _mix_out_bar(profile: TrackProfile) -> int:
    exits = [
        landmark.bar
        for landmark in profile.landmarks
        if landmark.kind == "mix_out" and _confidence_ready(landmark.confidence)
    ]
    if exits:
        return min(exits)
    return max((phrase.start_bar for phrase in profile.phrase_boundaries), default=1)


def _outgoing_start_phrase(
    profile: TrackProfile,
    *,
    required_bars: int,
) -> PhraseBoundary:
    limit = _mix_out_bar(profile) - required_bars
    choices = [
        phrase
        for phrase in profile.phrase_boundaries
        if phrase.start_bar <= limit
        and phrase.beat_in_bar == 1
        and _confidence_ready(phrase.confidence)
    ]
    if not choices:
        raise ValueError(
            f"{profile.title} has no verified outgoing phrase with "
            f"{required_bars} bars of runway"
        )
    return max(choices, key=lambda phrase: phrase.start_bar)


def _phrase_average(
    profile: TrackProfile,
    phrase: PhraseBoundary,
    attribute: str = "median",
    bars: int = 8,
) -> float:
    curve = {item.bar: item for item in profile.bass_energy_by_bar}
    values = [
        float(getattr(curve[bar], attribute))
        for bar in range(phrase.start_bar, phrase.start_bar + bars)
        if bar in curve
    ]
    return sum(values) / len(values) if values else 0.0


def _outgoing_release_score(
    profile: TrackProfile,
    phrase: PhraseBoundary,
) -> float:
    previous = [
        item
        for item in profile.phrase_boundaries
        if item.start_bar < phrase.start_bar
        and item.beat_in_bar == 1
        and _confidence_ready(item.confidence)
    ]
    previous_energy = (
        _phrase_average(profile, previous[-1])
        if previous
        else _phrase_average(profile, phrase)
    )
    current_energy = _phrase_average(profile, phrase)
    energy_release = max(-10.0, min(35.0, (previous_energy - current_energy) * 5.0))
    label_score = {
        "down": 58.0,
        "outro": 45.0,
        "chorus": 15.0,
        "up": 5.0,
        "intro": -25.0,
    }.get(phrase.label.casefold(), 0.0)
    return label_score + energy_release


def _vocal_overlap_bars(
    outgoing: TrackProfile,
    incoming: TrackProfile,
    handoff: EnergyHandoff,
) -> int:
    count = 0
    for offset in range(
        handoff.audible_entry_offset,
        handoff.critical_offset + 1,
    ):
        outgoing_bar = handoff.outgoing_start.start_bar + offset
        incoming_bar = (
            1 + offset
            if handoff.incoming_entry is None
            else handoff.incoming_entry.bar + offset
        )
        outgoing_vocal = any(
            segment.kind == "vocal"
            and segment.start_bar <= outgoing_bar <= segment.end_bar
            for segment in outgoing.segments
        )
        incoming_vocal = any(
            segment.kind == "vocal"
            and segment.start_bar <= incoming_bar <= segment.end_bar
            for segment in incoming.segments
        )
        if outgoing_vocal and incoming_vocal:
            count += 1
    return count


def _verified_instrumental_loop(
    profile: TrackProfile,
    phrase: PhraseBoundary,
) -> bool:
    """Return whether four bars at a verified phrase start contain no vocal."""
    if profile.vocal_confidence not in {"high", "verified"}:
        return False
    loop_start = phrase.start_bar
    loop_end = loop_start + 3
    return not any(
        segment.kind == "vocal"
        and segment.start_bar <= loop_end
        and segment.end_bar >= loop_start
        for segment in profile.segments
    )


def _energy_handoffs(
    outgoing: TrackProfile,
    incoming: TrackProfile,
) -> list[EnergyHandoff]:
    """Pair an outgoing energy release with an incoming chorus/drop downbeat."""
    phrases = [
        phrase
        for phrase in incoming.phrase_boundaries
        if phrase.beat_in_bar == 1 and _confidence_ready(phrase.confidence)
    ]
    file_start = next(
        (
            landmark
            for landmark in incoming.landmarks
            if landmark.kind in {"mix_in", "phrase_start"}
            and landmark.bar == 1
            and (landmark.beat or 1) == 1
            and _confidence_ready(landmark.confidence)
        ),
        None,
    )
    automation_entries = _automation_entries(incoming)
    mix_out = _mix_out_bar(outgoing)
    results: list[EnergyHandoff] = []
    for critical in phrases:
        evidence = energy_phrase_evidence(incoming, critical.start_bar)
        if not evidence.get("verified"):
            continue
        bass_landmark = next(
            (
                landmark
                for landmark in incoming.landmarks
                if landmark.kind in {"drop", "bass_in"}
                and landmark.bar == critical.start_bar
                and (landmark.beat or 1) == 1
                and _confidence_ready(landmark.confidence)
            ),
            None,
        )
        entry_options: list[TrackLandmark | None] = [
            entry
            for entry in automation_entries
            if critical.start_bar - entry.bar in {8, 16}
        ]
        if file_start is not None:
            entry_options.append(None)
        for entry in entry_options:
            entry_bar = entry.bar if entry is not None else 1
            critical_offset = critical.start_bar - entry_bar
            if critical_offset < 4 or critical_offset > 64:
                continue
            audible_candidates = [
                phrase
                for phrase in phrases
                if phrase.start_bar >= entry_bar
                and critical.start_bar - phrase.start_bar in {8, 16, 4}
            ]
            if not audible_candidates:
                continue
            # Eight bars is the default when it preserves a useful build. It
            # avoids exposing a long vocal/melodic overlap merely because a
            # file-start pre-roll began much earlier while muted.
            audible = max(
                audible_candidates,
                key=lambda phrase: (
                    3
                    if critical.start_bar - phrase.start_bar == 8
                    else 2
                    if critical.start_bar - phrase.start_bar == 16
                    else 1,
                    phrase.start_bar,
                ),
            )
            audible_entry_offset = audible.start_bar - entry_bar
            outgoing_pairs: list[tuple[PhraseBoundary, PhraseBoundary]] = []
            for outgoing_critical in outgoing.phrase_boundaries:
                if (
                    outgoing_critical.beat_in_bar != 1
                    or not _confidence_ready(outgoing_critical.confidence)
                    or outgoing_critical.start_bar + 8 > mix_out
                ):
                    continue
                outgoing_start = _phrase_at(
                    outgoing,
                    outgoing_critical.start_bar - critical_offset,
                )
                if outgoing_start is not None:
                    outgoing_pairs.append((outgoing_start, outgoing_critical))
            # A verified four-bar build can land on the second half of an
            # eight-bar musical sentence. Preserve that deliberate subphrase
            # case while longer handoffs still require full phrase-to-phrase
            # alignment.
            if critical_offset == 4 and not outgoing_pairs:
                for outgoing_start in outgoing.phrase_boundaries:
                    if (
                        outgoing_start.beat_in_bar == 1
                        and _confidence_ready(outgoing_start.confidence)
                        and outgoing_start.start_bar + 12 <= mix_out
                    ):
                        outgoing_pairs.append(
                            (
                                outgoing_start,
                                outgoing_start.model_copy(
                                    update={
                                        "start_bar": outgoing_start.start_bar + 4,
                                        "start_beat": outgoing_start.start_beat + 16,
                                    }
                                ),
                            )
                        )
            for outgoing_start, outgoing_critical in outgoing_pairs:
                label = critical.label.casefold()
                incoming_label_score = {
                    "drop": 62.0,
                    "chorus": 58.0,
                    "up": 18.0,
                    "down": -8.0,
                    "intro": -20.0,
                    "outro": -30.0,
                }.get(label, 0.0)
                user_verified = bool(
                    bass_landmark
                    and bass_landmark.confidence == "verified"
                    and bass_landmark.name.casefold().startswith("user verified")
                )
                score = (
                    float(evidence.get("score", 0))
                    + incoming_label_score
                    + _outgoing_release_score(outgoing, outgoing_critical)
                    + (80.0 if user_verified else 0.0)
                    + (10.0 if entry is not None else 0.0)
                    + min(12.0, outgoing_start.start_bar / 16.0)
                )
                results.append(
                    EnergyHandoff(
                        outgoing_start=outgoing_start,
                        outgoing_critical=outgoing_critical,
                        incoming_entry=entry,
                        incoming_audible=audible,
                        incoming_critical=critical,
                        bass_landmark=bass_landmark,
                        critical_offset=critical_offset,
                        audible_entry_offset=audible_entry_offset,
                        score=score,
                        reason=(
                            f"align {outgoing.title} {outgoing_critical.label} "
                            f"bar {outgoing_critical.start_bar} with "
                            f"{incoming.title} {critical.label} bar "
                            f"{critical.start_bar}; energy source "
                            f"{evidence.get('energy_source') or 'waveform'}"
                        ),
                    )
                )
    return sorted(
        results,
        key=lambda item: (
            item.score,
            item.outgoing_critical.start_bar,
            item.incoming_critical.start_bar,
        ),
        reverse=True,
    )


def _load_spec(
    profile: TrackProfile, cue: TrackLandmark | None = None
) -> TrackLoadSpec:
    return TrackLoadSpec(
        track_id=profile.track_id,
        title=profile.title,
        artist=profile.artist,
        source=profile.source,
        cue=cue.cue if cue is not None else None,
        cue_time_ms=cue.time_ms if cue is not None else None,
    )


def _progressive_house_events(
    *,
    outgoing_deck: int,
    incoming_deck: int,
    launch: MusicalEvent,
    audible_entry_offset: int,
    critical_offset: int,
) -> list[MusicalEvent]:
    """Build the deterministic channel-fader/EQ/CFX envelope for a handoff.

    A file-start launch may be an inaudible pre-roll.  ``audible_entry_offset``
    is the verified incoming phrase where the channel first opens, while
    ``critical_offset`` is the verified bass phrase.  This distinction lets a
    track without G/H cues receive a proper phrase-led blend instead of the old
    two-fader hard cut.
    """
    audible_lead = critical_offset - audible_entry_offset
    if audible_lead not in {4, 8, 16}:
        raise ValueError("audible entry must be 4, 8, or 16 bars before bass")

    events = [
        MusicalEvent(
            bar_offset=0,
            action="eq_low",
            parameters={"deck": incoming_deck, "value": -1},
        ),
    ]
    if audible_entry_offset > 0:
        events.append(
            MusicalEvent(
                bar_offset=0,
                action="channel_fader",
                parameters={"deck": incoming_deck, "value": 0},
            )
        )
    events.append(launch)

    fade_positions = [
        (audible_entry_offset, 0.18),
        (audible_entry_offset + max(1, audible_lead // 4), 0.35),
        (audible_entry_offset + max(2, audible_lead // 2), 0.72),
        (critical_offset - max(1, audible_lead // 4), 0.86),
    ]
    for bar_offset, value in fade_positions:
        events.append(
            MusicalEvent(
                bar_offset=bar_offset,
                action="channel_fader",
                parameters={"deck": incoming_deck, "value": value},
            )
        )

    # Beat 1 of the analyzed bass phrase owns the low-EQ transfer.  The
    # outgoing channel stays fully open until this exact musical boundary.
    events.extend(
        [
            MusicalEvent(
                bar_offset=critical_offset,
                action="eq_low",
                parameters={"deck": outgoing_deck, "value": -1},
            ),
            MusicalEvent(
                bar_offset=critical_offset,
                action="eq_low",
                parameters={"deck": incoming_deck, "value": 0},
            ),
            MusicalEvent(
                bar_offset=critical_offset,
                action="channel_fader",
                parameters={"deck": incoming_deck, "value": 1},
            ),
            # FILTER is an absolute, observable-safe CFX control.  It gives a
            # guaranteed high-pass tail without blindly toggling an unknown
            # Beat FX state or cycling an unobserved effect selector.
            MusicalEvent(
                bar_offset=critical_offset + 1,
                action="filter",
                parameters={"deck": outgoing_deck, "value": 0.12},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 2,
                action="channel_fader",
                parameters={"deck": outgoing_deck, "value": 0.86},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 2,
                action="filter",
                parameters={"deck": outgoing_deck, "value": 0.22},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 4,
                action="channel_fader",
                parameters={"deck": outgoing_deck, "value": 0.62},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 4,
                action="filter",
                parameters={"deck": outgoing_deck, "value": 0.38},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 6,
                action="channel_fader",
                parameters={"deck": outgoing_deck, "value": 0.32},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 6,
                action="filter",
                parameters={"deck": outgoing_deck, "value": 0.58},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 8,
                action="channel_fader",
                parameters={"deck": outgoing_deck, "value": 0},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 8,
                action="filter",
                parameters={"deck": outgoing_deck, "value": 0},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 8,
                action="eq_low",
                parameters={"deck": outgoing_deck, "value": 0},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 8,
                beat_offset=1,
                action="cue",
                parameters={"deck": outgoing_deck},
            ),
        ]
    )
    return sorted(events, key=lambda event: (event.bar_offset, event.beat_offset))


def _compact_handoff_events(
    *,
    outgoing_deck: int,
    incoming_deck: int,
    launch: MusicalEvent,
    audible_entry_offset: int,
    critical_offset: int,
    filter_exit: bool = False,
) -> list[MusicalEvent]:
    events = _progressive_house_events(
        outgoing_deck=outgoing_deck,
        incoming_deck=incoming_deck,
        launch=launch,
        audible_entry_offset=audible_entry_offset,
        critical_offset=critical_offset,
    )
    # Replace the eight-bar long-blend retirement with a decisive four-bar
    # post-swap tail.  The incoming build and exact low-EQ transfer are shared.
    events = [
        event
        for event in events
        if not (
            event.parameters.get("deck") == outgoing_deck
            and event.bar_offset > critical_offset
        )
    ]
    filter_values = (0.28, 0.48, 0.68) if filter_exit else (0.12, 0.28, 0.48)
    events.extend(
        [
            MusicalEvent(
                bar_offset=critical_offset + 1,
                action="filter",
                parameters={"deck": outgoing_deck, "value": filter_values[0]},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 1,
                action="channel_fader",
                parameters={"deck": outgoing_deck, "value": 0.76},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 2,
                action="filter",
                parameters={"deck": outgoing_deck, "value": filter_values[1]},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 2,
                action="channel_fader",
                parameters={"deck": outgoing_deck, "value": 0.42},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 3,
                action="filter",
                parameters={"deck": outgoing_deck, "value": filter_values[2]},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 4,
                action="channel_fader",
                parameters={"deck": outgoing_deck, "value": 0},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 4,
                action="filter",
                parameters={"deck": outgoing_deck, "value": 0},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 4,
                action="eq_low",
                parameters={"deck": outgoing_deck, "value": 0},
            ),
            MusicalEvent(
                bar_offset=critical_offset + 4,
                beat_offset=1,
                action="cue",
                parameters={"deck": outgoing_deck},
            ),
        ]
    )
    return sorted(events, key=lambda event: (event.bar_offset, event.beat_offset))


def _breakdown_events(
    *,
    outgoing_deck: int,
    incoming_deck: int,
    launch: MusicalEvent,
) -> list[MusicalEvent]:
    return [
        MusicalEvent(
            bar_offset=0,
            action="channel_fader",
            parameters={"deck": incoming_deck, "value": 0.2},
        ),
        launch,
        MusicalEvent(
            bar_offset=2,
            action="channel_fader",
            parameters={"deck": incoming_deck, "value": 0.48},
        ),
        MusicalEvent(
            bar_offset=4,
            action="filter",
            parameters={"deck": outgoing_deck, "value": 0.22},
        ),
        MusicalEvent(
            bar_offset=4,
            action="channel_fader",
            parameters={"deck": outgoing_deck, "value": 0.78},
        ),
        MusicalEvent(
            bar_offset=4,
            action="channel_fader",
            parameters={"deck": incoming_deck, "value": 0.78},
        ),
        MusicalEvent(
            bar_offset=6,
            action="filter",
            parameters={"deck": outgoing_deck, "value": 0.52},
        ),
        MusicalEvent(
            bar_offset=6,
            action="channel_fader",
            parameters={"deck": outgoing_deck, "value": 0.38},
        ),
        MusicalEvent(
            bar_offset=8,
            action="channel_fader",
            parameters={"deck": incoming_deck, "value": 1},
        ),
        MusicalEvent(
            bar_offset=8,
            action="channel_fader",
            parameters={"deck": outgoing_deck, "value": 0},
        ),
        MusicalEvent(
            bar_offset=8,
            action="filter",
            parameters={"deck": outgoing_deck, "value": 0},
        ),
        MusicalEvent(
            bar_offset=8,
            beat_offset=1,
            action="cue",
            parameters={"deck": outgoing_deck},
        ),
    ]


def _phrase_cut_events(
    *,
    outgoing_deck: int,
    incoming_deck: int,
    launch: MusicalEvent,
) -> list[MusicalEvent]:
    return [
        launch,
        MusicalEvent(
            bar_offset=0,
            action="channel_fader",
            parameters={"deck": incoming_deck, "value": 1},
        ),
        MusicalEvent(
            bar_offset=0,
            action="channel_fader",
            parameters={"deck": outgoing_deck, "value": 0},
        ),
        MusicalEvent(
            bar_offset=0,
            beat_offset=1,
            action="cue",
            parameters={"deck": outgoing_deck},
        ),
    ]


class _LegacyTransitionKingCompiler:
    """Compile one evidence-backed handoff with a deterministic energy policy."""

    def compile(
        self,
        outgoing: TrackProfile,
        incoming: TrackProfile,
        *,
        outgoing_deck: int,
    ) -> CompiledHandoff:
        incoming_deck = 2 if outgoing_deck == 1 else 1
        anchored = _verified_bass_handoff(incoming)
        if anchored is not None:
            entry, bass, lead = anchored
            tail = 8
            start = _outgoing_start_phrase(
                outgoing,
                required_bars=lead + tail,
            )
            events = _progressive_house_events(
                outgoing_deck=outgoing_deck,
                incoming_deck=incoming_deck,
                launch=MusicalEvent(
                    bar_offset=0,
                    action="hot_cue",
                    parameters={"deck": incoming_deck, "cue": entry.cue},
                ),
                audible_entry_offset=0,
                critical_offset=lead,
            )
            card = TransitionCard(
                name=f"{outgoing.track_id}-{incoming.track_id}-bass-swap",
                outgoing_track_id=outgoing.track_id,
                incoming_track_id=incoming.track_id,
                anchor_deck=outgoing_deck,
                outgoing_deck=outgoing_deck,
                incoming_deck=incoming_deck,
                transition_family="long_blend",
                start_quantum_bars=16 if lead == 16 else 8,
                minimum_lead_bars=16,
                start_phrase_index=start.index,
                beat_sync_required=True,
                quantize_required=True,
                phrase_alignment_verified=True,
                vocal_plan_verified=True,
                bass_plan_verified=True,
                incoming_loaded_verified=True,
                intended_vocal_owner="intentional_overlap",
                critical_bar_offset=lead,
                events=events,
                abort_plan=(
                    "Keep the outgoing channel and bass full; mute and stop the "
                    "incoming deck, then hold the prepared rescue loop."
                ),
                notes=(
                    f"Verified cue {entry.cue} at bar {entry.bar}; waveform "
                    f"bass handoff at bar {bass.bar}. Outgoing fader remains "
                    "full until after the low-EQ swap, then retires across an "
                    "eight-bar high-pass CFX tail."
                ),
            )
            self._validate(card, outgoing, incoming)
            return CompiledHandoff(
                card=card,
                load=_load_spec(incoming, entry),
                technique="verified_bass_swap",
                cue_landmark=entry,
                bass_landmark=bass,
            )
        return self._compile_safe_fallback(
            outgoing,
            incoming,
            outgoing_deck=outgoing_deck,
        )

    def _compile_safe_fallback(
        self,
        outgoing: TrackProfile,
        incoming: TrackProfile,
        *,
        outgoing_deck: int,
    ) -> CompiledHandoff:
        incoming_deck = 2 if outgoing_deck == 1 else 1
        file_start = next(
            (
                landmark
                for landmark in incoming.landmarks
                if landmark.kind in {"mix_in", "phrase_start"}
                and landmark.bar == 1
                and (landmark.beat or 1) == 1
                and _confidence_ready(landmark.confidence)
            ),
            None,
        )
        if file_start is None:
            raise ValueError(
                f"{incoming.title} needs a verified G/H entry cue before live use"
            )
        candidates = [
            landmark
            for landmark in incoming.landmarks
            if landmark.kind in {"drop", "bass_in"}
            and landmark.bar > file_start.bar
            and (landmark.beat or 1) == 1
            and _confidence_ready(landmark.confidence)
            and _phrase_at(incoming, landmark.bar) is not None
            and (
                (
                    landmark.kind == "drop"
                    and landmark.confidence == "verified"
                    and landmark.name.casefold().startswith("user verified")
                )
                or bass_phrase_evidence(incoming, landmark.bar).get("verified")
            )
        ]
        if not candidates:
            raise ValueError(
                f"{incoming.title} has no verified bass phrase for a live handoff"
            )
        bass = min(candidates, key=lambda item: item.bar)
        audible_entries = [
            phrase
            for phrase in incoming.phrase_boundaries
            if phrase.start_bar < bass.bar
            and phrase.beat_in_bar == 1
            and _confidence_ready(phrase.confidence)
            and bass.bar - phrase.start_bar in {16, 8, 4}
        ]
        if not audible_entries:
            raise ValueError(
                f"{incoming.title} has no verified 4/8/16-bar entry phrase "
                f"before bass bar {bass.bar}"
            )
        audible_entry = max(
            audible_entries,
            key=lambda phrase: (
                1 if bass.bar - phrase.start_bar == 16 else 0,
                phrase.start_bar,
            ),
        )
        critical = bass.bar - file_start.bar
        audible_entry_offset = audible_entry.start_bar - file_start.bar
        start = _outgoing_start_phrase(
            outgoing,
            required_bars=critical + 8,
        )
        events = _progressive_house_events(
            outgoing_deck=outgoing_deck,
            incoming_deck=incoming_deck,
            launch=MusicalEvent(
                bar_offset=0,
                action="play_pause",
                parameters={"deck": incoming_deck},
            ),
            audible_entry_offset=audible_entry_offset,
            critical_offset=critical,
        )
        card = TransitionCard(
            name=f"{outgoing.track_id}-{incoming.track_id}-file-start-bass-swap",
            outgoing_track_id=outgoing.track_id,
            incoming_track_id=incoming.track_id,
            anchor_deck=outgoing_deck,
            outgoing_deck=outgoing_deck,
            incoming_deck=incoming_deck,
            transition_family="long_blend",
            start_quantum_bars=16,
            minimum_lead_bars=16,
            start_phrase_index=start.index,
            beat_sync_required=True,
            quantize_required=True,
            phrase_alignment_verified=True,
            vocal_plan_verified=True,
            bass_plan_verified=True,
            incoming_loaded_verified=True,
            intended_vocal_owner="incoming",
            critical_bar_offset=critical,
            events=events,
            abort_plan=(
                "Keep the outgoing channel at full level and stop the muted "
                "incoming deck; engage the prepared rescue loop."
            ),
            notes=(
                f"Verified file-start pre-roll; incoming becomes audible at "
                f"bar {audible_entry.start_bar}, swaps lows at analyzed "
                f"{bass.kind} bar {bass.bar}, and retires the outgoing deck "
                "across an eight-bar high-pass CFX tail."
            ),
        )
        self._validate(card, outgoing, incoming)
        return CompiledHandoff(
            card=card,
            load=_load_spec(incoming),
            technique="file_start_bass_swap",
            cue_landmark=None,
            bass_landmark=bass,
        )

    @staticmethod
    def _validate(
        card: TransitionCard,
        outgoing: TrackProfile,
        incoming: TrackProfile,
    ) -> None:
        errors = validate_transition_card(card, outgoing, incoming)
        if errors:
            raise ValueError("; ".join(errors))


class TransitionKingCompiler:
    """Score multiple evidence-backed techniques and compile the best one."""

    def assess(
        self,
        outgoing: TrackProfile,
        incoming: TrackProfile,
        *,
        outgoing_deck: int,
        recent_techniques: tuple[str, ...] = (),
        proficient_techniques: frozenset[str] = frozenset(),
        vibe: str = "maintain",
        allow_safe_cuts: bool = True,
    ) -> list[CompiledHandoff]:
        proposals: list[CompiledHandoff] = []

        def consider(builder) -> None:
            # A technique can be structurally inapplicable (including a
            # harmonic-overlap rejection) without invalidating other safe
            # techniques for the same pair. Each candidate fails closed on
            # its own; the pair fails only when no candidate validates.
            try:
                proposal = builder()
            except ValueError:
                return
            if proposal is not None:
                proposals.append(proposal)

        energy_pairs = _energy_handoffs(outgoing, incoming)
        if energy_pairs:
            pair = energy_pairs[0]
            overlap = _vocal_overlap_bars(outgoing, incoming, pair)
            long_score = pair.score + (
                32.0
                if pair.outgoing_critical.label.casefold() == "down"
                and pair.incoming_critical.label.casefold() == "chorus"
                else 18.0
            ) - overlap * 12.0
            consider(
                lambda: self._compile_energy_card(
                    outgoing,
                    incoming,
                    outgoing_deck=outgoing_deck,
                    pair=pair,
                    family="long_blend",
                    technique="long_blend",
                    score=long_score,
                    reason=(
                        f"Long energy-preserving blend: {pair.reason}. "
                        "The outgoing channel stays full through the swap and "
                        "retires over eight bars."
                    ),
                )
            )
            consider(
                lambda: self._compile_energy_card(
                    outgoing,
                    incoming,
                    outgoing_deck=outgoing_deck,
                    pair=pair,
                    family="bass_swap",
                    technique="compact_bass_swap",
                    score=pair.score
                    + (
                        20.0
                        if pair.incoming_critical.start_bar
                        - pair.incoming_audible.start_bar
                        == 8
                        else 8.0
                    )
                    - overlap * 8.0,
                    reason=(
                        f"Compact bass swap: {pair.reason}. The incoming track "
                        "is established first and the outgoing tail clears in "
                        "four bars."
                    ),
                )
            )
            if pair.outgoing_critical.label.casefold() == "outro":
                consider(
                    lambda: self._compile_energy_card(
                        outgoing,
                        incoming,
                        outgoing_deck=outgoing_deck,
                        pair=pair,
                        family="echo_exit",
                        technique="filter_exit",
                        score=pair.score + 25.0,
                        reason=(
                            f"High-pass exit: {pair.reason}. A compact filter "
                            "tail clears the short outgoing phrase after the swap."
                        ),
                    )
                )
            if (
                overlap >= 4
                and pair.audible_entry_offset > 0
                and "stem_vocal_blend" in proficient_techniques
            ):
                consider(
                    lambda: self._compile_energy_card(
                        outgoing,
                        incoming,
                        outgoing_deck=outgoing_deck,
                        pair=pair,
                        family="long_blend",
                        technique="stem_vocal_blend",
                        score=pair.score + 15.0 + min(8.0, overlap / 2.0),
                        reason=(
                            f"Stem-assisted blend: {pair.reason}. Incoming vocals "
                            f"are suppressed for {overlap} overlap bars and restored "
                            "after the outgoing deck is silent."
                        ),
                        use_incoming_vocal_stem=True,
                    )
                )

            consider(
                lambda: self._compile_vocal_loop_blend(
                    outgoing,
                    incoming,
                    outgoing_deck=outgoing_deck,
                    pair=pair,
                    vocal_overlap_bars=overlap,
                )
            )

        consider(
            lambda: self._compile_loop_bridge(
                outgoing,
                incoming,
                outgoing_deck=outgoing_deck,
            )
        )

        consider(
            lambda: self._compile_breakdown(
                outgoing,
                incoming,
                outgoing_deck=outgoing_deck,
                vibe=vibe,
            )
        )

        if allow_safe_cuts:
            consider(
                lambda: self._compile_phrase_cut(
                    outgoing,
                    incoming,
                    outgoing_deck=outgoing_deck,
                )
            )

        if not proposals:
            raise ValueError(
                f"{outgoing.title} -> {incoming.title} has no verified transition route"
            )

        adjusted = []
        for proposal in proposals:
            repeated = recent_techniques.count(proposal.technique)
            immediate_repeat = bool(
                recent_techniques and recent_techniques[-1] == proposal.technique
            )
            penalty = repeated * 7.0 + (8.0 if immediate_repeat else 0.0)
            adjusted.append(
                CompiledHandoff(
                    **{
                        **proposal.__dict__,
                        "suitability_score": proposal.suitability_score - penalty,
                    }
                )
            )
        ranked = sorted(
            adjusted,
            key=lambda item: (item.suitability_score, item.technique),
            reverse=True,
        )
        names = tuple(item.technique for item in ranked)
        return [
            CompiledHandoff(
                **{
                    **item.__dict__,
                    "alternatives": tuple(
                        name for name in names if name != item.technique
                    ),
                }
            )
            for item in ranked
        ]

    def compile(
        self,
        outgoing: TrackProfile,
        incoming: TrackProfile,
        *,
        outgoing_deck: int,
        recent_techniques: tuple[str, ...] = (),
        proficient_techniques: frozenset[str] = frozenset(),
        vibe: str = "maintain",
        allow_safe_cuts: bool = True,
    ) -> CompiledHandoff:
        return self.assess(
            outgoing,
            incoming,
            outgoing_deck=outgoing_deck,
            recent_techniques=recent_techniques,
            proficient_techniques=proficient_techniques,
            vibe=vibe,
            allow_safe_cuts=allow_safe_cuts,
        )[0]

    def _compile_vocal_loop_blend(
        self,
        outgoing: TrackProfile,
        incoming: TrackProfile,
        *,
        outgoing_deck: int,
        pair: EnergyHandoff,
        vocal_overlap_bars: int,
    ) -> CompiledHandoff | None:
        """Hold a clean outgoing pocket when its advancing tail would clash."""
        if vocal_overlap_bars < 2 or not _verified_instrumental_loop(
            outgoing,
            pair.outgoing_start,
        ):
            return None
        incoming_deck = 2 if outgoing_deck == 1 else 1
        if pair.incoming_entry is not None:
            launch = MusicalEvent(
                bar_offset=0,
                action="hot_cue",
                parameters={"deck": incoming_deck, "cue": pair.incoming_entry.cue},
            )
            load = _load_spec(incoming, pair.incoming_entry)
        else:
            launch = MusicalEvent(
                bar_offset=0,
                action="play_pause",
                parameters={"deck": incoming_deck},
            )
            load = _load_spec(incoming)
        events = _progressive_house_events(
            outgoing_deck=outgoing_deck,
            incoming_deck=incoming_deck,
            launch=launch,
            audible_entry_offset=pair.audible_entry_offset,
            critical_offset=pair.critical_offset,
        )
        events.extend(
            [
                MusicalEvent(
                    bar_offset=0,
                    action="loop_16",
                    parameters={"deck": outgoing_deck},
                ),
                MusicalEvent(
                    bar_offset=pair.critical_offset + 8,
                    action="loop_toggle",
                    parameters={"deck": outgoing_deck},
                ),
            ]
        )
        events.sort(key=lambda event: (event.bar_offset, event.beat_offset))
        card = TransitionCard(
            name=f"{outgoing.track_id}-{incoming.track_id}-vocal-safe-loop",
            outgoing_track_id=outgoing.track_id,
            incoming_track_id=incoming.track_id,
            anchor_deck=outgoing_deck,
            outgoing_deck=outgoing_deck,
            incoming_deck=incoming_deck,
            transition_family="loop_bridge",
            start_quantum_bars=16 if pair.critical_offset >= 16 else 8,
            minimum_lead_bars=16,
            start_phrase_index=pair.outgoing_start.index,
            beat_sync_required=True,
            quantize_required=True,
            phrase_alignment_verified=True,
            vocal_plan_verified=True,
            bass_plan_verified=True,
            loop_plan_verified=True,
            incoming_loaded_verified=True,
            intended_vocal_owner="incoming",
            critical_bar_offset=pair.critical_offset,
            events=events,
            abort_plan=(
                "Keep the verified outgoing loop and bass active; stop the muted "
                "incoming deck before its fader rises."
            ),
            notes=(
                f"Four-bar instrumental loop at outgoing bar "
                f"{pair.outgoing_start.start_bar} prevents {vocal_overlap_bars} "
                "bars of analyzed vocal collision while preserving runway."
            ),
        )
        self._validate(card, outgoing, incoming)
        return CompiledHandoff(
            card=card,
            load=load,
            technique="vocal_safe_loop_blend",
            cue_landmark=pair.incoming_entry,
            bass_landmark=pair.bass_landmark,
            reason=(
                "A verified instrumental outgoing loop preserves energy and "
                f"removes {vocal_overlap_bars} bars of vocal collision."
            ),
            suitability_score=pair.score + 42.0 + vocal_overlap_bars * 7.0,
        )

    def _compile_energy_card(
        self,
        outgoing: TrackProfile,
        incoming: TrackProfile,
        *,
        outgoing_deck: int,
        pair: EnergyHandoff,
        family: Literal["long_blend", "bass_swap", "echo_exit"],
        technique: str,
        score: float,
        reason: str,
        use_incoming_vocal_stem: bool = False,
    ) -> CompiledHandoff:
        incoming_deck = 2 if outgoing_deck == 1 else 1
        if pair.incoming_entry is not None:
            launch = MusicalEvent(
                bar_offset=0,
                action="hot_cue",
                parameters={"deck": incoming_deck, "cue": pair.incoming_entry.cue},
            )
            load = _load_spec(incoming, pair.incoming_entry)
        else:
            launch = MusicalEvent(
                bar_offset=0,
                action="play_pause",
                parameters={"deck": incoming_deck},
            )
            load = _load_spec(incoming)
        if family == "long_blend":
            events = _progressive_house_events(
                outgoing_deck=outgoing_deck,
                incoming_deck=incoming_deck,
                launch=launch,
                audible_entry_offset=pair.audible_entry_offset,
                critical_offset=pair.critical_offset,
            )
        else:
            events = _compact_handoff_events(
                outgoing_deck=outgoing_deck,
                incoming_deck=incoming_deck,
                launch=launch,
                audible_entry_offset=pair.audible_entry_offset,
                critical_offset=pair.critical_offset,
                filter_exit=family == "echo_exit",
            )
        if use_incoming_vocal_stem:
            events.extend(
                [
                    MusicalEvent(
                        bar_offset=0,
                        action="stem_vocal",
                        parameters={"deck": incoming_deck},
                    ),
                    MusicalEvent(
                        bar_offset=pair.critical_offset + 8,
                        action="stem_vocal",
                        parameters={"deck": incoming_deck},
                    ),
                ]
            )
            events = sorted(
                events, key=lambda event: (event.bar_offset, event.beat_offset)
            )
        card = TransitionCard(
            name=f"{outgoing.track_id}-{incoming.track_id}-{technique}",
            outgoing_track_id=outgoing.track_id,
            incoming_track_id=incoming.track_id,
            anchor_deck=outgoing_deck,
            outgoing_deck=outgoing_deck,
            incoming_deck=incoming_deck,
            transition_family=family,
            start_quantum_bars=(
                32
                if pair.critical_offset >= 32
                else 16
                if pair.critical_offset >= 16
                else 8
            ),
            minimum_lead_bars=16,
            start_phrase_index=pair.outgoing_start.index,
            beat_sync_required=True,
            quantize_required=True,
            phrase_alignment_verified=True,
            vocal_plan_verified=True,
            bass_plan_verified=True,
            incoming_loaded_verified=True,
            intended_vocal_owner=(
                "outgoing" if use_incoming_vocal_stem else "intentional_overlap"
            ),
            critical_bar_offset=pair.critical_offset,
            events=events,
            abort_plan=(
                "Keep the outgoing channel and bass full; mute and stop the "
                "incoming deck, then hold the prepared rescue loop."
            ),
            notes=(
                f"{reason} Audible incoming phrase begins at bar "
                f"{pair.incoming_audible.start_bar}; exact energy transfer is "
                f"outgoing bar {pair.outgoing_critical.start_bar} to incoming "
                f"bar {pair.incoming_critical.start_bar}."
            ),
        )
        self._validate(card, outgoing, incoming)
        return CompiledHandoff(
            card=card,
            load=load,
            technique=technique,
            cue_landmark=pair.incoming_entry,
            bass_landmark=pair.bass_landmark,
            reason=reason,
            suitability_score=score,
        )

    def _compile_loop_bridge(
        self,
        outgoing: TrackProfile,
        incoming: TrackProfile,
        *,
        outgoing_deck: int,
    ) -> CompiledHandoff | None:
        anchored = _verified_bass_handoff(incoming)
        if anchored is None or _energy_handoffs(outgoing, incoming):
            return None
        entry, bass, lead = anchored
        outgoing_deck = int(outgoing_deck)
        incoming_deck = 2 if outgoing_deck == 1 else 1
        start = _outgoing_start_phrase(outgoing, required_bars=1)
        events = _progressive_house_events(
            outgoing_deck=outgoing_deck,
            incoming_deck=incoming_deck,
            launch=MusicalEvent(
                bar_offset=0,
                action="hot_cue",
                parameters={"deck": incoming_deck, "cue": entry.cue},
            ),
            audible_entry_offset=0,
            critical_offset=lead,
        )
        events.extend(
            [
                MusicalEvent(
                    bar_offset=0,
                    action="loop_16",
                    parameters={"deck": outgoing_deck},
                ),
                MusicalEvent(
                    bar_offset=lead + 8,
                    action="loop_toggle",
                    parameters={"deck": outgoing_deck},
                ),
            ]
        )
        events = sorted(events, key=lambda event: (event.bar_offset, event.beat_offset))
        card = TransitionCard(
            name=f"{outgoing.track_id}-{incoming.track_id}-loop-bridge",
            outgoing_track_id=outgoing.track_id,
            incoming_track_id=incoming.track_id,
            anchor_deck=outgoing_deck,
            outgoing_deck=outgoing_deck,
            incoming_deck=incoming_deck,
            transition_family="loop_bridge",
            start_quantum_bars=16 if lead == 16 else 8,
            minimum_lead_bars=1,
            start_phrase_index=start.index,
            beat_sync_required=True,
            quantize_required=True,
            phrase_alignment_verified=True,
            vocal_plan_verified=True,
            bass_plan_verified=True,
            loop_plan_verified=True,
            incoming_loaded_verified=True,
            intended_vocal_owner="intentional_overlap",
            critical_bar_offset=lead,
            events=events,
            abort_plan=(
                "Keep the verified outgoing loop active and stop the muted "
                "incoming deck."
            ),
            notes=(
                "A verified four-bar outgoing loop supplies missing runway; "
                f"bass transfers to incoming bar {bass.bar} before the loop releases."
            ),
        )
        self._validate(card, outgoing, incoming)
        return CompiledHandoff(
            card=card,
            load=_load_spec(incoming, entry),
            technique="loop_bridge",
            cue_landmark=entry,
            bass_landmark=bass,
            reason=(
                "The phrase structures do not align naturally, so a verified "
                "loop supplies runway."
            ),
            suitability_score=155.0,
        )

    def _compile_breakdown(
        self,
        outgoing: TrackProfile,
        incoming: TrackProfile,
        *,
        outgoing_deck: int,
        vibe: str,
    ) -> CompiledHandoff | None:
        entries = _automation_entries(incoming)
        entry = next(
            (
                item
                for item in entries
                if _phrase_at(incoming, item.bar) is not None
                and _phrase_at(incoming, item.bar).label.casefold() == "down"
            ),
            None,
        )
        file_start = next(
            (
                item
                for item in incoming.landmarks
                if item.kind in {"mix_in", "phrase_start"}
                and item.bar == 1
                and (item.beat or 1) == 1
                and _confidence_ready(item.confidence)
            ),
            None,
        )
        if entry is None and file_start is None:
            return None
        phrase = (
            _phrase_at(incoming, entry.bar)
            if entry is not None
            else _phrase_at(incoming, 1)
        )
        if phrase is None or phrase.label.casefold() not in {"down", "intro"}:
            return None
        incoming_deck = 2 if outgoing_deck == 1 else 1
        start = _outgoing_start_phrase(outgoing, required_bars=8)
        launch = MusicalEvent(
            bar_offset=0,
            action="hot_cue" if entry is not None else "play_pause",
            parameters=(
                {"deck": incoming_deck, "cue": entry.cue}
                if entry is not None
                else {"deck": incoming_deck}
            ),
        )
        score = 112.0 + (30.0 if vibe in {"downtempo", "deeper"} else 0.0)
        card = TransitionCard(
            name=f"{outgoing.track_id}-{incoming.track_id}-breakdown-handoff",
            outgoing_track_id=outgoing.track_id,
            incoming_track_id=incoming.track_id,
            anchor_deck=outgoing_deck,
            outgoing_deck=outgoing_deck,
            incoming_deck=incoming_deck,
            transition_family="breakdown_handoff",
            start_quantum_bars=8,
            minimum_lead_bars=8,
            start_phrase_index=start.index,
            beat_sync_required=True,
            quantize_required=True,
            phrase_alignment_verified=True,
            vocal_plan_verified=True,
            bass_plan_verified=True,
            intentional_energy_drop=True,
            incoming_loaded_verified=True,
            intended_vocal_owner="incoming",
            critical_bar_offset=8,
            events=_breakdown_events(
                outgoing_deck=outgoing_deck,
                incoming_deck=incoming_deck,
                launch=launch,
            ),
            abort_plan=(
                "Keep the outgoing deck full and stop the incoming deck before "
                "its fader rises."
            ),
            notes="Deliberate breakdown handoff selected for an atmospheric reset.",
        )
        self._validate(card, outgoing, incoming)
        return CompiledHandoff(
            card=card,
            load=_load_spec(incoming, entry)
            if entry is not None
            else _load_spec(incoming),
            technique="breakdown_handoff",
            cue_landmark=entry,
            bass_landmark=None,
            reason=(
                "Incoming verified down/intro phrase supports a deliberate "
                "emotional reset."
            ),
            suitability_score=score,
        )

    def _compile_phrase_cut(
        self,
        outgoing: TrackProfile,
        incoming: TrackProfile,
        *,
        outgoing_deck: int,
    ) -> CompiledHandoff | None:
        entry = next(iter(_automation_entries(incoming)), None)
        file_start = next(
            (
                item
                for item in incoming.landmarks
                if item.kind in {"mix_in", "phrase_start"}
                and item.bar == 1
                and (item.beat or 1) == 1
                and _confidence_ready(item.confidence)
            ),
            None,
        )
        if entry is None and file_start is None:
            return None
        incoming_deck = 2 if outgoing_deck == 1 else 1
        harmonic = camelot_compatibility(outgoing.key, incoming.key)
        start = _outgoing_start_phrase(outgoing, required_bars=1)
        launch = MusicalEvent(
            bar_offset=0,
            action="hot_cue" if entry is not None else "play_pause",
            parameters=(
                {"deck": incoming_deck, "cue": entry.cue}
                if entry is not None
                else {"deck": incoming_deck}
            ),
        )
        card = TransitionCard(
            name=f"{outgoing.track_id}-{incoming.track_id}-phrase-cut",
            outgoing_track_id=outgoing.track_id,
            incoming_track_id=incoming.track_id,
            anchor_deck=outgoing_deck,
            outgoing_deck=outgoing_deck,
            incoming_deck=incoming_deck,
            transition_family="phrase_cut",
            start_quantum_bars=1,
            minimum_lead_bars=1,
            start_phrase_index=start.index,
            beat_sync_required=True,
            quantize_required=True,
            phrase_alignment_verified=True,
            vocal_plan_verified=True,
            bass_plan_verified=True,
            incoming_loaded_verified=True,
            harmonic_risk_accepted=not harmonic.get("compatible", False),
            intended_vocal_owner="incoming",
            critical_bar_offset=0,
            events=_phrase_cut_events(
                outgoing_deck=outgoing_deck,
                incoming_deck=incoming_deck,
                launch=launch,
            ),
            abort_plan=(
                "Do not cut; keep the outgoing channel full and stop the "
                "incoming deck."
            ),
            notes=(
                "Verified phrase-boundary cut retained as a low-complexity "
                "fallback. The non-overlap cut explicitly isolates incompatible "
                "keys."
                if not harmonic.get("compatible", False)
                else "Verified phrase-boundary cut retained as a low-complexity "
                "fallback."
            ),
        )
        self._validate(card, outgoing, incoming)
        return CompiledHandoff(
            card=card,
            load=_load_spec(incoming, entry)
            if entry is not None
            else _load_spec(incoming),
            technique="phrase_cut",
            cue_landmark=entry,
            bass_landmark=None,
            reason="A decisive verified phrase cut is the safest non-overlap fallback.",
            suitability_score=95.0,
        )

    @staticmethod
    def _validate(
        card: TransitionCard,
        outgoing: TrackProfile,
        incoming: TrackProfile,
    ) -> None:
        errors = validate_transition_card(card, outgoing, incoming)
        if errors:
            raise ValueError("; ".join(errors))


class LocalDJPlanner:
    """Choose a repeatable route and compile every adjacent transition."""

    def __init__(
        self,
        profiles: Iterable[TrackProfile],
        *,
        proficient_techniques: Iterable[str] = (),
    ) -> None:
        self.profiles = {
            profile.track_id: profile
            for profile in profiles
            if profile.readiness()["ready"]
        }
        self.proficient_techniques = frozenset(proficient_techniques)
        self.compiler = TransitionKingCompiler()
        self._handoff_cache: dict[
            tuple[str, str, int, str, bool], CompiledHandoff | None
        ] = {}

    def build_plan(
        self,
        brief: DJBrief,
        *,
        opening_deck: int = 1,
    ) -> AutonomousSetPlan:
        if opening_deck not in {1, 2}:
            raise ValueError("opening_deck must be 1 or 2")
        if brief.start_track_id not in self.profiles:
            raise ValueError("opening track is not Tier-A automation ready")
        target = None
        if brief.target_track_id is not None:
            if brief.target_track_id not in self.profiles:
                raise ValueError("destination track is not Tier-A automation ready")
            if brief.target_track_id == brief.start_track_id:
                raise ValueError("destination track must differ from the opening track")
            if brief.target_track_id in brief.excluded_track_ids:
                raise ValueError("destination track has already played in this set")
            target = self.profiles[brief.target_track_id]
        allowed = (
            set(brief.allowed_track_ids)
            if brief.allowed_track_ids
            else set(self.profiles)
        )
        missing_allowed = allowed - self.profiles.keys()
        if missing_allowed:
            raise ValueError(
                "candidate pool contains tracks that are not Tier-A automation ready: "
                + ", ".join(sorted(missing_allowed))
            )
        if brief.start_track_id not in allowed:
            raise ValueError("opening track is outside the selected candidate pool")
        if target is not None and target.track_id not in allowed:
            raise ValueError("destination track is outside the selected candidate pool")
        route, route_handoffs = self._search_route(
            brief,
            opening_deck=opening_deck,
            start_track_ids=(brief.start_track_id,),
            allowed_track_ids=allowed,
            target=target,
        )

        return self._plan_from_route(brief, route, route_handoffs, target=target)

    def build_playlist_plan(
        self,
        track_ids: Iterable[str],
        *,
        opening_track_id: str | None = None,
        opening_deck: int = 1,
        vibe: str = "maintain",
        name: str = "RekordBot playlist set",
    ) -> AutonomousSetPlan:
        """Find the strongest safe ordering that uses every playlist track once."""
        ordered_ids = tuple(dict.fromkeys(str(track_id) for track_id in track_ids))
        if len(ordered_ids) < 2:
            raise ValueError("playlist needs at least two automation-ready tracks")
        missing = [track_id for track_id in ordered_ids if track_id not in self.profiles]
        if missing:
            raise ValueError(
                "playlist contains tracks that are not Tier-A automation ready: "
                + ", ".join(missing)
            )
        if opening_track_id is not None and opening_track_id not in ordered_ids:
            raise ValueError("selected opening track is not in the playlist")
        brief = DJBrief(
            start_track_id=opening_track_id or ordered_ids[0],
            target_track_count=len(ordered_ids),
            name=name,
            vibe=vibe,
            allowed_track_ids=list(ordered_ids),
            max_native_bpm_delta=8.0,
        )
        starts = (opening_track_id,) if opening_track_id else ordered_ids
        route, handoffs = self._search_route(
            brief,
            opening_deck=opening_deck,
            start_track_ids=tuple(item for item in starts if item is not None),
            allowed_track_ids=set(ordered_ids),
            target=None,
            beam_width=96,
        )
        return self._plan_from_route(brief, route, handoffs, target=None)

    def _search_route(
        self,
        brief: DJBrief,
        *,
        opening_deck: int,
        start_track_ids: tuple[str, ...],
        allowed_track_ids: set[str],
        target: TrackProfile | None,
        beam_width: int = 64,
    ) -> tuple[list[TrackProfile], list[CompiledHandoff]]:
        """Beam-search the transition graph instead of committing greedily."""
        excluded = set(brief.excluded_track_ids) - set(start_track_ids)
        states: list[tuple[float, tuple[str, ...], tuple[CompiledHandoff, ...]]] = [
            (0.0, (track_id,), ()) for track_id in start_track_ids
        ]
        for depth in range(1, brief.target_track_count):
            expanded: list[
                tuple[float, tuple[str, ...], tuple[CompiledHandoff, ...]]
            ] = []
            final_slot = depth == brief.target_track_count - 1
            for score, route_ids, handoffs in states:
                current = self.profiles[route_ids[-1]]
                if final_slot and target is not None:
                    candidate_ids = (target.track_id,)
                else:
                    candidate_ids = tuple(
                        track_id
                        for track_id in allowed_track_ids
                        if track_id not in route_ids
                        and track_id not in excluded
                        and (target is None or track_id != target.track_id)
                    )
                outgoing_deck = (
                    opening_deck if depth % 2 else (2 if opening_deck == 1 else 1)
                )
                for candidate_id in candidate_ids:
                    if candidate_id in route_ids or candidate_id in excluded:
                        continue
                    candidate = self.profiles[candidate_id]
                    if abs(candidate.bpm - current.bpm) > brief.max_native_bpm_delta:
                        continue
                    handoff = self._cached_handoff(
                        current,
                        candidate,
                        outgoing_deck=outgoing_deck,
                        brief=brief,
                    )
                    if handoff is None:
                        continue
                    selection = self._score_candidate(
                        current,
                        candidate,
                        depth=depth,
                        total=brief.target_track_count,
                        target_bpm=brief.target_bpm,
                        target_track=target,
                        vibe=brief.vibe,
                    )[0]
                    repeated = sum(
                        item.technique == handoff.technique for item in handoffs[-3:]
                    )
                    diversity_penalty = repeated * 7.0 + (
                        8.0
                        if handoffs and handoffs[-1].technique == handoff.technique
                        else 0.0
                    )
                    expanded.append(
                        (
                            score
                            + selection
                            + handoff.suitability_score * 0.35
                            - diversity_penalty,
                            (*route_ids, candidate_id),
                            (*handoffs, handoff),
                        )
                    )
            if not expanded:
                direction = (
                    f" while steering toward {target.title}" if target is not None else ""
                )
                raise ValueError(
                    "no complete safe route exists through the prepared transition "
                    f"graph{direction}"
                )
            expanded.sort(key=lambda item: (item[0], item[1]), reverse=True)
            states = expanded[:beam_width]
        _, route_ids, handoffs = max(states, key=lambda item: (item[0], item[1]))
        return [self.profiles[track_id] for track_id in route_ids], list(handoffs)

    def _cached_handoff(
        self,
        outgoing: TrackProfile,
        incoming: TrackProfile,
        *,
        outgoing_deck: int,
        brief: DJBrief,
    ) -> CompiledHandoff | None:
        key = (
            outgoing.track_id,
            incoming.track_id,
            outgoing_deck,
            brief.vibe,
            brief.allow_safe_cuts,
        )
        if key not in self._handoff_cache:
            try:
                self._handoff_cache[key] = self.compiler.compile(
                    outgoing,
                    incoming,
                    outgoing_deck=outgoing_deck,
                    proficient_techniques=self.proficient_techniques,
                    vibe=brief.vibe,
                    allow_safe_cuts=brief.allow_safe_cuts,
                )
            except ValueError:
                self._handoff_cache[key] = None
        return self._handoff_cache[key]

    def _plan_from_route(
        self,
        brief: DJBrief,
        route: list[TrackProfile],
        route_handoffs: list[CompiledHandoff],
        *,
        target: TrackProfile | None,
    ) -> AutonomousSetPlan:

        transitions: list[TransitionOption] = []
        for index, ((outgoing, incoming), handoff) in enumerate(
            zip(pairwise(route), route_handoffs, strict=True)
        ):
            transitions.append(
                TransitionOption(
                    id=f"{index + 1}-{outgoing.track_id}-{incoming.track_id}",
                    card=handoff.card,
                    incoming=handoff.load,
                    priority=0,
                    technique=handoff.technique,
                    reason=handoff.reason,
                    alternatives=list(handoff.alternatives),
                    fx_effect={
                        "filter_exit": "echo",
                        "breakdown_handoff": "reverb",
                        "loop_bridge": "spiral",
                        "vocal_safe_loop_blend": "spiral",
                        "phrase_cut": "vinyl_brake",
                    }.get(handoff.technique),
                )
            )

        plan = AutonomousSetPlan(
            name=brief.name,
            opening=_load_spec(route[0]),
            transitions=transitions,
            target_track_count=len(route),
            stage_deadline_bars=64,
            reserve_deadline_bars=32,
            rescue_loop_trigger_bars=16,
            rescue_loop_beats=16,
            retry_limit=3,
            tempo_strategy="auto",
            tempo_target_bpm=(
                brief.target_bpm
                if brief.target_bpm is not None
                else target.bpm
                if target is not None
                else None
            ),
        )
        return plan.materialize_tempo_arc(
            {track.track_id: track.bpm for track in route}
        )

    @staticmethod
    def _score_candidate(
        current: TrackProfile,
        candidate: TrackProfile,
        *,
        depth: int,
        total: int,
        target_bpm: float | None,
        target_track: TrackProfile | None,
        vibe: str,
    ) -> tuple[float, str, str]:
        harmonic = camelot_compatibility(current.key, candidate.key)
        harmonic_score = 40.0 if harmonic.get("compatible") else -60.0
        bpm_score = 30.0 - abs(candidate.bpm - current.bpm) * 7.5
        cue_score = 20.0 if _verified_bass_handoff(candidate) is not None else 0.0
        final_bpm = (
            target_bpm
            if target_bpm is not None
            else target_track.bpm
            if target_track is not None
            else current.bpm
        )
        desired = current.bpm + (final_bpm - current.bpm) * depth / max(1, total - 1)
        arc_score = 15.0 - abs(candidate.bpm - desired) * 3.0
        destination_score = 0.0
        if target_track is not None:
            toward_target = camelot_compatibility(candidate.key, target_track.key)
            destination_score = 12.0 if toward_target.get("compatible") else -8.0
        vocal_bars = sum(
            segment.end_bar - segment.start_bar
            for segment in candidate.segments
            if segment.kind == "vocal"
        )
        total_bars = max(
            (phrase.start_bar for phrase in candidate.phrase_boundaries),
            default=1,
        )
        vocal_density = min(1.0, vocal_bars / total_bars)
        vibe_score = {
            "maintain": -abs(candidate.bpm - current.bpm) * 2.0,
            "downtempo": (current.bpm - candidate.bpm) * 5.0,
            "energetic": (candidate.bpm - current.bpm) * 5.0,
            "deeper": (10.0 if (candidate.key or "").endswith("A") else 0.0)
            + (current.bpm - candidate.bpm) * 2.0,
            "vocal": vocal_density * 24.0,
            "instrumental": (1.0 - vocal_density) * 24.0,
        }[vibe]
        return (
            harmonic_score
            + bpm_score
            + cue_score
            + arc_score
            + destination_score
            + vibe_score,
            candidate.artist.casefold(),
            candidate.title.casefold(),
        )
