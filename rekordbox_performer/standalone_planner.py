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
    validate_transition_card,
)
from .set_runner import AutonomousSetPlan, TrackLoadSpec, TransitionOption


class DJBrief(BaseModel):
    start_track_id: str
    target_track_count: int = Field(default=6, ge=2, le=20)
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
    max_native_bpm_delta: float = Field(default=4.0, gt=0, le=12)
    allow_safe_cuts: bool = True


@dataclass(frozen=True)
class CompiledHandoff:
    card: TransitionCard
    load: TrackLoadSpec
    technique: str
    cue_landmark: TrackLandmark | None
    bass_landmark: TrackLandmark | None


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


class TransitionKingCompiler:
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


class LocalDJPlanner:
    """Choose a repeatable route and compile every adjacent transition."""

    def __init__(self, profiles: Iterable[TrackProfile]) -> None:
        self.profiles = {
            profile.track_id: profile
            for profile in profiles
            if profile.readiness()["ready"]
        }
        self.compiler = TransitionKingCompiler()

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
        excluded = set(brief.excluded_track_ids) - {brief.start_track_id}
        route = [self.profiles[brief.start_track_id]]
        while len(route) < brief.target_track_count:
            current = route[-1]
            used = {track.track_id for track in route}
            final_slot = len(route) == brief.target_track_count - 1
            if final_slot and target is not None:
                candidates = [target]
            else:
                candidates = [
                    profile
                    for profile in self.profiles.values()
                    if profile.track_id not in used
                    and profile.track_id not in excluded
                    and (target is None or profile.track_id != target.track_id)
                ]
            candidates = [
                profile
                for profile in candidates
                if abs(profile.bpm - current.bpm) <= brief.max_native_bpm_delta
            ]
            ranked = sorted(
                candidates,
                key=lambda profile: self._score_candidate(
                    current,
                    profile,
                    depth=len(route),
                    total=brief.target_track_count,
                    target_bpm=brief.target_bpm,
                    target_track=target,
                    vibe=brief.vibe,
                ),
                reverse=True,
            )
            selected = None
            for candidate in ranked:
                if (
                    not brief.allow_safe_cuts
                    and _verified_bass_handoff(candidate) is None
                ):
                    continue
                try:
                    self.compiler.compile(
                        current,
                        candidate,
                        outgoing_deck=(
                            opening_deck
                            if len(route) % 2
                            else (2 if opening_deck == 1 else 1)
                        ),
                    )
                except ValueError:
                    continue
                selected = candidate
                break
            if selected is None:
                direction = (
                    f" while steering toward {target.title}"
                    if target is not None
                    else ""
                )
                raise ValueError(
                    f"no safe prepared successor exists for {current.title}{direction}"
                )
            route.append(selected)

        transitions: list[TransitionOption] = []
        for index, (outgoing, incoming) in enumerate(pairwise(route)):
            outgoing_deck = (
                opening_deck if index % 2 == 0 else (2 if opening_deck == 1 else 1)
            )
            handoff = self.compiler.compile(
                outgoing,
                incoming,
                outgoing_deck=outgoing_deck,
            )
            transitions.append(
                TransitionOption(
                    id=f"{index + 1}-{outgoing.track_id}-{incoming.track_id}",
                    card=handoff.card,
                    incoming=handoff.load,
                    priority=0,
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
