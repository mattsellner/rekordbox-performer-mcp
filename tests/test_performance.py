from pathlib import Path

from rekordbox_performer.intelligence import (
    AnalysisBeatGridPoint,
    BassEnergyBar,
    MusicalEvent,
    PhraseBoundary,
    TrackLandmark,
    TrackProfile,
    TrackSegment,
    TransitionCard,
)
from rekordbox_performer.performance import (
    SetSessionManager,
    cue_preparation_plan,
    fx_recipe,
    observation_from_elapsed,
    sync_report,
    transition_qa,
    vocal_handoff,
)


def profile(track_id: str) -> TrackProfile:
    return TrackProfile(
        track_id=track_id,
        title=track_id.upper(),
        artist="Artist",
        bpm=120,
        beatgrid_confidence="high",
        phrase_confidence="high",
        vocal_confidence="high",
        beat_grid=[
            AnalysisBeatGridPoint(index=1, bar=1, beat=1, bpm=120, time_ms=0),
            AnalysisBeatGridPoint(index=65, bar=17, beat=1, bpm=120, time_ms=32000),
            AnalysisBeatGridPoint(index=241, bar=61, beat=1, bpm=120, time_ms=120000),
        ],
        phrase_boundaries=[
            PhraseBoundary(
                index=1,
                start_beat=65,
                end_beat=96,
                start_bar=17,
                beat_in_bar=1,
                length_beats=32,
                length_bars=8,
                kind_code=1,
                label="UP 1",
                confidence="high",
            ),
            PhraseBoundary(
                index=2,
                start_beat=129,
                end_beat=192,
                start_bar=33,
                beat_in_bar=1,
                length_beats=64,
                length_bars=16,
                kind_code=5,
                label="DROP",
                confidence="high",
            ),
        ],
        bass_energy_by_bar=[
            BassEnergyBar(bar=bar, median=10, mean=12, peak=20)
            for bar in range(1, 81)
        ],
        landmarks=[
            TrackLandmark(name="drop", kind="drop", bar=33, confidence="high"),
        ],
        segments=[TrackSegment(kind="vocal", start_bar=17, end_bar=25, confidence="high")],
    )


def card() -> TransitionCard:
    return TransitionCard(
        name="handoff",
        outgoing_track_id="a",
        incoming_track_id="b",
        anchor_deck=1,
        outgoing_deck=1,
        incoming_deck=2,
        intended_vocal_owner="none",
        critical_bar_offset=8,
        phrase_alignment_verified=True,
        vocal_plan_verified=True,
        bass_plan_verified=True,
        incoming_loaded_verified=True,
        events=[MusicalEvent(bar_offset=0, action="hot_cue", parameters={"deck": 2, "cue": 1})],
        abort_plan="Keep outgoing playing",
    )


def test_elapsed_reconciliation_uses_absolute_grid_position() -> None:
    observed = observation_from_elapsed(
        deck=1,
        profile=profile("a"),
        elapsed_seconds=120,
        playing=True,
        sync_enabled=True,
        quantize_enabled=True,
    )
    assert observed.track_beat == 241
    assert observed.bar == 61
    assert observed.source == "native"


def test_elapsed_reconciliation_preserves_pickup_grid_position() -> None:
    candidate = profile("a")
    candidate.beat_grid = [
        AnalysisBeatGridPoint(index=1, bar=1, beat=3, bpm=120, time_ms=0),
        AnalysisBeatGridPoint(index=3, bar=2, beat=1, bpm=120, time_ms=1000),
    ]

    observed = observation_from_elapsed(
        deck=1,
        profile=candidate,
        elapsed_seconds=0,
        playing=True,
        sync_enabled=True,
        quantize_enabled=True,
    )

    assert observed.track_beat == 1
    assert observed.bar == 1
    assert observed.beat == 3


def test_vocal_overlap_is_soft_and_measured() -> None:
    report = vocal_handoff(
        profile("a"), profile("b"),
        outgoing_start_bar=17,
        incoming_start_bar=17,
        overlap_bars=8,
    )
    assert report["overlap_bars"] == 8
    assert report["blocking"] is False


def test_cue_plan_selects_sixteen_bars_before_drop() -> None:
    result = cue_preparation_plan(profile("a"))
    assert result["suggestions"][0]["bar"] == 17
    assert result["suggestions"][0]["role"] == "16_bars_before_drop"
    assert result["suggestions"][0]["cue"] == 7
    assert result["suggestions"][0]["verified_bass_phrase_start"] is True
    assert result["cue_policy"] == "automation uses Hot Cue G/H only"


def test_cue_plan_rejects_non_phrase_downbeats_and_never_uses_user_cues() -> None:
    candidate = profile("a")
    candidate.landmarks = [
        TrackLandmark(name="early bass", kind="bass_in", bar=29, confidence="high"),
        TrackLandmark(name="verified bass", kind="bass_in", bar=33, confidence="high"),
        TrackLandmark(name="user cue A", kind="phrase_start", bar=1, cue=1, confidence="verified"),
    ]

    result = cue_preparation_plan(candidate)

    assert [item["bar"] for item in result["suggestions"]] == [17]
    assert [item["cue"] for item in result["suggestions"]] == [7]


def test_cue_plan_prefers_the_stronger_later_bass_phrase() -> None:
    candidate = profile("a")
    candidate.phrase_boundaries.append(
        PhraseBoundary(
            index=3,
            start_beat=193,
            end_beat=224,
            start_bar=49,
            beat_in_bar=1,
            length_beats=32,
            length_bars=8,
            kind_code=5,
            label="DROP",
            confidence="high",
        )
    )
    candidate.landmarks.append(
        TrackLandmark(name="later drop", kind="drop", bar=49, confidence="high")
    )
    candidate.bass_energy_by_bar = [
        BassEnergyBar(
            bar=bar,
            median=(2 if 33 <= bar <= 40 else 20 if 49 <= bar <= 56 else 10),
            mean=(3 if 33 <= bar <= 40 else 22 if 49 <= bar <= 56 else 12),
            peak=(6 if 33 <= bar <= 40 else 30 if 49 <= bar <= 56 else 20),
        )
        for bar in range(1, 81)
    ]

    result = cue_preparation_plan(candidate)

    assert result["suggestions"][0]["target"]["bar"] == 49
    assert result["suggestions"][0]["bass_waveform_evidence"]["verified"] is True


def test_sync_report_flags_phase_or_mode() -> None:
    report = sync_report(
        {"bpm": 120, "beat_phase": 0.0},
        {"bpm": 120, "beat_phase": 0.25, "sync_enabled": False},
    )
    assert report["verified"] is False
    assert report["beat_phase_error_ms"] == 125


def test_sync_report_rejects_matching_beats_on_different_bar_positions() -> None:
    report = sync_report(
        {"bpm": 120, "beat": 3, "beat_phase": 0.02},
        {"bpm": 120, "beat": 1, "beat_phase": 0.02, "sync_enabled": True},
    )

    assert report["verified"] is False
    assert report["beat_phase_error_ms"] == 0
    assert report["bar_phase_error_beats"] == 2
    assert "beat-in-bar alignment error is 2.00 beats" in report["errors"]


def test_fx_recipe_has_explicit_reset() -> None:
    events = fx_recipe(card())["events"]
    assert fx_recipe(card())["effect"] == "echo"
    assert fx_recipe(card())["selection"]["requires_observed_current_effect"] is True
    assert any(event["action"] == "fx_wet_dry" and event["parameters"]["value"] == 0 for event in events)
    assert sum(event["action"] == "fx_toggle" for event in events) == 2


def test_set_session_persists_and_advances(tmp_path: Path) -> None:
    first = SetSessionManager(tmp_path / "set.json")
    created = first.create("night", ["a", "b", "c"])
    assert created["following"] == "c"
    first.start()
    second = SetSessionManager(tmp_path / "set.json")
    advanced = second.advance("job-1", True)
    assert advanced["current"] == "b"
    assert advanced["next"] == "c"


def test_transition_qa_fails_bad_dispatch() -> None:
    report = transition_qa({
        "id": "job-1",
        "status": "completed",
        "p99_event_lateness_ms": 25,
        "completed_events": 4,
        "verification": {"verified": True, "errors": []},
    })
    assert report["passed"] is False
    assert report["score"] < 100


def test_transition_qa_fails_unverified_beat_phase() -> None:
    report = transition_qa({
        "id": "job-phase-error",
        "status": "completed",
        "p99_event_lateness_ms": 2,
        "completed_events": 17,
        "verification": {
            "verified": True,
            "errors": [],
            "sync": {
                "verified": False,
                "errors": ["beat phase error is 83.1 ms"],
                "beat_phase_error_ms": 83.1,
            },
        },
    })

    assert report["passed"] is False
    assert "beat phase error is 83.1 ms" in report["faults"]
    assert report["metrics"]["beat_phase_error_ms"] == 83.1
