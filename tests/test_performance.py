from pathlib import Path

from rekordbox_performer.intelligence import (
    AnalysisBeatGridPoint,
    MusicalEvent,
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
            AnalysisBeatGridPoint(index=241, bar=61, beat=1, bpm=120, time_ms=120000),
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


def test_sync_report_flags_phase_or_mode() -> None:
    report = sync_report(
        {"bpm": 120, "beat_phase": 0.0},
        {"bpm": 120, "beat_phase": 0.25, "sync_enabled": False},
    )
    assert report["verified"] is False
    assert report["beat_phase_error_ms"] == 125


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
