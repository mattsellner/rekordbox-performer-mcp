import json
from pathlib import Path

from rekordbox_performer.intelligence import (
    DeckObservation,
    AnalysisBeatGridPoint,
    LiveState,
    MusicalEvent,
    PlaylistTrackMetadata,
    ProfileStore,
    PhraseBoundary,
    RekordboxAnalysisImport,
    RehearsalReview,
    TrackLandmark,
    TrackProfile,
    TrackSegment,
    TransitionCard,
    SetPlan,
    audit_set_plan,
    compile_transition_card,
    camelot_compatibility,
    normalize_camelot,
    validate_transition_card,
)


def prepared_profile(track_id: str, title: str) -> TrackProfile:
    return TrackProfile(
        track_id=track_id,
        title=title,
        artist="Artist",
        bpm=128,
        key="9A",
        beatgrid_confidence="verified",
        phrase_confidence="verified",
        vocal_confidence="verified",
        phrase_boundaries=[
            PhraseBoundary(
                index=1,
                start_beat=1,
                end_beat=64,
                start_bar=1,
                beat_in_bar=1,
                length_beats=64,
                length_bars=16,
                kind_code=1,
                label="intro",
                confidence="verified",
            ),
            PhraseBoundary(
                index=2,
                start_beat=65,
                end_beat=128,
                start_bar=17,
                beat_in_bar=1,
                length_beats=64,
                length_bars=16,
                kind_code=5,
                label="chorus",
                confidence="verified",
            ),
        ],
        landmarks=[
            TrackLandmark(
                name="phrase one",
                kind="mix_in",
                bar=1,
                cue=1,
                confidence="verified",
            ),
            TrackLandmark(
                name="bass entry",
                kind="bass_in",
                bar=17,
                confidence="verified",
            ),
            TrackLandmark(
                name="verified drop",
                kind="drop",
                bar=17,
                confidence="verified",
            ),
            TrackLandmark(
                name="clean exit",
                kind="mix_out",
                bar=65,
                confidence="verified",
            ),
        ],
        segments=[
            TrackSegment(
                kind="vocal",
                start_bar=17,
                end_bar=33,
                confidence="verified",
            )
        ],
    )


def valid_card() -> TransitionCard:
    return TransitionCard(
        name="A to B",
        outgoing_track_id="a",
        incoming_track_id="b",
        anchor_deck=1,
        outgoing_deck=1,
        incoming_deck=2,
        start_quantum_bars=16,
        phrase_alignment_verified=True,
        vocal_plan_verified=True,
        bass_plan_verified=True,
        incoming_loaded_verified=True,
        intended_vocal_owner="incoming",
        critical_bar_offset=16,
        abort_plan="Keep A playing and cancel B.",
        events=[
            MusicalEvent(
                bar_offset=0,
                action="hot_cue",
                parameters={"deck": 2, "cue": 1},
            ),
            MusicalEvent(
                bar_offset=16,
                action="eq_low",
                parameters={"deck": 1, "value": -1},
            ),
            MusicalEvent(
                bar_offset=16,
                action="eq_low",
                parameters={"deck": 2, "value": 0},
            ),
            MusicalEvent(
                bar_offset=20,
                action="channel_fader",
                parameters={"deck": 1, "value": 0},
            ),
            MusicalEvent(
                bar_offset=20,
                beat_offset=1,
                action="cue",
                parameters={"deck": 1},
            ),
        ],
    )


def test_ready_profile_requires_vocal_map() -> None:
    profile = prepared_profile("a", "A")
    assert profile.readiness()["ready"] is True
    assert profile.readiness()["tier"] == "A"
    profile.segments = []
    assert profile.readiness()["checks"]["vocals"] is False
    assert profile.readiness()["tier"] == "B"


def test_normalize_camelot_accepts_rekordbox_and_note_keys() -> None:
    assert normalize_camelot("7A") == "7A"
    assert normalize_camelot("Ebm") == "2A"
    assert normalize_camelot("F#") == "2B"
    assert normalize_camelot(None) is None


def test_camelot_compatibility_rejects_seven_a_to_two_a() -> None:
    result = camelot_compatibility("7A", "Ebm")
    assert result["outgoing"] == "7A"
    assert result["incoming"] == "2A"
    assert result["compatible"] is False


def test_card_rejects_incompatible_harmonic_move_by_default() -> None:
    outgoing = prepared_profile("a", "A")
    incoming = prepared_profile("b", "B")
    outgoing.key = "7A"
    incoming.key = "Ebm"
    errors = validate_transition_card(valid_card(), outgoing, incoming)
    assert any("7A -> 2A" in error for error in errors)


def test_card_can_explicitly_accept_harmonic_risk() -> None:
    outgoing = prepared_profile("a", "A")
    incoming = prepared_profile("b", "B")
    outgoing.key = "7A"
    incoming.key = "2A"
    card = valid_card()
    card.harmonic_risk_accepted = True
    errors = validate_transition_card(card, outgoing, incoming)
    assert not any("harmonic" in error for error in errors)


def test_live_state_rolls_rounded_phase_into_next_beat() -> None:
    state = LiveState()
    state.update(
        DeckObservation(
            deck=1,
            track_id="a",
            title="A",
            bpm=128,
            playing=False,
            bar=1,
            beat=1,
            track_beat=1,
            beat_phase=0.99996,
            source="native",
            confidence="high",
        )
    )

    current = state.get(1)
    assert current["track_beat"] == 2
    assert current["beat"] == 2
    assert current["beat_phase"] == 0.0


def test_phrase_cut_can_launch_verified_file_start_without_hot_cue() -> None:
    outgoing = prepared_profile("a", "A")
    incoming = prepared_profile("b", "B")
    card = valid_card()
    card.transition_family = "phrase_cut"
    card.events[0] = MusicalEvent(
        bar_offset=0,
        action="play_pause",
        parameters={"deck": 2},
    )
    errors = validate_transition_card(card, outgoing, incoming)
    assert not any("launch" in error.lower() for error in errors)


def test_bass_swap_still_requires_verified_hot_cue() -> None:
    outgoing = prepared_profile("a", "A")
    incoming = prepared_profile("b", "B")
    card = valid_card()
    card.events[0] = MusicalEvent(
        bar_offset=0,
        action="play_pause",
        parameters={"deck": 2},
    )
    errors = validate_transition_card(card, outgoing, incoming)
    assert any("requires a verified Hot Cue" in error for error in errors)


def test_card_compiles_to_next_phrase_boundary() -> None:
    state = LiveState()
    state.update(
        DeckObservation(
            deck=1,
            track_id="a",
            title="A",
            bpm=128,
            playing=True,
            bar=6,
            beat=2,
            beat_phase=0.25,
            source="native",
            confidence="verified",
            sync_enabled=True,
            quantize_enabled=True,
        )
    )
    state.update(
        DeckObservation(
            deck=2,
            track_id="b",
            title="B",
            bpm=128,
            playing=False,
            bar=1,
            beat=1,
            source="native",
            confidence="verified",
            sync_enabled=True,
            quantize_enabled=True,
        )
    )
    compiled = compile_transition_card(
        valid_card(),
        prepared_profile("a", "A"),
        prepared_profile("b", "B"),
        state,
    )
    assert compiled["ready"] is True
    assert compiled["start_basis"] == "rekordbox_phrase_analysis"
    assert compiled["start_track_beat"] == 65
    assert compiled["events"][1]["at_ms"] > compiled["events"][0]["at_ms"]


def test_card_rejects_unverified_vocal_plan() -> None:
    state = LiveState()
    state.update(
        DeckObservation(
            deck=1,
            track_id="a",
            title="A",
            bpm=128,
            playing=True,
            bar=1,
            beat=1,
            source="native",
            confidence="verified",
            sync_enabled=True,
            quantize_enabled=True,
        )
    )
    state.update(
        DeckObservation(
            deck=2,
            track_id="b",
            title="B",
            bpm=128,
            playing=False,
            bar=1,
            beat=1,
            source="native",
            confidence="verified",
            sync_enabled=True,
            quantize_enabled=True,
        )
    )
    card = valid_card()
    card.vocal_plan_verified = False
    compiled = compile_transition_card(
        card,
        prepared_profile("a", "A"),
        prepared_profile("b", "B"),
        state,
    )
    assert compiled["ready"] is False
    assert (
        "vocal plan is not verified and vocal risk is not accepted"
        in compiled["errors"]
    )


def test_card_accepts_explicit_vocal_risk_without_vocal_segments() -> None:
    state = LiveState()
    state.update(
        DeckObservation(
            deck=1,
            track_id="a",
            title="A",
            bpm=128,
            playing=True,
            bar=1,
            beat=1,
            source="native",
            confidence="verified",
            sync_enabled=True,
            quantize_enabled=True,
        )
    )
    state.update(
        DeckObservation(
            deck=2,
            track_id="b",
            title="B",
            bpm=128,
            playing=False,
            bar=1,
            beat=1,
            source="native",
            confidence="verified",
            sync_enabled=True,
            quantize_enabled=True,
        )
    )
    outgoing = prepared_profile("a", "A")
    incoming = prepared_profile("b", "B")
    outgoing.segments = [
        segment for segment in outgoing.segments if segment.kind != "vocal"
    ]
    incoming.segments = [
        segment for segment in incoming.segments if segment.kind != "vocal"
    ]
    outgoing.vocal_confidence = "unknown"
    incoming.vocal_confidence = "unknown"
    card = valid_card()
    card.vocal_plan_verified = False
    card.vocal_risk_accepted = True

    compiled = compile_transition_card(card, outgoing, incoming, state)

    assert compiled["ready"] is True


def test_intentional_overlap_is_backward_compatible_vocal_risk_acceptance() -> None:
    state = LiveState()
    state.update(
        DeckObservation(
            deck=1,
            track_id="a",
            title="A",
            bpm=128,
            playing=True,
            bar=1,
            beat=1,
            source="native",
            confidence="verified",
            sync_enabled=True,
            quantize_enabled=True,
        )
    )
    state.update(
        DeckObservation(
            deck=2,
            track_id="b",
            title="B",
            bpm=128,
            playing=False,
            bar=1,
            beat=1,
            source="native",
            confidence="verified",
            sync_enabled=True,
            quantize_enabled=True,
        )
    )
    outgoing = prepared_profile("a", "A")
    incoming = prepared_profile("b", "B")
    for profile in (outgoing, incoming):
        profile.segments = [
            segment for segment in profile.segments if segment.kind != "vocal"
        ]
        profile.vocal_confidence = "unknown"
    card = valid_card()
    card.intended_vocal_owner = "intentional_overlap"

    compiled = compile_transition_card(card, outgoing, incoming, state)

    assert compiled["ready"] is True


def test_profile_and_rehearsal_store(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path)
    result = store.upsert(prepared_profile("a", "A"))
    assert result["readiness"]["ready"] is True
    audit = store.audit(["a", "missing"])
    assert audit["ready"] is False
    review = RehearsalReview(
        transition_name="A to B",
        outgoing_track_id="a",
        incoming_track_id="b",
        beat_phase_error_ms=12,
        bar_error=0,
        bass_swap_error_beats=0.25,
        vocal_clash=False,
        energy_continuity=9,
        cleanliness=9,
        user_rating=9,
    )
    stored = store.record_rehearsal(review)
    assert stored["score"] >= 8
    assert store.rehearsal_summary("a", "b")["count"] == 1


def test_profile_store_clients_do_not_overwrite_each_other(tmp_path: Path) -> None:
    first = ProfileStore(tmp_path)
    second = ProfileStore(tmp_path)

    first.upsert(prepared_profile("a", "A"))
    second.upsert(prepared_profile("b", "B"))

    assert first.get("a").title == "A"
    assert first.get("b").title == "B"
    assert second.audit(["a", "b"])["ready"] is True


def test_profile_store_migrates_legacy_json_once(tmp_path: Path) -> None:
    legacy = prepared_profile("legacy", "Legacy")
    (tmp_path / "track-profiles.json").write_text(
        json.dumps({"legacy": legacy.model_dump()}),
        encoding="utf-8",
    )

    store = ProfileStore(tmp_path)
    assert store.get("legacy").title == "Legacy"

    (tmp_path / "track-profiles.json").write_text("{}", encoding="utf-8")
    reopened = ProfileStore(tmp_path)
    assert reopened.get("legacy").title == "Legacy"


def test_playlist_seed_preserves_verified_profile(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path)
    store.upsert(prepared_profile("a", "A"))
    result = store.seed_playlist(
        [
            PlaylistTrackMetadata(
                track_id="a",
                title="A",
                artist="Artist",
                bpm=128,
            ),
            PlaylistTrackMetadata(
                track_id="b",
                title="B",
                artist="Artist",
                bpm=129,
            ),
        ]
    )
    assert result["preserved"] == 1
    assert result["created"] == 1
    assert store.get("a").beatgrid_confidence == "verified"
    assert store.get("b").readiness()["ready"] is False


def test_playlist_seed_skips_zero_bpm_without_rejecting_batch(
    tmp_path: Path,
) -> None:
    store = ProfileStore(tmp_path)
    result = store.seed_playlist(
        [
            PlaylistTrackMetadata(
                track_id="clip",
                title="Voice memo",
                artist="",
                bpm=0,
            ),
            PlaylistTrackMetadata(
                track_id="track",
                title="Track",
                artist="Artist",
                bpm=128,
            ),
        ]
    )
    assert result["created"] == 1
    assert result["skipped"][0]["track_id"] == "clip"
    assert store.get("track").title == "Track"


def test_complete_set_plan_audit(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path)
    store.upsert(prepared_profile("a", "A"))
    store.upsert(prepared_profile("b", "B"))
    plan = SetPlan(
        name="two tracks",
        track_ids=["a", "b"],
        cards=[valid_card()],
    )
    result = audit_set_plan(plan, store)
    assert result["ready"] is True


def test_set_plan_audit_accepts_only_waived_vocal_gap(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path)
    for track_id, title in (("a", "A"), ("b", "B")):
        profile = prepared_profile(track_id, title)
        profile.segments = [
            segment for segment in profile.segments if segment.kind != "vocal"
        ]
        profile.vocal_confidence = "unknown"
        store.upsert(profile)
    card = valid_card()
    card.vocal_plan_verified = False
    card.vocal_risk_accepted = True
    plan = SetPlan(
        name="two tracks with accepted vocal risk",
        track_ids=["a", "b"],
        cards=[card],
    )

    result = audit_set_plan(plan, store)

    assert result["ready"] is True


def test_native_analysis_ingest_preserves_manual_structure(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path)
    existing = prepared_profile("a", "A")
    store.upsert(existing)
    result = store.ingest_analysis(
        RekordboxAnalysisImport(
            track_id="a",
            title="A",
            artist="Artist",
            source="spotify",
            service_uri="spotify:track:abc",
            bpm=128,
            beatgrid_available=True,
            phrase_analysis_available=True,
            beat_count=2,
            phrase_count=1,
            beat_grid=[
                AnalysisBeatGridPoint(
                    index=1,
                    bar=1,
                    beat=1,
                    bpm=128,
                    time_ms=50,
                ),
                AnalysisBeatGridPoint(
                    index=2,
                    bar=1,
                    beat=2,
                    bpm=128,
                    time_ms=519,
                ),
            ],
            phrases=[
                PhraseBoundary(
                    index=1,
                    start_beat=1,
                    end_beat=64,
                    start_bar=1,
                    beat_in_bar=1,
                    length_beats=64,
                    length_bars=16,
                    kind_code=1,
                    label="intro",
                )
            ],
        )
    )
    profile = TrackProfile.model_validate(result["profile"])
    assert profile.source == "spotify"
    assert profile.beatgrid_confidence == "high"
    assert profile.phrase_boundaries[0].label == "intro"
    assert profile.beat_grid[0].time_ms == 50
    assert any(item.kind == "mix_out" for item in profile.landmarks)
    assert any(item.kind == "bass_in" for item in profile.landmarks)


def test_native_analysis_adds_analyzed_outro_mix_out(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path)
    result = store.ingest_analysis(
        RekordboxAnalysisImport(
            track_id="a",
            title="A",
            artist="Artist",
            bpm=128,
            beatgrid_available=True,
            phrase_analysis_available=True,
            beat_count=2,
            phrase_count=2,
            beat_grid=[
                AnalysisBeatGridPoint(
                    index=1,
                    bar=1,
                    beat=1,
                    bpm=128,
                    time_ms=50,
                ),
                AnalysisBeatGridPoint(
                    index=65,
                    bar=17,
                    beat=1,
                    bpm=128,
                    time_ms=30050,
                ),
            ],
            phrases=[
                PhraseBoundary(
                    index=1,
                    start_beat=1,
                    end_beat=64,
                    start_bar=1,
                    beat_in_bar=1,
                    length_beats=64,
                    length_bars=16,
                    kind_code=1,
                    label="intro",
                ),
                PhraseBoundary(
                    index=2,
                    start_beat=65,
                    end_beat=128,
                    start_bar=17,
                    beat_in_bar=1,
                    length_beats=64,
                    length_bars=16,
                    kind_code=6,
                    label="outro",
                ),
            ],
        )
    )
    profile = TrackProfile.model_validate(result["profile"])
    mix_out = next(
        item
        for item in profile.landmarks
        if item.kind == "mix_out"
    )
    assert mix_out.bar == 17
    assert mix_out.time_ms == 30050
    assert profile.analysis_readiness()["ready"] is True


def test_native_vocal_and_low_band_segments_complete_profile_readiness(
    tmp_path: Path,
) -> None:
    store = ProfileStore(tmp_path)
    result = store.ingest_analysis(
        RekordboxAnalysisImport(
            track_id="a",
            title="A",
            artist="Artist",
            bpm=128,
            beatgrid_available=True,
            phrase_analysis_available=True,
            beat_count=128,
            phrase_count=2,
            beat_grid=[
                AnalysisBeatGridPoint(
                    index=1,
                    bar=1,
                    beat=1,
                    bpm=128,
                    time_ms=0,
                ),
                AnalysisBeatGridPoint(
                    index=65,
                    bar=17,
                    beat=1,
                    bpm=128,
                    time_ms=30000,
                ),
            ],
            phrases=[
                PhraseBoundary(
                    index=1,
                    start_beat=1,
                    end_beat=64,
                    start_bar=1,
                    beat_in_bar=1,
                    length_beats=64,
                    length_bars=16,
                    kind_code=1,
                    label="intro",
                ),
                PhraseBoundary(
                    index=2,
                    start_beat=65,
                    end_beat=128,
                    start_bar=17,
                    beat_in_bar=1,
                    length_beats=64,
                    length_bars=16,
                    kind_code=6,
                    label="outro",
                ),
            ],
            vocal_analysis_available=True,
            vocal_segments=[
                TrackSegment(
                    kind="vocal",
                    start_bar=5,
                    end_bar=12,
                    confidence="high",
                )
            ],
            bass_analysis_available=True,
            bass_segments=[
                TrackSegment(
                    kind="bass",
                    start_bar=1,
                    end_bar=17,
                    confidence="high",
                )
            ],
        )
    )
    profile = TrackProfile.model_validate(result["profile"])
    assert profile.vocal_confidence == "high"
    assert any(item.kind == "bass_in" for item in profile.landmarks)
    assert any(item.kind == "bass_out" for item in profile.landmarks)
    assert profile.readiness()["ready"] is True


def test_card_rejects_unobserved_or_disabled_sync() -> None:
    state = LiveState()
    state.update(
        DeckObservation(
            deck=1,
            track_id="a",
            title="A",
            bpm=128,
            playing=True,
            bar=1,
            beat=1,
            source="native",
            confidence="verified",
            sync_enabled=True,
            quantize_enabled=True,
        )
    )
    state.update(
        DeckObservation(
            deck=2,
            track_id="b",
            title="B",
            bpm=128,
            playing=False,
            bar=1,
            beat=1,
            source="native",
            confidence="verified",
            sync_enabled=False,
            quantize_enabled=True,
        )
    )
    compiled = compile_transition_card(
        valid_card(),
        prepared_profile("a", "A"),
        prepared_profile("b", "B"),
        state,
    )
    assert compiled["ready"] is False
    assert "incoming Beat Sync is off" in compiled["errors"]


def test_card_rejects_bass_swap_split_across_critical_bar() -> None:
    card = valid_card()
    card.events[2].beat_offset = 2
    state = LiveState()
    for deck, track_id, playing in ((1, "a", True), (2, "b", False)):
        state.update(
            DeckObservation(
                deck=deck,
                track_id=track_id,
                title=track_id.upper(),
                bpm=128,
                playing=playing,
                bar=1,
                beat=1,
                source="native",
                confidence="verified",
                sync_enabled=True,
                quantize_enabled=True,
            )
        )
    compiled = compile_transition_card(
        card,
        prepared_profile("a", "A"),
        prepared_profile("b", "B"),
        state,
    )
    assert compiled["ready"] is False
    assert (
        "critical downbeat lacks a simultaneous two-deck bass swap"
        in compiled["errors"]
    )


def test_card_rejects_unverified_incoming_launch_cue() -> None:
    card = valid_card()
    card.events[0].parameters["cue"] = 2
    state = LiveState()
    for deck, track_id, playing in ((1, "a", True), (2, "b", False)):
        state.update(
            DeckObservation(
                deck=deck,
                track_id=track_id,
                title=track_id.upper(),
                bpm=128,
                playing=playing,
                bar=1,
                beat=1,
                source="native",
                confidence="verified",
                sync_enabled=True,
                quantize_enabled=True,
            )
        )
    compiled = compile_transition_card(
        card,
        prepared_profile("a", "A"),
        prepared_profile("b", "B"),
        state,
    )
    assert compiled["ready"] is False
    assert (
        "incoming launch cue is not a high-confidence mix-in or "
        "phrase-start landmark"
        in compiled["errors"]
    )


def test_bass_swap_rejects_critical_bar_without_verified_drop() -> None:
    incoming = prepared_profile("b", "B")
    incoming.landmarks = [
        landmark for landmark in incoming.landmarks if landmark.kind != "drop"
    ]
    errors = validate_transition_card(
        valid_card(),
        prepared_profile("a", "A"),
        incoming,
    )
    assert (
        "critical bass swap does not land on a verified incoming drop"
        in errors
    )


def test_tempo_mismatch_requires_verified_beat_sync() -> None:
    incoming = prepared_profile("b", "B")
    incoming.bpm = 129
    card = valid_card()
    card.beat_sync_required = False
    errors = validate_transition_card(
        card,
        prepared_profile("a", "A"),
        incoming,
    )
    assert "tempo-mismatched tracks require verified Beat Sync" in errors


def test_card_rejects_manual_clock_and_playing_incoming_deck() -> None:
    state = LiveState()
    state.update(
        DeckObservation(
            deck=1,
            track_id="a",
            title="A",
            bpm=128,
            playing=True,
            bar=1,
            beat=1,
            source="manual",
            confidence="verified",
            sync_enabled=True,
            quantize_enabled=True,
        )
    )
    state.update(
        DeckObservation(
            deck=2,
            track_id="b",
            title="B",
            bpm=128,
            playing=True,
            bar=1,
            beat=1,
            source="native",
            confidence="verified",
            sync_enabled=True,
            quantize_enabled=True,
        )
    )
    compiled = compile_transition_card(
        valid_card(),
        prepared_profile("a", "A"),
        prepared_profile("b", "B"),
        state,
    )
    assert compiled["ready"] is False
    assert "manual observations are not authoritative live clocks" in compiled["errors"]
    assert "incoming deck must be stopped before its scheduled launch" in compiled["errors"]


def test_phrase_compiler_respects_non_bar_one_boundary() -> None:
    outgoing = prepared_profile("a", "A")
    outgoing.phrase_boundaries = [
        PhraseBoundary(
            index=1,
            start_beat=1,
            end_beat=66,
            start_bar=1,
            beat_in_bar=1,
            length_beats=66,
            kind_code=1,
            label="intro",
        ),
        PhraseBoundary(
            index=2,
            start_beat=67,
            end_beat=98,
            start_bar=17,
            beat_in_bar=3,
            length_beats=32,
            length_bars=8,
            kind_code=5,
            label="chorus",
        ),
    ]
    state = LiveState()
    state.update(
        DeckObservation(
            deck=1,
            track_id="a",
            title="A",
            bpm=128,
            playing=True,
            bar=1,
            beat=1,
            track_beat=1,
            source="native",
            confidence="verified",
            sync_enabled=True,
            quantize_enabled=True,
        )
    )
    state.update(
        DeckObservation(
            deck=2,
            track_id="b",
            title="B",
            bpm=128,
            playing=False,
            bar=1,
            beat=1,
            source="native",
            confidence="verified",
            sync_enabled=True,
            quantize_enabled=True,
        )
    )
    compiled = compile_transition_card(
        valid_card(),
        outgoing,
        prepared_profile("b", "B"),
        state,
    )
    assert compiled["ready"] is True
    assert compiled["start_track_beat"] == 67
    assert compiled["start_beat_in_bar"] == 3


def test_stopped_anchor_compiles_verified_dual_hot_cue_launch() -> None:
    outgoing = prepared_profile("a", "A")
    outgoing.landmarks.append(
        TrackLandmark(
            name="bridge",
            kind="phrase_start",
            bar=65,
            beat=1,
            cue=2,
            confidence="verified",
        )
    )
    card = valid_card()
    card.beat_sync_required = False
    card.critical_bar_offset = 4
    incoming = prepared_profile("b", "B")
    incoming.landmarks.append(
        TrackLandmark(
            name="short-entry drop",
            kind="drop",
            bar=5,
            confidence="verified",
        )
    )
    card.events = [
        MusicalEvent(
            bar_offset=0,
            action="hot_cue",
            parameters={"deck": 1, "cue": 2},
        ),
        MusicalEvent(
            bar_offset=0,
            action="hot_cue",
            parameters={"deck": 2, "cue": 1},
        ),
        MusicalEvent(
            bar_offset=4,
            action="eq_low",
            parameters={"deck": 1, "value": -1},
        ),
        MusicalEvent(
            bar_offset=4,
            action="eq_low",
            parameters={"deck": 2, "value": 0},
        ),
        MusicalEvent(
            bar_offset=8,
            action="channel_fader",
            parameters={"deck": 1, "value": 0},
        ),
        MusicalEvent(
            bar_offset=8,
            beat_offset=1,
            action="cue",
            parameters={"deck": 1},
        ),
    ]
    state = LiveState()
    for deck, track_id in ((1, "a"), (2, "b")):
        state.update(
            DeckObservation(
                deck=deck,
                track_id=track_id,
                title=track_id.upper(),
                bpm=128,
                playing=False,
                bar=1,
                beat=1,
                source="native",
                confidence="verified",
                sync_enabled=False,
                quantize_enabled=True,
            )
        )

    compiled = compile_transition_card(card, outgoing, incoming, state)

    assert compiled["ready"] is True
    assert compiled["start_basis"] == "verified_dual_hot_cue"
    assert compiled["start_bar"] == 65
    assert compiled["events"][0]["at_ms"] == 250
    assert compiled["events"][1]["at_ms"] == 250
    assert compiled["events"][2]["at_ms"] == 7750
