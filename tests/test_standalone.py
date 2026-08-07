import asyncio

from rekordbox_performer.app import format_snapshot
from rekordbox_performer.intelligence import (
    BassEnergyBar,
    PhraseBoundary,
    ProfileStore,
    TrackLandmark,
    TrackProfile,
    TrackSegment,
    validate_transition_card,
)
from rekordbox_performer.standalone_engine import StandaloneDJEngine
from rekordbox_performer.standalone_planner import (
    DJBrief,
    LocalDJPlanner,
    TransitionKingCompiler,
    _breakdown_events,
)
from rekordbox_performer.standalone_status import (
    RuntimePhase,
    StandaloneStatusStore,
    TrackRole,
)


def profile(track_id: str, title: str, bpm: float, key: str) -> TrackProfile:
    return TrackProfile(
        track_id=track_id,
        title=title,
        artist=f"Artist {title}",
        bpm=bpm,
        key=key,
        source="local",
        time_signature=4,
        beat_count=320,
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
            PhraseBoundary(
                index=3,
                start_beat=129,
                end_beat=192,
                start_bar=33,
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
                name="automation G",
                kind="mix_in",
                bar=1,
                beat=1,
                cue=7,
                time_ms=0,
                confidence="verified",
            ),
            TrackLandmark(
                name="verified bass phrase",
                kind="bass_in",
                bar=17,
                beat=1,
                confidence="verified",
            ),
            TrackLandmark(
                name="clean exit",
                kind="mix_out",
                bar=65,
                beat=1,
                confidence="verified",
            ),
        ],
        segments=[
            TrackSegment(
                kind="vocal",
                start_bar=17,
                end_bar=24,
                confidence="verified",
            ),
            TrackSegment(
                kind="bass",
                start_bar=17,
                end_bar=32,
                confidence="verified",
            ),
        ],
        bass_energy_by_bar=[
            BassEnergyBar(bar=bar, median=10, mean=12, peak=20) for bar in range(1, 81)
        ],
    )


def test_transition_compiler_preserves_energy_until_verified_bass_swap() -> None:
    outgoing = profile("a", "A", 128, "9A")
    incoming = profile("b", "B", 129, "10A")

    handoff = TransitionKingCompiler().compile(
        outgoing,
        incoming,
        outgoing_deck=1,
    )

    assert handoff.technique == "long_blend"
    assert handoff.load.cue == 7
    assert handoff.card.critical_bar_offset == 16
    assert validate_transition_card(handoff.card, outgoing, incoming) == []
    early_outgoing_fades = [
        event
        for event in handoff.card.events
        if event.action == "channel_fader"
        and event.parameters.get("deck") == 1
        and event.bar_offset <= 16
    ]
    assert early_outgoing_fades == []
    incoming_established = [
        event
        for event in handoff.card.events
        if event.action == "channel_fader"
        and event.parameters.get("deck") == 2
        and event.parameters.get("value", 0) >= 0.7
        and event.bar_offset <= 12
    ]
    assert incoming_established


def test_file_start_fallback_uses_low_eq_swap_and_progressive_cfx_tail() -> None:
    outgoing = profile("a", "A", 128, "9A")
    incoming = profile("b", "B", 129, "10A")
    incoming.landmarks = [
        TrackLandmark(
            name="Rekordbox phrase 1: intro",
            kind="phrase_start",
            bar=1,
            beat=1,
            time_ms=0,
            confidence="high",
        ),
        *[
            item
            for item in incoming.landmarks
            if item.kind not in {"mix_in", "phrase_start"}
        ],
    ]

    handoff = TransitionKingCompiler().compile(
        outgoing,
        incoming,
        outgoing_deck=1,
    )

    assert handoff.technique == "long_blend"
    assert handoff.card.transition_family == "long_blend"
    assert handoff.card.critical_bar_offset == 16
    assert validate_transition_card(handoff.card, outgoing, incoming) == []
    assert any(
        event.action == "eq_low"
        and event.bar_offset == 0
        and event.parameters == {"deck": 2, "value": -1}
        for event in handoff.card.events
    )
    critical = [
        event
        for event in handoff.card.events
        if event.bar_offset == 16 and event.beat_offset == 0
    ]
    assert {
        (event.action, event.parameters.get("deck"), event.parameters.get("value"))
        for event in critical
    } >= {
        ("eq_low", 1, -1),
        ("eq_low", 2, 0),
        ("channel_fader", 2, 1),
    }
    outgoing_fades = [
        event.parameters["value"]
        for event in handoff.card.events
        if event.action == "channel_fader" and event.parameters.get("deck") == 1
    ]
    assert outgoing_fades == [0.86, 0.62, 0.32, 0]
    outgoing_filter = [
        event.parameters["value"]
        for event in handoff.card.events
        if event.action == "filter" and event.parameters.get("deck") == 1
    ]
    assert outgoing_filter == [0.12, 0.22, 0.38, 0.58, 0]


def test_energy_router_aligns_outgoing_down_with_incoming_chorus() -> None:
    outgoing = profile("ocean", "Lost In The Ocean", 125, "3A")
    outgoing.phrase_boundaries = [
        PhraseBoundary(
            index=index,
            start_beat=(bar - 1) * 4 + 1,
            end_beat=(bar - 1) * 4 + 32,
            start_bar=bar,
            beat_in_bar=1,
            length_beats=32,
            length_bars=8,
            kind_code=3 if label == "down" else 5,
            label=label,
            confidence="verified",
        )
        for index, (bar, label) in enumerate(
            [(1, "intro"), (89, "up"), (113, "chorus"), (121, "down")],
            start=1,
        )
    ]
    outgoing.landmarks = [
        TrackLandmark(
            name="file start",
            kind="phrase_start",
            bar=1,
            time_ms=0,
            confidence="verified",
        ),
        TrackLandmark(
            name="bass",
            kind="bass_in",
            bar=113,
            confidence="verified",
        ),
        TrackLandmark(
            name="outro",
            kind="mix_out",
            bar=145,
            confidence="verified",
        ),
    ]
    outgoing.bass_energy_by_bar = [
        BassEnergyBar(
            bar=bar,
            median=8 if 113 <= bar < 121 else 2,
            mean=14 if 113 <= bar < 121 else 7,
            peak=80,
        )
        for bar in range(1, 153)
    ]

    incoming = profile("faces", "Blow Ya Faces Off", 125, "4A")
    incoming.phrase_boundaries = [
        PhraseBoundary(
            index=index,
            start_beat=(bar - 1) * 4 + 1,
            end_beat=(bar - 1) * 4 + length * 4,
            start_bar=bar,
            beat_in_bar=1,
            length_beats=length * 4,
            length_bars=length,
            kind_code=5 if label == "chorus" else 2,
            label=label,
            confidence="verified",
        )
        for index, (bar, label, length) in enumerate(
            [
                (1, "intro", 16),
                (17, "up", 4),
                (21, "up", 4),
                (25, "up", 8),
                (33, "chorus", 16),
            ],
            start=1,
        )
    ]
    incoming.landmarks = [
        TrackLandmark(
            name="file start",
            kind="phrase_start",
            bar=1,
            time_ms=0,
            confidence="verified",
        ),
        TrackLandmark(
            name="strong early bass",
            kind="bass_in",
            bar=21,
            confidence="verified",
        ),
        TrackLandmark(
            name="outro",
            kind="mix_out",
            bar=65,
            confidence="verified",
        ),
    ]
    incoming.bass_energy_by_bar = [
        BassEnergyBar(
            bar=bar,
            median=20 if 21 <= bar < 33 else 7,
            mean=21 if 33 <= bar < 41 else 18,
            peak=105 if 33 <= bar < 41 else 75,
        )
        for bar in range(1, 81)
    ]

    handoff = TransitionKingCompiler().compile(
        outgoing,
        incoming,
        outgoing_deck=1,
    )

    assert handoff.technique == "long_blend"
    assert handoff.card.start_phrase_index == 2
    assert handoff.card.critical_bar_offset == 32
    assert "down bar 121" in handoff.reason
    assert "chorus bar 33" in handoff.reason
    assert any(
        event.action == "channel_fader"
        and event.parameters == {"deck": 2, "value": 0.18}
        and event.bar_offset == 24
        for event in handoff.card.events
    )


def test_user_verified_four_bar_drop_gets_blend_not_hard_cut() -> None:
    outgoing = profile("a", "A", 130, "9A")
    incoming = profile("system", "System", 130, "9A")
    incoming.phrase_boundaries[1] = incoming.phrase_boundaries[1].model_copy(
        update={"start_bar": 5, "start_beat": 17}
    )
    incoming.landmarks = [
        TrackLandmark(
            name="Rekordbox phrase 1: intro",
            kind="phrase_start",
            bar=1,
            beat=1,
            time_ms=0,
            confidence="high",
        ),
        TrackLandmark(
            name="User verified System UP1 drop",
            kind="drop",
            bar=5,
            beat=1,
            time_ms=7500,
            confidence="verified",
        ),
        TrackLandmark(
            name="clean exit",
            kind="mix_out",
            bar=65,
            beat=1,
            confidence="verified",
        ),
    ]

    handoff = TransitionKingCompiler().compile(
        outgoing,
        incoming,
        outgoing_deck=1,
    )

    assert handoff.card.critical_bar_offset == 4
    assert validate_transition_card(handoff.card, outgoing, incoming) == []
    assert not any(
        event.action == "channel_fader"
        and event.parameters.get("deck") == 1
        and event.bar_offset <= 4
        for event in handoff.card.events
    )
    assert [
        event.parameters["value"]
        for event in handoff.card.events
        if event.action == "channel_fader" and event.parameters.get("deck") == 2
    ] == [0.18, 0.35, 0.72, 0.86, 1]


def test_incompatible_keys_fall_through_to_nonoverlap_phrase_cut() -> None:
    outgoing = profile("a", "A", 128, "9A")
    incoming = profile("b", "B", 128, "6A")

    handoff = TransitionKingCompiler().compile(
        outgoing,
        incoming,
        outgoing_deck=1,
    )

    assert handoff.technique == "phrase_cut"
    assert handoff.card.harmonic_risk_accepted is True
    assert handoff.card.transition_family == "phrase_cut"
    assert validate_transition_card(handoff.card, outgoing, incoming) == []


def test_local_planner_builds_repeatable_harmonic_tempo_route() -> None:
    profiles = [
        profile("a", "A", 128, "9A"),
        profile("b", "B", 129, "10A"),
        profile("c", "C", 130, "11A"),
    ]
    brief = DJBrief(
        start_track_id="a",
        target_track_count=3,
        target_bpm=130,
    )

    first = LocalDJPlanner(profiles).build_plan(brief)
    second = LocalDJPlanner(reversed(profiles)).build_plan(brief)

    assert first.model_dump() == second.model_dump()
    assert first.opening.track_id == "a"
    assert [item.incoming.track_id for item in first.transitions] == ["b", "c"]
    assert first.transitions[-1].tempo_after.target_bpm == 130
    assert first.rescue_loop_trigger_bars == 16


def test_local_planner_compiles_an_eight_track_rising_set() -> None:
    profiles = [
        profile(str(index), f"Track {index}", 120 + index, "9A") for index in range(8)
    ]

    plan = LocalDJPlanner(profiles).build_plan(
        DJBrief(
            start_track_id="0",
            target_track_count=8,
            target_bpm=130,
        )
    )

    assert plan.target_track_count == 8
    assert len(plan.transitions) == 7
    targets = [item.tempo_after.target_bpm for item in plan.transitions]
    assert targets == sorted(targets)
    assert targets[-1] == 130
    profiles_by_id = {item.track_id: item for item in profiles}
    for item in plan.transitions:
        assert (
            validate_transition_card(
                item.card,
                profiles_by_id[item.card.outgoing_track_id],
                profiles_by_id[item.card.incoming_track_id],
            )
            == []
        )


def test_local_planner_arrives_at_requested_track_on_requested_deck_path() -> None:
    profiles = [
        profile("a", "Opening", 126, "9A"),
        profile("b", "Bridge", 127, "10A"),
        profile("c", "Destination", 128, "11A"),
    ]

    plan = LocalDJPlanner(profiles).build_plan(
        DJBrief(
            start_track_id="a",
            target_track_count=3,
            target_track_id="c",
            vibe="energetic",
        ),
        opening_deck=2,
    )

    assert [item.incoming.track_id for item in plan.transitions] == ["b", "c"]
    assert plan.transitions[0].card.outgoing_deck == 2
    assert plan.transitions[1].card.outgoing_deck == 1
    assert plan.tempo_target_bpm == 128


def test_vibe_direction_changes_the_next_selection() -> None:
    profiles = [
        profile("a", "Opening", 128, "9A"),
        profile("lower", "Lower", 127, "9A"),
        profile("higher", "Higher", 129, "9A"),
    ]
    planner = LocalDJPlanner(profiles)

    down = planner.build_plan(
        DJBrief(start_track_id="a", target_track_count=2, vibe="downtempo")
    )
    up = planner.build_plan(
        DJBrief(start_track_id="a", target_track_count=2, vibe="energetic")
    )

    assert down.transitions[0].incoming.track_id == "lower"
    assert up.transitions[0].incoming.track_id == "higher"


def test_status_store_is_atomic_and_formats_corner_view(tmp_path) -> None:
    store = StandaloneStatusStore(tmp_path)
    store.publish(
        phase=RuntimePhase.WAITING,
        headline="Preparing to bring in Track B",
        detail="Launch in 16 bars.",
        current=TrackRole(title="Track A", artist="Artist A", deck=1, bpm=128),
        staged=TrackRole(title="Track B", artist="Artist B", deck=2, bpm=129),
        bpm=128,
        bar=49,
        beat=1,
        action_in_bars=16,
    )

    snapshot = store.read()
    view = format_snapshot(snapshot)

    assert view.headline == "Preparing to bring in Track B"
    assert view.now_playing == "Track A — Artist A"
    assert "action in 16.0 bars" in view.clock
    assert len(store.recent_events()) == 1


class FakeAdapter:
    def __init__(self) -> None:
        self.armed_seconds = None
        self.recovery_planner = None
        self.statuses = [
            {
                "active": True,
                "status": "running",
                "current_track_id": "a",
                "current_deck": 1,
                "active_option_id": "1-a-b",
                "staged_option_id": "1-a-b",
                "active_job_id": "job",
                "transition_start_at": None,
                "transition_critical_at": None,
                "failures": [],
                "deadline_phase": "normal",
                "rescue_loop_active": False,
            },
            {
                "active": False,
                "status": "completed",
                "current_track_id": "b",
                "current_deck": 2,
                "active_option_id": None,
                "staged_option_id": "1-a-b",
                "active_job_id": None,
                "transition_start_at": None,
                "transition_critical_at": None,
                "failures": [],
                "deadline_phase": "normal",
                "rescue_loop_active": False,
            },
        ]

    def connect(self):
        return {"connected": True}

    def arm(self, seconds):
        self.armed_seconds = seconds
        return {"armed": True, "seconds": seconds}

    async def preflight(self, plan):
        return {"ready": True}

    async def start(self, plan):
        return {"ready": True}

    async def continue_set(self, plan):
        self.continued_plan = plan
        return {"ready": True}

    def runner_status(self):
        return self.statuses.pop(0)

    def control_status(self):
        return {
            "connected": True,
            "live_state": {
                "decks": [
                    {
                        "deck": 1,
                        "bpm": 128,
                        "bar": 49,
                        "beat": 1,
                        "playing": True,
                        "observation_age_ms": 25,
                    }
                ]
            },
        }

    def rekordbox_status(self):
        return {
            "decks": [
                {"deck": 1, "title": "A"},
                {"deck": 2, "title": "B"},
            ]
        }

    def queue_steering(self, plan):
        return {"queued": True, "projected_plan": plan.model_dump()}

    def register_recovery_planner(self, callback):
        self.recovery_planner = callback

    async def hold_loop(self, deck, beats):
        return {"verified": True}

    def stop_automation(self):
        return {"status": "stopped"}

    def emergency_stop(self):
        return {"status": "stopped"}


def test_standalone_engine_runs_without_codex_round_trips(tmp_path) -> None:
    async def scenario() -> None:
        profiles = ProfileStore(tmp_path / "profiles")
        profiles.upsert(profile("a", "A", 128, "9A"))
        profiles.upsert(profile("b", "B", 129, "10A"))
        statuses = StandaloneStatusStore(tmp_path / "status")
        engine = StandaloneDJEngine(
            profile_store=profiles,
            status_store=statuses,
            adapter=FakeAdapter(),
            monitor_seconds=0.001,
        )

        result = await engine.start_set(
            DJBrief(start_track_id="a", target_track_count=2)
        )
        await engine.monitor_task

        assert result["ready"] is True
        assert engine.adapter.armed_seconds == 30 * 60
        assert callable(engine.adapter.recovery_planner)
        assert statuses.read().phase == RuntimePhase.COMPLETE
        assert any(
            event.phase == RuntimePhase.PREFLIGHT for event in statuses.recent_events()
        )

    asyncio.run(scenario())


def test_breakdown_handoff_establishes_incoming_before_smooth_retirement() -> None:
    from rekordbox_performer.intelligence import MusicalEvent

    events = _breakdown_events(
        outgoing_deck=1,
        incoming_deck=2,
        launch=MusicalEvent(
            bar_offset=0,
            action="play_pause",
            parameters={"deck": 2},
        ),
    )
    outgoing_faders = [
        event
        for event in events
        if event.action == "channel_fader" and event.parameters.get("deck") == 1
    ]
    incoming_faders = [
        event
        for event in events
        if event.action == "channel_fader" and event.parameters.get("deck") == 2
    ]

    assert min(event.bar_offset for event in outgoing_faders) == 8
    assert incoming_faders[0].parameters["value"] == 0.08
    assert incoming_faders[-1].parameters["value"] == 1.0
    assert len(incoming_faders) == 33
    assert len(outgoing_faders) == 17
    assert any(
        event.action == "eq_low"
        and event.bar_offset == 8
        and event.parameters == {"deck": 1, "value": -1}
        for event in events
    )
    assert any(
        event.action == "eq_low"
        and event.bar_offset == 8
        and event.parameters == {"deck": 2, "value": 0}
        for event in events
    )


def test_playlist_planner_uses_every_track_once() -> None:
    profiles = [
        profile("a", "A", 124, "9A"),
        profile("b", "B", 125, "10A"),
        profile("c", "C", 126, "11A"),
        profile("d", "D", 127, "12A"),
    ]

    plan = LocalDJPlanner(profiles).build_playlist_plan(
        ["c", "a", "d", "b"],
        opening_track_id="a",
    )
    route = [plan.opening.track_id] + [
        option.incoming.track_id for option in plan.transitions
    ]

    assert route[0] == "a"
    assert len(route) == 4
    assert set(route) == {"a", "b", "c", "d"}


def test_completed_set_can_attach_to_the_still_playing_final_track(tmp_path) -> None:
    class ContinuationAdapter(FakeAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.statuses = []
            self.continued_plan = None

        def runner_status(self):
            return {
                "active": False,
                "status": "completed",
                "current_track_id": "a",
                "current_deck": 1,
                "played_track_ids": ["a"],
                "active_option_id": None,
            }

    async def scenario() -> None:
        profiles = ProfileStore(tmp_path / "profiles")
        for item in (
            profile("a", "A", 124, "9A"),
            profile("b", "B", 125, "10A"),
            profile("c", "C", 126, "11A"),
        ):
            profiles.upsert(item)
        adapter = ContinuationAdapter()
        engine = StandaloneDJEngine(
            profile_store=profiles,
            status_store=StandaloneStatusStore(tmp_path / "status"),
            adapter=adapter,
            monitor_seconds=60,
        )

        result = await engine.continue_set(
            target_query="C",
            transition_count=2,
        )

        assert result["continued"] is True
        assert adapter.continued_plan.opening.track_id == "a"
        assert adapter.continued_plan.transitions[-1].incoming.track_id == "c"
        engine.monitor_task.cancel()

    asyncio.run(scenario())


def test_endless_route_reuses_an_exhausted_small_pool_safely(tmp_path) -> None:
    profiles = ProfileStore(tmp_path / "profiles")
    track_ids = [str(index) for index in range(6)]
    for index, track_id in enumerate(track_ids):
        profiles.upsert(profile(track_id, f"Track {index}", 124 + index, "9A"))
    engine = StandaloneDJEngine(
        profile_store=profiles,
        status_store=StandaloneStatusStore(tmp_path / "status"),
        adapter=FakeAdapter(),
    )
    engine._candidate_track_ids = set(track_ids)

    future, transitions = engine._build_future_route(
        anchor_id="5",
        anchor_deck=2,
        target=None,
        vibe="maintain",
        transition_count=5,
        excluded=track_ids,
    )

    assert transitions == 5
    assert future.target_track_count == 6
    assert len({future.opening.track_id, *[
        option.incoming.track_id for option in future.transitions
    ]}) == 6


def test_embedded_replacement_planner_excludes_exhausted_track(tmp_path) -> None:
    async def scenario() -> None:
        profiles = ProfileStore(tmp_path / "profiles")
        for item in (
            profile("a", "Current", 126, "9A"),
            profile("b", "Failed", 126, "9A"),
            profile("c", "Replacement One", 127, "10A"),
            profile("d", "Replacement Two", 128, "11A"),
        ):
            profiles.upsert(item)
        statuses = StandaloneStatusStore(tmp_path / "status")
        engine = StandaloneDJEngine(
            profile_store=profiles,
            status_store=statuses,
            adapter=FakeAdapter(),
        )

        future = await engine._build_replacement_route(
            {
                "current_track_id": "a",
                "current_deck": 1,
                "played_track_ids": ["a"],
                "exhausted_incoming_track_ids": ["b"],
                "target_track_count": 3,
            }
        )

        assert future is not None
        assert future.opening.track_id == "a"
        assert "b" not in [item.incoming.track_id for item in future.transitions]
        assert statuses.read().headline == "Selecting a replacement transition"

    asyncio.run(scenario())


def test_engine_resolves_accent_insensitive_search_and_loaded_deck(tmp_path) -> None:
    profiles = ProfileStore(tmp_path / "profiles")
    ready = profile("a", "Lately (Remix)", 126, "9A")
    ready.artist = "RÜFÜS DU SOL"
    profiles.upsert(ready)
    engine = StandaloneDJEngine(
        profile_store=profiles,
        status_store=StandaloneStatusStore(tmp_path / "status"),
        adapter=FakeAdapter(),
    )
    engine.adapter.rekordbox_status = lambda: {
        "decks": [{"deck": 1, "title": "Lately (Remix)"}]
    }

    assert engine.resolve_track("rufus").track_id == "a"
    assert engine.loaded_track(1).track_id == "a"


def test_engine_detects_manually_started_rekordbox_track(tmp_path) -> None:
    profiles = ProfileStore(tmp_path / "profiles")
    profiles.upsert(profile("a", "Manual Track", 126, "9A"))
    statuses = StandaloneStatusStore(tmp_path / "status")
    engine = StandaloneDJEngine(
        profile_store=profiles,
        status_store=statuses,
        adapter=FakeAdapter(),
    )
    first = {
        "mode": "PERFORMANCE",
        "decks": [
            {
                "deck": 1,
                "title": "Manual Track",
                "artist": "Artist Manual Track",
                "bpm": 126.0,
                "elapsed_seconds": 12,
            }
        ],
    }
    second = {
        **first,
        "decks": [{**first["decks"][0], "elapsed_seconds": 13}],
    }

    engine._publish_manual_transport(first)
    engine._publish_manual_transport(second)

    snapshot = statuses.read()
    assert snapshot.phase == RuntimePhase.IDLE
    assert snapshot.headline == "Manual playback detected on Deck 1"
    assert snapshot.current.track_id == "a"
    assert snapshot.current.state == "manual"
    assert snapshot.bpm == 126.0
    assert snapshot.health.midi == "down"


def test_manual_monitor_does_not_overwrite_launch_failure(tmp_path) -> None:
    statuses = StandaloneStatusStore(tmp_path / "status")
    engine = StandaloneDJEngine(
        profile_store=ProfileStore(tmp_path / "profiles"),
        status_store=statuses,
        adapter=FakeAdapter(),
    )
    statuses.publish(
        phase=RuntimePhase.FAILED,
        headline="Rekordbox launch failed",
        detail="exact staging failed",
        severity="error",
    )

    assert engine._manual_monitor_can_publish() is False
    assert statuses.read().headline == "Rekordbox launch failed"
