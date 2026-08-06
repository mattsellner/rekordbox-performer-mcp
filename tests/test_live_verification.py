from contextlib import contextmanager
from types import SimpleNamespace
import asyncio
import time

import pytest

from rekordbox_performer import server
from rekordbox_performer.intelligence import (
    DeckObservation,
    MusicalEvent,
    TransitionCard,
)
from rekordbox_performer.set_runner import (
    TempoPlan,
    TrackLoadSpec,
    TransitionOption,
)


def card() -> TransitionCard:
    return TransitionCard(
        name="Verified transition",
        outgoing_track_id="a",
        incoming_track_id="b",
        anchor_deck=1,
        outgoing_deck=1,
        incoming_deck=2,
        phrase_alignment_verified=True,
        vocal_risk_accepted=True,
        bass_plan_verified=True,
        incoming_loaded_verified=True,
        intended_vocal_owner="incoming",
        critical_bar_offset=0,
        abort_plan="Keep deck one playing.",
        events=[
            MusicalEvent(
                bar_offset=0,
                action="hot_cue",
                parameters={"deck": 2, "cue": 1},
            ),
            MusicalEvent(
                bar_offset=0,
                action="eq_low",
                parameters={"deck": 1, "value": -1},
            ),
            MusicalEvent(
                bar_offset=0,
                action="eq_low",
                parameters={"deck": 2, "value": 0},
            ),
            MusicalEvent(
                bar_offset=1,
                action="channel_fader",
                parameters={"deck": 1, "value": 0},
            ),
            MusicalEvent(
                bar_offset=1,
                beat_offset=1,
                action="cue",
                parameters={"deck": 1},
            ),
        ],
    )


def opening_card() -> TransitionCard:
    transition = card()
    transition.start_phrase_index = 2
    transition.events[0] = MusicalEvent(
        bar_offset=0,
        action="play_pause",
        parameters={"deck": 2},
    )
    return transition


def hot_cue_opening_card() -> TransitionCard:
    transition = card()
    transition.start_phrase_index = 2
    transition.events[0] = MusicalEvent(
        bar_offset=0,
        action="hot_cue",
        parameters={"deck": 2, "cue": 7},
    )
    return transition


def test_optional_fx_failure_cannot_block_runner_transition(monkeypatch) -> None:
    transition = card()
    option = TransitionOption(
        id="a-b",
        card=transition,
        incoming=TrackLoadSpec(
            track_id="b",
            title="Incoming",
            cue=1,
            cue_time_ms=0,
        ),
        technique="vocal_safe_loop_blend",
        fx_effect="spiral",
    )
    staged = []

    async def fail_fx(_card, _effect):
        raise RuntimeError("Deck 1 first-slot FX selector is not visible")

    async def schedule_dry(**kwargs):
        staged.append(kwargs)
        return {"ready": True, "schedule": {"job": {"id": "dry-job"}}}

    monkeypatch.setattr(server, "_prepare_option_fx", fail_fx)
    monkeypatch.setattr(server, "stage_and_schedule_transition_card", schedule_dry)

    result = asyncio.run(server._runner_schedule(option, False))

    assert result["ready"] is True
    assert result["job"]["id"] == "dry-job"
    assert staged[0]["card"] == transition
    assert result["fx_preparation"] == {
        "verified": False,
        "desired": "spiral",
        "error": "Deck 1 first-slot FX selector is not visible",
        "fallback": "dry transition card",
    }


class OpeningProfileStore:
    def get(self, track_id: str):
        titles = {"a": "Outgoing", "b": "Incoming"}
        return SimpleNamespace(
            track_id=track_id,
            title=titles[track_id],
            bpm=128.0,
            time_signature=4,
            beat_grid=[],
        )


class OpeningUI:
    def __init__(self, incoming_bpm: float = 128.0) -> None:
        self.incoming_bpm = incoming_bpm
        self.sync = {1: True, 2: True}
        self.quantize = {1: True, 2: True}

    def status(self) -> dict:
        return {
            "decks": [
                {
                    "deck": 1,
                    "title": "Outgoing",
                    "elapsed_seconds": 0,
                    "bpm": 128.0,
                    "beat_sync_enabled": self.sync[1],
                    "quantize_enabled": self.quantize[1],
                },
                {
                    "deck": 2,
                    "title": "Incoming",
                    "elapsed_seconds": 0,
                    "bpm": self.incoming_bpm,
                    "beat_sync_enabled": self.sync[2],
                    "quantize_enabled": self.quantize[2],
                },
            ]
        }

    def ensure_pc_master_out(self, enabled: bool) -> dict:
        assert enabled is False
        return {"before": True, "after": False, "changed": True}


class OpeningEngine:
    def __init__(self) -> None:
        self.actions: list[str] = []

    async def send_action(self, action: str, parameters: dict) -> list[str]:
        self.actions.append(action)
        return [action]


class OpeningSession:
    def start(self) -> dict:
        return {"status": "running"}


def _install_opening_fakes(monkeypatch, *, incoming_bpm: float = 128.0):
    engine = OpeningEngine()
    ui = OpeningUI(incoming_bpm)
    monkeypatch.setattr(server, "profile_store", OpeningProfileStore())
    monkeypatch.setattr(server, "rekordbox_ui", ui)
    monkeypatch.setattr(server, "deck_observer", CompletionObserver())
    monkeypatch.setattr(server, "engine", engine)
    monkeypatch.setattr(server, "live_state", server.LiveState())
    monkeypatch.setattr(server, "set_sessions", OpeningSession())

    async def modes(**kwargs):
        deck = kwargs["deck"]
        changed = False
        if kwargs.get("beat_sync") is not None:
            changed = changed or ui.sync[deck] != kwargs["beat_sync"]
            ui.sync[deck] = kwargs["beat_sync"]
        if kwargs.get("quantize") is not None:
            changed = changed or ui.quantize[deck] != kwargs["quantize"]
            ui.quantize[deck] = kwargs["quantize"]
        return {"deck": deck, "changed": changed, "actions": []}

    async def master(deck):
        return {
            "deck": deck,
            "changed": False,
            "before": {"target": True, "other": False},
            "after": {"target": True, "other": False},
            "messages": [],
        }

    async def stage(**kwargs):
        return {
            "deck": kwargs["deck"],
            "track_id": kwargs["track_id"],
            "verified": True,
            "already_loaded": True,
            "browser_search_skipped": True,
        }

    async def master(deck):
        messages = await engine.send_action("master", {"deck": deck})
        return {
            "deck": deck,
            "before": {"target": False, "other": True},
            "after": {"target": True, "other": False},
            "changed": True,
            "messages": messages,
        }

    monkeypatch.setattr(server, "ensure_deck_modes", modes)
    monkeypatch.setattr(server, "ensure_master_deck", master)
    monkeypatch.setattr(server, "_stage_track", stage)
    return engine


def test_atomic_opening_stops_both_decks_when_job_is_not_scheduled(
    monkeypatch,
) -> None:
    engine = _install_opening_fakes(monkeypatch)

    async def launch(**kwargs):
        return {"live_state": {"playing": True}}

    async def refresh(**kwargs):
        return {"outgoing": {"playing": True}, "incoming": {"playing": False}}

    async def schedule(**kwargs):
        return {"ready": False, "errors": ["scheduler unavailable"]}

    monkeypatch.setattr(server, "launch_staged_track", launch)
    monkeypatch.setattr(server, "refresh_transition_state", refresh)
    monkeypatch.setattr(server, "perform_transition_card", schedule)

    result = asyncio.run(
        server.launch_and_schedule_opening_transition(
            card=opening_card(),
            outgoing_title="Outgoing",
            incoming_title="Incoming",
            execution_id="opening-1",
        )
    )

    assert result["ready"] is False
    assert result["started"] is False
    assert result["aborted_after_launch"] is True
    assert "loop_8" not in engine.actions
    assert engine.actions[-4:] == [
        "channel_fader",
        "cue",
        "channel_fader",
        "cue",
    ]


def test_atomic_opening_cancels_unreserved_job_before_stopping_decks(
    monkeypatch,
) -> None:
    _install_opening_fakes(monkeypatch)
    cancelled = []

    async def launch(**kwargs):
        return {"live_state": {"playing": True}}

    async def refresh(**kwargs):
        return {"outgoing": {"playing": True}, "incoming": {"playing": False}}

    async def schedule(**kwargs):
        return {
            "ready": True,
            "job": {
                "id": "unreserved-job",
                "execution_id": "opening-unreserved",
                "status": "running",
                "event_count": 5,
                "control_reserved_until_monotonic": None,
            },
        }

    class CancelScheduler:
        def cancel(self, job_id):
            cancelled.append(job_id)
            return {"id": job_id, "status": "cancelled"}

    monkeypatch.setattr(server, "launch_staged_track", launch)
    monkeypatch.setattr(server, "refresh_transition_state", refresh)
    monkeypatch.setattr(server, "perform_transition_card", schedule)
    monkeypatch.setattr(server, "scheduler", CancelScheduler())

    result = asyncio.run(
        server.launch_and_schedule_opening_transition(
            card=opening_card(),
            outgoing_title="Outgoing",
            incoming_title="Incoming",
            execution_id="opening-unreserved",
        )
    )

    assert result["ready"] is False
    assert result["started"] is False
    assert cancelled == ["unreserved-job"]
    assert result["cancelled_job"]["status"] == "cancelled"


def test_atomic_opening_requires_an_explicit_analyzed_phrase() -> None:
    transition = opening_card()
    transition.start_phrase_index = None

    with pytest.raises(ValueError, match="explicit analyzed start_phrase_index"):
        asyncio.run(
            server.launch_and_schedule_opening_transition(
                card=transition,
                outgoing_title="Outgoing",
                incoming_title="Incoming",
                execution_id="opening-without-phrase",
            )
        )


def test_atomic_opening_sets_master_after_launch_and_arms_scheduled_job(
    monkeypatch,
) -> None:
    engine = _install_opening_fakes(monkeypatch, incoming_bpm=129.0)

    async def launch(**kwargs):
        assert "master" not in engine.actions
        return {"live_state": {"playing": True}}

    async def refresh(**kwargs):
        assert "master" in engine.actions
        return {"outgoing": {"playing": True}, "incoming": {"playing": False}}

    async def schedule(**kwargs):
        return {
            "ready": True,
            "job": {
                "id": "job-1",
                "execution_id": "opening-2",
                "status": "running",
                "event_count": 5,
                "control_reserved_until_monotonic": 123.0,
            },
        }

    monkeypatch.setattr(server, "launch_staged_track", launch)
    monkeypatch.setattr(server, "refresh_transition_state", refresh)
    monkeypatch.setattr(server, "perform_transition_card", schedule)

    result = asyncio.run(
        server.launch_and_schedule_opening_transition(
            card=opening_card(),
            outgoing_title="Outgoing",
            incoming_title="Incoming",
            execution_id="opening-2",
        )
    )

    assert result["ready"] is True
    assert result["started"] is True
    assert result["audio_route"]["after"] is False
    assert result["schedule"]["job"]["id"] == "job-1"
    assert engine.actions == [
        "channel_fader",
        "gain",
        "eq_high",
        "eq_mid",
        "eq_low",
        "filter",
        "fx_wet_dry",
        "channel_fader",
        "gain",
        "eq_high",
        "eq_mid",
        "eq_low",
        "filter",
        "fx_wet_dry",
        "tempo",
        "tempo",
        "tempo",
        "tempo",
        "tempo",
        "tempo",
        "master",
    ]


def test_atomic_opening_disables_sync_before_launch_then_uses_native_tempo(
    monkeypatch,
) -> None:
    _install_opening_fakes(monkeypatch, incoming_bpm=130.0)
    ui = server.rekordbox_ui
    order = []

    original_modes = server.ensure_deck_modes

    async def modes(**kwargs):
        order.append(f"sync:{kwargs['deck']}:{kwargs.get('beat_sync')}")
        return await original_modes(**kwargs)

    async def launch(**kwargs):
        order.append("launch")
        assert ui.sync == {1: False, 2: False}
        return {"live_state": {"playing": True}}

    refresh_count = 0

    async def refresh(**kwargs):
        nonlocal refresh_count
        refresh_count += 1
        order.append(f"refresh:{refresh_count}")
        assert "launch" in order
        return {
            "outgoing": {"playing": True, "bpm": 128.0},
            "incoming": {"playing": False, "bpm": 130.0},
        }

    async def schedule(**kwargs):
        return {
            "ready": True,
            "job": {
                "id": "native-opening",
                "execution_id": "native-opening",
                "status": "running",
                "event_count": 5,
                "control_reserved_until_monotonic": 999.0,
            },
        }

    monkeypatch.setattr(server, "ensure_deck_modes", modes)
    monkeypatch.setattr(server, "launch_staged_track", launch)
    monkeypatch.setattr(server, "refresh_transition_state", refresh)
    monkeypatch.setattr(server, "perform_transition_card", schedule)

    result = asyncio.run(
        server.launch_and_schedule_opening_transition(
            card=opening_card(),
            outgoing_title="Outgoing",
            incoming_title="Incoming",
            execution_id="native-opening",
        )
    )

    assert result["ready"] is True
    assert result["opening_native_bpm"] == 128.0
    assert order[:3] == ["sync:1:False", "sync:2:False", "launch"]
    assert order.index("refresh:1") < order.index("sync:1:True")
    assert ui.sync == {1: True, 2: True}


def test_atomic_opening_aborts_if_rekordbox_changes_opening_tempo(
    monkeypatch,
) -> None:
    _install_opening_fakes(monkeypatch, incoming_bpm=130.0)

    async def launch(**kwargs):
        return {"live_state": {"playing": True}}

    async def refresh(**kwargs):
        return {
            "outgoing": {"playing": True, "bpm": 130.0},
            "incoming": {"playing": False, "bpm": 130.0},
        }

    async def schedule(**kwargs):
        raise AssertionError("a non-native opening must never be scheduled")

    monkeypatch.setattr(server, "launch_staged_track", launch)
    monkeypatch.setattr(server, "refresh_transition_state", refresh)
    monkeypatch.setattr(server, "perform_transition_card", schedule)

    result = asyncio.run(
        server.launch_and_schedule_opening_transition(
            card=opening_card(),
            outgoing_title="Outgoing",
            incoming_title="Incoming",
            execution_id="wrong-opening-tempo",
        )
    )

    assert result["ready"] is False
    assert result["aborted_after_launch"] is True
    assert "did not start at its native BPM" in result["errors"][0]


def test_tempo_ramp_brackets_soft_takeover_pickup(monkeypatch) -> None:
    class TempoEngine:
        def __init__(self) -> None:
            self.values = []

        async def send_action(self, action, parameters):
            assert action == "tempo"
            self.values.append(parameters["value"])
            return [parameters["value"]]

    async def no_sleep(_seconds):
        return None

    engine = TempoEngine()
    monkeypatch.setattr(server, "engine", engine)
    monkeypatch.setattr(server.asyncio, "sleep", no_sleep)
    plan = TempoPlan(target_bpm=122.5, duration_bars=32)

    result = asyncio.run(
        server._acquire_tempo_soft_takeover(
            plan=plan,
            deck=2,
            native_bpm=123.0,
            live_bpm=121.0,
        )
    )

    current = plan.control_value(native_bpm=123.0, bpm=121.0)
    assert engine.values == [current - 0.02, current + 0.02, current]
    assert result["pickup_values"] == engine.values


def test_atomic_opening_prepares_and_verifies_hot_cue_before_playback(
    monkeypatch,
) -> None:
    _install_opening_fakes(monkeypatch)
    order = []

    async def stage(**kwargs):
        order.append("stage")
        return {"verified": True, "already_loaded": True}

    async def verify(**kwargs):
        order.append("verify")
        assert kwargs["cue"] == 7
        assert kwargs["expected_time_ms"] == 112_087
        return {"verified": True}

    async def launch(**kwargs):
        order.append("launch")
        assert order == ["stage", "verify", "launch"]
        return {"live_state": {"playing": True}}

    async def refresh(**kwargs):
        assert kwargs["incoming_hot_cue"] == 7
        return {"outgoing": {"playing": True}, "incoming": {"playing": False}}

    async def schedule(**kwargs):
        return {
            "ready": True,
            "job": {
                "id": "job-hot-cue",
                "execution_id": "opening-hot-cue",
                "status": "scheduled",
                "event_count": 5,
                "control_reserved_until_monotonic": 456.0,
            },
        }

    monkeypatch.setattr(server, "_stage_track", stage)
    monkeypatch.setattr(server, "verify_hot_cue", verify)
    monkeypatch.setattr(server, "launch_staged_track", launch)
    monkeypatch.setattr(server, "refresh_transition_state", refresh)
    monkeypatch.setattr(server, "perform_transition_card", schedule)

    result = asyncio.run(
        server.launch_and_schedule_opening_transition(
            card=hot_cue_opening_card(),
            outgoing_title="Outgoing",
            incoming_title="Incoming",
            incoming_artist="Artist",
            incoming_cue=7,
            incoming_cue_time_ms=112_087,
            execution_id="opening-hot-cue",
        )
    )

    assert result["ready"] is True
    assert result["started"] is True
    assert result["schedule"]["job"]["id"] == "job-hot-cue"


def test_refresh_preserves_native_subsecond_phase_for_running_deck(
    monkeypatch,
) -> None:
    class Profiles:
        def get(self, track_id: str):
            return SimpleNamespace(
                track_id=track_id,
                title={"a": "Outgoing", "b": "Incoming"}[track_id],
                bpm=128.0,
                time_signature=4,
                beat_grid=[],
            )

    class CoarseClockUI:
        def __init__(self) -> None:
            self.calls = 0

        def status(self) -> dict:
            self.calls += 1
            outgoing = 12 if self.calls == 1 else 13
            return {
                "decks": [
                    {
                        "deck": 1,
                        "title": "Outgoing",
                        "elapsed_seconds": outgoing,
                        "bpm": 128.0,
                        "beat_sync_enabled": True,
                        "quantize_enabled": True,
                    },
                    {
                        "deck": 2,
                        "title": "Incoming",
                        "elapsed_seconds": 0,
                        "bpm": 128.0,
                        "beat_sync_enabled": True,
                        "quantize_enabled": True,
                    },
                ]
            }

    async def no_sleep(_seconds):
        return None

    state = server.LiveState()
    state.update(
        DeckObservation(
            deck=1,
            track_id="a",
            title="Outgoing",
            bpm=128.0,
            playing=True,
            bar=7,
            beat=1,
            track_beat=25,
            beat_phase=0.6,
            sync_enabled=True,
            quantize_enabled=True,
            source="native",
            confidence="high",
        )
    )
    monkeypatch.setattr(server, "profile_store", Profiles())
    monkeypatch.setattr(server, "rekordbox_ui", CoarseClockUI())
    monkeypatch.setattr(server, "deck_observer", CompletionObserver())
    monkeypatch.setattr(server, "live_state", state)
    monkeypatch.setattr(server.asyncio, "sleep", no_sleep)

    result = asyncio.run(
        server.refresh_transition_state(
            outgoing_deck=1,
            outgoing_track_id="a",
            outgoing_title="Outgoing",
            incoming_deck=2,
            incoming_track_id="b",
            incoming_title="Incoming",
        )
    )

    # The UI reports only whole seconds (13s would snap near beat 28).  Keep
    # the native launch clock at beat 25 + its fractional phase instead.
    assert result["outgoing"]["track_beat"] == 25
    assert 0.59 <= result["outgoing"]["beat_phase"] <= 0.65
    assert result["outgoing"]["source"] == "native"


def test_rolling_scheduler_skips_blind_master_toggle_and_requires_job(
    monkeypatch,
) -> None:
    monkeypatch.setattr(server, "profile_store", OpeningProfileStore())
    master_calls = []

    async def stage(**kwargs):
        return {"verified": True, "already_loaded": True}

    async def verify(**kwargs):
        return {"verified": True}

    async def refresh(**kwargs):
        return {"outgoing": {"playing": True}, "incoming": {"playing": False}}

    async def master(deck):
        master_calls.append(deck)
        return {
            "deck": deck,
            "changed": False,
            "before": {"target": True, "other": False},
            "after": {"target": True, "other": False},
            "messages": [],
        }

    async def modes(**kwargs):
        return {"deck": kwargs["deck"], "changed": False, "actions": []}

    async def schedule(**kwargs):
        return {
            "ready": True,
            "job": {
                "id": "rolling-job",
                "execution_id": "rolling-1",
                "status": "running",
                "event_count": 5,
                "control_reserved_until_monotonic": 789.0,
            },
        }

    async def blind_toggle(*args, **kwargs):
        raise AssertionError("Master must never be blindly toggled")

    monkeypatch.setattr(server, "_stage_track", stage)
    monkeypatch.setattr(server, "verify_hot_cue", verify)
    monkeypatch.setattr(server, "refresh_transition_state", refresh)
    monkeypatch.setattr(server, "ensure_master_deck", master)
    monkeypatch.setattr(server, "ensure_deck_modes", modes)
    monkeypatch.setattr(server, "perform_transition_card", schedule)
    monkeypatch.setattr(server, "trigger_control", blind_toggle)

    result = asyncio.run(
        server.stage_and_schedule_transition_card(
            card=card(),
            incoming_title="Incoming",
            incoming_artist="Artist",
            incoming_cue=1,
            incoming_cue_time_ms=10_000,
            execution_id="rolling-1",
        )
    )

    assert result["ready"] is True
    assert result["schedule"]["job"]["id"] == "rolling-job"
    assert master_calls == [1]


def test_rolling_scheduler_observes_clock_after_master_verification(
    monkeypatch,
) -> None:
    monkeypatch.setattr(server, "profile_store", OpeningProfileStore())
    order = []

    async def stage(**kwargs):
        return {"verified": True, "already_loaded": True}

    async def verify(**kwargs):
        return {"verified": True}

    async def master(deck):
        order.append("master")
        return {
            "deck": deck,
            "changed": False,
            "before": {"target": True, "other": False},
            "after": {"target": True, "other": False},
            "messages": [],
        }

    async def refresh(**kwargs):
        order.append("refresh")
        assert order[0] == "master"
        return {"outgoing": {"playing": True}, "incoming": {"playing": False}}

    async def modes(**kwargs):
        return {"deck": kwargs["deck"], "changed": False, "actions": []}

    async def schedule(**kwargs):
        return {
            "ready": True,
            "job": {
                "id": "ordered-job",
                "execution_id": "ordered-rolling",
                "status": "running",
                "event_count": 5,
                "control_reserved_until_monotonic": 789.0,
            },
        }

    monkeypatch.setattr(server, "_stage_track", stage)
    monkeypatch.setattr(server, "verify_hot_cue", verify)
    monkeypatch.setattr(server, "ensure_master_deck", master)
    monkeypatch.setattr(server, "refresh_transition_state", refresh)
    monkeypatch.setattr(server, "ensure_deck_modes", modes)
    monkeypatch.setattr(server, "perform_transition_card", schedule)

    result = asyncio.run(
        server.stage_and_schedule_transition_card(
            card=card(),
            incoming_title="Incoming",
            incoming_artist="Artist",
            incoming_cue=1,
            incoming_cue_time_ms=10_000,
            execution_id="ordered-rolling",
        )
    )

    assert result["ready"] is True
    assert order == ["master", "refresh", "refresh"]


def test_rolling_scheduler_accepts_verified_file_start_without_hot_cue(
    monkeypatch,
) -> None:
    monkeypatch.setattr(server, "profile_store", OpeningProfileStore())
    transition = opening_card()
    transition.transition_family = "phrase_cut"
    refresh_calls = []

    async def stage(**kwargs):
        return {"verified": True, "already_loaded": True}

    async def refresh(**kwargs):
        refresh_calls.append(kwargs)
        return {"outgoing": {"playing": True}, "incoming": {"playing": False}}

    async def modes(**kwargs):
        return {"deck": kwargs["deck"], "changed": False, "actions": []}

    async def file_start_master(deck):
        return {
            "deck": deck,
            "changed": False,
            "before": {"target": True, "other": False},
            "after": {"target": True, "other": False},
            "messages": [],
        }

    async def schedule(**kwargs):
        return {
            "ready": True,
            "job": {
                "id": "file-start-job",
                "execution_id": "file-start-1",
                "status": "running",
                "event_count": 5,
                "control_reserved_until_monotonic": 789.0,
            },
        }

    async def unexpected_cue_verification(**kwargs):
        raise AssertionError("file-start launch must not verify a Hot Cue")

    monkeypatch.setattr(server, "_stage_track", stage)
    monkeypatch.setattr(server, "verify_hot_cue", unexpected_cue_verification)
    monkeypatch.setattr(server, "refresh_transition_state", refresh)
    monkeypatch.setattr(server, "ensure_master_deck", file_start_master)
    monkeypatch.setattr(server, "ensure_deck_modes", modes)
    monkeypatch.setattr(server, "perform_transition_card", schedule)

    result = asyncio.run(
        server.stage_and_schedule_transition_card(
            card=transition,
            incoming_title="Incoming",
            incoming_artist="Artist",
            execution_id="file-start-1",
        )
    )

    assert result["ready"] is True
    assert result["cue_verification"] is None
    assert all(call["incoming_hot_cue"] is None for call in refresh_calls)


def test_ensure_stem_state_observes_toggles_and_verifies(monkeypatch) -> None:
    class StemUI:
        def __init__(self) -> None:
            self.calls = 0

        def invalidate_status_cache(self) -> None:
            pass

        def status(self) -> dict:
            self.calls += 1
            vocal = self.calls == 1
            return {
                "decks": [
                    {"deck": 1},
                    {
                        "deck": 2,
                        "stem_vocal_enabled": vocal,
                        "stem_instrumental_enabled": True,
                        "stem_drums_enabled": True,
                    },
                ]
            }

    async def no_sleep(_seconds):
        return None

    engine = OpeningEngine()
    monkeypatch.setattr(server, "rekordbox_ui", StemUI())
    monkeypatch.setattr(server, "deck_observer", CompletionObserver())
    monkeypatch.setattr(server, "engine", engine)
    monkeypatch.setattr(server.asyncio, "sleep", no_sleep)

    result = asyncio.run(server.ensure_stem_state(deck=2, vocal=False))

    assert result["verified"] is True
    assert result["changed"] is True
    assert engine.actions == ["stem_vocal"]


def test_sync_guard_cancels_before_fader_rise_on_live_bpm_mismatch(
    monkeypatch,
) -> None:
    class GuardUI:
        def __init__(self) -> None:
            self.calls = 0
            self.invalidated = False

        def invalidate_status_cache(self) -> None:
            self.invalidated = True

        def status(self) -> dict:
            self.calls += 1
            return {
                "decks": [
                    {
                        "deck": 1,
                        "title": "Outgoing",
                        "elapsed_seconds": 10 + self.calls,
                        "bpm": 121.0,
                        "beat_sync_enabled": True,
                        "quantize_enabled": True,
                    },
                    {
                        "deck": 2,
                        "title": "Incoming",
                        "elapsed_seconds": self.calls,
                        "bpm": 123.0,
                        "beat_sync_enabled": True,
                        "quantize_enabled": True,
                    },
                ]
            }

    class GuardScheduler:
        def __init__(self) -> None:
            self.cancelled: list[str] = []

        def cancel(self, job_id: str) -> dict:
            self.cancelled.append(job_id)
            return {"id": job_id, "status": "cancelled"}

    async def no_sleep(_seconds):
        return None

    engine = OpeningEngine()
    scheduler = GuardScheduler()
    monkeypatch.setattr(server, "rekordbox_ui", GuardUI())
    monkeypatch.setattr(server, "deck_observer", CompletionObserver())
    monkeypatch.setattr(server, "engine", engine)
    monkeypatch.setattr(server, "scheduler", scheduler)
    monkeypatch.setattr(server.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(server, "sync_guard_status", {})

    asyncio.run(
        server._guard_incoming_sync(
            job_id="guarded-job",
            card=opening_card(),
            launch_delay_ms=0,
            outgoing_title="Outgoing",
            incoming_title="Incoming",
        )
    )

    assert scheduler.cancelled == ["guarded-job"]
    assert server.sync_guard_status["guarded-job"]["status"] == "failed_safe"
    assert engine.actions[-3:] == ["channel_fader", "eq_low", "cue"]


def test_sync_guard_status_capture_does_not_block_scheduler_loop(
    monkeypatch,
) -> None:
    class SlowUI:
        @staticmethod
        def status():
            time.sleep(0.08)
            return {"decks": []}

        @staticmethod
        def invalidate_status_cache():
            pass

    monkeypatch.setattr(server, "rekordbox_ui", SlowUI())
    monkeypatch.setattr(server, "deck_observer", CompletionObserver())

    async def scenario() -> None:
        capture = asyncio.create_task(server._guard_status(invalidate=True))
        await asyncio.sleep(0.01)
        assert capture.done() is False
        await capture

    asyncio.run(scenario())


def test_sync_guard_rejects_transport_prearmed_before_phrase_beat_one() -> None:
    with pytest.raises(RuntimeError, match="exact phrase beat-1 boundary"):
        server._arm_sync_guard(
            job_id="early-launch",
            card=opening_card(),
            events=[
                {
                    "at_ms": 19_653,
                    "action": "hot_cue",
                    "parameters": {"deck": 2, "cue": 1},
                }
            ],
            outgoing_title="Outgoing",
            incoming_title="Incoming",
            expected_phrase_boundary_ms=20_000,
        )


def test_sync_guard_repairs_two_beat_bar_error_before_fader_rise(
    monkeypatch,
) -> None:
    class GuardUI:
        def __init__(self) -> None:
            self.calls = 0

        def invalidate_status_cache(self) -> None:
            pass

        def status(self) -> dict:
            self.calls += 1
            corrected = self.calls >= 3
            return {
                "bar_alignment": {
                    "verified": True,
                    "error_beats": 0 if corrected else 2,
                    "signed_error_beats": 0 if corrected else 2,
                },
                "decks": [
                    {
                        "deck": 1,
                        "title": "Outgoing",
                        "elapsed_seconds": 10 + self.calls,
                        "bpm": 120.0,
                        "beat_sync_enabled": True,
                        "quantize_enabled": True,
                    },
                    {
                        "deck": 2,
                        "title": "Incoming",
                        "elapsed_seconds": self.calls,
                        "bpm": 120.0,
                        "beat_sync_enabled": True,
                        "quantize_enabled": True,
                    },
                ],
            }

    class GuardScheduler:
        def __init__(self) -> None:
            self.cancelled: list[str] = []

        def cancel(self, job_id: str) -> dict:
            self.cancelled.append(job_id)
            return {"id": job_id, "status": "cancelled"}

    async def no_sleep(_seconds):
        return None

    engine = OpeningEngine()
    scheduler = GuardScheduler()
    monkeypatch.setattr(server, "rekordbox_ui", GuardUI())
    monkeypatch.setattr(server, "deck_observer", CompletionObserver())
    monkeypatch.setattr(server, "engine", engine)
    monkeypatch.setattr(server, "scheduler", scheduler)
    monkeypatch.setattr(server.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(server, "sync_guard_status", {})

    asyncio.run(
        server._guard_incoming_sync(
            job_id="repair-job",
            card=opening_card(),
            launch_delay_ms=0,
            outgoing_title="Outgoing",
            incoming_title="Incoming",
        )
    )

    assert scheduler.cancelled == []
    assert "beat_jump_2_forward" in engine.actions
    assert server.sync_guard_status["repair-job"]["status"] == "passed"
    assert (
        server.sync_guard_status["repair-job"]["bar_alignment_correction"]["after"][
            "error_beats"
        ]
        == 0
    )


class FakeProfileStore:
    def get(self, track_id: str):
        titles = {"a": "Outgoing", "b": "Incoming"}
        return SimpleNamespace(title=titles[track_id])


class CompletionProfileStore:
    def get(self, track_id: str):
        titles = {"a": "Outgoing", "b": "Incoming"}
        return SimpleNamespace(
            title=titles[track_id],
            bpm=130.0,
            time_signature=4,
            landmarks=[
                SimpleNamespace(
                    cue=1,
                    kind="phrase_start",
                    confidence="verified",
                    bar=1,
                    beat=1,
                )
            ],
        )


class CompletionUI:
    def __init__(self) -> None:
        self.calls = 0

    def status(self) -> dict:
        self.calls += 1
        if self.calls == 1:
            return status(outgoing_time=30, incoming_time=0)
        return status(outgoing_time=30, incoming_time=1)


class CompletionObserver:
    @contextmanager
    def exclusive_adapter(self):
        yield


def status(outgoing_time: int, incoming_time: int) -> dict:
    return {
        "decks": [
            {
                "deck": 1,
                "title": "Outgoing",
                "elapsed_seconds": outgoing_time,
            },
            {
                "deck": 2,
                "title": "Incoming",
                "elapsed_seconds": incoming_time,
            },
        ]
    }


def test_transition_postconditions_require_actual_transport_handoff(
    monkeypatch,
) -> None:
    monkeypatch.setattr(server, "profile_store", FakeProfileStore())
    result = server._verify_transition_postconditions(
        card(),
        status(outgoing_time=30, incoming_time=0),
        status(outgoing_time=30, incoming_time=1),
    )
    assert result["verified"] is True


def test_bar_alignment_consensus_rejects_a_single_unstable_read() -> None:
    stable = {
        "bar_alignment": {
            "verified": True,
            "error_beats": 0.02,
            "signed_error_beats": 0.02,
        }
    }
    shifted = {
        "bar_alignment": {
            "verified": True,
            "error_beats": 1.0,
            "signed_error_beats": 1.0,
        }
    }

    result = server._bar_alignment_consensus(stable, shifted)

    assert result["verified"] is False
    assert "disagree" in result["error"]


def test_transition_postconditions_reject_midi_false_success(monkeypatch) -> None:
    monkeypatch.setattr(server, "profile_store", FakeProfileStore())
    result = server._verify_transition_postconditions(
        card(),
        status(outgoing_time=30, incoming_time=0),
        status(outgoing_time=31, incoming_time=0),
    )
    assert result["verified"] is False
    assert "outgoing deck is still playing after retirement" in result["errors"]
    assert "incoming deck did not start playing" in result["errors"]


def test_completion_promotes_incoming_launch_to_live_clock(monkeypatch) -> None:
    monkeypatch.setattr(server, "profile_store", CompletionProfileStore())
    monkeypatch.setattr(server, "rekordbox_ui", CompletionUI())
    monkeypatch.setattr(server, "deck_observer", CompletionObserver())
    monkeypatch.setattr(server.time, "sleep", lambda _: None)
    monkeypatch.setattr(server, "live_state", server.LiveState())

    result = server._card_completion_verifier(
        card(),
        launch_clock={"incoming_monotonic": time.monotonic() - 1.0},
    )

    assert result["verified"] is True
    promoted = result["promoted_live_clock"]
    assert promoted["deck"] == 2
    assert promoted["track_id"] == "b"
    assert promoted["source"] == "native"
    assert promoted["confidence"] == "high"
    assert promoted["playing"] is True


def test_outbound_midi_cannot_be_promoted_to_high_confidence_observation() -> None:
    observation = DeckObservation(
        deck=1,
        track_id="a",
        title="Outgoing",
        bpm=128,
        playing=True,
        bar=1,
        beat=1,
        source="midi",
        confidence="high",
    )
    with pytest.raises(ValueError, match="outbound MIDI only"):
        server.observe_deck_state(observation)


def test_transition_candidate_ranker_excludes_seven_a_to_two_a() -> None:
    result = server.rank_transition_candidates(
        outgoing_bpm=126,
        outgoing_key="7A",
        candidates=[
            server.TransitionCandidate(
                track_id="bad",
                title="Bad",
                bpm=128,
                key="Ebm",
            ),
            server.TransitionCandidate(
                track_id="good",
                title="Good",
                bpm=127,
                key="8A",
            ),
        ],
    )
    assert [item["track_id"] for item in result["ranked"]] == ["good"]
    assert [item["track_id"] for item in result["excluded"]] == ["bad"]
    assert result["vocal_clash_scoring"] == "not_requested"


def test_hot_cue_position_window_handles_slow_ui_snapshot() -> None:
    lower, upper = server._hot_cue_position_window(
        observed_seconds=1,
        snapshot_started_after_seconds=0.01,
        snapshot_finished_after_seconds=2.391,
    )

    assert lower == pytest.approx(-1.391)
    assert upper == pytest.approx(1.99)
    assert lower <= 0.022 <= upper


def test_hot_cue_position_window_still_rejects_wrong_cue() -> None:
    lower, upper = server._hot_cue_position_window(
        observed_seconds=61,
        snapshot_started_after_seconds=0.01,
        snapshot_finished_after_seconds=2.391,
    )

    assert not lower - 1.0 <= 0.022 <= upper + 1.0


def test_nonzero_staged_position_requires_verified_hot_cue(monkeypatch) -> None:
    monkeypatch.setattr(server, "verified_hot_cues", {})

    with pytest.raises(RuntimeError, match="has not been verified"):
        server._require_staged_incoming_position(
            {"elapsed_seconds": 62},
            deck=1,
            track_id="b",
            hot_cue=2,
        )


def test_verified_hot_cue_allows_nonzero_staged_position(monkeypatch) -> None:
    monkeypatch.setattr(
        server,
        "verified_hot_cues",
        {
            (1, "b", 2): {
                "deck": 1,
                "track_id": "b",
                "cue": 2,
                "verified_monotonic": time.monotonic(),
            }
        },
    )

    server._require_staged_incoming_position(
        {"elapsed_seconds": 62},
        deck=1,
        track_id="b",
        hot_cue=2,
    )


def test_file_start_launch_still_rejects_nonzero_position() -> None:
    with pytest.raises(RuntimeError, match="file start"):
        server._require_staged_incoming_position(
            {"elapsed_seconds": 62},
            deck=1,
            track_id="b",
            hot_cue=None,
        )


def test_dual_hot_cue_card_requires_both_session_verifications(monkeypatch) -> None:
    transition = card()
    transition.events.insert(
        0,
        MusicalEvent(
            bar_offset=0,
            action="hot_cue",
            parameters={"deck": 1, "cue": 2},
        ),
    )
    now = time.monotonic()
    monkeypatch.setattr(
        server,
        "verified_hot_cues",
        {
            (1, "a", 2): {"deck": 1, "cue": 2, "verified_monotonic": now},
        },
    )

    with pytest.raises(RuntimeError, match="Hot Cue 1 on deck 2"):
        server._require_verified_hot_cue(transition)

    server.verified_hot_cues[(2, "b", 1)] = {
        "deck": 2,
        "cue": 1,
        "verified_monotonic": now,
    }
    verified = server._require_verified_hot_cue(transition)
    assert verified is not None
    assert {(item["deck"], item["cue"]) for item in verified} == {(1, 2), (2, 1)}
