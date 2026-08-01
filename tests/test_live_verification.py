from contextlib import contextmanager
from types import SimpleNamespace
import time

import pytest

from rekordbox_performer import server
from rekordbox_performer.intelligence import (
    DeckObservation,
    MusicalEvent,
    TransitionCard,
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
