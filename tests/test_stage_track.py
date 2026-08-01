import asyncio
from contextlib import contextmanager

import pytest

from rekordbox_performer import server
from rekordbox_performer.rekordbox_ui import DeckSnapshot


class FakeUI:
    def __init__(self) -> None:
        self.wait_calls = 0
        self.dragged = False
        self.search_queries = []
        self.stage_route_cache = {}

    def deck_is_playing(self, deck: int, *, initial=None) -> bool:
        return False

    def select_exact_track(
        self,
        title: str,
        search_query=None,
        result_index=0,
    ):
        self.search_queries.append(search_query)
        return {
            "title": title,
            "search_query": search_query,
            "result_count": 1,
            "result_index": result_index,
            "selected_row_top": 800,
            "focused": True,
        }

    def wait_for_deck_title(self, deck: int, title: str, timeout_seconds: float):
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise RuntimeError("MIDI load did not take")
        return DeckSnapshot(
            deck=deck,
            title=title,
            artist=getattr(self, "expected_artist", "Artist"),
            bpm=128,
            key="5A",
            elapsed_seconds=0,
            beat_sync_enabled=True,
            quantize_enabled=True,
        )

    def drag_selected_track_to_deck(self, **kwargs):
        self.dragged = True
        return {"method": "drag_to_deck"}


class AutoStartUI(FakeUI):
    def __init__(self) -> None:
        super().__init__()
        self.play_checks = 0

    def deck_is_playing(self, deck: int, *, initial=None) -> bool:
        self.play_checks += 1
        return self.play_checks >= 3


class FakeEngine:
    def __init__(self) -> None:
        self.actions = []

    async def send_action(self, action: str, parameters=None):
        self.actions.append((action, parameters))
        return [action]


class FakeObserver:
    def __init__(self) -> None:
        self.invalidated = False

    def invalidate(self) -> None:
        self.invalidated = True

    @contextmanager
    def exclusive_adapter(self):
        yield


def test_stage_track_uses_verified_drag_fallback(monkeypatch) -> None:
    ui = FakeUI()
    observer = FakeObserver()
    monkeypatch.setattr(server, "rekordbox_ui", ui)
    engine = FakeEngine()
    monkeypatch.setattr(server, "engine", engine)
    monkeypatch.setattr(server, "deck_observer", observer)

    result = asyncio.run(
        server._stage_track(
            deck=1,
            track_id="rekordbox:123",
            title="Track",
            artist="Artist",
            source="local",
        )
    )

    assert result["verified"] is True
    assert result["track_id"] == "rekordbox:123"
    assert result["attempts"][0]["method"] == "drag_to_deck"
    assert result["selection"]["search_query"] == "Track Artist"
    assert ui.dragged is True
    assert observer.invalidated is True
    assert engine.actions[:2] == [
        ("channel_fader", {"deck": 1, "value": 0}),
        ("cue", {"deck": 1}),
    ]
    assert result["transport_verified_stopped"] is True


def test_stage_track_broadens_parenthetical_search(monkeypatch) -> None:
    ui = FakeUI()
    ui.expected_artist = "John Summit, Inéz"
    original = ui.select_exact_track

    def select(title, search_query=None, result_index=0):
        ui.search_queries.append(search_query)
        if search_query != "light years":
            raise RuntimeError(
                f"Exact search for {title!r} returned no visible rows"
            )
        result = original(title, search_query=search_query, result_index=result_index)
        ui.search_queries.pop()
        return result

    ui.select_exact_track = select
    observer = FakeObserver()
    monkeypatch.setattr(server, "rekordbox_ui", ui)
    monkeypatch.setattr(server, "engine", FakeEngine())
    monkeypatch.setattr(server, "deck_observer", observer)

    result = asyncio.run(
        server._stage_track(
            deck=1,
            track_id="52342283",
            title="light years (Matt Sassari Remix)",
            artist="John Summit, Inéz",
        )
    )

    assert result["selection"]["search_query"] == "light years"
    assert ui.search_queries[:3] == [
        "light years (Matt Sassari Remix) John Summit, Inéz",
        "light years (Matt Sassari Remix)",
        "light years",
    ]


def test_stage_track_mutes_stops_and_rejects_autoplay(monkeypatch) -> None:
    ui = AutoStartUI()
    observer = FakeObserver()
    engine = FakeEngine()
    monkeypatch.setattr(server, "rekordbox_ui", ui)
    monkeypatch.setattr(server, "engine", engine)
    monkeypatch.setattr(server, "deck_observer", observer)

    with pytest.raises(RuntimeError, match="auto-started"):
        asyncio.run(
            server._stage_track(
                deck=1,
                track_id="rekordbox:123",
                title="Track",
                artist="Artist",
            )
        )

    assert engine.actions[:2] == [
        ("channel_fader", {"deck": 1, "value": 0}),
        ("cue", {"deck": 1}),
    ]
    assert engine.actions[-2:] == [
        ("channel_fader", {"deck": 1, "value": 0}),
        ("cue", {"deck": 1}),
    ]


def test_stage_track_tries_ascii_apostrophe_search_variant(monkeypatch) -> None:
    ui = FakeUI()
    ui.expected_artist = "AYYBO"
    original = ui.select_exact_track

    def select(title, search_query=None, result_index=0):
        ui.search_queries.append(search_query)
        if search_query != "I Don't Want To AYYBO":
            raise RuntimeError(
                f"Exact search for {title!r} returned no visible rows"
            )
        result = original(title, search_query=search_query, result_index=result_index)
        ui.search_queries.pop()
        return result

    ui.select_exact_track = select
    monkeypatch.setattr(server, "rekordbox_ui", ui)
    monkeypatch.setattr(server, "engine", FakeEngine())
    monkeypatch.setattr(server, "deck_observer", FakeObserver())

    result = asyncio.run(
        server._stage_track(
            deck=2,
            track_id="7466681",
            title="I Don’t Want To",
            artist="AYYBO",
        )
    )
    assert result["selection"]["search_query"] == "I Don't Want To AYYBO"


def test_stage_track_reuses_verified_search_and_drag_route(monkeypatch) -> None:
    ui = FakeUI()
    observer = FakeObserver()
    engine = FakeEngine()
    monkeypatch.setattr(server, "rekordbox_ui", ui)
    monkeypatch.setattr(server, "engine", engine)
    monkeypatch.setattr(server, "deck_observer", observer)

    first = asyncio.run(
        server._stage_track(
            deck=1,
            track_id="rekordbox:cached",
            title="Track",
            artist="Artist",
        )
    )
    # The cached route skips the known-failing MIDI attempt and goes straight
    # to drag verification, so make that single verification succeed.
    ui.wait_calls = 1
    engine.actions.clear()
    second = asyncio.run(
        server._stage_track(
            deck=1,
            track_id="rekordbox:cached",
            title="Track",
            artist="Artist",
        )
    )

    assert first["route_cache_hit"] is False
    assert second["route_cache_hit"] is True
    assert second["attempts"][0]["method"] == "drag_to_deck"
    assert not any(action == "load_deck_1" for action, _ in engine.actions)
