import json
from types import SimpleNamespace

from rekordbox_performer.stage_capture import stage_to


def snapshot(title: str, artist: str):
    payload = {
        "deck": 1,
        "title": title,
        "artist": artist,
        "elapsed_seconds": 0,
    }
    return SimpleNamespace(
        title=title,
        artist=artist,
        elapsed_seconds=0,
        public=lambda: payload,
    )


class ResidentAdapter:
    def deck_snapshot(self, _deck: int):
        return snapshot("System", "Odd Mob, OMNOM, HYPERBEAM")

    def deck_is_playing(self, _deck: int, *, initial=None):
        return False


class DragAdapter:
    def __init__(self) -> None:
        self.loaded = snapshot("Pop Up (Original Mix)", "Kyle Watson")

    def deck_snapshot(self, _deck: int):
        return snapshot("System", "Odd Mob, OMNOM, HYPERBEAM")

    def deck_is_playing(self, _deck: int, *, initial=None):
        return False

    def select_exact_track(self, title: str, *, search_query: str, result_index: int):
        assert title == "Pop Up (Original Mix)"
        assert result_index == 0
        return {
            "title": title,
            "search_query": search_query,
            "result_count": 1,
            "result_index": 0,
            "selected_row_top": 700,
            "selected_row_point": [300, 720],
            "deck_drop_points": {1: [350, 300]},
        }

    def drag_selected_track_to_deck(self, **_kwargs):
        return {"method": "drag_to_deck"}

    def wait_for_deck_title(self, _deck: int, _title: str, *, timeout_seconds: float):
        assert timeout_seconds == 6.0
        return self.loaded


def test_stage_capture_skips_browser_for_verified_resident_track(tmp_path) -> None:
    output = tmp_path / "stage.json"
    assert (
        stage_to(
            output,
            deck=1,
            title="System",
            artist="Odd Mob, OMNOM, HYPERBEAM",
            adapter=ResidentAdapter(),
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["verified"] is True
    assert payload["already_loaded"] is True


def test_stage_capture_drags_exact_artist_and_verifies_stopped(tmp_path) -> None:
    output = tmp_path / "stage.json"
    assert (
        stage_to(
            output,
            deck=1,
            title="Pop Up (Original Mix)",
            artist="Kyle Watson",
            adapter=DragAdapter(),
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["verified"] is True
    assert payload["already_loaded"] is False
    assert payload["attempts"][0]["artist_matches"] is True
