from __future__ import annotations

from PIL import Image
import pytest

from rekordbox_performer.rekordbox_ui import (
    ControlSample,
    DeckSnapshot,
    RekordboxUIAdapter,
    blue_ratio,
    normalize_title,
    parse_clock_seconds,
    unique_row_tops,
)


def test_deck_snapshot_prefers_live_jog_bpm_over_native_metadata_bpm() -> None:
    samples = [
        ControlSample(None, "Text", "121.00", 180, 302, 224, 316),
        ControlSample(None, "Text", "123.44", 770, 451, 824, 474),
        ControlSample(None, "Text", "2.0", 762, 474, 788, 485),
    ]
    snapshot = RekordboxUIAdapter()._deck_snapshot(
        samples,
        Image.new("RGB", (1920, 1009)),
        deck=1,
        window_width=1920,
    )
    assert snapshot.bpm == 123.44


def test_deck_snapshot_derives_live_bpm_from_pitch_when_jog_bpm_is_omitted() -> None:
    samples = [
        ControlSample(None, "Text", "121.00", 180, 302, 224, 316),
        ControlSample(None, "Text", "2.0", 762, 474, 788, 485),
    ]
    snapshot = RekordboxUIAdapter()._deck_snapshot(
        samples,
        Image.new("RGB", (1920, 1009)),
        deck=1,
        window_width=1920,
    )
    assert snapshot.bpm == 123.42


def test_parse_clock_seconds() -> None:
    assert parse_clock_seconds("01:09") == 69
    assert parse_clock_seconds("-03:05") == -185
    with pytest.raises(ValueError):
        parse_clock_seconds("1.2Bars")


def test_normalize_title_handles_rekordbox_punctuation() -> None:
    assert normalize_title("I Don’t  Want To") == normalize_title(
        "I Don't Want To"
    )


def test_unique_row_tops_collapses_controls_from_the_same_row() -> None:
    assert unique_row_tops([813, 813, 814, 835, 857]) == [813, 835, 857]


def test_result_rows_excludes_rekordbox_column_header() -> None:
    class Control:
        def __init__(self, top: int) -> None:
            self.element_info = type("Info", (), {"control_type": "Custom"})()
            self.top = top

        def rectangle(self):
            return type(
                "Rect",
                (),
                {"left": 334, "top": self.top, "right": 1964,
                 "bottom": self.top + 22},
            )()

    class Root:
        def descendants(self):
            return [Control(778), Control(800), Control(800), Control(822)]

        def rectangle(self):
            return type(
                "Rect",
                (),
                {"left": 0, "top": 0, "width": lambda self: 1920},
            )()

    adapter = RekordboxUIAdapter()
    adapter._relative_rectangle = lambda root, control: (
        334,
        control.top,
        1964,
        control.top + 22,
    )
    assert adapter._result_rows(Root()) == [800, 822]


def test_browser_row_sampling_skips_non_custom_geometry() -> None:
    class Rect:
        left = 0
        top = 0
        right = 1920
        bottom = 1000

        def width(self):
            return 1920

    class Control:
        def __init__(self, control_type, top):
            self.element_info = type("Info", (), {"control_type": control_type})()
            self.top = top
            self.rectangle_calls = 0

        def rectangle(self):
            self.rectangle_calls += 1
            return type(
                "ControlRect",
                (),
                {"left": 300, "top": self.top, "right": 1900, "bottom": self.top + 22},
            )()

    text_control = Control("Text", 800)
    row_control = Control("Custom", 800)

    class Root:
        def rectangle(self):
            return Rect()

        def descendants(self):
            return [text_control, row_control]

    samples, _ = RekordboxUIAdapter()._sample_browser_rows(Root())

    assert len(samples) == 1
    assert samples[0].control is row_control
    assert text_control.rectangle_calls == 0
    assert row_control.rectangle_calls == 1


def test_result_row_point_uses_non_editable_left_gutter() -> None:
    class Control:
        def __init__(self, top: int, left: int = 334) -> None:
            self.element_info = type("Info", (), {"control_type": "Custom"})()
            self.top = top
            self.left = left

        def rectangle(self):
            return type(
                "ControlRect",
                (),
                {
                    "left": 100 + self.left,
                    "top": 50 + self.top,
                    "right": 100 + 1964,
                    "bottom": 50 + self.top + 22,
                },
            )()

    class Rect:
        left = 100
        top = 50

        def width(self):
            return 1920

    class Root:
        def descendants(self):
            return [Control(800)]

        def rectangle(self):
            return Rect()

    adapter = RekordboxUIAdapter()
    adapter._relative_rectangle = lambda root, control: (
        control.left,
        control.top,
        1964,
        control.top + 22,
    )

    x, y = adapter._result_row_point(Root(), 800)
    assert x == 446
    assert y == 861
    assert x < 100 + int(1920 * 0.45)


def test_wait_for_deck_title_retries_transient_uia_failure() -> None:
    adapter = RekordboxUIAdapter()
    expected = DeckSnapshot(
        deck=2,
        title="Vertigo",
        artist="Odd Mob",
        bpm=128,
        key=None,
        elapsed_seconds=0,
        beat_sync_enabled=False,
        quantize_enabled=True,
    )
    responses = iter([RuntimeError("tooltip owns foreground"), expected])

    def deck_snapshot(deck: int):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    adapter.deck_snapshot = deck_snapshot
    assert adapter.wait_for_deck_title(
        2,
        "Vertigo",
        timeout_seconds=0.5,
    ) == expected


def test_status_samples_ui_tree_once_for_both_decks() -> None:
    class WindowRect:
        left = 0
        top = 0

        def width(self):
            return 1920

    class ControlRect:
        left = 10
        top = 10
        right = 210
        bottom = 40

    class Control:
        element_info = type("Info", (), {"control_type": "ComboBox"})()

        def rectangle(self):
            return ControlRect()

        def window_text(self):
            return "PERFORMANCE"

    class Root:
        def __init__(self):
            self.descendant_calls = 0
            self.rectangle_calls = 0

        def descendants(self):
            self.descendant_calls += 1
            return [Control()]

        def rectangle(self):
            self.rectangle_calls += 1
            return WindowRect()

    root = Root()
    adapter = RekordboxUIAdapter()
    adapter._root = lambda: root
    adapter._capture_focused = lambda _: Image.new("RGB", (1920, 1009))

    result = adapter.status()
    second = adapter.status()

    assert result["mode"] == "PERFORMANCE"
    assert second["mode"] == "PERFORMANCE"
    assert root.descendant_calls == 1
    assert root.rectangle_calls == 4


def test_deck_is_playing_reuses_supplied_initial_snapshot(monkeypatch) -> None:
    adapter = RekordboxUIAdapter()
    initial = DeckSnapshot(1, "Track", "Artist", 130, "9A", 10, False, True)
    later = DeckSnapshot(1, "Track", "Artist", 130, "9A", 11, False, True)
    calls = []
    adapter.deck_snapshot = lambda deck: calls.append(deck) or later
    monkeypatch.setattr("rekordbox_performer.rekordbox_ui.time.sleep", lambda _: None)

    assert adapter.deck_is_playing(1, initial=initial) is True
    assert calls == [1]


def test_blue_ratio_distinguishes_active_rekordbox_blue() -> None:
    active = Image.new("RGB", (10, 10), (32, 122, 235))
    inactive = Image.new("RGB", (10, 10), (78, 78, 78))
    assert blue_ratio(active) > 0.9
    assert blue_ratio(inactive) == 0
