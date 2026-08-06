from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from rekordbox_performer import rekordbox_ui
from rekordbox_performer.rekordbox_ui import (
    ControlSample,
    DeckSnapshot,
    RekordboxUIAdapter,
    analyze_bar_grid_alignment,
    blue_ratio,
    normalize_title,
    parse_clock_seconds,
    unique_row_tops,
    vivid_color_ratio,
)


def test_select_exact_track_reacquires_search_after_collection_rebuild(
    monkeypatch,
) -> None:
    class WindowRect:
        left = 0
        top = 0

        @staticmethod
        def width():
            return 1920

    class Control:
        def __init__(self) -> None:
            self.clicked = False

        def click_input(self) -> None:
            self.clicked = True

    class Root:
        def __init__(self) -> None:
            self.focused = False
            self.clicked_at = None

        @staticmethod
        def descendants():
            return []

        @staticmethod
        def rectangle():
            return WindowRect()

        def set_focus(self) -> None:
            self.focused = True

        def click_input(self, *, coords) -> None:
            self.clicked_at = coords

    stale_search = Control()
    collection = Control()
    fresh_search = Control()
    first_root = Root()
    rebuilt_root = Root()
    adapter = RekordboxUIAdapter()
    roots = iter((first_root, rebuilt_root))
    adapter._root = lambda: next(roots)
    initial_samples = [
        ControlSample(None, "ComboBox", "PERFORMANCE", 0, 10, 100, 40),
        ControlSample(collection, "Text", "Collection", 0, 700, 100, 720),
        ControlSample(stale_search, "Edit", "old", 1600, 720, 1900, 750),
    ]
    rebuilt_samples = [
        ControlSample(None, "ComboBox", "PERFORMANCE", 0, 10, 100, 40),
        ControlSample(fresh_search, "Edit", "", 1600, 720, 1900, 750),
    ]
    sampled = iter(((initial_samples, 1920), (rebuilt_samples, 1920)))
    adapter._sample_controls = lambda *_args: next(sampled)
    row_samples = [
        ControlSample(None, "Custom", "header", 0, 780, 1800, 802),
        ControlSample(None, "Custom", "System", 0, 807, 1800, 829),
    ]
    adapter._sample_browser_rows = lambda *_args: (row_samples, 1920)
    adapter._set_clipboard_text = lambda _query: None
    adapter._restore_clipboard_text = lambda _previous: None
    monkeypatch.setattr(rekordbox_ui.time, "sleep", lambda _seconds: None)
    sent = []
    monkeypatch.setattr(rekordbox_ui, "send_keys", sent.append)

    result = adapter.select_exact_track("System", result_index=0)

    assert collection.clicked is True
    assert stale_search.clicked is False
    assert fresh_search.clicked is True
    assert rebuilt_root.focused is True
    assert sent == ["{END}", "+{HOME}", "{BACKSPACE}", "^v"]
    assert result["result_count"] == 1


def test_single_browser_result_is_not_mistaken_for_header() -> None:
    samples = [
        ControlSample(None, "Custom", "", 334, 785, 1895, 807),
    ]

    assert RekordboxUIAdapter._result_rows_from_samples(samples, 1920) == [785]


def test_visual_bar_grid_alignment_detects_two_beat_offset() -> None:
    image = Image.new("RGB", (1672, 566))
    draw = ImageDraw.Draw(image)
    for x in range(141, 1672, 159):
        draw.rectangle((x - 1, 118, x + 1, 121), fill=(255, 0, 0))
    for x in range(62, 1672, 159):
        draw.rectangle((x - 1, 187, x + 1, 190), fill=(255, 0, 0))

    result = analyze_bar_grid_alignment(image)

    assert result["verified"] is True
    assert result["error_beats"] == pytest.approx(2.0, abs=0.03)


def test_visual_bar_grid_alignment_accepts_matching_downbeats() -> None:
    image = Image.new("RGB", (1672, 566))
    draw = ImageDraw.Draw(image)
    for y in (118, 187):
        for x in range(141, 1672, 159):
            draw.rectangle((x - 1, y, x + 1, y + 3), fill=(255, 0, 0))

    result = analyze_bar_grid_alignment(image)

    assert result["verified"] is True
    assert result["error_beats"] == pytest.approx(0.0, abs=0.03)


def test_visual_bar_grid_alignment_handles_taller_fx_header() -> None:
    image = Image.new("RGB", (1902, 579))
    draw = ImageDraw.Draw(image)
    for y in (164, 232):
        for x in range(326, 1902, 157):
            draw.rectangle((x - 1, y, x + 1, y + 3), fill=(255, 0, 0))

    result = analyze_bar_grid_alignment(image)

    assert result["verified"] is True
    assert result["error_beats"] == pytest.approx(0.0, abs=0.03)


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


def test_deck_snapshot_does_not_overwrite_right_jog_bpm_with_file_bpm() -> None:
    samples = [
        ControlSample(None, "Text", "123.44", 1095, 451, 1149, 474),
        ControlSample(None, "Text", "123.00", 1085, 474, 1119, 485),
    ]
    snapshot = RekordboxUIAdapter()._deck_snapshot(
        samples,
        Image.new("RGB", (1902, 999)),
        deck=2,
        window_width=1902,
    )
    assert snapshot.bpm == 123.44


def test_deck_snapshot_accepts_edit_control_for_left_jog_bpm() -> None:
    samples = [
        ControlSample(None, "Text", "121.00", 180, 302, 224, 316),
        ControlSample(None, "Edit", "123.44", 763, 451, 817, 474),
        ControlSample(None, "Text", "2.0", 753, 474, 779, 485),
    ]
    snapshot = RekordboxUIAdapter()._deck_snapshot(
        samples,
        Image.new("RGB", (1902, 999)),
        deck=1,
        window_width=1902,
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


def test_deck_snapshot_observes_master_button_state() -> None:
    image = Image.new("RGB", (1920, 1009))
    ImageDraw.Draw(image).rectangle((817, 305, 887, 318), fill=(0, 120, 255))
    samples = [
        ControlSample(None, "Button", "MASTER", 817, 305, 887, 318),
    ]

    snapshot = RekordboxUIAdapter()._deck_snapshot(
        samples,
        image,
        deck=1,
        window_width=1920,
    )

    assert snapshot.master_enabled is True


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


def test_transport_status_never_focuses_or_captures_rekordbox() -> None:
    class WindowRect:
        @staticmethod
        def height():
            return 1000

    class Root:
        @staticmethod
        def rectangle():
            return WindowRect()

    samples = [
        ControlSample(None, "ComboBox", "PERFORMANCE", 0, 10, 100, 40),
        ControlSample(None, "Text", "Manual Track", 20, 275, 500, 295),
        ControlSample(None, "Text", "Artist", 20, 305, 150, 320),
        ControlSample(None, "Text", "00:42", 500, 305, 550, 320),
        ControlSample(None, "Edit", "126.00", 760, 450, 820, 470),
    ]
    adapter = RekordboxUIAdapter()
    adapter._root = lambda: Root()
    adapter._transport_controls = lambda _root: []
    adapter._sample_controls = lambda _root, _controls: (samples, 1920)
    adapter._capture_focused = lambda _root: pytest.fail("captured Rekordbox")

    result = adapter.transport_status()

    assert result["mode"] is None
    assert result["decks"][0] == {
        "deck": 1,
        "title": "Manual Track",
        "artist": "Artist",
        "bpm": 126.0,
        "key": None,
        "elapsed_seconds": 42,
    }


def test_blue_ratio_distinguishes_active_rekordbox_blue() -> None:
    active = Image.new("RGB", (10, 10), (32, 122, 235))
    inactive = Image.new("RGB", (10, 10), (78, 78, 78))
    assert blue_ratio(active) > 0.9
    assert blue_ratio(inactive) == 0


def test_vivid_color_ratio_distinguishes_active_stem_button() -> None:
    active = Image.new("RGB", (10, 10), (18, 170, 45))
    inactive = Image.new("RGB", (10, 10), (78, 78, 78))
    assert vivid_color_ratio(active) > 0.9
    assert vivid_color_ratio(inactive) == 0
