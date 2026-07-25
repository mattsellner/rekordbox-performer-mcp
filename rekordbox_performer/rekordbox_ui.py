"""Observable Rekordbox 7 selection and deck loading through Windows UIA."""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

import win32clipboard
from PIL import Image
from pywinauto import Application, Desktop
from pywinauto.keyboard import send_keys


_TIME_PATTERN = re.compile(r"^\s*(-?)(\d{1,3}):(\d{2})\s*$")


def parse_clock_seconds(value: str) -> int:
    match = _TIME_PATTERN.fullmatch(value)
    if not match:
        raise ValueError(f"Unrecognized Rekordbox clock: {value!r}")
    sign, minutes, seconds = match.groups()
    total = int(minutes) * 60 + int(seconds)
    return -total if sign else total


def normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.translate(
        str.maketrans(
            {
                "\u2018": "'",
                "\u2019": "'",
                "\uff07": "'",
                "\u201c": '"',
                "\u201d": '"',
            }
        )
    )
    return " ".join(normalized.split()).casefold()


def unique_row_tops(values: list[int], tolerance: int = 2) -> list[int]:
    result: list[int] = []
    for value in sorted(values):
        if not result or value - result[-1] > tolerance:
            result.append(value)
    return result


def blue_ratio(image: Image.Image) -> float:
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width == 0 or height == 0:
        return 0.0
    access = rgb.load()
    blue = sum(
        1
        for y in range(height)
        for x in range(width)
        for red, green, value in (access[x, y],)
        if value > 100 and value > red * 1.3 and value > green * 1.05
    )
    return blue / (width * height)


@dataclass(frozen=True)
class DeckSnapshot:
    deck: int
    title: str
    artist: str
    bpm: float | None
    key: str | None
    elapsed_seconds: int | None
    beat_sync_enabled: bool | None
    quantize_enabled: bool | None

    def public(self) -> dict[str, Any]:
        return {
            "deck": self.deck,
            "title": self.title,
            "artist": self.artist,
            "bpm": self.bpm,
            "key": self.key,
            "elapsed_seconds": self.elapsed_seconds,
            "beat_sync_enabled": self.beat_sync_enabled,
            "quantize_enabled": self.quantize_enabled,
        }


class RekordboxUIAdapter:
    """Read and manipulate only stable, observable Rekordbox UI controls."""

    def __init__(self, window_title: str = "rekordbox") -> None:
        self.window_title = window_title

    def _root(self):
        window = next(
            (
                item
                for item in Desktop(backend="uia").windows()
                if item.window_text().casefold()
                == self.window_title.casefold()
            ),
            None,
        )
        if window is None:
            raise RuntimeError("The Rekordbox main window is not visible")
        app = Application(backend="uia").connect(handle=window.handle)
        return app.window(handle=window.handle)

    @staticmethod
    def _relative_rectangle(root, control) -> tuple[int, int, int, int]:
        window = root.rectangle()
        rectangle = control.rectangle()
        return (
            rectangle.left - window.left,
            rectangle.top - window.top,
            rectangle.right - window.left,
            rectangle.bottom - window.top,
        )

    def _mode(self, root) -> str:
        controls = [
            control
            for control in root.descendants()
            if getattr(control.element_info, "control_type", None) == "ComboBox"
            and 0 <= self._relative_rectangle(root, control)[1] <= 70
        ]
        known_modes = {"performance", "export", "edit", "lighting"}
        return next(
            (
                control.window_text()
                for control in controls
                if control.window_text().casefold() in known_modes
            ),
            "",
        )

    def _search_edit(self, root):
        controls = []
        for control in root.descendants():
            if getattr(control.element_info, "control_type", None) != "Edit":
                continue
            left, top, right, _ = self._relative_rectangle(root, control)
            if 700 <= top <= 780 and left >= root.rectangle().width() - 350:
                controls.append(control)
        if len(controls) != 1:
            raise RuntimeError(
                "Expected one Rekordbox browser search field, "
                f"found {len(controls)}"
            )
        return controls[0]

    @staticmethod
    def _set_clipboard_text(value: str) -> str | None:
        previous = None
        win32clipboard.OpenClipboard()
        try:
            if win32clipboard.IsClipboardFormatAvailable(
                win32clipboard.CF_UNICODETEXT
            ):
                previous = win32clipboard.GetClipboardData(
                    win32clipboard.CF_UNICODETEXT
                )
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(
                value,
                win32clipboard.CF_UNICODETEXT,
            )
        finally:
            win32clipboard.CloseClipboard()
        return previous

    @staticmethod
    def _restore_clipboard_text(previous: str | None) -> None:
        if previous is None:
            return
        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(
                previous,
                win32clipboard.CF_UNICODETEXT,
            )
        finally:
            win32clipboard.CloseClipboard()

    def _result_rows(self, root) -> list[int]:
        tops = []
        for control in root.descendants():
            control_type = getattr(
                control.element_info,
                "control_type",
                None,
            )
            if control_type != "Custom":
                continue
            left, top, right, bottom = self._relative_rectangle(root, control)
            if (
                770 <= top <= 930
                and 18 <= bottom - top <= 28
                and right - left >= root.rectangle().width() * 0.75
            ):
                tops.append(top)
        return unique_row_tops(tops)

    def select_exact_track(
        self,
        title: str,
        *,
        result_index: int | None = None,
        timeout_seconds: float = 6.0,
    ) -> dict[str, Any]:
        root = self._root()
        mode = self._mode(root)
        if mode.casefold() != "performance":
            raise RuntimeError(
                f"Rekordbox must be in Performance mode, found {mode!r}"
            )
        search = self._search_edit(root)
        previous = self._set_clipboard_text(title)
        try:
            root.set_focus()
            search.click_input()
            send_keys("^a^v")
        finally:
            self._restore_clipboard_text(previous)

        deadline = time.monotonic() + timeout_seconds
        rows: list[int] = []
        while time.monotonic() < deadline:
            rows = self._result_rows(root)
            if rows:
                break
            time.sleep(0.1)
        if not rows:
            raise RuntimeError(
                f"Exact search for {title!r} returned no visible rows"
            )
        if result_index is None:
            if len(rows) != 1:
                raise RuntimeError(
                    f"Exact search for {title!r} returned "
                    f"{len(rows)} visible rows; provide a result index "
                    "and verify deck metadata after loading"
                )
            result_index = 0
        if not 0 <= result_index < len(rows):
            raise ValueError(
                f"result_index must be between 0 and {len(rows) - 1}"
            )
        window = root.rectangle()
        row_y = window.top + rows[result_index] + 10
        row_x = window.left + int(window.width() * 0.45)
        root.click_input(coords=(row_x - window.left, row_y - window.top))
        return {
            "title": title,
            "result_count": len(rows),
            "result_index": result_index,
            "selected_row_top": rows[result_index],
            "focused": True,
        }

    def _deck_controls(
        self,
        root,
        deck: int,
        descendants: list[Any] | None = None,
    ) -> list[Any]:
        if deck not in (1, 2):
            raise ValueError("deck must be 1 or 2")
        width = root.rectangle().width()
        midpoint = width / 2
        controls = []
        for control in descendants or root.descendants():
            left, _, right, _ = self._relative_rectangle(root, control)
            center = (left + right) / 2
            if (deck == 1 and center < midpoint) or (
                deck == 2 and center >= midpoint
            ):
                controls.append(control)
        return controls

    def _button_active(
        self,
        root,
        control,
        image: Image.Image | None = None,
    ) -> bool:
        image = image or root.capture_as_image()
        if image is None:
            raise RuntimeError("Unable to capture the Rekordbox window")
        left, top, right, bottom = self._relative_rectangle(root, control)
        return blue_ratio(image.crop((left, top, right, bottom))) >= 0.25

    @staticmethod
    def _capture_focused(root) -> Image.Image:
        # Windows can return a blank/white PrintWindow capture for Rekordbox
        # while another process owns the foreground window. Focus it before
        # sampling pixels used for observable mode-state verification.
        root.set_focus()
        time.sleep(0.12)
        image = root.capture_as_image()
        if image is None:
            raise RuntimeError("Unable to capture the Rekordbox window")
        return image

    def _deck_snapshot(
        self,
        root,
        descendants: list[Any],
        image: Image.Image,
        deck: int,
    ) -> DeckSnapshot:
        controls = self._deck_controls(root, deck, descendants)
        title_candidates = []
        artist_candidates = []
        bpm = None
        key = None
        elapsed = None
        sync = None
        quantize = None
        for control in controls:
            left, top, right, _ = self._relative_rectangle(root, control)
            text = control.window_text()
            control_type = getattr(control.element_info, "control_type", None)
            if (
                control_type == "Text"
                and 260 <= top <= 310
                and right - left >= root.rectangle().width() * 0.2
                and text
            ):
                title_candidates.append((right - left, text))
            if (
                control_type == "Text"
                and 285 <= top <= 330
                and text
                and not text.startswith("-")
            ):
                try:
                    parsed = parse_clock_seconds(text)
                except ValueError:
                    try:
                        value = float(text.strip())
                    except ValueError:
                        pass
                    else:
                        if 40 <= value <= 500:
                            bpm = value
                else:
                    elapsed = parsed
            if control_type == "Text" and 295 <= top <= 325 and text:
                stripped = text.strip()
                try:
                    numeric = float(stripped)
                except ValueError:
                    numeric = None
                if numeric is not None and 40 <= numeric <= 500:
                    bpm = numeric
                elif re.fullmatch(
                    r"(?:\d{1,2}[AB]|[A-G](?:#|b)?)",
                    stripped,
                ):
                    key = stripped
                elif not stripped.startswith("-") and ":" not in stripped:
                    artist_candidates.append((left, stripped))
            if text == "BEAT\nSYNC":
                sync = self._button_active(root, control, image)
            if text == "Q" and 420 <= top <= 500:
                quantize = self._button_active(root, control, image)
        title = max(title_candidates, default=(0, ""))[1]
        artist = min(artist_candidates, default=(0, ""))[1]
        return DeckSnapshot(
            deck=deck,
            title=title,
            artist=artist,
            bpm=bpm,
            key=key,
            elapsed_seconds=elapsed,
            beat_sync_enabled=sync,
            quantize_enabled=quantize,
        )

    def deck_snapshot(self, deck: int) -> DeckSnapshot:
        root = self._root()
        descendants = root.descendants()
        image = self._capture_focused(root)
        return self._deck_snapshot(root, descendants, image, deck)

    def deck_is_playing(
        self,
        deck: int,
        *,
        sample_seconds: float = 1.1,
    ) -> bool:
        first = self.deck_snapshot(deck)
        time.sleep(sample_seconds)
        second = self.deck_snapshot(deck)
        if first.title != second.title:
            raise RuntimeError("Deck title changed during transport observation")
        if (
            first.elapsed_seconds is None
            or second.elapsed_seconds is None
        ):
            raise RuntimeError("Unable to observe the deck elapsed-time display")
        return second.elapsed_seconds != first.elapsed_seconds

    def wait_for_deck_title(
        self,
        deck: int,
        expected_title: str,
        *,
        timeout_seconds: float = 8.0,
    ) -> DeckSnapshot:
        deadline = time.monotonic() + timeout_seconds
        last = self.deck_snapshot(deck)
        while time.monotonic() < deadline:
            if normalize_title(last.title) == normalize_title(expected_title):
                return last
            time.sleep(0.25)
            last = self.deck_snapshot(deck)
        raise RuntimeError(
            f"Deck {deck} loaded {last.title!r}, expected {expected_title!r}"
        )

    def status(self) -> dict[str, Any]:
        root = self._root()
        descendants = root.descendants()
        image = self._capture_focused(root)
        return {
            "mode": self._mode(root),
            "decks": [
                self._deck_snapshot(
                    root,
                    descendants,
                    image,
                    deck,
                ).public()
                for deck in (1, 2)
            ],
        }
