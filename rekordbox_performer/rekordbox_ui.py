"""Observable Rekordbox 7 selection and deck loading through Windows UIA."""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

import win32clipboard
from PIL import Image
from pywinauto import Application, Desktop, mouse
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


def vivid_color_ratio(image: Image.Image) -> float:
    """Measure Rekordbox's colored active-state outline/text."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width == 0 or height == 0:
        return 0.0
    access = rgb.load()
    vivid = sum(
        1
        for y in range(height)
        for x in range(width)
        for pixel in (access[x, y],)
        if max(pixel) > 70 and max(pixel) - min(pixel) > 35
    )
    return vivid / (width * height)


def _red_marker_centers(image: Image.Image, y: int) -> list[float]:
    """Return horizontal centers of Rekordbox's red beat-grid markers."""
    rgb = image.convert("RGB")
    runs: list[list[int]] = []
    for x in range(rgb.width):
        red, green, blue = rgb.getpixel((x, y))
        if red < 180 or green > 70 or blue > 70:
            continue
        if not runs or x > runs[-1][-1] + 1:
            runs.append([x])
        else:
            runs[-1].append(x)
    return [(run[0] + run[-1]) / 2.0 for run in runs]


def _marker_lattice(
    image: Image.Image,
    start_y: int,
    end_y: int,
) -> dict[str, Any] | None:
    """Fit the cleanest regularly spaced downbeat-marker row in one deck."""
    width = image.width
    best: tuple[tuple[float, int, float], dict[str, Any]] | None = None
    for y in range(max(0, start_y), min(image.height, end_y)):
        centers = _red_marker_centers(image, y)
        if len(centers) < 6:
            continue
        for left_index, left in enumerate(centers):
            for right in centers[left_index + 1:]:
                step = right - left
                if not width * 0.07 <= step <= width * 0.12:
                    continue
                by_index: dict[int, tuple[float, float]] = {}
                for center in centers:
                    index = round((center - left) / step)
                    residual = abs(center - (left + index * step))
                    if residual > 2.5:
                        continue
                    previous = by_index.get(index)
                    if previous is None or residual < previous[1]:
                        by_index[index] = (center, residual)
                if len(by_index) < 6:
                    continue
                ordered = sorted(by_index.items())
                longest: list[tuple[int, tuple[float, float]]] = []
                current: list[tuple[int, tuple[float, float]]] = []
                for item in ordered:
                    if current and item[0] != current[-1][0] + 1:
                        if len(current) > len(longest):
                            longest = current
                        current = []
                    current.append(item)
                if len(current) > len(longest):
                    longest = current
                if len(longest) < 6:
                    continue
                xs = [item[1][0] for item in longest]
                qs = [item[0] for item in longest]
                q_mean = sum(qs) / len(qs)
                x_mean = sum(xs) / len(xs)
                denominator = sum((q - q_mean) ** 2 for q in qs)
                if denominator == 0:
                    continue
                fitted_step = sum(
                    (q - q_mean) * (x - x_mean)
                    for q, x in zip(qs, xs)
                ) / denominator
                intercept = x_mean - fitted_step * q_mean
                rms = (
                    sum(
                        (x - (intercept + fitted_step * q)) ** 2
                        for q, x in zip(qs, xs)
                    ) / len(qs)
                ) ** 0.5
                coverage = len(xs) / len(centers)
                score = (coverage, len(xs), -rms)
                result = {
                    "y": y,
                    "marker_count": len(xs),
                    "spacing_px": fitted_step,
                    "intercept_px": intercept,
                    "rms_px": rms,
                }
                if best is None or score > best[0]:
                    best = (score, result)
    return None if best is None else best[1]


def analyze_bar_grid_alignment(image: Image.Image) -> dict[str, Any]:
    """Measure deck-to-deck beat-in-bar alignment from visible downbeats.

    Rekordbox paints red downbeat triangles above each stacked enlarged
    waveform. Their horizontal phase is authoritative for bar alignment and
    catches the exact failure that Beat Sync alone cannot: beats can be phase
    locked while beat 1 on one deck is aligned to beat 3 on the other.
    """
    outgoing = _marker_lattice(
        image,
        108,
        132,
    )
    incoming = _marker_lattice(
        image,
        184,
        196,
    )
    if outgoing is None or incoming is None:
        return {
            "verified": False,
            "error": "visible downbeat marker rows were not detected",
        }
    first_step = float(outgoing["spacing_px"])
    second_step = float(incoming["spacing_px"])
    spacing = (first_step + second_step) / 2.0
    if spacing <= 0 or abs(first_step - second_step) / spacing > 0.05:
        return {
            "verified": False,
            "error": "deck waveform zoom levels do not match",
            "outgoing": outgoing,
            "incoming": incoming,
        }
    raw = float(incoming["intercept_px"]) - float(outgoing["intercept_px"])
    phase_px = (raw + spacing / 2.0) % spacing - spacing / 2.0
    error_beats = abs(phase_px) / spacing * 4.0
    return {
        "verified": True,
        "error_beats": round(error_beats, 3),
        "signed_error_beats": round(phase_px / spacing * 4.0, 3),
        "spacing_px": round(spacing, 3),
        "outgoing": outgoing,
        "incoming": incoming,
    }


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
    master_enabled: bool | None = None
    stem_vocal_enabled: bool | None = None
    stem_instrumental_enabled: bool | None = None
    stem_drums_enabled: bool | None = None

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
            "master_enabled": self.master_enabled,
            "stem_vocal_enabled": self.stem_vocal_enabled,
            "stem_instrumental_enabled": self.stem_instrumental_enabled,
            "stem_drums_enabled": self.stem_drums_enabled,
        }


@dataclass(frozen=True)
class ControlSample:
    control: Any
    control_type: str | None
    text: str
    left: int
    top: int
    right: int
    bottom: int


class RekordboxUIAdapter:
    """Read and manipulate only stable, observable Rekordbox UI controls."""

    def __init__(self, window_title: str = "rekordbox") -> None:
        self.window_title = window_title
        # Successful stage routes are learned by server._stage_track. Keeping
        # the cache on the adapter scopes it to one Rekordbox/Codex session and
        # naturally discards it after either process restarts.
        self.stage_route_cache: dict[str, dict[str, Any]] = {}
        # Rekordbox exposes thousands of UIA descendants.  Enumerating that
        # tree for every deck-clock read can consume an entire musical phrase.
        # Cache only the small set of controls used by status/deck snapshots;
        # their text and pixels are still sampled fresh on every call.
        self._status_control_cache: list[Any] | None = None
        self._status_cache_signature: tuple[int, int] | None = None

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
        window = root.rectangle()
        controls = [
            control
            for control in root.descendants()
            if getattr(control.element_info, "control_type", None) == "ComboBox"
            and 0 <= control.rectangle().top - window.top <= 70
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

    def _sample_controls(
        self,
        root,
        descendants: list[Any] | None = None,
    ) -> tuple[list[ControlSample], int]:
        """Read UIA geometry/text once for a complete Rekordbox snapshot."""
        window = root.rectangle()
        samples = []
        for control in descendants or root.descendants():
            info = control.element_info
            control_type = getattr(info, "control_type", None)
            rectangle = control.rectangle()
            # Rekordbox exposes BEAT SYNC and Quantize labels through Button
            # or Custom controls depending on the build, so retain names for
            # every sampled control. This is still one UIA read per control.
            text_reader = getattr(control, "window_text", None)
            text = text_reader() if text_reader is not None else ""
            samples.append(
                ControlSample(
                    control=control,
                    control_type=control_type,
                    text=text,
                    left=rectangle.left - window.left,
                    top=rectangle.top - window.top,
                    right=rectangle.right - window.left,
                    bottom=rectangle.bottom - window.top,
                )
            )
        return samples, window.width()

    def _status_controls(self, root) -> list[Any]:
        window = root.rectangle()
        signature = (int(getattr(root, "handle", id(root))), window.width())
        if (
            self._status_control_cache is not None
            and self._status_cache_signature == signature
        ):
            return self._status_control_cache
        controls = []
        for control in root.descendants():
            info = control.element_info
            control_type = getattr(info, "control_type", None)
            rectangle = control.rectangle()
            top = rectangle.top - window.top
            if (
                (control_type == "ComboBox" and 0 <= top <= 70)
                or (control_type == "Button" and 0 <= top <= 70)
                or (
                    control_type in {"Text", "Edit", "Button", "Custom"}
                    and (
                        250 <= top <= 335
                        or 360 <= top <= 410
                        or 410 <= top <= 510
                    )
                )
            ):
                controls.append(control)
        self._status_control_cache = controls
        self._status_cache_signature = signature
        return controls

    def invalidate_status_cache(self) -> None:
        """Force the next status read to reacquire Rekordbox UIA controls.

        Rekordbox can recreate the large live-BPM controls when transport or
        deck content changes. Holding the old UIA objects would silently turn
        a live-clock observation back into analyzed-file metadata.
        """
        self._status_control_cache = None
        self._status_cache_signature = None

    @staticmethod
    def _mode_from_samples(samples: list[ControlSample]) -> str:
        known_modes = {"performance", "export", "edit", "lighting"}
        return next(
            (
                sample.text
                for sample in samples
                if sample.control_type == "ComboBox"
                and 0 <= sample.top <= 70
                and sample.text.casefold() in known_modes
            ),
            "",
        )

    def _search_edit(self, root):
        window = root.rectangle()
        width = window.width()
        controls = []
        for control in root.descendants():
            if getattr(control.element_info, "control_type", None) != "Edit":
                continue
            rectangle = control.rectangle()
            left = rectangle.left - window.left
            top = rectangle.top - window.top
            if 700 <= top <= 780 and left >= width - 350:
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
        samples, width = self._sample_browser_rows(root)
        return self._result_rows_from_samples(samples, width)

    def _sample_browser_rows(
        self,
        root,
        descendants: list[Any] | None = None,
    ) -> tuple[list[ControlSample], int]:
        """Sample only row-shaped Custom controls during search polling.

        A complete Rekordbox tree contains hundreds of controls. Reading text
        and geometry from every one on each poll dominated exact-load time,
        especially while another deck was playing. Browser result detection
        needs only Custom-control geometry, so keep the polling path narrow.
        """
        window = root.rectangle()
        width = window.width()
        samples = []
        for control in descendants or root.descendants():
            control_type = getattr(control.element_info, "control_type", None)
            if control_type != "Custom":
                continue
            rectangle = control.rectangle()
            left = rectangle.left - window.left
            top = rectangle.top - window.top
            right = rectangle.right - window.left
            bottom = rectangle.bottom - window.top
            if (
                770 <= top <= 930
                and 18 <= bottom - top <= 28
                and right - left >= width * 0.75
            ):
                samples.append(
                    ControlSample(
                        control=control,
                        control_type=control_type,
                        text="",
                        left=left,
                        top=top,
                        right=right,
                        bottom=bottom,
                    )
                )
        return samples, width

    @staticmethod
    def _result_rows_from_samples(
        samples: list[ControlSample],
        window_width: int,
    ) -> list[int]:
        tops = []
        for sample in samples:
            if (
                sample.control_type == "Custom"
                and 770 <= sample.top <= 930
                and 18 <= sample.bottom - sample.top <= 28
                and sample.right - sample.left >= window_width * 0.75
            ):
                tops.append(sample.top)
        row_tops = unique_row_tops(tops)
        # Rekordbox exposes the browser's column-header band as a row-shaped
        # Custom control.  It is always the first wide 22 px band and is not a
        # draggable track.  Returning it caused an empty/header drag that left
        # the target deck at "Not Loaded.".
        return row_tops[1:] if row_tops else []

    def _result_row_point(self, root, row_top: int) -> tuple[int, int]:
        """Return a safe absolute hit point in the row's non-editable gutter."""
        window = root.rectangle()
        samples, _ = self._sample_controls(root)
        return self._result_row_point_from_samples(samples, window, row_top)

    @staticmethod
    def _result_row_point_from_samples(
        samples: list[ControlSample],
        window,
        row_top: int,
    ) -> tuple[int, int]:
        width = window.width()
        candidates = []
        for sample in samples:
            if (
                abs(sample.top - row_top) <= 1
                and 18 <= sample.bottom - sample.top <= 28
                and sample.right - sample.left >= width * 0.75
            ):
                candidates.append(
                    (sample.left, sample.top, sample.right, sample.bottom)
                )
        if candidates:
            left, top, right, bottom = max(
                candidates,
                key=lambda item: item[2] - item[0],
            )
            # The title cell is editable.  Twelve pixels inside the wide row
            # container lands in Rekordbox's selector/artwork gutter instead.
            return (
                window.left + left + min(12, max(1, right - left - 1)),
                window.top + (top + bottom) // 2,
            )
        # Conservative fallback for layouts that do not expose the row Custom
        # control: use the browser's far-left gutter, never the title column.
        return (
            window.left + int(window.width() * 0.18),
            window.top + row_top + 10,
        )

    def select_exact_track(
        self,
        title: str,
        *,
        search_query: str | None = None,
        result_index: int | None = None,
        timeout_seconds: float = 3.0,
    ) -> dict[str, Any]:
        root = self._root()
        descendants = root.descendants()
        samples, window_width = self._sample_controls(root, descendants)
        mode = self._mode_from_samples(samples)
        if mode.casefold() != "performance":
            raise RuntimeError(
                f"Rekordbox must be in Performance mode, found {mode!r}"
            )
        collection_controls = [
            sample.control
            for sample in samples
            if sample.control_type == "Text"
            and normalize_title(sample.text) == "collection"
        ]
        if len(collection_controls) != 1:
            raise RuntimeError(
                "Expected one Rekordbox Collection browser source, "
                f"found {len(collection_controls)}"
            )
        collection_controls[0].click_input()
        time.sleep(0.15)
        search_controls = [
            sample.control
            for sample in samples
            if sample.control_type == "Edit"
            and 700 <= sample.top <= 780
            and sample.left >= window_width - 350
        ]
        if len(search_controls) != 1:
            raise RuntimeError(
                "Expected one Rekordbox browser search field, "
                f"found {len(search_controls)}"
            )
        search = search_controls[0]
        query = search_query or title
        previous = self._set_clipboard_text(query)
        try:
            root.set_focus()
            search.click_input()
            # Clear and paste as two settled operations. Rekordbox applies its
            # browser filter asynchronously; a combined Ctrl+A/Ctrl+V could
            # leave the previous result rows visible long enough to be
            # mistaken for the new query, and restoring the clipboard
            # immediately could race the paste on busy UI threads.
            send_keys("^a{BACKSPACE}")
            time.sleep(0.08)
            send_keys("^v")
            time.sleep(0.12)
        finally:
            self._restore_clipboard_text(previous)

        # Rekordbox updates the result grid asynchronously. A short fixed
        # settle avoids accepting geometry left over from the previous query.
        time.sleep(0.25)
        deadline = time.monotonic() + timeout_seconds
        rows: list[int] = []
        result_samples: list[ControlSample] = []
        while time.monotonic() < deadline:
            descendants = root.descendants()
            result_samples, window_width = self._sample_browser_rows(
                root,
                descendants,
            )
            rows = self._result_rows_from_samples(
                result_samples,
                window_width,
            )
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
        row_x, row_y = self._result_row_point_from_samples(
            result_samples,
            window,
            rows[result_index],
        )
        deck_drop_points = {}
        for deck in (1, 2):
            deck_samples = [
                sample
                for sample in samples
                if sample.control_type == "Text"
                and 260 <= sample.top <= 330
                and sample.right - sample.left >= window.width() * 0.2
                and (
                    (deck == 1 and (sample.left + sample.right) / 2 < window.width() / 2)
                    or (deck == 2 and (sample.left + sample.right) / 2 >= window.width() / 2)
                )
            ]
            if deck_samples:
                target = max(deck_samples, key=lambda item: item.right - item.left)
                deck_drop_points[deck] = (
                    window.left + (target.left + target.right) // 2,
                    window.top + (target.top + target.bottom) // 2,
                )
        root.click_input(coords=(row_x - window.left, row_y - window.top))
        return {
            "title": title,
            "search_query": query,
            "result_count": len(rows),
            "result_index": result_index,
            "selected_row_top": rows[result_index],
            "selected_row_point": (row_x, row_y),
            "deck_drop_points": deck_drop_points,
            "focused": True,
        }

    def drag_selected_track_to_deck(
        self,
        *,
        selected_row_top: int,
        deck: int,
        selected_row_point: tuple[int, int] | None = None,
        deck_drop_point: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Load the selected browser row by dragging it onto a stopped deck."""
        if deck not in (1, 2):
            raise ValueError("deck must be 1 or 2")
        root = self._root()
        window = root.rectangle()
        source = selected_row_point or self._result_row_point(
            root,
            selected_row_top,
        )
        target = deck_drop_point or (
            window.left + int(window.width() * (0.25 if deck == 1 else 0.75)),
            window.top + min(310, int(window.height() * 0.3)),
        )
        root.set_focus()
        mouse.move(coords=source)
        mouse.press(button="left", coords=source)
        time.sleep(0.25)
        for step in range(1, 7):
            point = (
                source[0] + (target[0] - source[0]) * step // 6,
                source[1] + (target[1] - source[1]) * step // 6,
            )
            mouse.move(coords=point)
            time.sleep(0.06)
        mouse.release(button="left", coords=target)
        return {
            "deck": deck,
            "source": source,
            "target": target,
            "method": "drag_to_deck",
        }

    def _deck_controls(
        self,
        root,
        deck: int,
        descendants: list[Any] | None = None,
    ) -> list[Any]:
        if deck not in (1, 2):
            raise ValueError("deck must be 1 or 2")
        window = root.rectangle()
        width = window.width()
        midpoint = width / 2
        controls = []
        for control in descendants or root.descendants():
            rectangle = control.rectangle()
            left = rectangle.left - window.left
            right = rectangle.right - window.left
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

    def _pc_master_out_button(self, root, descendants=None):
        """Locate Rekordbox's top-bar PC MASTER OUT toggle.

        Rekordbox exposes this icon as an unnamed button.  Anchor it between
        the named Free-plan button and AudioCpuGraphButton so the lookup is
        independent of absolute screen resolution.
        """
        descendants = descendants or root.descendants()
        free = next(
            (
                control
                for control in descendants
                if control.element_info.control_type == "Button"
                and control.window_text().startswith("Free")
            ),
            None,
        )
        cpu = next(
            (
                control
                for control in descendants
                if control.element_info.control_type == "Button"
                and control.window_text() == "AudioCpuGraphButton"
            ),
            None,
        )
        if free is None or cpu is None:
            raise RuntimeError("Unable to locate Rekordbox audio-route controls")
        candidates = [
            control
            for control in descendants
            if control.element_info.control_type == "Button"
            and not control.window_text()
            and free.rectangle().right < control.rectangle().left
            and control.rectangle().right < cpu.rectangle().left
            and control.rectangle().top < cpu.rectangle().bottom + 12
        ]
        if not candidates:
            raise RuntimeError("Unable to locate Rekordbox PC MASTER OUT")
        return max(candidates, key=lambda control: control.rectangle().left)

    def pc_master_out_enabled(
        self,
        root=None,
        image: Image.Image | None = None,
        descendants=None,
    ) -> bool:
        root = root or self._root()
        image = image or self._capture_focused(root)
        return self._button_active(
            root,
            self._pc_master_out_button(root, descendants),
            image,
        )

    def ensure_pc_master_out(self, enabled: bool) -> dict[str, Any]:
        root = self._root()
        descendants = self._status_controls(root)
        image = self._capture_focused(root)
        button = self._pc_master_out_button(root, descendants)
        before = self._button_active(root, button, image)
        if before != enabled:
            button.click_input()
            time.sleep(0.35)
        after = self.pc_master_out_enabled(root, descendants=descendants)
        if after != enabled:
            raise RuntimeError(
                "Rekordbox PC MASTER OUT did not reach the requested state"
            )
        return {"before": before, "after": after, "changed": before != after}

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
        samples: list[ControlSample],
        image: Image.Image,
        deck: int,
        window_width: int,
    ) -> DeckSnapshot:
        midpoint = window_width / 2
        title_candidates = []
        artist_candidates = []
        native_bpm = None
        live_bpm = None
        pitch_percent = None
        key = None
        elapsed = None
        sync = None
        quantize = None
        master = None
        stems: dict[str, bool | None] = {
            "VOCAL": None,
            "INST": None,
            "DRUMS": None,
        }
        for sample in samples:
            left = sample.left
            top = sample.top
            right = sample.right
            text = sample.text
            control_type = sample.control_type
            center = (left + right) / 2
            if (deck == 1 and center >= midpoint) or (
                deck == 2 and center < midpoint
            ):
                continue
            if (
                control_type == "Text"
                and 260 <= top <= 310
                and right - left >= window_width * 0.2
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
                            native_bpm = value
                else:
                    elapsed = parsed
            if control_type == "Text" and 295 <= top <= 325 and text:
                stripped = text.strip()
                try:
                    numeric = float(stripped)
                except ValueError:
                    numeric = None
                if numeric is not None and 40 <= numeric <= 500:
                    native_bpm = numeric
                elif re.fullmatch(
                    r"(?:\d{1,2}[AB]|[A-G](?:#|b)?)",
                    stripped,
                ):
                    key = stripped
                elif not stripped.startswith("-") and ":" not in stripped:
                    artist_candidates.append((left, stripped))
            if text == "BEAT\nSYNC":
                sync = blue_ratio(
                    image.crop((left, sample.top, right, sample.bottom))
                ) >= 0.25
            if text == "MASTER" and control_type == "Button":
                master = blue_ratio(
                    image.crop((left, sample.top, right, sample.bottom))
                ) >= 0.08
            if text == "Q" and 420 <= top <= 500:
                quantize = blue_ratio(
                    image.crop((left, sample.top, right, sample.bottom))
                ) >= 0.25
            if (
                control_type == "Button"
                and 360 <= top <= 410
                and text in stems
            ):
                stems[text] = vivid_color_ratio(
                    image.crop((left, sample.top, right, sample.bottom))
                ) >= 0.08
            # Rekordbox exposes the large jog-display BPM separately from the
            # analyzed/native BPM in the metadata row.  This is the scheduler
            # clock after tempo and Beat Sync are applied.
            if control_type in {"Text", "Edit"} and 430 <= top < 470 and text:
                try:
                    numeric = float(text.strip())
                except ValueError:
                    numeric = None
                if numeric is not None and 40 <= numeric <= 500:
                    live_bpm = numeric
            # On the left deck Rekordbox's accessibility tree can omit the
            # large live BPM but still exposes the jog pitch percentage.
            if control_type == "Text" and 465 <= top <= 495 and text:
                try:
                    numeric = float(text.strip())
                except ValueError:
                    numeric = None
                if numeric is not None and -25 <= numeric <= 25:
                    pitch_percent = numeric
        title = max(title_candidates, default=(0, ""))[1]
        artist = min(artist_candidates, default=(0, ""))[1]
        bpm = live_bpm
        if bpm is None and native_bpm is not None and pitch_percent is not None:
            bpm = round(native_bpm * (1.0 + pitch_percent / 100.0), 2)
        if bpm is None:
            bpm = native_bpm
        return DeckSnapshot(
            deck=deck,
            title=title,
            artist=artist,
            bpm=bpm,
            key=key,
            elapsed_seconds=elapsed,
            beat_sync_enabled=sync,
            quantize_enabled=quantize,
            master_enabled=master,
            stem_vocal_enabled=stems["VOCAL"],
            stem_instrumental_enabled=stems["INST"],
            stem_drums_enabled=stems["DRUMS"],
        )

    def deck_snapshot(self, deck: int) -> DeckSnapshot:
        root = self._root()
        descendants = self._status_controls(root)
        samples, window_width = self._sample_controls(root, descendants)
        image = self._capture_focused(root)
        return self._deck_snapshot(samples, image, deck, window_width)

    def deck_is_playing(
        self,
        deck: int,
        *,
        sample_seconds: float = 1.1,
        initial: DeckSnapshot | None = None,
    ) -> bool:
        first = initial or self.deck_snapshot(deck)
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
        last: DeckSnapshot | None = None
        last_error: RuntimeError | None = None
        while time.monotonic() < deadline:
            try:
                last = self.deck_snapshot(deck)
                last_error = None
            except RuntimeError as exc:
                last_error = exc
                time.sleep(0.1)
                continue
            if normalize_title(last.title) == normalize_title(expected_title):
                return last
            time.sleep(0.25)
        if (
            last is not None
            and normalize_title(last.title) == normalize_title(expected_title)
        ):
            return last
        if last is None and last_error is not None:
            raise RuntimeError(
                f"Deck {deck} could not be observed while waiting for "
                f"{expected_title!r}: {last_error}"
            ) from last_error
        raise RuntimeError(
            f"Deck {deck} loaded {last.title!r}, expected {expected_title!r}"
        )

    def status(self) -> dict[str, Any]:
        root = self._root()
        descendants = self._status_controls(root)
        samples, window_width = self._sample_controls(root, descendants)
        image = self._capture_focused(root)
        try:
            pc_master_out_enabled = self.pc_master_out_enabled(
                root,
                image,
                descendants,
            )
        except RuntimeError:
            pc_master_out_enabled = None
        return {
            "mode": self._mode_from_samples(samples),
            "pc_master_out_enabled": pc_master_out_enabled,
            "bar_alignment": analyze_bar_grid_alignment(image),
            "decks": [
                self._deck_snapshot(
                    samples,
                    image,
                    deck,
                    window_width,
                ).public()
                for deck in (1, 2)
            ],
        }
