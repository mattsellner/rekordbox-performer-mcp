from __future__ import annotations

from PIL import Image
import pytest

from rekordbox_performer.rekordbox_ui import (
    blue_ratio,
    normalize_title,
    parse_clock_seconds,
    unique_row_tops,
)


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


def test_blue_ratio_distinguishes_active_rekordbox_blue() -> None:
    active = Image.new("RGB", (10, 10), (32, 122, 235))
    inactive = Image.new("RGB", (10, 10), (78, 78, 78))
    assert blue_ratio(active) > 0.9
    assert blue_ratio(inactive) == 0
