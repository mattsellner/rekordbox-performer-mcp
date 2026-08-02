from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "mapping" / "Codex-Rekordbox-Performer.midi.csv"


def test_mapping_has_rekordbox_shape() -> None:
    with MAPPING.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0][:3] == ["@file", "1", "Codex Rekordbox Performer"]
    assert rows[1] == [
        "#name", "function", "type", "input",
        "deck1", "deck2", "deck3", "deck4",
        "output", "deck1", "deck2", "deck3", "deck4",
        "option", "comment",
    ]
    assert all(len(row) == 15 for row in rows)

    functions = {row[1] for row in rows[2:]}
    names = {row[0] for row in rows[2:]}
    assert {
        "PlayPause", "Cue", "Sync", "ChannelFader", "Gain",
        "EQHigh", "EQMid", "EQLow", "TempoSlider", "CrossFader",
        "Quantize",
    } <= functions
    assert {
        "MixPointSelectPrev", "MixPointSelectNext", "MixPointLink",
        "ActivePartVocal", "ActivePartInst", "ActivePartDrums",
    } <= names
