"""Stable virtual-MIDI contract used by Rekordbox MIDI Learn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mido import Message


class ProtocolError(ValueError):
    """Raised when a control action is invalid."""


@dataclass(frozen=True)
class EncodedMessage:
    message: Message
    delay_after_ms: int = 0


DECK_NOTES = {
    "play_pause": 0,
    "cue": 1,
    "sync": 2,
    "master": 3,
    "load": 4,
    "loop_toggle": 5,
    "loop_half": 6,
    "loop_double": 7,
    "loop_4": 8,
    "loop_8": 9,
    "loop_16": 10,
    "quantize": 11,
    "fx_toggle": 32,
    "mix_point_previous": 96,
    "mix_point_next": 97,
    "mix_point_set": 98,
}

DECK_CONTROLS = {
    "channel_fader": (0, "unit"),
    "gain": (1, "bipolar"),
    "eq_high": (2, "bipolar"),
    "eq_mid": (3, "bipolar"),
    "eq_low": (4, "bipolar"),
    "filter": (5, "bipolar"),
    "tempo": (6, "bipolar"),
    "fx_wet_dry": (32, "unit"),
}

GLOBAL_NOTES = {
    "browse_up": 0,
    "browse_down": 1,
    "load_deck_1": 2,
    "load_deck_2": 3,
    "browser_back": 4,
    "browser_forward": 5,
}

GLOBAL_CONTROLS = {
    "crossfader": (0, "bipolar"),
}

TRIGGER_ACTIONS = frozenset((*DECK_NOTES, "hot_cue", *GLOBAL_NOTES))
CONTINUOUS_ACTIONS = frozenset((*DECK_CONTROLS, *GLOBAL_CONTROLS))
ALL_ACTIONS = frozenset((*TRIGGER_ACTIONS, *CONTINUOUS_ACTIONS))


def _deck_channel(parameters: dict[str, Any]) -> int:
    deck = int(parameters.get("deck", 0))
    if deck not in (1, 2):
        raise ProtocolError("deck must be 1 or 2")
    return deck - 1


def _scale_value(mode: str, raw: Any) -> int:
    value = float(raw)
    if mode == "unit":
        if not 0.0 <= value <= 1.0:
            raise ProtocolError("value must be between 0.0 and 1.0")
        return round(value * 127)
    if mode == "bipolar":
        if not -1.0 <= value <= 1.0:
            raise ProtocolError("value must be between -1.0 and 1.0")
        return round((value + 1.0) * 63.5)
    raise ProtocolError(f"unknown value mode: {mode}")


def encode_action(
    action: str,
    parameters: dict[str, Any] | None = None,
    note_hold_ms: int = 80,
) -> list[EncodedMessage]:
    """Encode a named action into one or more MIDI messages."""
    parameters = dict(parameters or {})
    if action not in ALL_ACTIONS:
        raise ProtocolError(
            f"unknown action '{action}'. Valid actions: {', '.join(sorted(ALL_ACTIONS))}"
        )

    if action in DECK_NOTES or action == "hot_cue":
        channel = _deck_channel(parameters)
        if action == "hot_cue":
            cue = int(parameters.get("cue", 0))
            if not 1 <= cue <= 8:
                raise ProtocolError("cue must be between 1 and 8")
            note = 15 + cue
        else:
            note = DECK_NOTES[action]
        return [
            EncodedMessage(
                Message("note_on", channel=channel, note=note, velocity=127),
                delay_after_ms=note_hold_ms,
            ),
            # Rekordbox controller mappings expect button release as Note On
            # velocity 0 on the same status byte, matching Pioneer hardware.
            EncodedMessage(
                Message("note_on", channel=channel, note=note, velocity=0)
            ),
        ]

    if action in DECK_CONTROLS:
        channel = _deck_channel(parameters)
        control, mode = DECK_CONTROLS[action]
        midi_value = _scale_value(mode, parameters.get("value"))
        return [
            EncodedMessage(
                Message(
                    "control_change",
                    channel=channel,
                    control=control,
                    value=midi_value,
                )
            )
        ]

    if action in GLOBAL_NOTES:
        note = GLOBAL_NOTES[action]
        channel = 2
        return [
            EncodedMessage(
                Message("note_on", channel=channel, note=note, velocity=127),
                delay_after_ms=note_hold_ms,
            ),
            EncodedMessage(
                Message("note_on", channel=channel, note=note, velocity=0)
            ),
        ]

    control, mode = GLOBAL_CONTROLS[action]
    midi_value = _scale_value(mode, parameters.get("value"))
    return [
        EncodedMessage(
            Message(
                "control_change",
                channel=3,
                control=control,
                value=midi_value,
            )
        )
    ]


def mapping_manifest() -> list[dict[str, Any]]:
    """Return every stable MIDI assignment for the Rekordbox mapping workflow."""
    rows: list[dict[str, Any]] = []
    for deck in (1, 2):
        for action, note in DECK_NOTES.items():
            rows.append(
                {
                    "action": action,
                    "deck": deck,
                    "midi_channel": deck,
                    "message": "note",
                    "number": note,
                    "value_range": "trigger",
                }
            )
        for cue in range(1, 9):
            rows.append(
                {
                    "action": "hot_cue",
                    "deck": deck,
                    "cue": cue,
                    "midi_channel": deck,
                    "message": "note",
                    "number": 15 + cue,
                    "value_range": "trigger",
                }
            )
        for action, (control, mode) in DECK_CONTROLS.items():
            rows.append(
                {
                    "action": action,
                    "deck": deck,
                    "midi_channel": deck,
                    "message": "control_change",
                    "number": control,
                    "value_range": "0..1" if mode == "unit" else "-1..1",
                }
            )
    for action, note in GLOBAL_NOTES.items():
        rows.append(
            {
                "action": action,
                "deck": None,
                "midi_channel": 3,
                "message": "note",
                "number": note,
                "value_range": "trigger",
            }
        )
    for action, (control, mode) in GLOBAL_CONTROLS.items():
        rows.append(
            {
                "action": action,
                "deck": None,
                "midi_channel": 4,
                "message": "control_change",
                "number": control,
                "value_range": "0..1" if mode == "unit" else "-1..1",
            }
        )
    return rows
