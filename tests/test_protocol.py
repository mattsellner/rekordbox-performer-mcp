import pytest

from rekordbox_performer.protocol import ProtocolError, encode_action


def test_deck_play_note() -> None:
    messages = encode_action("play_pause", {"deck": 2})
    assert messages[0].message.type == "note_on"
    assert messages[0].message.channel == 1
    assert messages[0].message.note == 0
    assert messages[1].message.type == "note_on"
    assert messages[1].message.velocity == 0


def test_eq_center_maps_to_64() -> None:
    message = encode_action("eq_low", {"deck": 1, "value": 0})[0].message
    assert message.type == "control_change"
    assert message.control == 4
    assert message.value == 64


def test_crossfader_extremes() -> None:
    left = encode_action("crossfader", {"value": -1})[0].message
    right = encode_action("crossfader", {"value": 1})[0].message
    assert left.value == 0
    assert right.value == 127


def test_hot_cue_bounds() -> None:
    assert encode_action("hot_cue", {"deck": 1, "cue": 8})[0].message.note == 23
    with pytest.raises(ProtocolError):
        encode_action("hot_cue", {"deck": 1, "cue": 9})


def test_invalid_deck_rejected() -> None:
    with pytest.raises(ProtocolError):
        encode_action("cue", {"deck": 3})
