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


def test_native_mix_point_controls_are_deck_scoped() -> None:
    quantize = encode_action("quantize", {"deck": 2})[0].message
    mix_point = encode_action("mix_point_set", {"deck": 1})[0].message
    assert quantize.channel == 1
    assert quantize.note == 11
    assert mix_point.channel == 0
    assert mix_point.note == 98


def test_flx4_browser_encoder_actions_are_global() -> None:
    up = encode_action("browse_up")[0].message
    down = encode_action("browse_down")[0].message
    assert (up.channel, up.note) == (2, 0)
    assert (down.channel, down.note) == (2, 1)


def test_flx4_load_buttons_are_global_and_deck_specific() -> None:
    load_1 = encode_action("load_deck_1")[0].message
    load_2 = encode_action("load_deck_2")[0].message
    assert (load_1.channel, load_1.note) == (2, 2)
    assert (load_2.channel, load_2.note) == (2, 3)


def test_full_fx_actions_have_stable_controls() -> None:
    select_next = encode_action("fx_select_next", {"deck": 1})[0].message
    select_back = encode_action("fx_select_back", {"deck": 2})[0].message
    beat_up = encode_action("fx_beat_up", {"deck": 1})[0].message
    assert (select_next.channel, select_next.note) == (0, 40)
    assert (select_back.channel, select_back.note) == (1, 41)
    assert (beat_up.channel, beat_up.note) == (0, 42)
