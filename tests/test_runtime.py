from pathlib import Path

import pytest

from rekordbox_performer.runtime import ControlLease


def test_control_lease_allows_only_one_owner(tmp_path: Path) -> None:
    first = ControlLease(tmp_path / "control.lock")
    second = ControlLease(tmp_path / "control.lock")

    first.acquire()
    try:
        assert first.status()["held_by_this_process"] is True
        assert first.status()["owner"]["pid"]
        with pytest.raises(RuntimeError, match="owns live MIDI control"):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    assert second.status()["held_by_this_process"] is True
    second.release()
