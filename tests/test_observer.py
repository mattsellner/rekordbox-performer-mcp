import time
from pathlib import Path

from rekordbox_performer.observer import SharedDeckObserver


class FakeAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def status(self):
        self.calls += 1
        return {"mode": "PERFORMANCE", "decks": []}


class SlowAdapter(FakeAdapter):
    def status(self):
        time.sleep(1.6)
        return super().status()


def test_shared_observer_returns_fresh_cached_snapshot(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    observer = SharedDeckObserver(adapter, tmp_path, interval_seconds=0.05)
    try:
        first = observer.status(max_age_ms=500)
        second = observer.status(max_age_ms=500)
        assert first["mode"] == "PERFORMANCE"
        assert second["observer_pid"] == first["observer_pid"]
        assert second["age_ms"] <= 500
        assert adapter.calls >= 1
    finally:
        observer.close()


def test_shared_observer_waits_for_slow_rekordbox_scan(tmp_path: Path) -> None:
    adapter = SlowAdapter()
    observer = SharedDeckObserver(
        adapter,
        tmp_path,
        interval_seconds=0.05,
        idle_seconds=0.05,
    )
    try:
        snapshot = observer.status(max_age_ms=500)
        assert snapshot["mode"] == "PERFORMANCE"
        assert adapter.calls == 1
    finally:
        observer.close()


def test_shared_observer_marks_stale_snapshot_age(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    observer = SharedDeckObserver(adapter, tmp_path, interval_seconds=10)
    try:
        observer.status(max_age_ms=500)
        time.sleep(0.02)
        snapshot = observer.status(max_age_ms=500)
        assert snapshot["age_ms"] >= 0
    finally:
        observer.close()


def test_shared_observer_stops_after_idle_window(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    observer = SharedDeckObserver(
        adapter,
        tmp_path,
        interval_seconds=0.02,
        idle_seconds=0.08,
    )
    try:
        observer.status(max_age_ms=500)
        time.sleep(0.2)
        calls_after_idle = adapter.calls
        time.sleep(0.1)
        assert adapter.calls == calls_after_idle
        assert not observer.lease.held
    finally:
        observer.close()


def test_shared_observer_restarts_after_idle_window(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    observer = SharedDeckObserver(
        adapter,
        tmp_path,
        interval_seconds=0.02,
        idle_seconds=0.05,
    )
    try:
        first = observer.status(max_age_ms=500)
        time.sleep(0.12)
        calls_after_idle = adapter.calls
        second = observer.status(max_age_ms=50)
        assert adapter.calls > calls_after_idle
        assert second["observed_at"] > first["observed_at"]
    finally:
        observer.close()


def test_shared_observer_never_returns_stale_snapshot(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    observer = SharedDeckObserver(
        adapter,
        tmp_path,
        interval_seconds=10,
        idle_seconds=0.01,
    )
    observer.snapshot_path.write_text(
        '{"mode":"PERFORMANCE","decks":[],"observed_at":1}',
        encoding="utf-8",
    )
    observer._ensure_started = lambda: None
    try:
        try:
            observer.status(max_age_ms=1, wait_seconds=0.01)
        except RuntimeError as exc:
            assert "stale" in str(exc)
        else:
            raise AssertionError("Expected stale observation rejection")
    finally:
        observer.close()
