"""Isolated Rekordbox status capture for phrase-critical sync guards."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .rekordbox_ui import RekordboxUIAdapter


def capture_to(path: str | Path) -> int:
    destination = Path(path)
    try:
        status = RekordboxUIAdapter().status()
        payload = {"ok": True, "status": status}
        exit_code = 0
    except Exception as exc:  # noqa: BLE001 - subprocess boundary
        payload = {"ok": False, "error": str(exc)}
        exit_code = 1
    destination.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return exit_code


def capture_pair_to(
    path: str | Path,
    *,
    capture_at_monotonic: float,
    sample_delay_seconds: float = 0.35,
) -> int:
    """Warm one helper process, then capture two pre-retirement frames."""
    destination = Path(path)
    try:
        time.sleep(max(0.0, capture_at_monotonic - time.monotonic()))
        adapter = RekordboxUIAdapter()
        adapter.invalidate_status_cache()
        first = adapter.status()
        time.sleep(sample_delay_seconds)
        second = adapter.status()
        payload = {"ok": True, "statuses": [first, second]}
        exit_code = 0
    except Exception as exc:  # noqa: BLE001 - subprocess boundary
        payload = {"ok": False, "error": str(exc)}
        exit_code = 1
    destination.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return exit_code


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--pair", action="store_true")
    parser.add_argument("--capture-at-monotonic", type=float)
    args = parser.parse_args(argv)
    if args.pair:
        if args.capture_at_monotonic is None:
            parser.error("--pair requires --capture-at-monotonic")
        return capture_pair_to(
            args.output,
            capture_at_monotonic=args.capture_at_monotonic,
        )
    return capture_to(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
