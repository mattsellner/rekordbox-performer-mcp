"""Bounded, isolated Rekordbox browser staging for the rolling set runner."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .rekordbox_ui import RekordboxUIAdapter, normalize_title


def _queries(title: str, artist: str | None) -> list[str]:
    apostrophe_safe = title.translate(
        str.maketrans({"\u2018": "'", "\u2019": "'", "\uff07": "'"})
    )
    result: list[str] = []
    for query in (
        f"{title} {artist}" if artist else title,
        title,
        f"{apostrophe_safe} {artist}" if artist else apostrophe_safe,
        apostrophe_safe,
        re.sub(r"\s*\([^)]*\)\s*$", "", title).strip(),
    ):
        if query and query not in result:
            result.append(query)
    return result


def stage_to(
    path: str | Path,
    *,
    deck: int,
    title: str,
    artist: str | None = None,
    adapter: RekordboxUIAdapter | None = None,
) -> int:
    """Stage one stopped deck through drag-and-drop and persist the proof."""
    destination = Path(path)
    ui = adapter or RekordboxUIAdapter()
    try:
        if deck not in {1, 2}:
            raise ValueError("deck must be 1 or 2")
        resident = ui.deck_snapshot(deck)
        if ui.deck_is_playing(deck, initial=resident):
            raise RuntimeError(f"Refusing to load playing deck {deck}")
        if (
            normalize_title(resident.title) == normalize_title(title)
            and (
                artist is None
                or normalize_title(resident.artist) == normalize_title(artist)
            )
            and resident.elapsed_seconds is not None
            and resident.elapsed_seconds <= 1
        ):
            payload: dict[str, Any] = {
                "ok": True,
                "verified": True,
                "already_loaded": True,
                "browser_search_skipped": True,
                "observed": resident.public(),
                "attempts": [],
            }
        else:
            selection = None
            empty_searches: list[str] = []
            for query in _queries(title, artist):
                try:
                    selection = ui.select_exact_track(
                        title,
                        search_query=query,
                        result_index=0,
                    )
                    break
                except RuntimeError as exc:
                    if "returned no visible rows" not in str(exc):
                        raise
                    empty_searches.append(query)
            if selection is None:
                raise RuntimeError(
                    f"No visible Rekordbox rows for {title!r}; "
                    f"tried {empty_searches!r}"
                )
            attempts: list[dict[str, Any]] = []
            observed = None
            chosen = None
            for candidate_index in range(int(selection["result_count"])):
                candidate = (
                    selection
                    if candidate_index == 0
                    else ui.select_exact_track(
                        title,
                        search_query=str(selection["search_query"]),
                        result_index=candidate_index,
                    )
                )
                drag = ui.drag_selected_track_to_deck(
                    selected_row_top=candidate["selected_row_top"],
                    deck=deck,
                    selected_row_point=candidate.get("selected_row_point"),
                    deck_drop_point=candidate.get("deck_drop_points", {}).get(deck),
                )
                try:
                    candidate_observed = ui.wait_for_deck_title(
                        deck,
                        title,
                        timeout_seconds=6.0,
                    )
                except RuntimeError as exc:
                    attempts.append(
                        {
                            "result_index": candidate_index,
                            "method": drag["method"],
                            "error": str(exc),
                        }
                    )
                    continue
                artist_matches = artist is None or normalize_title(
                    candidate_observed.artist
                ) == normalize_title(artist)
                attempts.append(
                    {
                        "result_index": candidate_index,
                        "method": drag["method"],
                        "observed": candidate_observed.public(),
                        "artist_matches": artist_matches,
                    }
                )
                if artist_matches:
                    observed = candidate_observed
                    chosen = candidate
                    break
            if observed is None or chosen is None:
                raise RuntimeError(
                    f"No visible {title!r} row loaded expected artist "
                    f"{artist!r}; observed attempts: {attempts}"
                )
            auto_started = ui.deck_is_playing(deck, initial=observed)
            payload = {
                "ok": not auto_started,
                "verified": not auto_started,
                "auto_started": auto_started,
                "already_loaded": False,
                "browser_search_skipped": False,
                "selection": chosen,
                "observed": observed.public(),
                "attempts": attempts,
                "error": (
                    f"Loaded deck {deck} auto-started"
                    if auto_started
                    else None
                ),
            }
        exit_code = 0 if payload.get("ok") else 1
    except Exception as exc:  # noqa: BLE001 - subprocess boundary
        payload = {"ok": False, "verified": False, "error": str(exc)}
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
    parser.add_argument("--deck", required=True, type=int)
    parser.add_argument("--title", required=True)
    parser.add_argument("--artist")
    args = parser.parse_args(argv)
    return stage_to(
        args.output,
        deck=args.deck,
        title=args.title,
        artist=args.artist,
    )


if __name__ == "__main__":
    raise SystemExit(main())
