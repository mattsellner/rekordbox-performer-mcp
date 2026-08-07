"""Read-only Rekordbox playlist discovery for the standalone application."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pyrekordbox


@dataclass(frozen=True)
class PlaylistSummary:
    playlist_id: str
    name: str
    track_ids: tuple[str, ...]
    ready_track_ids: tuple[str, ...]

    @property
    def track_count(self) -> int:
        return len(self.track_ids)

    @property
    def ready_count(self) -> int:
        return len(self.ready_track_ids)

    @property
    def label(self) -> str:
        return f"{self.name}  ({self.ready_count}/{self.track_count} ready)"


class RekordboxPlaylistCatalog:
    """Expose playlist membership without mutating Rekordbox's database."""

    @staticmethod
    def _close(database: Any) -> None:
        close = getattr(database, "close", None)
        if callable(close):
            close()

    def list_playlists(self, ready_track_ids: Iterable[str]) -> list[PlaylistSummary]:
        ready = set(ready_track_ids)
        database = pyrekordbox.Rekordbox6Database()
        try:
            result = []
            for playlist in database.get_playlist():
                if getattr(playlist, "rb_local_deleted", 0):
                    continue
                is_folder = bool(getattr(playlist, "is_folder", False)) or (
                    getattr(playlist, "Attribute", 0) == 1
                )
                if is_folder:
                    continue
                songs = [
                    song
                    for song in database.get_playlist_songs(PlaylistID=playlist.ID)
                    if not getattr(song, "rb_local_deleted", 0)
                ]
                songs.sort(key=lambda song: getattr(song, "TrackNo", 0) or 0)
                track_ids = tuple(str(song.ContentID) for song in songs)
                if not track_ids:
                    continue
                result.append(
                    PlaylistSummary(
                        playlist_id=str(playlist.ID),
                        name=str(playlist.Name or "Untitled playlist"),
                        track_ids=track_ids,
                        ready_track_ids=tuple(
                            track_id for track_id in track_ids if track_id in ready
                        ),
                    )
                )
            return sorted(result, key=lambda item: item.name.casefold())
        finally:
            self._close(database)
