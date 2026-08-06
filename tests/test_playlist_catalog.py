from types import SimpleNamespace

from rekordbox_performer.playlist_catalog import RekordboxPlaylistCatalog


def test_playlist_catalog_preserves_order_and_reports_readiness(monkeypatch) -> None:
    playlists = [
        SimpleNamespace(
            ID="playlist-1",
            Name="Friday Set",
            Attribute=0,
            is_folder=False,
            rb_local_deleted=0,
        ),
        SimpleNamespace(
            ID="folder-1",
            Name="Folder",
            Attribute=1,
            is_folder=True,
            rb_local_deleted=0,
        ),
    ]
    songs = [
        SimpleNamespace(ContentID="b", TrackNo=2, rb_local_deleted=0),
        SimpleNamespace(ContentID="a", TrackNo=1, rb_local_deleted=0),
        SimpleNamespace(ContentID="deleted", TrackNo=3, rb_local_deleted=1),
    ]

    class FakeDatabase:
        closed = False

        def get_playlist(self):
            return playlists

        def get_playlist_songs(self, *, PlaylistID):
            assert PlaylistID == "playlist-1"
            return songs

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        "rekordbox_performer.playlist_catalog.pyrekordbox.Rekordbox6Database",
        FakeDatabase,
    )

    result = RekordboxPlaylistCatalog().list_playlists({"a"})

    assert len(result) == 1
    assert result[0].name == "Friday Set"
    assert result[0].track_ids == ("a", "b")
    assert result[0].ready_track_ids == ("a",)
    assert result[0].label == "Friday Set  (1/2 ready)"
