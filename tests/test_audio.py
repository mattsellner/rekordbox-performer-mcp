from pathlib import Path

import numpy as np
import soundfile as sf

from rekordbox_performer.audio import analyze_rehearsal_audio


def test_rehearsal_audio_analysis(tmp_path: Path) -> None:
    sample_rate = 48_000
    audio = np.zeros((sample_rate * 4, 2), dtype=np.float32)
    for second in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        start = int(second * sample_rate)
        audio[start : start + 240, :] = 0.8
    path = tmp_path / "clicks.wav"
    sf.write(path, audio, sample_rate)

    result = analyze_rehearsal_audio(
        str(path),
        120,
        expected_downbeats_ms=[1000, 2000, 3000],
    )
    assert result["duration_seconds"] == 4.0
    assert result["clipping_percent"] == 0.0
    assert result["mean_downbeat_error_ms"] <= 10
