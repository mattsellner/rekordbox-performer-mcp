"""Explicitly started WASAPI loopback capture and lightweight rehearsal analysis."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any


class RehearsalCapture:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._chunks: list[Any] = []
        self._sample_rate = 48_000
        self._name = ""
        self._started_at: float | None = None
        self._error: str | None = None

    def status(self) -> dict[str, Any]:
        return {
            "recording": bool(self._thread and self._thread.is_alive()),
            "name": self._name or None,
            "started_at": self._started_at,
            "error": self._error,
        }

    def start(self, name: str, sample_rate: int = 48_000) -> dict[str, Any]:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("A rehearsal capture is already running")
        try:
            import numpy  # noqa: F401
            import soundcard  # noqa: F401
            import soundfile  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Install the rehearsal extra: pip install -e .[rehearsal]"
            ) from exc
        self._name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in name.strip()
        )[:80] or "rehearsal"
        self._sample_rate = sample_rate
        self._chunks = []
        self._stop.clear()
        self._error = None
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()
        # Surface endpoint/open failures during the tool call instead of
        # reporting a misleading successful start that dies immediately.
        time.sleep(0.15)
        if not self._thread.is_alive() and self._error:
            raise RuntimeError(self._error)
        return self.status()

    def _record(self) -> None:
        try:
            import soundcard as sc

            speaker = sc.default_speaker()
            if speaker is None:
                raise RuntimeError("No default Windows speaker was found")
            microphone = sc.get_microphone(
                # Names can collide with a physical microphone endpoint (the
                # FLX4 exposes both as "Line"). The render endpoint ID selects
                # the actual WASAPI loopback device unambiguously.
                id=str(speaker.id),
                include_loopback=True,
            )
            with microphone.recorder(
                samplerate=self._sample_rate,
                channels=2,
            ) as recorder:
                while not self._stop.is_set():
                    self._chunks.append(recorder.record(numframes=4800))
        except Exception as exc:
            self._error = str(exc)

    def stop(self) -> dict[str, Any]:
        if not self._thread:
            raise RuntimeError("No rehearsal capture has been started")
        self._stop.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("Rehearsal capture did not stop cleanly")
        if self._error:
            raise RuntimeError(self._error)
        if not self._chunks:
            raise RuntimeError("Rehearsal capture produced no audio")

        import numpy as np
        import soundfile as sf

        audio = np.concatenate(self._chunks, axis=0)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = self.data_dir / f"{timestamp}-{self._name}.wav"
        sf.write(path, audio, self._sample_rate, subtype="PCM_24")
        result = {
            "path": str(path),
            "duration_seconds": round(len(audio) / self._sample_rate, 3),
            "sample_rate": self._sample_rate,
            "channels": int(audio.shape[1]) if audio.ndim > 1 else 1,
        }
        self._thread = None
        self._chunks = []
        return result


def analyze_rehearsal_audio(
    path: str,
    bpm: float,
    expected_downbeats_ms: list[float] | None = None,
) -> dict[str, Any]:
    """Measure level, clipping, onset regularity, and expected-downbeat error."""
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    hop = max(1, round(sample_rate * 0.01))
    usable = len(mono) - (len(mono) % hop)
    frames = mono[:usable].reshape(-1, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    onset = np.maximum(0.0, np.diff(rms, prepend=rms[0]))

    beat_period_frames = max(1, round((60.0 / bpm) / (hop / sample_rate)))
    phase_scores = [
        float(onset[phase::beat_period_frames].sum())
        for phase in range(beat_period_frames)
    ]
    best_phase = int(np.argmax(phase_scores))
    beat_onsets = onset[best_phase::beat_period_frames]
    regularity = (
        float(np.mean(beat_onsets) / (np.std(beat_onsets) + 1e-9))
        if len(beat_onsets)
        else 0.0
    )

    onset_threshold = float(np.percentile(onset, 85))
    peak_indices = np.flatnonzero(onset >= onset_threshold)
    peak_times_ms = peak_indices * hop * 1000.0 / sample_rate
    downbeat_errors = []
    for expected in expected_downbeats_ms or []:
        if len(peak_times_ms) == 0:
            break
        downbeat_errors.append(float(np.min(np.abs(peak_times_ms - expected))))

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    clipping = float(np.mean(np.abs(audio) >= 0.999) * 100.0)
    return {
        "path": str(Path(path)),
        "duration_seconds": round(len(audio) / sample_rate, 3),
        "sample_rate": sample_rate,
        "bpm_reference": bpm,
        "estimated_beat_phase_offset_ms": round(
            best_phase * hop * 1000.0 / sample_rate,
            2,
        ),
        "onset_regularity": round(regularity, 3),
        "peak_dbfs": round(20.0 * float(np.log10(max(peak, 1e-9))), 2),
        "clipping_percent": round(clipping, 5),
        "expected_downbeat_count": len(expected_downbeats_ms or []),
        "mean_downbeat_error_ms": (
            round(float(np.mean(downbeat_errors)), 2)
            if downbeat_errors
            else None
        ),
        "max_downbeat_error_ms": (
            round(float(np.max(downbeat_errors)), 2)
            if downbeat_errors
            else None
        ),
        "vocal_clash": "manual review required",
    }
