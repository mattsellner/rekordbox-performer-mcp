"""MIDI output, arming, and learn-mode helpers."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Protocol

import mido

from .intelligence import default_data_dir
from .protocol import CONTINUOUS_ACTIONS, encode_action
from .runtime import ControlLease


DEFAULT_PORT_NAME = "Codex Rekordbox Performer"


class MidiOutput(Protocol):
    name: str

    def send(self, message: mido.Message) -> None: ...

    def close(self) -> None: ...


class MidiEngine:
    def __init__(self, lease: ControlLease | None = None) -> None:
        self.output: MidiOutput | None = None
        self.port_name: str | None = None
        self.armed_until = 0.0
        self.job_armed_until = 0.0
        self.set_armed_until = 0.0
        self.sent_messages = 0
        self.continuous_control_state: dict[str, dict[str, Any]] = {}
        self.lease = lease or ControlLease(
            default_data_dir() / "live-midi-control.lock"
        )

    @staticmethod
    def list_output_ports() -> list[str]:
        return list(mido.get_output_names())

    def connect(self, requested_name: str | None = None) -> dict[str, Any]:
        requested_name = (
            requested_name
            or os.environ.get("REKORDBOX_MIDI_PORT")
            or DEFAULT_PORT_NAME
        )
        available = self.list_output_ports()
        exact = [name for name in available if name == requested_name]
        partial = [
            name for name in available if requested_name.casefold() in name.casefold()
        ]
        matches = exact or partial
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one MIDI output matching '{requested_name}', found "
                f"{len(matches)}. Available outputs: {available}"
            )
        if self.output is not None and self.port_name == matches[0]:
            return self.status()
        self.disconnect()
        self.lease.acquire()
        try:
            self.output = mido.open_output(matches[0])
            self.port_name = matches[0]
        except Exception:
            self.lease.release()
            raise
        return self.status()

    def disconnect(self) -> None:
        self.disarm()
        if self.output is not None:
            self.output.close()
        self.output = None
        self.port_name = None
        self.lease.release()

    def arm(self, seconds: int = 300) -> dict[str, Any]:
        if self.output is None:
            raise RuntimeError("Connect a MIDI output before arming control")
        if not 10 <= seconds <= 1800:
            raise ValueError("seconds must be between 10 and 1800")
        self.armed_until = time.monotonic() + seconds
        return self.status()

    def disarm(self) -> dict[str, Any]:
        self.armed_until = 0.0
        self.job_armed_until = 0.0
        self.set_armed_until = 0.0
        return self.status()

    def is_armed(self) -> bool:
        deadline = max(
            self.armed_until,
            self.job_armed_until,
            self.set_armed_until,
        )
        return self.output is not None and time.monotonic() < deadline

    def authorize_set(self, duration_seconds: float) -> float:
        """Reserve control for one explicitly started autonomous set."""
        if self.output is None or not self.is_armed():
            raise RuntimeError(
                "Connect and arm live MIDI before authorizing an autonomous set"
            )
        if duration_seconds <= 0 or duration_seconds > 4 * 60 * 60:
            raise ValueError("set duration must be between 1 second and 4 hours")
        self.set_armed_until = max(
            self.set_armed_until,
            time.monotonic() + duration_seconds,
        )
        return self.set_armed_until

    def release_set_control(self) -> None:
        self.set_armed_until = 0.0

    def reserve_job_control(self, duration_seconds: float) -> float:
        """Keep an already-authorized scheduled job armed through verification."""
        if not self.is_armed():
            raise RuntimeError("Live MIDI control must be armed before scheduling")
        if duration_seconds < 0 or duration_seconds > 20 * 60:
            raise ValueError("job control duration must be between 0 and 1200 seconds")
        self.job_armed_until = max(
            self.job_armed_until,
            time.monotonic() + duration_seconds + 15.0,
        )
        return self.job_armed_until

    def _require_output(self) -> MidiOutput:
        if self.output is None:
            raise RuntimeError("MIDI output is not connected")
        return self.output

    def _require_armed(self) -> MidiOutput:
        output = self._require_output()
        if not self.is_armed():
            raise RuntimeError("Live MIDI control is not armed")
        return output

    async def send_action(
        self,
        action: str,
        parameters: dict[str, Any] | None = None,
        *,
        require_armed: bool = True,
    ) -> list[str]:
        output = self._require_armed() if require_armed else self._require_output()
        encoded = encode_action(action, parameters)
        descriptions: list[str] = []
        for item in encoded:
            output.send(item.message)
            self.sent_messages += 1
            descriptions.append(str(item.message))
            if item.delay_after_ms:
                await asyncio.sleep(item.delay_after_ms / 1000)
        if action in CONTINUOUS_ACTIONS:
            parameters = dict(parameters or {})
            deck = parameters.get("deck")
            key = f"deck_{deck}.{action}" if deck is not None else action
            self.continuous_control_state[key] = {
                "value": float(parameters["value"]),
                "commanded_at": time.time(),
                "verification": "commanded_not_observed",
            }
        return descriptions

    async def send_learn_signal(
        self, action: str, parameters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send a recognizable signal while Rekordbox MIDI Learn is listening."""
        parameters = dict(parameters or {})
        if action in CONTINUOUS_ACTIONS:
            original = parameters.get("value")
            values = (
                (0.0, 1.0, 0.5)
                if action in {"channel_fader", "fx_wet_dry"}
                else (-1.0, 1.0, 0.0)
            )
            sent = []
            for value in values:
                parameters["value"] = value
                sent.extend(
                    await self.send_action(
                        action, parameters, require_armed=False
                    )
                )
                await asyncio.sleep(0.08)
            if original is not None:
                parameters["value"] = original
            return {"action": action, "messages": sent}
        sent = await self.send_action(action, parameters, require_armed=False)
        return {"action": action, "messages": sent}

    def status(self) -> dict[str, Any]:
        remaining = max(0.0, self.armed_until - time.monotonic())
        job_remaining = max(0.0, self.job_armed_until - time.monotonic())
        set_remaining = max(0.0, self.set_armed_until - time.monotonic())
        return {
            "connected": self.output is not None,
            "port_name": self.port_name,
            "armed": self.is_armed(),
            "armed_seconds_remaining": round(remaining, 1),
            "job_armed_seconds_remaining": round(job_remaining, 1),
            "set_armed_seconds_remaining": round(set_remaining, 1),
            "sent_messages": self.sent_messages,
            "continuous_control_state": dict(self.continuous_control_state),
            "control_lease": self.lease.status(),
        }
