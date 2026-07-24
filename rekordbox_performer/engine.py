"""MIDI output, arming, and learn-mode helpers."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Protocol

import mido

from .protocol import CONTINUOUS_ACTIONS, encode_action


DEFAULT_PORT_NAME = "Codex Rekordbox Performer"


class MidiOutput(Protocol):
    name: str

    def send(self, message: mido.Message) -> None: ...

    def close(self) -> None: ...


class MidiEngine:
    def __init__(self) -> None:
        self.output: MidiOutput | None = None
        self.port_name: str | None = None
        self.armed_until = 0.0
        self.sent_messages = 0

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
        self.disconnect()
        self.output = mido.open_output(matches[0])
        self.port_name = matches[0]
        return self.status()

    def disconnect(self) -> None:
        self.disarm()
        if self.output is not None:
            self.output.close()
        self.output = None
        self.port_name = None

    def arm(self, seconds: int = 300) -> dict[str, Any]:
        if self.output is None:
            raise RuntimeError("Connect a MIDI output before arming control")
        if not 10 <= seconds <= 1800:
            raise ValueError("seconds must be between 10 and 1800")
        self.armed_until = time.monotonic() + seconds
        return self.status()

    def disarm(self) -> dict[str, Any]:
        self.armed_until = 0.0
        return self.status()

    def is_armed(self) -> bool:
        return self.output is not None and time.monotonic() < self.armed_until

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
        return {
            "connected": self.output is not None,
            "port_name": self.port_name,
            "armed": self.is_armed(),
            "armed_seconds_remaining": round(remaining, 1),
            "sent_messages": self.sent_messages,
        }

