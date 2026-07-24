"""FastMCP server for live Rekordbox control through virtual MIDI."""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from .engine import MidiEngine
from .protocol import (
    ALL_ACTIONS,
    CONTINUOUS_ACTIONS,
    TRIGGER_ACTIONS,
    mapping_manifest,
)
from .scheduler import TransitionScheduler


class TransitionEvent(BaseModel):
    at_ms: int = Field(ge=0)
    action: str
    parameters: dict[str, Any] = Field(default_factory=dict)


mcp = FastMCP("Rekordbox Performer")
engine = MidiEngine()
scheduler = TransitionScheduler(engine)


@mcp.tool(annotations={"readOnlyHint": True})
def list_midi_outputs() -> dict[str, Any]:
    """List MIDI outputs visible to the server."""
    return {"outputs": engine.list_output_ports()}


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def connect_midi(port_name: str | None = None) -> dict[str, Any]:
    """Connect to the loopMIDI output. Does not send control messages."""
    return engine.connect(port_name)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def disconnect_midi() -> dict[str, Any]:
    """Disconnect MIDI and disarm live control."""
    engine.disconnect()
    return engine.status()


@mcp.tool(annotations={"readOnlyHint": True})
def control_status() -> dict[str, Any]:
    """Return connection, arming, and scheduler status."""
    return {
        **engine.status(),
        "active_jobs": [
            job.public()
            for job in scheduler.jobs.values()
            if job.status in {"scheduled", "running"}
        ],
    }


@mcp.tool(annotations={"readOnlyHint": True})
def get_mapping_manifest() -> dict[str, Any]:
    """Return the complete stable MIDI mapping for Rekordbox MIDI Learn."""
    return {
        "trigger_actions": sorted(TRIGGER_ACTIONS),
        "continuous_actions": sorted(CONTINUOUS_ACTIONS),
        "assignments": mapping_manifest(),
    }


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def send_midi_learn_signal(
    action: str,
    deck: int | None = None,
    cue: int | None = None,
) -> dict[str, Any]:
    """Send a setup-only signal while Rekordbox MIDI Learn is listening."""
    parameters: dict[str, Any] = {}
    if deck is not None:
        parameters["deck"] = deck
    if cue is not None:
        parameters["cue"] = cue
    return await engine.send_learn_signal(action, parameters)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def arm_control(seconds: int = 300) -> dict[str, Any]:
    """Arm live control for a bounded period after the user requests performance."""
    return engine.arm(seconds)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def disarm_control() -> dict[str, Any]:
    """Disarm live controls without changing current Rekordbox mixer state."""
    return engine.disarm()


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def trigger_control(
    action: str,
    deck: int | None = None,
    cue: int | None = None,
) -> dict[str, Any]:
    """Trigger a mapped button action such as play, cue, sync, loop, or hot cue."""
    if action not in TRIGGER_ACTIONS:
        raise ValueError(
            f"action must be a trigger action: {', '.join(sorted(TRIGGER_ACTIONS))}"
        )
    parameters: dict[str, Any] = {}
    if deck is not None:
        parameters["deck"] = deck
    if cue is not None:
        parameters["cue"] = cue
    messages = await engine.send_action(action, parameters)
    return {"action": action, "parameters": parameters, "messages": messages}


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def set_continuous_control(
    action: str,
    value: float,
    deck: int | None = None,
) -> dict[str, Any]:
    """Set a mapped fader, EQ, filter, tempo, gain, or FX control."""
    if action not in CONTINUOUS_ACTIONS:
        raise ValueError(
            "action must be a continuous action: "
            + ", ".join(sorted(CONTINUOUS_ACTIONS))
        )
    parameters: dict[str, Any] = {"value": value}
    if deck is not None:
        parameters["deck"] = deck
    messages = await engine.send_action(action, parameters)
    return {"action": action, "parameters": parameters, "messages": messages}


@mcp.tool(annotations={"readOnlyHint": True})
def preview_transition(
    name: str, events: list[TransitionEvent]
) -> dict[str, Any]:
    """Validate and preview a transition without sending MIDI."""
    return scheduler.preview(name, [event.model_dump() for event in events])


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
async def perform_transition(
    name: str,
    events: list[TransitionEvent],
    commit: bool = False,
) -> dict[str, Any]:
    """Schedule a transition locally. Set commit=true only for requested live play."""
    raw_events = [event.model_dump() for event in events]
    if not commit:
        return {
            **scheduler.preview(name, raw_events),
            "message": "Dry run only. Set commit=true after review to perform.",
        }
    return scheduler.start(name, raw_events)


@mcp.tool(annotations={"readOnlyHint": True})
def transition_status(job_id: str) -> dict[str, Any]:
    """Get a scheduled transition's progress."""
    return scheduler.get(job_id)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def cancel_transition(job_id: str) -> dict[str, Any]:
    """Cancel future events in one transition without moving any controls."""
    return scheduler.cancel(job_id)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False})
def emergency_stop() -> dict[str, Any]:
    """Cancel all automation and disarm; leave current Rekordbox state untouched."""
    cancelled = scheduler.cancel_all()
    status = engine.disarm()
    return {
        "cancelled_jobs": cancelled,
        "control_status": status,
        "message": "Automation cancelled. Physical FLX4 control remains available.",
    }


def main() -> None:
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
