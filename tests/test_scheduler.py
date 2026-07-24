import asyncio

from rekordbox_performer.engine import MidiEngine
from rekordbox_performer.scheduler import TransitionScheduler, validate_events


def test_preview_is_side_effect_free() -> None:
    engine = MidiEngine()
    scheduler = TransitionScheduler(engine)
    result = scheduler.preview(
        "bass swap",
        [
            {
                "at_ms": 0,
                "action": "eq_low",
                "parameters": {"deck": 2, "value": -1},
            },
            {
                "at_ms": 1000,
                "action": "eq_low",
                "parameters": {"deck": 2, "value": 0},
            },
        ],
    )
    assert result["live_effect"] is False
    assert result["duration_ms"] == 1000


def test_unsorted_events_rejected() -> None:
    try:
        validate_events(
            [
                {"at_ms": 100, "action": "cue", "parameters": {"deck": 1}},
                {"at_ms": 0, "action": "cue", "parameters": {"deck": 2}},
            ]
        )
    except ValueError as exc:
        assert "ordered" in str(exc)
    else:
        raise AssertionError("Expected validation failure")


class FakeOutput:
    name = "fake"

    def __init__(self) -> None:
        self.messages = []

    def send(self, message) -> None:
        self.messages.append(message)

    def close(self) -> None:
        pass


def test_scheduler_executes_locally() -> None:
    async def scenario() -> None:
        engine = MidiEngine()
        output = FakeOutput()
        engine.output = output
        engine.port_name = output.name
        engine.arm(30)
        scheduler = TransitionScheduler(engine)
        result = scheduler.start(
            "quick cut",
            [
                {
                    "at_ms": 0,
                    "action": "channel_fader",
                    "parameters": {"deck": 1, "value": 1},
                },
                {
                    "at_ms": 5,
                    "action": "channel_fader",
                    "parameters": {"deck": 1, "value": 0},
                },
            ],
        )
        await scheduler.jobs[result["id"]].task
        final = scheduler.get(result["id"])
        assert final["status"] == "completed"
        assert final["completed_events"] == 2
        assert len(output.messages) == 2

    asyncio.run(scenario())
