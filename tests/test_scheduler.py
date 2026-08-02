import asyncio

from rekordbox_performer.engine import MidiEngine
from rekordbox_performer.scheduler import (
    TransitionScheduler,
    percentile,
    validate_events,
)


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


def test_percentile_reports_tail_latency() -> None:
    values = [1.0, 2.0, 3.0, 20.0]
    assert percentile(values, 0.95) == 20.0
    assert percentile([], 0.99) == 0.0


class FakeOutput:
    name = "fake"

    def __init__(self) -> None:
        self.messages = []

    def send(self, message) -> None:
        self.messages.append(message)

    def close(self) -> None:
        pass


def test_engine_reports_last_commanded_continuous_state() -> None:
    async def scenario() -> None:
        engine = MidiEngine()
        output = FakeOutput()
        engine.output = output
        engine.port_name = output.name
        engine.arm(30)
        await engine.send_action(
            "channel_fader", {"deck": 2, "value": 0.55}
        )
        state = engine.status()["continuous_control_state"]
        assert state["deck_2.channel_fader"]["value"] == 0.55
        assert (
            state["deck_2.channel_fader"]["verification"]
            == "commanded_not_observed"
        )

    asyncio.run(scenario())


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
        assert "p99_event_lateness_ms" in final
        assert scheduler.metrics()["completed_jobs"] == 1

    asyncio.run(scenario())


def test_scheduler_fails_when_rekordbox_postcondition_is_unverified() -> None:
    async def scenario() -> None:
        engine = MidiEngine()
        output = FakeOutput()
        engine.output = output
        engine.port_name = output.name
        engine.arm(30)
        scheduler = TransitionScheduler(engine)
        result = scheduler.start(
            "false success guard",
            [
                {
                    "at_ms": 0,
                    "action": "channel_fader",
                    "parameters": {"deck": 1, "value": 0},
                }
            ],
            completion_verifier=lambda: {
                "verified": False,
                "errors": ["incoming deck is not playing"],
            },
        )
        await scheduler.jobs[result["id"]].task
        final = scheduler.get(result["id"])
        assert final["status"] == "failed"
        assert final["verification"]["verified"] is False
        assert "incoming deck is not playing" in final["error"]

    asyncio.run(scenario())


def test_scheduler_completes_only_after_rekordbox_postcondition() -> None:
    async def scenario() -> None:
        engine = MidiEngine()
        output = FakeOutput()
        engine.output = output
        engine.port_name = output.name
        engine.arm(30)
        scheduler = TransitionScheduler(engine)
        result = scheduler.start(
            "verified handoff",
            [
                {
                    "at_ms": 0,
                    "action": "channel_fader",
                    "parameters": {"deck": 1, "value": 0},
                }
            ],
            completion_verifier=lambda: {
                "verified": True,
                "errors": [],
            },
        )
        await scheduler.jobs[result["id"]].task
        final = scheduler.get(result["id"])
        assert final["status"] == "completed"
        assert final["verification"]["verified"] is True

    asyncio.run(scenario())


def test_scheduler_deduplicates_retried_execution_id() -> None:
    async def scenario() -> None:
        engine = MidiEngine()
        output = FakeOutput()
        engine.output = output
        engine.port_name = output.name
        engine.arm(30)
        scheduler = TransitionScheduler(engine)
        events = [
            {
                "at_ms": 0,
                "action": "channel_fader",
                "parameters": {"deck": 1, "value": 0},
            }
        ]
        first = scheduler.start(
            "idempotent handoff",
            events,
            execution_id="pass-123",
        )
        retry = scheduler.start(
            "idempotent handoff",
            events,
            execution_id="pass-123",
        )
        assert retry["id"] == first["id"]
        assert retry["deduplicated"] is True
        assert len(scheduler.jobs) == 1
        await scheduler.jobs[first["id"]].task
        assert len(output.messages) == 1

    asyncio.run(scenario())


def test_scheduler_rejects_execution_id_reuse_for_different_card() -> None:
    async def scenario() -> None:
        engine = MidiEngine()
        output = FakeOutput()
        engine.output = output
        engine.port_name = output.name
        engine.arm(30)
        scheduler = TransitionScheduler(engine)
        scheduler.start(
            "first handoff",
            [{"at_ms": 1000, "action": "cue", "parameters": {"deck": 1}}],
            execution_id="pass-123",
        )
        try:
            scheduler.start(
                "different handoff",
                [{"at_ms": 1000, "action": "cue", "parameters": {"deck": 2}}],
                execution_id="pass-123",
            )
        except RuntimeError as exc:
            assert "different transition" in str(exc)
        else:
            raise AssertionError("Expected execution_id collision failure")
        scheduler.cancel_all()

    asyncio.run(scenario())


def test_scheduler_reports_exact_dispatch_to_event_observer() -> None:
    async def scenario() -> None:
        engine = MidiEngine()
        output = FakeOutput()
        engine.output = output
        engine.port_name = output.name
        engine.arm(30)
        scheduler = TransitionScheduler(engine)
        observed = []
        result = scheduler.start(
            "observed launch",
            [{"at_ms": 0, "action": "hot_cue", "parameters": {"deck": 2, "cue": 1}}],
            event_observer=lambda event, dispatched: observed.append(
                (event, dispatched)
            ),
        )
        await scheduler.jobs[result["id"]].task
        assert len(observed) == 1
        assert observed[0][0]["action"] == "hot_cue"
        assert observed[0][0]["parameters"] == {"deck": 2, "cue": 1}
        assert observed[0][1] > 0

    asyncio.run(scenario())


def test_scheduler_reserves_control_for_entire_job() -> None:
    async def scenario() -> None:
        engine = MidiEngine()
        output = FakeOutput()
        engine.output = output
        engine.port_name = output.name
        engine.arm(10)
        scheduler = TransitionScheduler(engine)
        result = scheduler.start(
            "long job",
            [{"at_ms": 11_000, "action": "cue", "parameters": {"deck": 1}}],
        )
        assert result["control_reserved_until_monotonic"] > engine.armed_until
        scheduler.cancel_all()

    asyncio.run(scenario())
