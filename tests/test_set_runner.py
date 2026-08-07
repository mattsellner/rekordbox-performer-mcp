import asyncio

from rekordbox_performer.intelligence import MusicalEvent, TransitionCard
from rekordbox_performer.set_runner import (
    AutonomousSetPlan,
    AutonomousSetRunner,
    TempoPlan,
    TrackLoadSpec,
    TransitionOption,
)


def option(
    option_id: str, outgoing: str, incoming: str, out_deck: int
) -> TransitionOption:
    in_deck = 2 if out_deck == 1 else 1
    return TransitionOption(
        id=option_id,
        incoming=TrackLoadSpec(track_id=incoming, title=incoming.upper()),
        card=TransitionCard(
            name=option_id,
            outgoing_track_id=outgoing,
            incoming_track_id=incoming,
            anchor_deck=out_deck,
            outgoing_deck=out_deck,
            incoming_deck=in_deck,
            transition_family="phrase_cut",
            phrase_alignment_verified=True,
            vocal_plan_verified=True,
            bass_plan_verified=True,
            incoming_loaded_verified=True,
            intended_vocal_owner="incoming",
            critical_bar_offset=1,
            abort_plan="Keep the outgoing track playing.",
            events=[
                MusicalEvent(
                    bar_offset=0,
                    action="play_pause",
                    parameters={"deck": in_deck},
                ),
                MusicalEvent(
                    bar_offset=1,
                    action="channel_fader",
                    parameters={"deck": out_deck, "value": 0},
                ),
                MusicalEvent(
                    bar_offset=1,
                    action="cue",
                    parameters={"deck": out_deck},
                ),
            ],
        ),
    )


def test_autonomous_runner_advances_without_client_round_trips(tmp_path) -> None:
    async def scenario() -> None:
        plan = AutonomousSetPlan(
            name="three tracks",
            opening=TrackLoadSpec(track_id="a", title="A"),
            transitions=[
                option("a-b", "a", "b", 1),
                option("b-c", "b", "c", 2),
            ],
            target_track_count=3,
        )
        jobs = {
            "job-1": {"id": "job-1", "status": "completed"},
        }
        scheduled = []
        finished = []

        async def schedule(next_option, release_loop):
            assert release_loop is False
            job_id = f"job-{len(jobs) + 1}"
            jobs[job_id] = {"id": job_id, "status": "completed"}
            scheduled.append(next_option.id)
            return {"ready": True, "job": jobs[job_id]}

        runner = AutonomousSetRunner(
            tmp_path / "state.json",
            schedule=schedule,
            job_status=lambda job_id: jobs[job_id],
            job_qa=lambda job_id: {"passed": True, "faults": []},
            remaining_bars=lambda track_id, deck: asyncio.sleep(0, result=64),
            engage_loop=lambda deck, beats: asyncio.sleep(0, result={"verified": True}),
            run_tempo=lambda plan, deck, track_id: asyncio.sleep(
                0, result={"status": "completed"}
            ),
            advance=lambda job_id, succeeded, error: {
                "job_id": job_id,
                "succeeded": succeeded,
            },
            finish=finished.append,
            poll_seconds=0.001,
        )
        runner.prepare(plan, opening_deck=1)
        runner.start_with_job("a-b", "job-1")
        await runner.task

        state = runner.public()
        assert state["status"] == "completed"
        assert state["played_track_ids"] == ["a", "b", "c"]
        assert scheduled == ["b-c"]
        assert finished == ["completed"]

    asyncio.run(scenario())


def test_autonomous_runner_completes_eight_tracks_after_transient_load_failure(
    tmp_path,
) -> None:
    async def scenario() -> None:
        track_ids = list("abcdefgh")
        transitions = [
            option(
                f"{outgoing}-{incoming}",
                outgoing,
                incoming,
                1 if index % 2 == 0 else 2,
            )
            for index, (outgoing, incoming) in enumerate(zip(track_ids, track_ids[1:]))
        ]
        plan = AutonomousSetPlan(
            name="eight-track endurance",
            opening=TrackLoadSpec(track_id="a", title="A"),
            transitions=transitions,
            target_track_count=8,
        )
        jobs = {"opening": {"id": "opening", "status": "completed"}}
        attempts = []

        async def schedule(next_option, release_loop):
            assert release_loop is False
            attempts.append(next_option.id)
            if next_option.id == "d-e" and attempts.count("d-e") == 1:
                raise RuntimeError("transient Rekordbox browser failure")
            job_id = f"job-{len(jobs)}"
            jobs[job_id] = {"id": job_id, "status": "completed"}
            return {"ready": True, "job": jobs[job_id]}

        runner = AutonomousSetRunner(
            tmp_path / "eight-track.json",
            schedule=schedule,
            job_status=lambda job_id: jobs[job_id],
            job_qa=lambda job_id: {"passed": True, "faults": []},
            remaining_bars=lambda track_id, deck: asyncio.sleep(0, result=64),
            engage_loop=lambda deck, beats: asyncio.sleep(0, result={"verified": True}),
            run_tempo=lambda plan, deck, track_id: asyncio.sleep(
                0, result={"status": "completed"}
            ),
            advance=lambda job_id, succeeded, error: {},
            poll_seconds=0.001,
        )
        runner.prepare(plan, opening_deck=1)
        runner.start_with_job("a-b", "opening")
        await runner.task

        state = runner.public()
        assert state["status"] == "completed"
        assert state["played_track_ids"] == track_ids
        assert attempts == ["b-c", "c-d", "d-e", "d-e", "e-f", "f-g", "g-h"]

    asyncio.run(scenario())


def test_stale_observations_do_not_exhaust_transition_retry_budget(tmp_path) -> None:
    async def scenario() -> None:
        plan = AutonomousSetPlan(
            name="observer outage",
            opening=TrackLoadSpec(track_id="a", title="A"),
            transitions=[option("a-b", "a", "b", 1)],
            target_track_count=2,
            retry_limit=3,
        )
        jobs = {"opening": {"id": "opening", "status": "failed"}}
        calls = 0

        async def schedule(next_option, release_loop):
            nonlocal calls
            calls += 1
            if calls <= 5:
                raise RuntimeError("deck observation is stale (5000 ms)")
            jobs["recovered"] = {"id": "recovered", "status": "completed"}
            return {"ready": True, "job": jobs["recovered"]}

        runner = AutonomousSetRunner(
            tmp_path / "observer-outage.json",
            schedule=schedule,
            job_status=lambda job_id: jobs[job_id],
            job_qa=lambda job_id: (
                {"passed": False, "faults": ["opening failed"]}
                if job_id == "opening"
                else {"passed": True, "faults": []}
            ),
            remaining_bars=lambda track_id, deck: asyncio.sleep(0, result=64),
            engage_loop=lambda deck, beats: asyncio.sleep(0, result={"verified": True}),
            run_tempo=lambda plan, deck, track_id: asyncio.sleep(
                0, result={"status": "completed"}
            ),
            advance=lambda job_id, succeeded, error: {},
            poll_seconds=0.001,
        )
        runner.prepare(plan, opening_deck=1)
        runner.start_with_job("a-b", "opening")
        await runner.task

        state = runner.public()
        assert calls == 6
        assert state["status"] == "completed"
        # The adopted opening job and the successful retry consume budget;
        # five stale observer reads do not.
        assert state["attempted_options"]["a-b"] == 2

    asyncio.run(scenario())


def test_exhausted_transition_is_replaced_without_stopping_set(tmp_path) -> None:
    async def scenario() -> None:
        original = AutonomousSetPlan(
            name="replace failed route",
            opening=TrackLoadSpec(track_id="a", title="A"),
            transitions=[
                option("a-b", "a", "b", 1),
                option("b-c", "b", "c", 2),
            ],
            target_track_count=3,
            retry_limit=3,
        )
        replacement = AutonomousSetPlan(
            name="replacement",
            opening=TrackLoadSpec(track_id="b", title="B"),
            transitions=[option("replacement-b-d", "b", "d", 2)],
            target_track_count=2,
        )
        jobs = {"opening": {"id": "opening", "status": "completed"}}
        scheduled = []
        recovery_calls = []

        async def schedule(next_option, release_loop):
            scheduled.append(next_option.id)
            if next_option.incoming.track_id == "c":
                raise RuntimeError("exact load route failed")
            jobs["replacement"] = {"id": "replacement", "status": "completed"}
            return {"ready": True, "job": jobs["replacement"]}

        async def recover(status):
            recovery_calls.append(status)
            assert status["current_track_id"] == "b"
            assert status["exhausted_incoming_track_ids"] == ["c"]
            return replacement

        runner = AutonomousSetRunner(
            tmp_path / "replacement.json",
            schedule=schedule,
            job_status=lambda job_id: jobs[job_id],
            job_qa=lambda job_id: {"passed": True, "faults": []},
            remaining_bars=lambda track_id, deck: asyncio.sleep(0, result=64),
            engage_loop=lambda deck, beats: asyncio.sleep(0, result={"verified": True}),
            run_tempo=lambda plan, deck, track_id: asyncio.sleep(
                0, result={"status": "completed"}
            ),
            advance=lambda job_id, succeeded, error: {},
            recover_route=recover,
            poll_seconds=0.001,
            recovery_retry_seconds=0.001,
        )
        runner.prepare(original, opening_deck=1)
        runner.start_with_job("a-b", "opening")
        await runner.task

        state = runner.public()
        assert state["status"] == "completed"
        assert state["played_track_ids"] == ["a", "b", "d"]
        assert scheduled == ["b-c", "b-c", "b-c", "replacement-b-d"]
        assert len(recovery_calls) == 1
        assert state["recovery_attempts"] == 1

    asyncio.run(scenario())


def test_title_repaint_does_not_consume_retry_budget(tmp_path) -> None:
    async def scenario() -> None:
        plan = AutonomousSetPlan(
            name="title repaint",
            opening=TrackLoadSpec(track_id="a", title="A"),
            transitions=[option("a-b", "a", "b", 1)],
            target_track_count=2,
            retry_limit=3,
        )
        jobs = {"opening": {"id": "opening", "status": "failed"}}
        calls = 0

        async def schedule(next_option, release_loop):
            nonlocal calls
            calls += 1
            if calls <= 5:
                raise RuntimeError("Deck title changed during transport observation")
            jobs["recovered"] = {"id": "recovered", "status": "completed"}
            return {"ready": True, "job": jobs["recovered"]}

        runner = AutonomousSetRunner(
            tmp_path / "title-repaint.json",
            schedule=schedule,
            job_status=lambda job_id: jobs[job_id],
            job_qa=lambda job_id: (
                {"passed": False, "faults": ["opening failed"]}
                if job_id == "opening"
                else {"passed": True, "faults": []}
            ),
            remaining_bars=lambda track_id, deck: asyncio.sleep(0, result=64),
            engage_loop=lambda deck, beats: asyncio.sleep(0, result={"verified": True}),
            run_tempo=lambda plan, deck, track_id: asyncio.sleep(
                0, result={"status": "completed"}
            ),
            advance=lambda job_id, succeeded, error: {},
            poll_seconds=0.001,
        )
        runner.prepare(plan, opening_deck=1)
        runner.start_with_job("a-b", "opening")
        await runner.task

        assert calls == 6
        assert runner.public()["status"] == "completed"
        assert runner.public()["attempted_options"]["a-b"] == 2

    asyncio.run(scenario())


def test_autonomous_runner_engages_loop_and_releases_it_in_retry(tmp_path) -> None:
    async def scenario() -> None:
        plan = AutonomousSetPlan(
            name="loop recovery",
            opening=TrackLoadSpec(track_id="a", title="A"),
            transitions=[option("a-b", "a", "b", 1)],
            target_track_count=2,
            retry_limit=3,
        )
        jobs = {"failed-opening": {"id": "failed-opening", "status": "failed"}}
        schedule_calls = []
        loop_calls = []

        async def schedule(next_option, release_loop):
            schedule_calls.append(release_loop)
            if len(schedule_calls) == 1:
                raise RuntimeError("browser route temporarily unavailable")
            jobs["recovered"] = {"id": "recovered", "status": "completed"}
            return {"ready": True, "job": jobs["recovered"]}

        async def engage(deck, beats):
            loop_calls.append((deck, beats))
            return {"verified": True}

        runner = AutonomousSetRunner(
            tmp_path / "state.json",
            schedule=schedule,
            job_status=lambda job_id: jobs[job_id],
            job_qa=lambda job_id: (
                {"passed": False, "faults": ["launch failed"]}
                if job_id == "failed-opening"
                else {"passed": True, "faults": []}
            ),
            remaining_bars=lambda track_id, deck: asyncio.sleep(0, result=6),
            engage_loop=engage,
            run_tempo=lambda plan, deck, track_id: asyncio.sleep(
                0, result={"status": "completed"}
            ),
            advance=lambda job_id, succeeded, error: {},
            poll_seconds=0.001,
        )
        runner.prepare(plan, opening_deck=1)
        runner.start_with_job("a-b", "failed-opening")
        await runner.task

        assert loop_calls == [(1, 4)]
        # With only six bars remaining, the runner loops before the first
        # retry attempt, so every accepted retry card carries the bar-0 loop
        # release instead of waiting for a staging failure to discover danger.
        assert schedule_calls == [True, True]
        assert runner.public()["status"] == "completed"
        assert runner.public()["deadline_phase"] == "rescue"

        # The adopted opening job counts toward the retry budget; it must not
        # silently grant one more attempt than the plan allows.
        assert runner.public()["attempted_options"]["a-b"] == 3

    asyncio.run(scenario())


def test_unverified_rescue_loop_does_not_terminate_the_set(tmp_path) -> None:
    async def scenario() -> None:
        plan = AutonomousSetPlan(
            name="loop verification fallback",
            opening=TrackLoadSpec(track_id="a", title="A"),
            transitions=[option("a-b", "a", "b", 1)],
            target_track_count=2,
            retry_limit=3,
        )
        jobs = {"opening": {"id": "opening", "status": "failed"}}

        async def schedule(next_option, release_loop):
            assert release_loop is False
            jobs["recovered"] = {"id": "recovered", "status": "completed"}
            return {"ready": True, "job": jobs["recovered"]}

        runner = AutonomousSetRunner(
            tmp_path / "unverified-loop.json",
            schedule=schedule,
            job_status=lambda job_id: jobs[job_id],
            job_qa=lambda job_id: (
                {"passed": False, "faults": ["opening failed"]}
                if job_id == "opening"
                else {"passed": True, "faults": []}
            ),
            remaining_bars=lambda track_id, deck: asyncio.sleep(0, result=6),
            engage_loop=lambda deck, beats: asyncio.sleep(
                0,
                result={"verified": False, "errors": ["transport read missed wrap"]},
            ),
            run_tempo=lambda plan, deck, track_id: asyncio.sleep(
                0, result={"status": "completed"}
            ),
            advance=lambda job_id, succeeded, error: {},
            poll_seconds=0.001,
        )
        runner.prepare(plan, opening_deck=1)
        runner.start_with_job("a-b", "opening")
        await runner.task

        state = runner.public()
        assert state["status"] == "completed"
        assert state["rescue_loop_active"] is False
        assert any(
            "continuing replacement planning" in warning
            for warning in state["warnings"]
        )

    asyncio.run(scenario())


def test_tempo_plan_enforces_stretch_and_maps_slider() -> None:
    plan = TempoPlan(target_bpm=130, tempo_range_percent=10)
    plan.validate_start(native_bpm=130, live_bpm=127)
    assert round(plan.control_value(native_bpm=130, bpm=130), 6) == 0

    unsafe = TempoPlan(target_bpm=130, max_stretch_percent=4)
    try:
        unsafe.validate_start(native_bpm=130, live_bpm=121)
    except ValueError as exc:
        assert "initial 6.92%" in str(exc)
    else:
        raise AssertionError("expected unsafe stretch to fail")

    unsafe_target = TempoPlan(target_bpm=130, max_stretch_percent=4)
    try:
        unsafe_target.validate_start(native_bpm=121, live_bpm=121)
    except ValueError as exc:
        assert "target 7.44%" in str(exc)
    else:
        raise AssertionError("expected an unsafe target stretch to fail")


def test_auto_tempo_arc_rises_gradually_across_the_primary_path() -> None:
    plan = AutonomousSetPlan(
        name="rising arc",
        opening=TrackLoadSpec(track_id="a", title="A"),
        transitions=[
            option("a-b", "a", "b", 1),
            option("b-c", "b", "c", 2),
            option("c-d", "c", "d", 1),
        ],
        target_track_count=4,
        tempo_ramp_bars=24,
    )

    resolved = plan.materialize_tempo_arc({"a": 120, "b": 121, "c": 125, "d": 129})

    assert [
        transition.tempo_after.target_bpm for transition in resolved.transitions
    ] == [123, 126, 129]
    assert all(
        transition.tempo_after.duration_bars == 24
        for transition in resolved.transitions
    )


def test_auto_tempo_arc_applies_to_reachable_fallback_branches() -> None:
    plan = AutonomousSetPlan(
        name="fallback arc",
        opening=TrackLoadSpec(track_id="a", title="A"),
        transitions=[
            option("a-b", "a", "b", 1),
            option("a-x", "a", "x", 1),
            option("b-c", "b", "c", 2),
            option("x-y", "x", "y", 2),
        ],
        target_track_count=3,
    )

    resolved = plan.materialize_tempo_arc(
        {"a": 120, "b": 123, "c": 126, "x": 122, "y": 125}
    )
    targets = {
        transition.id: transition.tempo_after.target_bpm
        for transition in resolved.transitions
    }

    assert targets == {"a-b": 123, "a-x": 123, "b-c": 126, "x-y": 126}


def test_manual_material_tempo_arc_requires_an_explicit_ramp() -> None:
    plan = AutonomousSetPlan(
        name="manual arc",
        opening=TrackLoadSpec(track_id="a", title="A"),
        transitions=[option("a-b", "a", "b", 1)],
        target_track_count=2,
        tempo_strategy="manual",
    )

    try:
        plan.materialize_tempo_arc({"a": 120, "b": 129})
    except ValueError as exc:
        assert "contains no TempoPlan" in str(exc)
    else:
        raise AssertionError("expected a missing manual tempo ramp to fail")


def test_hold_tempo_strategy_does_not_generate_ramps() -> None:
    plan = AutonomousSetPlan(
        name="fixed tempo",
        opening=TrackLoadSpec(track_id="a", title="A"),
        transitions=[option("a-b", "a", "b", 1)],
        target_track_count=2,
        tempo_strategy="hold",
    )

    resolved = plan.materialize_tempo_arc({"a": 120, "b": 129})

    assert resolved.transitions[0].tempo_after is None


def test_runner_executes_post_handoff_tempo_plan(tmp_path) -> None:
    async def scenario() -> None:
        transition = option("a-b", "a", "b", 1)
        transition.tempo_after = TempoPlan(target_bpm=123, duration_bars=16)
        plan = AutonomousSetPlan(
            name="tempo execution",
            opening=TrackLoadSpec(track_id="a", title="A"),
            transitions=[transition],
            target_track_count=2,
        )
        jobs = {"opening": {"id": "opening", "status": "completed"}}
        ramps = []

        async def run_tempo(tempo, deck, track_id):
            ramps.append((tempo.target_bpm, deck, track_id))
            return {"status": "completed"}

        runner = AutonomousSetRunner(
            tmp_path / "tempo-execution.json",
            schedule=lambda next_option, release_loop: asyncio.sleep(0),
            job_status=lambda job_id: jobs[job_id],
            job_qa=lambda job_id: {"passed": True, "faults": []},
            remaining_bars=lambda track_id, deck: asyncio.sleep(0, result=64),
            engage_loop=lambda deck, beats: asyncio.sleep(0, result={"verified": True}),
            run_tempo=run_tempo,
            advance=lambda job_id, succeeded, error: {},
            poll_seconds=0.001,
        )
        runner.prepare(plan, opening_deck=1)
        runner.start_with_job("a-b", "opening")
        await runner.task

        assert ramps == [(123, 2, "b")]
        assert runner.public()["status"] == "completed"

    asyncio.run(scenario())


def test_runner_prestages_next_track_while_tempo_ramp_is_running(tmp_path) -> None:
    async def scenario() -> None:
        first = option("a-b", "a", "b", 1)
        first.tempo_after = TempoPlan(target_bpm=123, duration_bars=32)
        second = option("b-c", "b", "c", 2)
        plan = AutonomousSetPlan(
            name="rolling lead",
            opening=TrackLoadSpec(track_id="a", title="A"),
            transitions=[first, second],
            target_track_count=3,
        )
        jobs = {"opening": {"id": "opening", "status": "completed"}}
        prestaged = asyncio.Event()
        timeline = []

        async def prestage(next_option):
            timeline.append(f"prestage:{next_option.id}")
            prestaged.set()
            return {"ready": True}

        async def run_tempo(tempo, deck, track_id):
            timeline.append("tempo:start")
            await asyncio.wait_for(prestaged.wait(), timeout=0.1)
            timeline.append("tempo:finish")
            return {"status": "completed"}

        async def schedule(next_option, release_loop):
            timeline.append(f"schedule:{next_option.id}")
            jobs["second"] = {"id": "second", "status": "completed"}
            return {
                "ready": True,
                "job": jobs["second"],
                "start_delay_ms": 1_000,
                "bpm": 123,
            }

        runner = AutonomousSetRunner(
            tmp_path / "rolling-lead.json",
            schedule=schedule,
            prestage=prestage,
            job_status=lambda job_id: jobs[job_id],
            job_qa=lambda job_id: {"passed": True, "faults": []},
            remaining_bars=lambda track_id, deck: asyncio.sleep(0, result=64),
            engage_loop=lambda deck, beats: asyncio.sleep(0, result={"verified": True}),
            run_tempo=run_tempo,
            advance=lambda job_id, succeeded, error: {},
            poll_seconds=0.001,
        )
        runner.prepare(plan, opening_deck=1)
        runner.start_with_job("a-b", "opening")
        await runner.task

        assert timeline.index("prestage:b-c") < timeline.index("tempo:finish")
        assert timeline.index("tempo:finish") < timeline.index("schedule:b-c")
        # Once C becomes current, it is no longer incorrectly reported as a
        # staged deck. During the ramp it was staged before scheduling.
        assert runner.public()["staged_option_id"] is None
        assert runner.public()["status"] == "completed"

    asyncio.run(scenario())


def test_runner_continues_when_noncritical_tempo_ramp_is_rejected(tmp_path) -> None:
    async def scenario() -> None:
        first = option("a-b", "a", "b", 1)
        first.tempo_after = TempoPlan(target_bpm=123, duration_bars=32)
        second = option("b-c", "b", "c", 2)
        plan = AutonomousSetPlan(
            name="tempo fallback",
            opening=TrackLoadSpec(track_id="a", title="A"),
            transitions=[first, second],
            target_track_count=3,
        )
        jobs = {"opening": {"id": "opening", "status": "completed"}}
        scheduled = []

        async def schedule(next_option, release_loop):
            scheduled.append(next_option.id)
            jobs["second"] = {"id": "second", "status": "completed"}
            return {"ready": True, "job": jobs["second"]}

        async def reject_tempo(_tempo, _deck, _track_id):
            raise RuntimeError("tempo pickup was not acquired")

        runner = AutonomousSetRunner(
            tmp_path / "tempo-fallback.json",
            schedule=schedule,
            prestage=lambda _option: asyncio.sleep(0, result={"ready": True}),
            job_status=lambda job_id: jobs[job_id],
            job_qa=lambda job_id: {"passed": True, "faults": []},
            remaining_bars=lambda track_id, deck: asyncio.sleep(0, result=64),
            engage_loop=lambda deck, beats: asyncio.sleep(0, result={"verified": True}),
            run_tempo=reject_tempo,
            advance=lambda job_id, succeeded, error: {},
            poll_seconds=0.001,
        )
        runner.prepare(plan, opening_deck=1)
        runner.start_with_job("a-b", "opening")
        await runner.task

        state = runner.public()
        assert state["status"] == "completed"
        assert state["played_track_ids"] == ["a", "b", "c"]
        assert scheduled == ["b-c"]
        assert "tempo ramp skipped" in state["warnings"][0]
        assert state["failures"] == []

    asyncio.run(scenario())


def test_runner_engages_rescue_loop_at_thirty_two_bar_boundary(tmp_path) -> None:
    async def scenario() -> None:
        plan = AutonomousSetPlan(
            name="thirty-two bar rescue",
            opening=TrackLoadSpec(track_id="a", title="A"),
            transitions=[option("a-b", "a", "b", 1)],
            target_track_count=2,
        )
        jobs = {"opening": {"id": "opening", "status": "failed"}}
        loops = []

        async def schedule(next_option, release_loop):
            assert release_loop is True
            jobs["recovered"] = {"id": "recovered", "status": "completed"}
            return {"ready": True, "job": jobs["recovered"]}

        async def engage(deck, beats):
            loops.append((deck, beats))
            return {"verified": True}

        runner = AutonomousSetRunner(
            tmp_path / "thirty-two-bar-rescue.json",
            schedule=schedule,
            job_status=lambda job_id: jobs[job_id],
            job_qa=lambda job_id: (
                {"passed": False, "faults": ["opening failed"]}
                if job_id == "opening"
                else {"passed": True, "faults": []}
            ),
            remaining_bars=lambda track_id, deck: asyncio.sleep(0, result=32),
            engage_loop=engage,
            run_tempo=lambda plan, deck, track_id: asyncio.sleep(
                0, result={"status": "completed"}
            ),
            advance=lambda job_id, succeeded, error: {},
            poll_seconds=0.001,
        )
        runner.prepare(plan, opening_deck=1)
        runner.start_with_job("a-b", "opening")
        await runner.task

        assert loops == [(1, 4)]
        assert runner.public()["deadline_phase"] == "rescue"
        assert runner.public()["status"] == "completed"

    asyncio.run(scenario())


def test_runner_applies_steering_only_after_already_armed_handoff(tmp_path) -> None:
    async def scenario() -> None:
        original = AutonomousSetPlan(
            name="original",
            opening=TrackLoadSpec(track_id="a", title="A"),
            transitions=[
                option("a-b", "a", "b", 1),
                option("b-c", "b", "c", 2),
            ],
            target_track_count=3,
        )
        redirected = AutonomousSetPlan(
            name="redirected",
            opening=TrackLoadSpec(track_id="b", title="B"),
            transitions=[
                option("b-x", "b", "x", 2),
                option("x-y", "x", "y", 1),
            ],
            target_track_count=3,
        )
        jobs = {"opening": {"id": "opening", "status": "completed"}}
        scheduled = []

        async def schedule(next_option, release_loop):
            scheduled.append(next_option.id)
            job_id = f"job-{len(scheduled)}"
            jobs[job_id] = {"id": job_id, "status": "completed"}
            return {"ready": True, "job": jobs[job_id]}

        runner = AutonomousSetRunner(
            tmp_path / "steering.json",
            schedule=schedule,
            job_status=lambda job_id: jobs[job_id],
            job_qa=lambda job_id: {"passed": True, "faults": []},
            remaining_bars=lambda track_id, deck: asyncio.sleep(0, result=64),
            engage_loop=lambda deck, beats: asyncio.sleep(0, result={"verified": True}),
            run_tempo=lambda plan, deck, track_id: asyncio.sleep(
                0, result={"status": "completed"}
            ),
            advance=lambda job_id, succeeded, error: {},
            poll_seconds=0.001,
        )
        runner.prepare(original, opening_deck=1)
        runner.start_with_job("a-b", "opening")
        queued = runner.queue_redirect(redirected)

        assert queued["queued"] is True
        assert queued["projected_plan"]["target_track_count"] == 4
        assert runner.public()["steering_queued"] is True

        await runner.task

        assert runner.public()["played_track_ids"] == ["a", "b", "x", "y"]
        assert scheduled == ["b-x", "x-y"]
        assert runner.public()["steering_queued"] is False

    asyncio.run(scenario())
