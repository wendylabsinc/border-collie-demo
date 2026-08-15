from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

from border_collie_demo.black_box import RunBlackBox
from border_collie_demo.controller_start import (
    ControllerSample,
    ControllerStartAdapter,
    Go2ControllerStartSource,
)
from border_collie_demo.mission import MissionMachine
from border_collie_demo.run_results import ActiveRunError, RunResultStore
from border_collie_demo.stage_demo import Activation, StageDemo


TOPIC = "rt/lf/lowstate"
START_MASK = 1 << 2


def sample(sequence: int, buttons: int, received_s: float) -> ControllerSample:
    remote = bytearray(40)
    remote[2:4] = buttons.to_bytes(2, "little")
    return ControllerSample(
        source=TOPIC,
        source_sequence=sequence,
        received_monotonic_s=received_s,
        wireless_remote=bytes(remote),
    )


def test_one_start_edge_launches_one_idempotent_pear_mission_with_evidence(
    tmp_path,
) -> None:
    async def scenario() -> None:
        run_id = str(uuid4())
        missions = []
        black_box = RunBlackBox(tmp_path)

        async def activate(mission):
            missions.append(mission)
            return Activation(
                {
                    "run_id": run_id,
                    "current_phase": "find_fruit",
                    "outcome": None,
                    "reason": None,
                },
                idempotent_replay=False,
            )

        adapter = ControllerStartAdapter(
            activate=activate,
            black_box=black_box,
            active_run_id=lambda: None,
        )

        assert (await adapter.observe(sample(10, 0, 100.0))).disposition == "armed"
        accepted = await adapter.observe(sample(11, START_MASK, 100.05))
        held = await adapter.observe(sample(12, START_MASK, 100.10))
        duplicate = await adapter.observe(sample(12, START_MASK, 100.10))

        assert accepted.disposition == "accepted"
        assert held.disposition == "held"
        assert duplicate.disposition == "duplicate"
        assert len(missions) == 1
        assert missions[0].target_fruit == "pear"
        assert missions[0].activation_source == "go2_controller"
        assert missions[0].activation_id == f"go2-controller-start:{TOPIC}:11"

        events = black_box.read(run_id)
        event = next(item for item in events if item["kind"] == "controller_start")
        assert event["payload"] == {
            "activation_id": f"go2-controller-start:{TOPIC}:11",
            "button": "start",
            "button_mask": START_MASK,
            "button_word": START_MASK,
            "disposition": "accepted",
            "received_monotonic_s": 100.05,
            "source": TOPIC,
            "source_sequence": 11,
            "start_pressed": True,
            "target_fruit": "pear",
        }
        black_box.close()

    asyncio.run(scenario())


def test_start_held_across_boot_is_inert_until_release_and_new_edge(tmp_path) -> None:
    async def scenario() -> None:
        missions = []

        async def activate(mission):
            missions.append(mission)
            return Activation(
                {
                    "run_id": str(uuid4()),
                    "current_phase": "find_fruit",
                    "outcome": None,
                    "reason": None,
                },
                idempotent_replay=False,
            )

        black_box = RunBlackBox(tmp_path)
        adapter = ControllerStartAdapter(
            activate=activate,
            black_box=black_box,
            active_run_id=lambda: None,
        )

        first = await adapter.observe(sample(20, START_MASK, 200.0))
        held = await adapter.observe(sample(21, START_MASK, 200.1))
        released = await adapter.observe(sample(22, 0, 200.2))
        pressed = await adapter.observe(sample(23, START_MASK, 200.3))

        assert first.disposition == "startup_held"
        assert held.disposition == "startup_held"
        assert released.disposition == "armed"
        assert pressed.disposition == "accepted"
        assert len(missions) == 1
        black_box.close()

    asyncio.run(scenario())


def test_active_run_rejects_new_edge_and_records_it_on_the_active_black_box(
    tmp_path,
) -> None:
    async def scenario() -> None:
        active_run_id = str(uuid4())
        black_box = RunBlackBox(tmp_path)
        black_box.record(
            active_run_id,
            "run_started",
            phase="find_fruit",
            payload={},
        )

        async def activate(_mission):
            raise ActiveRunError("a Demo Run is already active")

        adapter = ControllerStartAdapter(
            activate=activate,
            black_box=black_box,
            active_run_id=lambda: active_run_id,
        )
        await adapter.observe(sample(30, 0, 300.0))
        decision = await adapter.observe(sample(31, START_MASK, 300.1))

        assert decision.disposition == "active_run"
        event = black_box.read(active_run_id)[-1]
        assert event["kind"] == "controller_start"
        assert event["payload"]["disposition"] == "active_run"
        assert event["payload"]["source_sequence"] == 31
        black_box.close()

    asyncio.run(scenario())


def test_failed_preflight_is_reported_as_not_ready_without_bypassing_activation(
    tmp_path,
) -> None:
    async def scenario() -> None:
        class NotReadyHardware:
            def __init__(self) -> None:
                self.capture_home_calls = 0
                self.stop_calls = 0

            async def start(self) -> None:
                pass

            async def close(self) -> list[str]:
                return []

            async def emergency_stop(self) -> list[str]:
                self.stop_calls += 1
                return []

            def capture_home(self):
                self.capture_home_calls += 1
                raise AssertionError("not-ready activation must not capture Home")

            def status(self) -> dict[str, object]:
                return {
                    "configured": True,
                    "connected": False,
                    "fault": "motion unavailable",
                    "active_operation": None,
                    "pose": {"healthy": False, "age_s": None, "pose": None},
                    "motion": {
                        "armed": False,
                        "last_command": {"forward_mps": 0.0, "yaw_rps": 0.0},
                    },
                }

        black_box = RunBlackBox(tmp_path)
        hardware = NotReadyHardware()
        results = RunResultStore(tmp_path, black_box=black_box)
        demo = StageDemo(
            MissionMachine(),
            results,
            hardware,
            lambda: {"ready": False, "detail": "camera unavailable"},
        )
        await demo.start()
        adapter = ControllerStartAdapter(
            activate=demo.activate,
            black_box=black_box,
            active_run_id=lambda: results.active_run_id,
        )
        await adapter.observe(sample(40, 0, 400.0))
        decision = await adapter.observe(sample(41, START_MASK, 400.1))

        assert decision.disposition == "not_ready"
        assert hardware.capture_home_calls == 0
        assert hardware.stop_calls == 1
        run = demo.result(decision.run_id)
        assert run["target_fruit"] == "pear"
        assert run["activation_source"] == "go2_controller"
        assert run["reason"] == "PREFLIGHT_FAILURE"
        assert run["final_safety_state"] == "DISARMED_CONFIRMED"
        event = black_box.read(decision.run_id)[-1]
        assert event["payload"]["disposition"] == "not_ready"
        assert event["payload"]["run_reason"] == "PREFLIGHT_FAILURE"
        await demo.close()
        black_box.close()

    asyncio.run(scenario())


def test_go2_source_preserves_lowstate_topic_tick_receive_time_and_button_bytes() -> None:
    async def scenario() -> None:
        observed = []

        class Subscriber:
            def __init__(self) -> None:
                self.handler = None
                self.queue_len = None
                self.closed = False

            def Init(self, handler, queue_len):
                self.handler = handler
                self.queue_len = queue_len

            def Close(self):
                self.closed = True

        subscriber = Subscriber()
        source = Go2ControllerStartSource(
            subscriber_factory=lambda topic, _message_type: (
                subscriber if topic == TOPIC else None
            ),
            message_type=object,
            monotonic_clock=lambda: 501.25,
        )

        await source.start(observed.append)
        remote = bytearray(40)
        remote[2:4] = START_MASK.to_bytes(2, "little")
        subscriber.handler(SimpleNamespace(tick=77, wireless_remote=remote))
        await source.close()

        assert subscriber.queue_len == 10
        assert subscriber.closed is True
        assert observed == [
            ControllerSample(
                source=TOPIC,
                source_sequence=77,
                received_monotonic_s=501.25,
                wireless_remote=bytes(remote),
            )
        ]

    asyncio.run(scenario())
