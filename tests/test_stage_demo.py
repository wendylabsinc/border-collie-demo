from __future__ import annotations

import asyncio

import pytest

from border_collie_demo.mission import MissionMachine
from border_collie_demo.orchestrator import SimulatedStageExecutor
from border_collie_demo.run_results import ActiveRunError, RunResultStore
from border_collie_demo.stage_demo import FruitMission, StageDemo


class ReadyHardware:
    def __init__(self) -> None:
        self.started = False
        self.stop_calls = 0
        self.forward_mps = 0.0
        self.yaw_rps = 0.0

    async def start(self) -> None:
        self.started = True

    async def close(self) -> list[str]:
        self.started = False
        return []

    async def emergency_stop(self) -> list[str]:
        self.stop_calls += 1
        self.forward_mps = 0.0
        self.yaw_rps = 0.0
        return []

    def capture_home(self) -> dict[str, object]:
        return {
            "x_m": 1.0,
            "y_m": 2.0,
            "yaw_rad": 0.25,
            "captured_monotonic_s": 3.0,
            "age_s": 0.01,
            "source": "test",
        }

    def status(self) -> dict[str, object]:
        return {
            "configured": True,
            "connected": self.started,
            "fault": None,
            "active_operation": None,
            "pose": {"healthy": True, "age_s": 0.01, "error": None},
            "motion": {
                "armed": False,
                "last_command": {
                    "forward_mps": self.forward_mps,
                    "yaw_rps": self.yaw_rps,
                    "reason": "test",
                },
            },
        }


def ready_camera() -> dict[str, object]:
    return {"ready": True, "detail": "camera ready"}


def test_activation_id_replays_the_same_durable_demo_run(tmp_path) -> None:
    async def scenario() -> None:
        selected: list[str] = []
        demo = StageDemo(
            MissionMachine(),
            RunResultStore(tmp_path),
            ReadyHardware(),
            ready_camera,
            select_perception_target=lambda fruit: selected.append(fruit),
        )
        await demo.start()
        mission = FruitMission(
            target_fruit="pear",
            activation_source="audience_ui",
            activation_id="ui-click-123",
        )

        first = await demo.activate(mission)
        replay = await demo.activate(mission)

        assert replay.idempotent_replay is True
        assert replay.run == first.run
        assert replay.run["activation_id"] == "ui-click-123"
        assert selected == ["pear"]
        with pytest.raises(ActiveRunError, match="already active"):
            await demo.activate(
                FruitMission(
                    target_fruit="pear",
                    activation_source="voice",
                    activation_id="voice-command-456",
                )
            )
        await demo.stop()
        await demo.close()

    asyncio.run(scenario())


def test_completed_run_is_read_through_interface_and_exactly_disarmed(tmp_path) -> None:
    async def scenario() -> None:
        hardware = ReadyHardware()
        hardware.forward_mps = 1.0
        hardware.yaw_rps = 0.3
        demo = StageDemo(
            MissionMachine(),
            RunResultStore(tmp_path),
            hardware,
            ready_camera,
            stage_executor=SimulatedStageExecutor(),
        )
        await demo.start()

        activation = await demo.activate(
            FruitMission("pear", "soak", "soak-attempt-1")
        )
        terminal = await demo.wait(activation.run["run_id"], timeout_s=1.0)

        assert terminal == demo.result(activation.run["run_id"])
        assert terminal["outcome"] == "COMPLETED"
        assert terminal["final_safety_state"] == "DISARMED_CONFIRMED"
        assert hardware.stop_calls >= 1
        assert hardware.status()["motion"]["last_command"] == {
            "forward_mps": 0.0,
            "yaw_rps": 0.0,
            "reason": "test",
        }
        await demo.close()

    asyncio.run(scenario())


def test_stop_cancels_execution_and_seals_once_without_late_resume(tmp_path) -> None:
    async def scenario() -> None:
        hardware = ReadyHardware()
        demo = StageDemo(
            MissionMachine(),
            RunResultStore(tmp_path),
            hardware,
            ready_camera,
            stage_executor=SimulatedStageExecutor(
                delay_at="turn_to_fruit",
                delay_s=0.25,
            ),
        )
        await demo.start()
        activation = await demo.activate(FruitMission("pear", "voice", "voice-1"))
        run_id = activation.run["run_id"]
        await asyncio.sleep(0)

        stopped = await demo.stop()
        await asyncio.sleep(0.30)

        assert stopped["run"]["outcome"] == "STOPPED"
        assert demo.result(run_id)["reason"] == "OPERATOR_STOP"
        assert demo.result(run_id)["current_phase"] == "stopped"
        assert demo.status()["active_run_id"] is None
        assert hardware.status()["motion"]["armed"] is False
        await demo.close()

    asyncio.run(scenario())


def test_terminal_activation_replay_survives_a_new_demo_instance(tmp_path) -> None:
    async def scenario() -> None:
        mission = FruitMission("pear", "soak", "durable-attempt")
        first_demo = StageDemo(
            MissionMachine(),
            RunResultStore(tmp_path),
            ReadyHardware(),
            ready_camera,
        )
        await first_demo.start()
        first = await first_demo.activate(mission)
        await first_demo.stop()
        await first_demo.close()

        second_demo = StageDemo(
            MissionMachine(),
            RunResultStore(tmp_path),
            ReadyHardware(),
            ready_camera,
        )
        await second_demo.start()
        replay = await second_demo.activate(mission)

        assert replay.idempotent_replay is True
        assert replay.run["run_id"] == first.run["run_id"]
        assert replay.run["outcome"] == "STOPPED"
        assert second_demo.status()["active_run_id"] is None
        await second_demo.close()

    asyncio.run(scenario())


def test_activation_id_cannot_be_reused_for_different_intent(tmp_path) -> None:
    async def scenario() -> None:
        demo = StageDemo(
            MissionMachine(),
            RunResultStore(tmp_path),
            ReadyHardware(),
            ready_camera,
        )
        await demo.start()
        await demo.activate(FruitMission("pear", "voice", "command-1"))

        with pytest.raises(ValueError, match="different Fruit Mission"):
            await demo.activate(FruitMission("apple", "voice", "command-1"))

        assert len(demo.list_results()) == 1
        await demo.stop()
        await demo.close()

    asyncio.run(scenario())


def test_completion_fails_closed_when_exact_zero_cannot_be_confirmed(tmp_path) -> None:
    class StickyHardware(ReadyHardware):
        async def emergency_stop(self) -> list[str]:
            self.stop_calls += 1
            return []

    async def scenario() -> None:
        hardware = StickyHardware()
        hardware.forward_mps = 1.0
        demo = StageDemo(
            MissionMachine(),
            RunResultStore(tmp_path),
            hardware,
            ready_camera,
            stage_executor=SimulatedStageExecutor(),
        )
        await demo.start()
        activation = await demo.activate(FruitMission("pear", "soak", "attempt-1"))

        terminal = await demo.wait(activation.run["run_id"], timeout_s=1.0)

        assert terminal["outcome"] == "FAILED"
        assert terminal["reason"] == "INTERNAL_ERROR"
        assert terminal["final_safety_state"] == "STOP_REQUESTED_UNCONFIRMED"
        assert "exact zero" in terminal["message"]
        await demo.close()

    asyncio.run(scenario())
