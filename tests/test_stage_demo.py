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
        self.x_m = 1.0
        self.y_m = 2.0
        self.home_x_m = 1.0
        self.home_y_m = 2.0
        self.home_yaw_rad = 0.25

    def drift_to(self, x_m: float, y_m: float, yaw_rad: float = 0.25) -> None:
        """Stand somewhere new, as a run that ended off-Home would leave Woof."""
        self.x_m = self.home_x_m = x_m
        self.y_m = self.home_y_m = y_m
        self.home_yaw_rad = yaw_rad

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
            "x_m": self.home_x_m,
            "y_m": self.home_y_m,
            "yaw_rad": self.home_yaw_rad,
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
            "pose": {
                "healthy": True,
                "age_s": 0.01,
                "error": None,
                "pose": {"x_m": self.x_m, "y_m": self.y_m},
            },
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


def test_next_activation_waits_for_fresh_return_to_prior_run_home(tmp_path) -> None:
    async def scenario() -> None:
        results = RunResultStore(tmp_path)
        prior = results.start_run(
            target_fruit="banana",
            activation_source="soak",
            activation_id="attempt-1",
        )
        results.record_home(
            prior["run_id"],
            {
                "x_m": 1.0,
                "y_m": 2.0,
                "yaw_rad": 0.0,
                "age_s": 0.01,
                "source": "test",
            },
        )
        results.seal(
            prior["run_id"],
            phase="failed",
            outcome="FAILED",
            reason="ARRIVAL_FAILURE",
            message="guidance timed out",
            final_safety_state="DISARMED_CONFIRMED",
        )
        hardware = ReadyHardware()
        hardware.x_m = 2.0
        demo = StageDemo(
            MissionMachine(),
            results,
            hardware,
            ready_camera,
            stage_home_margin_m=0.50,
        )
        await demo.start()

        blocked = demo.status()

        assert blocked["activation"]["ready"] is False
        assert blocked["activation"]["inter_run"] == {
            "required": True,
            "prior_run_id": prior["run_id"],
            "prior_outcome": "FAILED",
            "home_distance_m": 1.0,
            "stage_home_margin_m": 0.5,
            "returned_home": False,
        }
        assert blocked["activation"]["blockers"][-1]["name"] == (
            "inter_run_home_clearance"
        )
        with pytest.raises(ActiveRunError, match="has not returned Home"):
            await demo.activate(FruitMission("pear", "soak", "attempt-2"))

        hardware.x_m = 1.2
        ready = demo.status()

        assert ready["activation"]["ready"] is True
        assert ready["activation"]["inter_run"]["returned_home"] is True
        activation = await demo.activate(FruitMission("pear", "soak", "attempt-2"))
        assert activation.run["target_fruit"] == "pear"
        await demo.stop()
        await demo.close()

    asyncio.run(scenario())


async def _complete_run(
    demo: StageDemo, mission: FruitMission
) -> dict[str, object]:
    activation = await demo.activate(mission)
    return await demo.wait(activation.run["run_id"], timeout_s=5.0)


def _stage_demo(results: RunResultStore, hardware: ReadyHardware) -> StageDemo:
    return StageDemo(
        MissionMachine(),
        results,
        hardware,
        ready_camera,
        stage_executor=SimulatedStageExecutor(),
    )


def test_second_activation_reuses_the_persisted_stage_home(tmp_path) -> None:
    """Home is captured once; a later run must not adopt where Woof stopped."""

    async def scenario() -> None:
        hardware = ReadyHardware()
        demo = _stage_demo(RunResultStore(tmp_path), hardware)
        await demo.start()

        first = await _complete_run(demo, FruitMission("pear", "soak", "attempt-1"))
        # Woof finished 0.10 m off Home, inside the stage margin.
        hardware.drift_to(1.10, 2.0)
        second = await _complete_run(demo, FruitMission("pear", "soak", "attempt-2"))

        assert first["home"]["x_m"] == 1.0
        assert second["home"] == first["home"]
        assert first["home_provenance"]["source"] == "CAPTURED"
        assert second["home_provenance"]["source"] == "REUSED"
        assert second["home_provenance"]["activation_offset_m"] == pytest.approx(0.10)
        assert second["home_provenance"]["measured_pose"]["x_m"] == pytest.approx(1.10)
        assert demo.status()["stage_home"] == first["home"]
        await demo.close()

    asyncio.run(scenario())


def test_stage_home_does_not_drift_across_a_cohort_of_runs(tmp_path) -> None:
    """Successive runs share one Home instead of compounding their own error."""

    async def scenario() -> None:
        hardware = ReadyHardware()
        demo = _stage_demo(RunResultStore(tmp_path), hardware)
        await demo.start()

        homes = []
        for number in range(1, 6):
            run = await _complete_run(
                demo, FruitMission("pear", "cohort", f"cohort-run-{number}")
            )
            homes.append(run["home"])
            # Each run ends 0.12 m further out than the last would have.
            hardware.drift_to(1.0 + 0.12 * number, 2.0)

        assert all(home == homes[0] for home in homes)
        assert {home["x_m"] for home in homes} == {1.0}
        await demo.close()

    asyncio.run(scenario())


def test_failed_run_does_not_reset_the_stage_home(tmp_path) -> None:
    """A failure epilogue that leaves Woof rotated must not redefine Home."""

    async def scenario() -> None:
        hardware = ReadyHardware()
        demo = StageDemo(
            MissionMachine(),
            RunResultStore(tmp_path),
            hardware,
            ready_camera,
            stage_executor=SimulatedStageExecutor(fail_at="approach_fruit"),
        )
        await demo.start()

        first = await _complete_run(demo, FruitMission("pear", "soak", "attempt-1"))
        # The failed run left Woof rotated 2.4 rad and 0.20 m out.
        hardware.drift_to(1.20, 2.0, yaw_rad=2.65)

        assert first["outcome"] == "FAILED"
        assert demo.status()["stage_home"] == first["home"]

        second = await _complete_run(demo, FruitMission("pear", "soak", "attempt-2"))

        assert second["outcome"] == "FAILED"
        assert second["home"] == first["home"]
        assert second["home"]["yaw_rad"] == pytest.approx(0.25)
        assert second["home"]["x_m"] == pytest.approx(1.0)
        assert second["home_provenance"]["source"] == "REUSED"
        await demo.close()

    asyncio.run(scenario())


def test_stage_home_survives_a_process_restart(tmp_path) -> None:
    async def scenario() -> None:
        hardware = ReadyHardware()
        demo = _stage_demo(RunResultStore(tmp_path), hardware)
        await demo.start()
        first = await _complete_run(demo, FruitMission("pear", "soak", "attempt-1"))
        await demo.close()

        hardware.drift_to(1.15, 2.0)
        restarted = _stage_demo(RunResultStore(tmp_path), hardware)
        await restarted.start()

        assert restarted.status()["stage_home"] == first["home"]
        resumed = await _complete_run(
            restarted, FruitMission("pear", "soak", "attempt-2")
        )

        assert resumed["home"] == first["home"]
        assert resumed["home_provenance"]["source"] == "REUSED"
        await restarted.close()

    asyncio.run(scenario())


def test_explicit_recapture_is_the_only_thing_that_moves_home(tmp_path) -> None:
    async def scenario() -> None:
        hardware = ReadyHardware()
        demo = _stage_demo(RunResultStore(tmp_path), hardware)
        await demo.start()

        first = await _complete_run(demo, FruitMission("pear", "soak", "attempt-1"))
        hardware.drift_to(1.30, 2.0)
        moved = await demo.recapture_home()

        assert moved["previous_home"] == first["home"]
        assert moved["home"]["x_m"] == pytest.approx(1.30)
        assert moved["offset_from_previous_m"] == pytest.approx(0.30)

        second = await _complete_run(demo, FruitMission("pear", "soak", "attempt-2"))

        assert second["home"]["x_m"] == pytest.approx(1.30)
        assert second["home_provenance"]["source"] == "REUSED"
        assert second["home_provenance"]["activation_offset_m"] == pytest.approx(0.0)
        await demo.close()

    asyncio.run(scenario())


def test_recapture_home_is_refused_while_a_demo_run_is_active(tmp_path) -> None:
    async def scenario() -> None:
        demo = _stage_demo(RunResultStore(tmp_path), ReadyHardware())
        await demo.start()
        await demo.activate(FruitMission("pear", "soak", "attempt-1"))

        with pytest.raises(ActiveRunError, match="Home cannot be recaptured"):
            await demo.recapture_home()

        await demo.stop()
        await demo.close()

    asyncio.run(scenario())
