from __future__ import annotations

import asyncio

import pytest

from border_collie_demo.hardware import CameraFailure
from border_collie_demo.models import MissionPhase
from border_collie_demo.orchestrator import StageContext, StageFailure
from border_collie_demo.production import ProductionStageExecutor


class FakeProductionHardware:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    async def turn_relative(self, angle_rad: float, **options: float) -> dict[str, object]:
        self.calls.append(("turn_relative", angle_rad, options))
        return {"measured_yaw_change_rad": angle_rad, "motion_commands_sent": True}

    async def emergency_stop(self) -> list[str]:
        self.calls.append(("emergency_stop",))
        return []

    async def find_target(self, status_reader, target_fruit: str, **options: float) -> dict[str, object]:
        self.calls.append(("find_target", status_reader, target_fruit, options))
        return {"label": target_fruit, "stable_detections": 5, "motion_commands_sent": True}

    async def approach_target(self, status_reader, target_fruit: str, **options: float) -> dict[str, object]:
        self.calls.append(("approach_target", status_reader, target_fruit, options))
        return {"arrival_confirmed": True, "motion_commands_sent": True}

    async def stand_down(self) -> dict[str, object]:
        self.calls.append(("stand_down",))
        return {"posture": "stand_down", "motion_commands_sent": True}

    async def stand_up(self, **options: float) -> dict[str, object]:
        self.calls.append(("stand_up", options))
        return {
            "posture": "balance_stand",
            "motion_commands_sent": True,
            **options,
        }

    async def turn_toward_home(self, home: dict[str, object], **options: float) -> dict[str, object]:
        self.calls.append(("turn_toward_home", home, options))
        return {"home_bearing_error_rad": 0.02, "motion_commands_sent": True}

    async def return_home(self, home: dict[str, object], **options: float) -> dict[str, object]:
        self.calls.append(("return_home", home, options))
        return {"home_distance_m": 0.08, "motion_commands_sent": True}

    async def restore_home_heading(self, home: dict[str, object], **options: float) -> dict[str, object]:
        self.calls.append(("restore_home_heading", home, options))
        return {"home_distance_m": 0.08, "heading_error_rad": 0.04, "motion_commands_sent": True}


class FakeBark:
    async def bark(self) -> dict[str, object]:
        return {"bark_played": True}


def context(*, outbound_forward_pulses: int = 0) -> StageContext:
    return StageContext(
        run_id="run-1",
        target_fruit="pear",
        home={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        outbound_forward_pulses=outbound_forward_pulses,
    )


def test_turn_to_fruit_rotates_until_the_pear_is_recognized() -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        status_reader = lambda: {"camera_healthy": True, "target_ready": False}
        stages = ProductionStageExecutor(hardware, status_reader, FakeBark())

        evidence = await stages.execute(MissionPhase.TURN_TO_FRUIT, context())

        name, reader, fruit, options = hardware.calls[0]
        assert (name, reader, fruit) == (
            "find_target",
            status_reader,
            "pear",
        )
        assert options == {
            "yaw_rps": 0.50,
            "sweep_rad": pytest.approx(2.0 * 3.141592653589793),
            "timeout_s": 30.0,
        }
        assert evidence["stable_detections"] == 5

    asyncio.run(scenario())


def test_find_fruit_runs_bounded_camera_guided_search() -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        status_reader = lambda: {"ready": True}
        stages = ProductionStageExecutor(hardware, status_reader, FakeBark())

        evidence = await stages.execute(MissionPhase.FIND_FRUIT, context())

        name, reader, fruit, options = hardware.calls[0]
        assert (name, reader, fruit) == ("find_target", status_reader, "pear")
        assert options == {
            "yaw_rps": 0.20,
            "sweep_rad": pytest.approx(1.308997),
            "timeout_s": 9.0,
        }
        assert evidence["stable_detections"] == 5

    asyncio.run(scenario())


def test_approach_uses_measured_factory_motion_and_one_final_push() -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        status_reader = lambda: {"ready": True}
        stages = ProductionStageExecutor(hardware, status_reader, FakeBark())

        evidence = await stages.execute(MissionPhase.APPROACH_FRUIT, context())

        name, reader, fruit, options = hardware.calls[0]
        assert (name, reader, fruit) == ("approach_target", status_reader, "pear")
        assert options == {
            # The factory-avoidance calibration established 0.50 m/s as the
            # deadband edge, not a production value with usable margin. The
            # approach uses the separately verified 1.0 m/s signal while the
            # camera controller continuously steers and retains its arrival
            # and timeout gates.
            "forward_mps": 1.0,
            "maximum_yaw_rps": 0.30,
            "near_bottom_ratio": 0.86,
            "near_center_ratio": 0.72,
            "near_confirmations": 3,
            "near_loss_grace_s": 0.75,
            "final_push_mps": 1.0,
            "final_push_duration_s": 1.0,
            "timeout_s": 20.0,
        }
        assert evidence["arrival_confirmed"] is True

    asyncio.run(scenario())


def test_audience_action_sits_barks_then_stands_in_separate_stages() -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        bark = FakeBark()
        holds: list[float] = []

        async def hold(duration_s: float) -> None:
            holds.append(duration_s)

        stages = ProductionStageExecutor(hardware, dict, bark, sleep=hold)

        action = await stages.execute(MissionPhase.SIT_AND_BARK, context())
        standing = await stages.execute(MissionPhase.STAND, context())

        assert hardware.calls == [
            ("emergency_stop",),
            ("stand_down",),
            ("stand_up", {"settle_s": 1.0}),
        ]
        assert holds == [1.0, 5.0]
        assert action == {
            "posture": "stand_down",
            "motion_commands_sent": True,
            "bark_played": True,
            "arrival_stop_confirmed": True,
            "arrival_stop_settle_s": 1.0,
            "down_hold_s": 5.0,
        }
        assert standing == {
            "posture": "balance_stand",
            "motion_commands_sent": True,
            "settle_s": 1.0,
        }

    asyncio.run(scenario())


def test_return_stages_use_captured_home_and_locked_arrival_rules() -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        stages = ProductionStageExecutor(hardware, dict, FakeBark())

        turn = await stages.execute(MissionPhase.TURN_TOWARD_HOME, context())
        returned = await stages.execute(
            MissionPhase.RETURN_HOME,
            context(outbound_forward_pulses=12),
        )
        restored = await stages.execute(MissionPhase.RESTORE_HEADING, context())

        assert [call[0] for call in hardware.calls] == [
            "turn_toward_home",
            "return_home",
            "restore_home_heading",
        ]
        assert all(call[1] == context().home for call in hardware.calls)
        assert hardware.calls[0][2] == {
            "yaw_rps": 0.50,
            "tolerance_rad": pytest.approx(0.0872665),
            "response_timeout_s": 0.75,
            "response_min_progress_rad": pytest.approx(0.0349066),
            "recovery_settle_s": 1.0,
            "timeout_s": 30.0,
        }
        assert hardware.calls[1][2] == {
            "forward_mps": 1.0,
            "forward_pulse_count": 12,
            "arrival_tolerance_m": 0.10,
            "heading_gate_rad": pytest.approx(0.349066),
            "maximum_yaw_rps": 0.30,
            "minimum_progress_m": 0.03,
            "stall_timeout_s": 2.0,
            "timeout_s": 30.0,
        }
        assert hardware.calls[2][2] == {
            "yaw_rps": 0.30,
            "heading_tolerance_rad": pytest.approx(0.0872665),
            "position_tolerance_m": 0.10,
            "timeout_s": 15.0,
        }
        assert turn["home_bearing_error_rad"] == 0.02
        assert returned["home_distance_m"] == 0.08
        assert restored["heading_error_rad"] == 0.04

    asyncio.run(scenario())


def test_camera_failure_is_preserved_as_the_terminal_stage_reason() -> None:
    class FailedCameraHardware(FakeProductionHardware):
        async def find_target(self, *_args, **_options):
            raise CameraFailure("source progress is stale")

    async def scenario() -> None:
        stages = ProductionStageExecutor(FailedCameraHardware(), dict, FakeBark())

        with pytest.raises(StageFailure) as failure:
            await stages.execute(MissionPhase.FIND_FRUIT, context())

        assert failure.value.reason == "CAMERA_FAILURE"
        assert failure.value.message == "source progress is stale"

    asyncio.run(scenario())
