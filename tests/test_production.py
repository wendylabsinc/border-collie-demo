from __future__ import annotations

import asyncio

import pytest

from border_collie_demo.hardware import CameraFailure, TargetLost
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


def context(
    *,
    outbound_forward_pulses: int = 0,
    orientation_degrees: float = 0.0,
) -> StageContext:
    return StageContext(
        run_id="run-1",
        target_fruit="pear",
        home={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        orientation_degrees=orientation_degrees,
        outbound_forward_pulses=outbound_forward_pulses,
    )


def test_orient_for_run_uses_the_recorded_relative_angle() -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        stages = ProductionStageExecutor(hardware, dict, FakeBark())

        evidence = await stages.execute(
            MissionPhase.ORIENT_FOR_RUN,
            context(orientation_degrees=137.0),
        )

        name, angle_rad, options = hardware.calls[0]
        assert name == "turn_relative"
        assert angle_rad == pytest.approx(2.391101)
        assert options == {
            "yaw_rps": 1.00,
            "tolerance_rad": pytest.approx(0.05235987756),
            "timeout_s": 30.0,
        }
        assert evidence["requested_angle_degrees"] == 137.0
        assert evidence["motion_commands_sent"] is True

    asyncio.run(scenario())


def test_zero_degree_orientation_is_recorded_without_motion() -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        stages = ProductionStageExecutor(hardware, dict, FakeBark())

        evidence = await stages.execute(MissionPhase.ORIENT_FOR_RUN, context())

        assert hardware.calls == []
        assert evidence == {
            "requested_angle_degrees": 0.0,
            "requested_angle_rad": 0.0,
            "measured_yaw_change_rad": 0.0,
            "orientation_skipped": True,
            "motion_commands_sent": False,
        }

    asyncio.run(scenario())


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
            "yaw_rps": 1.00,
            "sweep_rad": pytest.approx(2.0 * 3.141592653589793),
            "timeout_s": 30.0,
        }
        assert evidence["stable_detections"] == 5

    asyncio.run(scenario())


def test_turn_to_fruit_uses_the_selected_red_apple_target() -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        status_reader = lambda: {"camera_healthy": True, "target_ready": False}
        stages = ProductionStageExecutor(hardware, status_reader, FakeBark())
        apple = StageContext(
            run_id="run-apple",
            target_fruit="apple",
            home={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        )

        evidence = await stages.execute(MissionPhase.TURN_TO_FRUIT, apple)

        assert hardware.calls[0][2] == "apple"
        assert evidence["label"] == "apple"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "phase",
    [MissionPhase.TURN_TO_FRUIT, MissionPhase.FIND_FRUIT],
)
def test_search_stage_skips_motion_when_target_is_already_visible(
    phase: MissionPhase,
) -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        status_reader = lambda: {
            "camera_healthy": True,
            "target_ready": True,
            "target_fruit": "pear",
            "detection": {
                "label": "pear",
                "confidence": 0.82,
                "consecutive_detections": 7,
                "age_s": 0.04,
                "center_x_ratio": 0.54,
                "center_y_ratio": 0.61,
                "bottom_ratio": 0.70,
            },
        }
        stages = ProductionStageExecutor(hardware, status_reader, FakeBark())

        evidence = await stages.execute(phase, context())

        assert hardware.calls == []
        assert evidence == {
            "label": "pear",
            "confidence": 0.82,
            "stable_detections": 7,
            "detection_age_s": 0.04,
            "center_x_ratio": 0.54,
            "center_y_ratio": 0.61,
            "bottom_ratio": 0.70,
            "search_progress_rad": 0.0,
            "search_skipped": True,
            "skip_reason": "target_already_visible",
            "motion_commands_sent": False,
        }

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


def test_stage_result_records_the_exact_velocity_commands() -> None:
    class TracedHardware(FakeProductionHardware):
        def __init__(self) -> None:
            super().__init__()
            self.phase: str | None = None

        def start_motion_trace(self, phase: str) -> None:
            self.phase = phase

        def motion_trace(self) -> list[dict[str, object]]:
            return [
                {
                    "sequence": 1,
                    "phase": self.phase,
                    "forward_mps": 0.0,
                    "yaw_rps": 0.5,
                    "reason": "find_target",
                }
            ]

    async def scenario() -> None:
        hardware = TracedHardware()
        stages = ProductionStageExecutor(hardware, dict, FakeBark())

        evidence = await stages.execute(MissionPhase.TURN_TO_FRUIT, context())

        assert evidence["motion_commands"] == [
            {
                "sequence": 1,
                "phase": "turn_to_fruit",
                "forward_mps": 0.0,
                "yaw_rps": 0.5,
                "reason": "find_target",
            }
        ]

    asyncio.run(scenario())


def test_approach_pins_full_and_close_range_speeds_independently() -> None:
    async def scenario() -> None:
        hardware = FakeProductionHardware()
        status_reader = lambda: {"ready": True}
        stages = ProductionStageExecutor(hardware, status_reader, FakeBark())

        evidence = await stages.execute(MissionPhase.APPROACH_FRUIT, context())

        name, reader, fruit, options = hardware.calls[0]
        assert (name, reader, fruit) == ("approach_target", status_reader, "pear")
        assert options == {
            # Normal tracking retains the qualified profile; close-range
            # geometry selects its own explicit speed before Arrival stops it.
            "forward_mps": 1.0,
            "maximum_yaw_rps": 0.5,
            "near_bottom_ratio": 0.86,
            "near_center_ratio": 0.72,
            "near_confirmations": 3,
            "near_loss_grace_s": 0.75,
            "close_range_mps": 0.55,
            "final_push_mps": 1.0,
            "final_push_duration_s": 1.0,
            "timeout_s": 20.0,
            "metric_arrival_required": True,
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


def test_search_target_loss_is_classified_with_recognition_evidence() -> None:
    class DistantPearHardware(FakeProductionHardware):
        async def find_target(self, *_args, **_options):
            raise TargetLost(
                "pear was not found in the bounded search sweep",
                evidence={
                    "samples": 42,
                    "pear_candidate_samples": 27,
                    "maximum_confidence": 0.019,
                    "maximum_bbox_area_ratio": 0.001,
                },
            )

    async def scenario() -> None:
        stages = ProductionStageExecutor(DistantPearHardware(), dict, FakeBark())

        with pytest.raises(StageFailure) as failure:
            await stages.execute(MissionPhase.TURN_TO_FRUIT, context())

        assert failure.value.reason == "TARGET_RECOGNITION_FAILURE"
        assert failure.value.message == (
            "pear was not found in the bounded search sweep"
        )
        assert failure.value.details == {
            "recognition": {
                "samples": 42,
                "pear_candidate_samples": 27,
                "maximum_confidence": 0.019,
                "maximum_bbox_area_ratio": 0.001,
            }
        }

    asyncio.run(scenario())
