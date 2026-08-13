import math

import pytest

from border_collie_demo.return_home import (
    Pose2D,
    ReturnMode,
    ReturnPlannerConfig,
    plan_position_return_step,
    plan_return_step,
)


def test_return_planner_turns_drives_at_reliable_speed_then_restores_heading() -> None:
    config = ReturnPlannerConfig(
        arrival_tolerance_m=0.10,
        heading_tolerance_rad=math.radians(5.0),
        heading_gate_rad=math.radians(20.0),
        forward_mps=0.50,
        maximum_yaw_rps=0.30,
    )
    home = Pose2D(0.0, 0.0, 0.0)

    turn = plan_return_step(home, Pose2D(1.0, 0.0, 0.0), config)
    drive = plan_return_step(home, Pose2D(1.0, 0.0, math.pi), config)
    restore = plan_return_step(home, Pose2D(0.08, 0.0, math.pi / 2.0), config)
    complete = plan_return_step(home, Pose2D(0.08, 0.0, math.radians(4.0)), config)

    assert turn.mode is ReturnMode.TURN_TO_HOME
    assert turn.forward_mps == 0.0
    assert drive.mode is ReturnMode.DRIVE_TO_HOME
    assert drive.forward_mps == 0.50
    assert restore.mode is ReturnMode.RESTORE_HEADING
    assert restore.distance_m == pytest.approx(0.08)
    assert complete.mode is ReturnMode.COMPLETE


def test_position_only_return_never_restores_heading_after_reaching_home() -> None:
    config = ReturnPlannerConfig(
        arrival_tolerance_m=0.10,
        heading_tolerance_rad=math.radians(5.0),
        heading_gate_rad=math.radians(20.0),
        forward_mps=1.0,
        maximum_yaw_rps=0.50,
    )

    step = plan_position_return_step(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.08, 0.0, 2.5),
        config,
    )

    assert step.mode is ReturnMode.COMPLETE
    assert step.distance_m == pytest.approx(0.08)
    assert step.forward_mps == 0.0
    assert step.yaw_rps == 0.0


def test_position_planner_exposes_heading_escape_instead_of_authorizing_forward() -> None:
    config = ReturnPlannerConfig(
        arrival_tolerance_m=0.10,
        heading_tolerance_rad=math.radians(5.0),
        heading_gate_rad=math.radians(20.0),
        forward_mps=1.0,
        maximum_yaw_rps=0.50,
    )
    home = Pose2D(0.0, 0.0, 0.0)

    escaped = plan_position_return_step(home, Pose2D(1.0, 0.0, 0.0), config)
    steerable = plan_position_return_step(
        home,
        Pose2D(1.0, 0.0, math.radians(170.0)),
        config,
    )

    assert escaped.mode is ReturnMode.TURN_TO_HOME
    assert escaped.forward_mps == 0.0
    assert escaped.yaw_rps != 0.0
    assert steerable.mode is ReturnMode.DRIVE_TO_HOME
    assert steerable.forward_mps == 1.0
    assert 0.0 < abs(steerable.yaw_rps) <= 0.50
