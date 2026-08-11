import math

import pytest

from border_collie_demo.return_home import (
    Pose2D,
    ReturnMode,
    ReturnPlannerConfig,
    plan_return_step,
)


def test_return_planner_turns_drives_at_reliable_speed_then_restores_heading() -> None:
    config = ReturnPlannerConfig(
        arrival_tolerance_m=0.10,
        heading_tolerance_rad=math.radians(5.0),
        heading_gate_rad=math.radians(20.0),
        forward_mps=0.55,
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
    assert drive.forward_mps == 0.55
    assert restore.mode is ReturnMode.RESTORE_HEADING
    assert restore.distance_m == pytest.approx(0.08)
    assert complete.mode is ReturnMode.COMPLETE
