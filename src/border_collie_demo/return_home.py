"""Pure closed-loop decisions for returning to the captured Home pose."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

VERIFIED_MINIMUM_YAW_RPS = 0.50


class ReturnMode(str, Enum):
    TURN_TO_HOME = "turn_to_home"
    DRIVE_TO_HOME = "drive_to_home"
    RESTORE_HEADING = "restore_heading"
    COMPLETE = "complete"


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) for value in (self.x_m, self.y_m, self.yaw_rad)
        ):
            raise ValueError("pose values must be finite")


@dataclass(frozen=True)
class ReturnPlannerConfig:
    arrival_tolerance_m: float
    heading_tolerance_rad: float
    heading_gate_rad: float
    forward_mps: float
    maximum_yaw_rps: float
    minimum_yaw_rps: float = 0.50

    def __post_init__(self) -> None:
        values = (
            self.arrival_tolerance_m,
            self.heading_tolerance_rad,
            self.heading_gate_rad,
            self.forward_mps,
            self.maximum_yaw_rps,
            self.minimum_yaw_rps,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("return planner values must be finite and positive")
        if self.heading_tolerance_rad > self.heading_gate_rad:
            raise ValueError("heading tolerance exceeds the course gate")
        if self.minimum_yaw_rps > self.maximum_yaw_rps:
            raise ValueError("minimum return yaw exceeds maximum return yaw")
        if self.minimum_yaw_rps < VERIFIED_MINIMUM_YAW_RPS:
            raise ValueError(
                "minimum return yaw is below the verified 0.50 rad/s turning signal"
            )


@dataclass(frozen=True)
class ReturnStep:
    mode: ReturnMode
    distance_m: float
    heading_error_rad: float
    forward_mps: float
    yaw_rps: float


def plan_return_step(
    home: Pose2D,
    current: Pose2D,
    config: ReturnPlannerConfig,
) -> ReturnStep:
    dx = home.x_m - current.x_m
    dy = home.y_m - current.y_m
    distance = math.hypot(dx, dy)
    if distance <= config.arrival_tolerance_m:
        heading_error = normalize_angle(home.yaw_rad - current.yaw_rad)
        mode = (
            ReturnMode.COMPLETE
            if abs(heading_error) <= config.heading_tolerance_rad
            else ReturnMode.RESTORE_HEADING
        )
        return ReturnStep(
            mode, distance, heading_error, 0.0, _yaw(heading_error, config)
        )

    target_yaw = math.atan2(dy, dx)
    heading_error = normalize_angle(target_yaw - current.yaw_rad)
    if abs(heading_error) > config.heading_gate_rad:
        return ReturnStep(
            ReturnMode.TURN_TO_HOME,
            distance,
            heading_error,
            0.0,
            _yaw(heading_error, config),
        )
    return ReturnStep(
        ReturnMode.DRIVE_TO_HOME,
        distance,
        heading_error,
        config.forward_mps,
        _yaw(heading_error, config),
    )


def plan_position_return_step(
    home: Pose2D,
    current: Pose2D,
    config: ReturnPlannerConfig,
) -> ReturnStep:
    """Plan toward Home while treating position as the only terminal gate.

    Heading is used only to steer toward the captured position. Once the
    measured position is inside the arrival tolerance, no heading-restoration
    command is authorized: turning after reaching Home can move the body back
    outside the position gate on the Go2.
    """
    step = plan_return_step(home, current, config)
    if step.distance_m <= config.arrival_tolerance_m:
        return ReturnStep(
            ReturnMode.COMPLETE,
            step.distance_m,
            step.heading_error_rad,
            0.0,
            0.0,
        )
    return step


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _yaw(error_rad: float, config: ReturnPlannerConfig) -> float:
    if abs(error_rad) <= config.heading_tolerance_rad:
        return 0.0
    magnitude = min(
        config.maximum_yaw_rps,
        max(config.minimum_yaw_rps, abs(error_rad)),
    )
    return math.copysign(magnitude, error_rad)
