"""Pure, auditable policies for fruit missions' short-term actions.

This module deliberately has no ROS2, audio, network, or model imports.  A
controller can call :func:`decide_mission_action` with a world snapshot and
turn the returned decision into an ordinary RobotKit ``Effect``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

from robotkit.contracts import ObservationRecord, WorldSnapshot, utc_now
from robotkit.fruits import DEFAULT_FRUIT, SUPPORTED_FRUITS, normalize_fruit


@dataclass(frozen=True)
class MissionConfig:
    search_angular_rps: float = 0.45
    max_linear_mps: float = 0.30
    max_angular_rps: float = 0.80
    steering_gain: float = 1.5
    camera_horizontal_fov_rad: float = math.radians(90.0)
    lidar_frame_id: str = "base_link"
    apple_stop_distance_m: float = 0.30
    fruit_stop_distance_m: float | None = None
    home_x_m: float = 0.0
    home_y_m: float = 0.0
    home_yaw_rad: float = 0.0
    home_position_tolerance_m: float = 0.15
    home_yaw_tolerance_rad: float = 0.15

    def __post_init__(self) -> None:
        positive = {
            "search_angular_rps": self.search_angular_rps,
            "max_linear_mps": self.max_linear_mps,
            "max_angular_rps": self.max_angular_rps,
            "steering_gain": self.steering_gain,
            "camera_horizontal_fov_rad": self.camera_horizontal_fov_rad,
            "fruit_stop_distance_m": self.stop_distance_m,
            "home_position_tolerance_m": self.home_position_tolerance_m,
            "home_yaw_tolerance_rad": self.home_yaw_tolerance_rad,
        }
        if any(not math.isfinite(value) or value <= 0 for value in positive.values()):
            raise ValueError("mission speed, geometry, and tolerance values must be positive")
        if not all(
            math.isfinite(value)
            for value in (self.home_x_m, self.home_y_m, self.home_yaw_rad)
        ):
            raise ValueError("home pose must be finite")
        if not self.lidar_frame_id:
            raise ValueError("lidar_frame_id must not be empty")

    @property
    def stop_distance_m(self) -> float:
        """Configured distance for any fruit, with the legacy apple setting as fallback."""

        return (
            self.fruit_stop_distance_m
            if self.fruit_stop_distance_m is not None
            else self.apple_stop_distance_m
        )


@dataclass(frozen=True)
class MissionDecision:
    """Controller-facing result for one control-loop iteration.

    ``completed`` tells the mission state machine that the action's physical
    postcondition already holds.  A zero velocity decision is still returned
    on completion so a prior velocity command is actively cancelled.
    """

    effect_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    require_fresh_streams: tuple[str, ...] = ()
    completed: bool = False
    reason: str = ""

    @property
    def stage_complete(self) -> bool:
        """State-machine-friendly alias for ``completed``."""

        return self.completed


def decide_mission_action(
    action: str,
    snapshot: WorldSnapshot,
    *,
    at: datetime | None = None,
    config: MissionConfig = MissionConfig(),
    target: str | None = None,
) -> MissionDecision:
    """Dispatch a semantic mission action to its deterministic policy."""

    normalized = action.strip().casefold().replace("-", "_").replace(" ", "_")
    at = at or snapshot.captured_at or utc_now()
    requested_target = normalize_fruit(target or "")
    if normalized.startswith("search_"):
        requested_target = normalize_fruit(normalized.removeprefix("search_"))
        if requested_target:
            return search_fruit(snapshot, requested_target, at=at, config=config)
    if normalized.startswith("approach_"):
        requested_target = normalize_fruit(normalized.removeprefix("approach_"))
        if requested_target:
            return approach_fruit(snapshot, requested_target, at=at, config=config)
    if normalized == "approach":
        return approach_fruit(
            snapshot, requested_target or DEFAULT_FRUIT, at=at, config=config
        )
    if normalized == "bark":
        return bark()
    if normalized in {"go_home", "return_home"}:
        return go_home(snapshot, at=at, config=config)
    if normalized in {"lie_down", "idle", "idle_at_home", "hold_position"}:
        return _stop(completed=normalized != "lie_down", reason="holding position")
    return _stop(reason=f"unsupported mission action: {action}")


def search_apple(
    snapshot: WorldSnapshot,
    *,
    at: datetime,
    config: MissionConfig = MissionConfig(),
) -> MissionDecision:
    """Backward-compatible apple-specific search entry point."""

    return search_fruit(snapshot, "apple", at=at, config=config)


def search_fruit(
    snapshot: WorldSnapshot,
    target: str,
    *,
    at: datetime,
    config: MissionConfig = MissionConfig(),
) -> MissionDecision:
    target = _require_fruit(target)
    vision = _fresh_observation(snapshot, "vision.fruits", at)
    if vision is not None and _best_fruit(vision.payload, target) is not None:
        return _stop(completed=True, reason=f"fresh YOLO observation contains {target}")
    return MissionDecision(
        "cmd_vel",
        {"linear_x_mps": 0.0, "angular_z_rps": config.search_angular_rps},
        completed=False,
        reason=f"rotate to acquire {target}",
    )


def approach_apple(
    snapshot: WorldSnapshot,
    *,
    at: datetime,
    config: MissionConfig = MissionConfig(),
) -> MissionDecision:
    """Backward-compatible apple-specific approach entry point."""

    return approach_fruit(snapshot, "apple", at=at, config=config)


def approach_fruit(
    snapshot: WorldSnapshot,
    target: str,
    *,
    at: datetime,
    config: MissionConfig = MissionConfig(),
) -> MissionDecision:
    """Steer toward the requested fruit using the LIDAR sector at its bearing.

    Forward velocity is produced only when *both* observations are fresh and a
    finite range exists for the target bearing.  A missing target/range actively
    produces zero velocity rather than allowing dead reckoning.
    """

    target = _require_fruit(target)
    vision = _fresh_observation(snapshot, "vision.fruits", at)
    proximity = _fresh_observation(snapshot, "lidar.proximity", at)
    if vision is None:
        return _stop(reason="vision.fruits is missing or stale")
    fruit = _best_fruit(vision.payload, target)
    if fruit is None:
        return _stop(reason=f"fresh YOLO observation does not contain {target}")
    if proximity is None:
        return _stop(reason="lidar.proximity is missing or stale; refusing blind advance")
    if proximity.frame_id != config.lidar_frame_id:
        return _stop(
            reason=(
                f"lidar.proximity frame {proximity.frame_id!r} is not the configured "
                f"fusion frame {config.lidar_frame_id!r}"
            )
        )

    center_x = _bbox_center_x(fruit)
    if center_x is None:
        return _stop(reason=f"{target} detection has no valid normalized bounding box")
    bearing = (0.5 - center_x) * config.camera_horizontal_fov_rad
    distance = _range_at_bearing(proximity.payload, bearing)
    if distance is None:
        return _stop(reason=f"no finite LIDAR sector range covers the {target} bearing")
    if distance <= config.stop_distance_m:
        return _stop(
            completed=True,
            reason=f"{target} is within {config.stop_distance_m:.2f} m",
        )

    angular = _clamp(
        config.steering_gain * bearing,
        -config.max_angular_rps,
        config.max_angular_rps,
    )
    # Slow near the target and when much of the command is devoted to turning.
    range_speed = max(0.0, distance - config.stop_distance_m)
    alignment = max(0.0, math.cos(bearing))
    linear = min(config.max_linear_mps, range_speed) * alignment
    return MissionDecision(
        "cmd_vel",
        {
            "linear_x_mps": linear,
            "angular_z_rps": angular,
            "target_bearing_rad": bearing,
            "target_range_m": distance,
            "target": target,
        },
        ("vision.fruits", "lidar.proximity"),
        reason=f"approaching {target} with fresh camera/LIDAR fusion",
    )


def bark() -> MissionDecision:
    """Request the configured bark asset; completion is the executor result."""

    return MissionDecision(
        "unitree_bark",
        {"sound": "bark"},
        completed=False,
        reason="play configured bark WAV",
    )


def go_home(
    snapshot: WorldSnapshot,
    *,
    at: datetime,
    config: MissionConfig = MissionConfig(),
) -> MissionDecision:
    pose = _fresh_observation(snapshot, "localization.pose", at)
    if pose is None:
        return _stop(reason="localization.pose is missing or stale")
    try:
        x = _finite_float(pose.payload["x_m"])
        y = _finite_float(pose.payload["y_m"])
        yaw = _finite_float(pose.payload["yaw_rad"])
    except (KeyError, TypeError, ValueError):
        return _stop(reason="localization.pose payload is invalid")

    dx = config.home_x_m - x
    dy = config.home_y_m - y
    distance = math.hypot(dx, dy)
    if distance <= config.home_position_tolerance_m:
        yaw_error = _normalize_angle(config.home_yaw_rad - yaw)
        if abs(yaw_error) <= config.home_yaw_tolerance_rad:
            return _stop(completed=True, reason="home position and yaw reached")
        return MissionDecision(
            "cmd_vel",
            {
                "linear_x_mps": 0.0,
                "angular_z_rps": _clamp(
                    config.steering_gain * yaw_error,
                    -config.max_angular_rps,
                    config.max_angular_rps,
                ),
            },
            ("localization.pose",),
            reason="aligning to home yaw",
        )

    heading_error = _normalize_angle(math.atan2(dy, dx) - yaw)
    angular = _clamp(
        config.steering_gain * heading_error,
        -config.max_angular_rps,
        config.max_angular_rps,
    )
    # Do not translate when home is behind the robot.  Turn in place first.
    alignment = max(0.0, math.cos(heading_error))
    linear = min(config.max_linear_mps, distance) * alignment
    return MissionDecision(
        "cmd_vel",
        {"linear_x_mps": linear, "angular_z_rps": angular},
        ("localization.pose",),
        reason="driving toward configured home pose",
    )


def _fresh_observation(
    snapshot: WorldSnapshot, stream: str, at: datetime
) -> ObservationRecord | None:
    candidates = (item for item in snapshot.observations if item.stream == stream)
    observation = max(candidates, key=lambda item: item.revision, default=None)
    if observation is None or observation.is_stale(at):
        return None
    return observation


def _require_fruit(target: str) -> str:
    normalized = normalize_fruit(target)
    if normalized is None:
        raise ValueError(
            f"unsupported fruit {target!r}; expected one of {', '.join(sorted(SUPPORTED_FRUITS))}"
        )
    return normalized


def _best_fruit(
    payload: Mapping[str, Any], target: str
) -> Mapping[str, Any] | None:
    detections = payload.get("detections")
    if not isinstance(detections, Sequence) or isinstance(detections, (str, bytes)):
        return None
    matches = [
        item
        for item in detections
        if isinstance(item, Mapping)
        and str(item.get("class_name", item.get("label", ""))).casefold() == target
    ]
    return max(matches, key=lambda item: _confidence(item), default=None)


def _confidence(item: Mapping[str, Any]) -> float:
    try:
        value = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _bbox_center_x(item: Mapping[str, Any]) -> float | None:
    box = item.get("bbox_xyxy_normalized", item.get("bbox_normalized"))
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
        return None
    try:
        x1, _, x2, _ = (_finite_float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x1 < x2 <= 1.0):
        return None
    return (x1 + x2) / 2.0


def _range_at_bearing(payload: Mapping[str, Any], bearing: float) -> float | None:
    """Read the closest valid range from the producer's angular-sector schema."""

    raw_sectors = payload.get("sectors", payload.get("angular_sectors"))
    if not isinstance(raw_sectors, Sequence) or isinstance(raw_sectors, (str, bytes)):
        return None
    matches: list[tuple[float, float]] = []
    for sector in raw_sectors:
        if not isinstance(sector, Mapping):
            continue
        try:
            start = _finite_float(
                sector.get(
                    "bearing_min_rad",
                    sector.get("start_angle_rad", sector.get("angle_min_rad")),
                )
            )
            end = _finite_float(
                sector.get(
                    "bearing_max_rad",
                    sector.get("end_angle_rad", sector.get("angle_max_rad")),
                )
            )
            distance = _finite_float(
                sector.get(
                    "nearest_distance_m",
                    sector.get("min_range_m", sector.get("distance_m")),
                )
            )
        except (TypeError, ValueError):
            continue
        if distance <= 0:
            continue
        center = _normalize_angle((start + end) / 2.0)
        if _angle_in_sector(bearing, start, end):
            matches.append((abs(_normalize_angle(bearing - center)), distance))
    if not matches:
        return None
    # Prefer the most centered overlapping sector, rather than a more distant
    # neighboring sector whose edge happens to overlap the bearing.
    return min(matches, key=lambda match: match[0])[1]


def _angle_in_sector(angle: float, start: float, end: float) -> bool:
    angle = _normalize_angle(angle)
    start = _normalize_angle(start)
    end = _normalize_angle(end)
    if start <= end:
        return start <= angle <= end
    return angle >= start or angle <= end


def _finite_float(value: Any) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("value must be finite")
    return converted


def _normalize_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _stop(*, completed: bool = False, reason: str) -> MissionDecision:
    return MissionDecision(
        "cmd_vel",
        {"linear_x_mps": 0.0, "angular_z_rps": 0.0},
        completed=completed,
        reason=reason,
    )
