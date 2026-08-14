"""Process-local, camera-observed fruit bearings anchored to qualified Home.

The map is intentionally advisory: it can choose a bounded yaw direction, but
it never authorizes translation, target identity, or Arrival.  No camera FOV is
assumed, so only centered full-frame observations contribute a body-yaw
bearing.  Ordinary selected-target guidance must reacquire after any map turn.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .fruits import SUPPORTED_FRUITS
from .return_home import normalize_angle

HOME_POSITION_TOLERANCE_M = 0.10
HOME_YAW_TOLERANCE_RAD = math.radians(5.0)
POSE_MAXIMUM_AGE_S = 0.50
CENTER_TOLERANCE_RATIO = 0.08
MINIMUM_OBSERVATION_CONFIDENCE = 0.01
MAXIMUM_BEARING_AGE_S = 300.0
CONTRADICTION_TOLERANCE_RAD = math.radians(45.0)


@dataclass(frozen=True)
class FruitBearingRoute:
    target_fruit: str
    available: bool
    reason: str
    angular_delta_rad: float | None = None
    direction: str | None = None
    evidence: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "target_fruit": self.target_fruit,
            "available": self.available,
            "reason": self.reason,
            "angular_delta_rad": self.angular_delta_rad,
            "direction": self.direction,
            "evidence": None if self.evidence is None else dict(self.evidence),
        }


@dataclass
class _BearingEvidence:
    weighted_sin: float = 0.0
    weighted_cos: float = 0.0
    confidence_total: float = 0.0
    sample_count: int = 0
    last_observed_s: float = 0.0

    @property
    def bearing_rad(self) -> float:
        return math.atan2(self.weighted_sin, self.weighted_cos)

    @property
    def average_confidence(self) -> float:
        return self.confidence_total / self.sample_count

    def add(self, bearing_rad: float, confidence: float, now_s: float) -> None:
        self.weighted_sin += confidence * math.sin(bearing_rad)
        self.weighted_cos += confidence * math.cos(bearing_rad)
        self.confidence_total += confidence
        self.sample_count += 1
        self.last_observed_s = now_s


class FruitBearingMap:
    """Collect centered all-fruit observations and provide advisory yaw routes."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        center_tolerance_ratio: float = CENTER_TOLERANCE_RATIO,
        maximum_bearing_age_s: float = MAXIMUM_BEARING_AGE_S,
        contradiction_tolerance_rad: float = CONTRADICTION_TOLERANCE_RAD,
    ) -> None:
        if not 0.0 < center_tolerance_ratio <= 0.25:
            raise ValueError("center tolerance must be in (0, 0.25]")
        if not math.isfinite(maximum_bearing_age_s) or maximum_bearing_age_s <= 0:
            raise ValueError("maximum bearing age must be finite and positive")
        if not 0.0 < contradiction_tolerance_rad <= math.pi:
            raise ValueError("contradiction tolerance must be in (0, pi]")
        self._clock = clock
        self._center_tolerance_ratio = center_tolerance_ratio
        self._maximum_bearing_age_s = maximum_bearing_age_s
        self._contradiction_tolerance_rad = contradiction_tolerance_rad
        self._anchor: dict[str, object] | None = None
        self._bearings: dict[str, _BearingEvidence] = {}
        self._seen_frames: set[tuple[str, str, int]] = set()
        self._invalidation_reason = "waiting_for_centered_observation"
        self._current_position_error_m: float | None = None
        self._current_yaw_error_rad: float | None = None
        self._last_observation_reason = "no_observation"

    def invalidate(self, reason: str) -> None:
        self._anchor = None
        self._bearings.clear()
        self._seen_frames.clear()
        self._invalidation_reason = reason

    def observe(
        self,
        home_pose: Mapping[str, object],
        robot_pose: Mapping[str, object],
        frame_identity: Mapping[str, object],
        observations: Mapping[str, object],
    ) -> dict[str, object]:
        """Associate centered observations from one exact frame with robot yaw."""
        now_s = self._clock()
        parsed_home = _pose(home_pose)
        parsed_robot = _pose(robot_pose)
        if parsed_home is None or parsed_robot is None:
            self._last_observation_reason = "pose_unavailable"
            return self.status()
        if not _fresh_pose(home_pose) or not _fresh_pose(robot_pose):
            self._last_observation_reason = "pose_stale"
            return self.status()
        frame = _frame_identity(frame_identity)
        if frame is None:
            self._last_observation_reason = "frame_identity_invalid"
            return self.status()
        generation, time_base, source_pts, odometry_epoch = frame
        frame_key = (generation, time_base, source_pts)
        if frame_key in self._seen_frames:
            self._last_observation_reason = "duplicate_frame"
            return self.status()

        if self._anchor is not None:
            mismatch = self._home_mismatch_reason(
                home_pose,
                generation=generation,
                odometry_epoch=odometry_epoch,
            )
            if mismatch is not None:
                self.invalidate(mismatch)

        qualified: list[tuple[str, float]] = []
        for raw_label, raw_observation in observations.items():
            label = str(raw_label).casefold().strip()
            if label not in SUPPORTED_FRUITS or not isinstance(raw_observation, Mapping):
                continue
            if not _observation_matches(raw_observation, frame):
                continue
            confidence = _number(raw_observation.get("confidence"))
            center_x = _number(raw_observation.get("center_x_ratio"))
            if (
                confidence is None
                or confidence < MINIMUM_OBSERVATION_CONFIDENCE
                or center_x is None
                or abs(center_x - 0.5) > self._center_tolerance_ratio
            ):
                continue
            qualified.append((label, confidence))

        if not qualified:
            self._last_observation_reason = "no_centered_observation"
            return self.status()
        if self._anchor is None:
            self._anchor = {
                "x_m": parsed_home[0],
                "y_m": parsed_home[1],
                "yaw_rad": parsed_home[2],
                "camera_generation": generation,
                "odometry_epoch": odometry_epoch,
                "anchored_monotonic_s": now_s,
            }
            self._invalidation_reason = None

        for label, confidence in qualified:
            evidence = self._bearings.get(label)
            if evidence is not None and abs(
                normalize_angle(parsed_robot[2] - evidence.bearing_rad)
            ) > self._contradiction_tolerance_rad:
                self.invalidate(f"contradictory_{label}_bearing")
                self._last_observation_reason = "contradictory_bearing"
                return self.status()
            evidence = self._bearings.setdefault(label, _BearingEvidence())
            evidence.add(parsed_robot[2], confidence, now_s)
        self._seen_frames.add(frame_key)
        self._last_observation_reason = "centered_observations_recorded"
        return self.status()

    def route_to(
        self,
        target_fruit: str,
        current_pose: Mapping[str, object],
    ) -> FruitBearingRoute:
        """Return an advisory shortest yaw turn, never motion authority."""
        target = target_fruit.casefold().strip()
        if target not in SUPPORTED_FRUITS:
            return FruitBearingRoute(target, False, "unsupported_fruit")
        if self._anchor is None:
            return FruitBearingRoute(target, False, "map_unanchored")
        generation = current_pose.get("generation")
        epoch = current_pose.get("odometry_epoch")
        mismatch = self._home_mismatch_reason(
            current_pose,
            generation=generation if isinstance(generation, str) else None,
            odometry_epoch=epoch if isinstance(epoch, str) else None,
        )
        if mismatch is not None:
            self.invalidate(mismatch)
            return FruitBearingRoute(target, False, mismatch)
        pose = _pose(current_pose)
        if pose is None or not _fresh_pose(current_pose):
            return FruitBearingRoute(target, False, "pose_unavailable_or_stale")
        evidence = self._bearings.get(target)
        if evidence is None:
            return FruitBearingRoute(target, False, "fruit_unmapped")
        age_s = self._clock() - evidence.last_observed_s
        if age_s < 0.0 or age_s > self._maximum_bearing_age_s:
            return FruitBearingRoute(target, False, "bearing_stale")
        delta = normalize_angle(evidence.bearing_rad - pose[2])
        direction = "aligned" if abs(delta) < 1e-9 else "left" if delta > 0 else "right"
        return FruitBearingRoute(
            target,
            True,
            "mapped_bearing_available",
            angular_delta_rad=delta,
            direction=direction,
            evidence={
                "bearing_rad": evidence.bearing_rad,
                "confidence": evidence.average_confidence,
                "sample_count": evidence.sample_count,
                "age_s": age_s,
                "camera_generation": self._anchor["camera_generation"],
                "odometry_epoch": self._anchor["odometry_epoch"],
                "authority": "yaw_route_only",
            },
        )

    def status(self) -> dict[str, object]:
        now_s = self._clock()
        return {
            "valid": self._anchor is not None,
            "invalidation_reason": self._invalidation_reason,
            "last_observation_reason": self._last_observation_reason,
            "anchor": None if self._anchor is None else dict(self._anchor),
            "current_home_position_error_m": self._current_position_error_m,
            "current_home_yaw_error_deg": (
                None
                if self._current_yaw_error_rad is None
                else math.degrees(self._current_yaw_error_rad)
            ),
            "fruits": {
                label: {
                    "bearing_rad": evidence.bearing_rad,
                    "confidence": evidence.average_confidence,
                    "sample_count": evidence.sample_count,
                    "age_s": max(0.0, now_s - evidence.last_observed_s),
                }
                for label, evidence in sorted(self._bearings.items())
            },
            "thresholds": {
                "center_tolerance_ratio": self._center_tolerance_ratio,
                "home_position_tolerance_m": HOME_POSITION_TOLERANCE_M,
                "home_yaw_tolerance_deg": math.degrees(HOME_YAW_TOLERANCE_RAD),
                "pose_maximum_age_s": POSE_MAXIMUM_AGE_S,
                "maximum_bearing_age_s": self._maximum_bearing_age_s,
                "contradiction_tolerance_deg": math.degrees(
                    self._contradiction_tolerance_rad
                ),
            },
            "authority": "yaw_route_only",
            "persistence": "process_local",
        }

    def _home_mismatch_reason(
        self,
        home_pose: Mapping[str, object],
        *,
        generation: str | None,
        odometry_epoch: str | None,
    ) -> str | None:
        assert self._anchor is not None
        pose = _pose(home_pose)
        if pose is None or not _fresh_pose(home_pose):
            return "home_pose_unavailable_or_stale"
        if generation != self._anchor["camera_generation"]:
            return "camera_generation_changed"
        if odometry_epoch != self._anchor["odometry_epoch"]:
            return "odometry_epoch_changed"
        self._current_position_error_m = math.hypot(
            pose[0] - float(self._anchor["x_m"]),
            pose[1] - float(self._anchor["y_m"]),
        )
        self._current_yaw_error_rad = abs(
            normalize_angle(pose[2] - float(self._anchor["yaw_rad"]))
        )
        if self._current_position_error_m > HOME_POSITION_TOLERANCE_M:
            return "home_position_mismatch"
        if self._current_yaw_error_rad > HOME_YAW_TOLERANCE_RAD:
            return "home_yaw_mismatch"
        return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _pose(value: Mapping[str, object]) -> tuple[float, float, float] | None:
    values = tuple(_number(value.get(key)) for key in ("x_m", "y_m", "yaw_rad"))
    if any(item is None for item in values):
        return None
    return tuple(float(item) for item in values if item is not None)  # type: ignore[return-value]


def _fresh_pose(value: Mapping[str, object]) -> bool:
    age = _number(value.get("age_s"))
    return age is not None and 0.0 <= age <= POSE_MAXIMUM_AGE_S


def _frame_identity(
    value: Mapping[str, object],
) -> tuple[str, str, int, str] | None:
    generation = value.get("generation")
    time_base = value.get("source_time_base")
    pts = value.get("source_pts")
    epoch = value.get("odometry_epoch")
    if (
        not isinstance(generation, str)
        or not generation
        or not isinstance(time_base, str)
        or not time_base
        or isinstance(pts, bool)
        or not isinstance(pts, int)
        or pts < 0
        or not isinstance(epoch, str)
        or not epoch
    ):
        return None
    return generation, time_base, pts, epoch


def _observation_matches(
    observation: Mapping[str, object],
    frame: tuple[str, str, int, str],
) -> bool:
    generation, time_base, pts, _epoch = frame
    return bool(
        observation.get("generation") == generation
        and observation.get("source_time_base") == time_base
        and observation.get("source_pts") == pts
    )
