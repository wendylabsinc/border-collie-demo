"""Confidence-bearing Home estimates from Go2 and sparse visual motion.

Go2 remains the metric scale source. Monocular vision contributes independent
yaw, motion continuity, and natural-scene loop-closure evidence without making
an uncalibrated claim about metres.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Protocol

from .return_home import Pose2D, normalize_angle


class HomeEstimateState(str, Enum):
    TRUSTED = "trusted"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class VisualOdometryObservation:
    generation: str
    frame_sequence: int
    motion_sequence: int
    captured_monotonic_s: float
    trajectory_x_px: float
    trajectory_y_px: float
    trajectory_yaw_rad: float
    motion_quality: float
    tracked_features: int
    inliers: int
    loop_closure: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.generation.strip():
            raise ValueError("visual odometry generation is required")
        if self.frame_sequence < 0 or self.motion_sequence < 0:
            raise ValueError("visual odometry sequences must be non-negative")
        numbers = (
            self.captured_monotonic_s,
            self.trajectory_x_px,
            self.trajectory_y_px,
            self.trajectory_yaw_rad,
            self.motion_quality,
        )
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("visual odometry values must be finite")
        if not 0.0 <= self.motion_quality <= 1.0:
            raise ValueError("visual odometry quality must be in [0, 1]")
        if self.tracked_features < 0 or self.inliers < 0:
            raise ValueError("visual odometry feature counts must be non-negative")


class VisualOdometryAdapter(Protocol):
    def observe_motion(self) -> VisualOdometryObservation | None: ...


@dataclass(frozen=True)
class HomeLocalizationConfig:
    maximum_odometry_age_s: float = 0.50
    maximum_visual_age_s: float = 0.75
    minimum_visual_quality: float = 0.55
    maximum_visual_yaw_disagreement_rad: float = math.radians(20.0)
    visual_yaw_weight: float = 0.35
    visual_image_yaw_sign: float = -1.0
    maximum_consecutive_visual_conflicts: int = 3
    base_position_sigma_m: float = 0.03
    position_sigma_per_m: float = 0.08
    base_heading_sigma_rad: float = math.radians(2.0)
    maximum_future_skew_s: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            self.maximum_odometry_age_s,
            self.maximum_visual_age_s,
            self.maximum_visual_yaw_disagreement_rad,
            self.base_position_sigma_m,
            self.position_sigma_per_m,
            self.base_heading_sigma_rad,
            self.maximum_future_skew_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Home localization limits must be finite and positive")
        if not 0.0 <= self.minimum_visual_quality <= 1.0:
            raise ValueError("minimum visual quality must be in [0, 1]")
        if not 0.0 <= self.visual_yaw_weight <= 1.0:
            raise ValueError("visual yaw weight must be in [0, 1]")
        if self.visual_image_yaw_sign not in {-1.0, 1.0}:
            raise ValueError("visual image yaw sign must be -1 or 1")
        if self.maximum_consecutive_visual_conflicts < 1:
            raise ValueError("visual conflict limit must be positive")


@dataclass(frozen=True)
class HomeEstimate:
    state: HomeEstimateState
    pose_from_home: Pose2D | None
    source: str | None
    unavailable_reason: str | None
    evidence: dict[str, object]

    @property
    def trusted(self) -> bool:
        return self.state is HomeEstimateState.TRUSTED

    @property
    def home_distance_m(self) -> float | None:
        if self.pose_from_home is None:
            return None
        return math.hypot(self.pose_from_home.x_m, self.pose_from_home.y_m)

    @property
    def heading_error_rad(self) -> float | None:
        if self.pose_from_home is None:
            return None
        return normalize_angle(-self.pose_from_home.yaw_rad)

    def to_dict(self) -> dict[str, object]:
        pose = self.pose_from_home
        return {
            "state": self.state.value,
            "trusted": self.trusted,
            "source": self.source,
            "unavailable_reason": self.unavailable_reason,
            "home_distance_m": self.home_distance_m,
            "heading_error_rad": self.heading_error_rad,
            "pose_from_home": (
                None
                if pose is None
                else {
                    "x_m": pose.x_m,
                    "y_m": pose.y_m,
                    "yaw_rad": pose.yaw_rad,
                }
            ),
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class _VisualBaseline:
    generation: str
    frame_sequence: int
    trajectory_x_px: float
    trajectory_y_px: float
    trajectory_yaw_rad: float


class HomeLocalizer:
    """Fuse all qualified relative-motion evidence behind one interface."""

    def __init__(
        self,
        visual_adapter: VisualOdometryAdapter | None = None,
        *,
        config: HomeLocalizationConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._visual_adapter = visual_adapter
        self.config = config or HomeLocalizationConfig()
        self._clock = clock
        self._lock = Lock()
        self._visual_baseline: _VisualBaseline | None = None
        self._visual_conflict_count = 0

    def capture_home(self) -> dict[str, object]:
        """Capture a visual origin without making Home depend on vision."""
        observation, error = self._observe_visual()
        with self._lock:
            self._visual_conflict_count = 0
            self._visual_baseline = (
                None
                if observation is None
                else _VisualBaseline(
                    generation=observation.generation,
                    frame_sequence=observation.frame_sequence,
                    trajectory_x_px=observation.trajectory_x_px,
                    trajectory_y_px=observation.trajectory_y_px,
                    trajectory_yaw_rad=observation.trajectory_yaw_rad,
                )
            )
        return {
            "state": "captured" if observation is not None else "unavailable",
            "generation": None if observation is None else observation.generation,
            "frame_sequence": (
                None if observation is None else observation.frame_sequence
            ),
            "error": error,
        }

    def describe(self) -> dict[str, object]:
        with self._lock:
            baseline = self._visual_baseline
        return {
            "metric_source": "rt/sportmodestate",
            "visual_source_configured": self._visual_adapter is not None,
            "visual_backend": "cpu_sparse_image_motion",
            "visual_metric_scale": False,
            "visual_home_baseline_captured": baseline is not None,
            "maximum_odometry_age_s": self.config.maximum_odometry_age_s,
            "maximum_visual_age_s": self.config.maximum_visual_age_s,
            "minimum_visual_quality": self.config.minimum_visual_quality,
            "maximum_visual_yaw_disagreement_rad": (
                self.config.maximum_visual_yaw_disagreement_rad
            ),
        }

    def estimate(
        self,
        home: Pose2D,
        current_odometry: Pose2D,
        *,
        odometry_age_s: float,
    ) -> HomeEstimate:
        odometry_pose = pose_from_home(home, current_odometry)
        distance_m = math.hypot(odometry_pose.x_m, odometry_pose.y_m)
        position_sigma_m = (
            self.config.base_position_sigma_m
            + distance_m * self.config.position_sigma_per_m
        )
        odometry_evidence = {
            "age_s": odometry_age_s,
            "pose_from_home": _pose_dict(odometry_pose),
        }
        uncertainty = {
            "position_sigma_m": position_sigma_m,
            "heading_sigma_rad": self.config.base_heading_sigma_rad,
            "model": "bounded_planar_heuristic",
        }
        if (
            not math.isfinite(odometry_age_s)
            or odometry_age_s < 0.0
            or odometry_age_s > self.config.maximum_odometry_age_s
        ):
            return self._unavailable(
                "odometry sample is stale or invalid",
                odometry=odometry_evidence,
                visual={"state": "not_checked"},
                uncertainty=uncertainty,
            )

        observation, adapter_error = self._observe_visual()
        with self._lock:
            baseline = self._visual_baseline
        if self._visual_adapter is None:
            return self._trusted_odometry(
                odometry_pose,
                odometry_evidence,
                uncertainty,
                visual={"state": "not_configured"},
            )
        if observation is None:
            return self._trusted_odometry(
                odometry_pose,
                odometry_evidence,
                uncertainty,
                visual={
                    "state": "unavailable",
                    "error": adapter_error,
                },
            )
        if baseline is None:
            return self._trusted_odometry(
                odometry_pose,
                odometry_evidence,
                uncertainty,
                visual={"state": "Home baseline is not captured"},
            )

        now = self._clock()
        age_s = now - observation.captured_monotonic_s
        visual_evidence: dict[str, object] = {
            "state": "observed",
            "generation": observation.generation,
            "frame_sequence": observation.frame_sequence,
            "motion_sequence": observation.motion_sequence,
            "age_s": age_s,
            "motion_quality": observation.motion_quality,
            "tracked_features": observation.tracked_features,
            "inliers": observation.inliers,
            "trajectory_image_space": {
                "x_px": observation.trajectory_x_px - baseline.trajectory_x_px,
                "y_px": observation.trajectory_y_px - baseline.trajectory_y_px,
                "yaw_rad": normalize_angle(
                    observation.trajectory_yaw_rad - baseline.trajectory_yaw_rad
                ),
            },
            "loop_closure": observation.loop_closure,
        }
        if observation.generation != baseline.generation:
            with self._lock:
                self._visual_conflict_count = 0
            visual_evidence["state"] = "generation_changed"
            return self._trusted_odometry(
                odometry_pose, odometry_evidence, uncertainty, visual=visual_evidence
            )
        if (
            not math.isfinite(now)
            or not math.isfinite(age_s)
            or age_s < -self.config.maximum_future_skew_s
            or age_s > self.config.maximum_visual_age_s
        ):
            visual_evidence["state"] = "stale"
            return self._trusted_odometry(
                odometry_pose, odometry_evidence, uncertainty, visual=visual_evidence
            )
        if observation.motion_quality < self.config.minimum_visual_quality:
            visual_evidence["state"] = "low_quality"
            return self._trusted_odometry(
                odometry_pose, odometry_evidence, uncertainty, visual=visual_evidence
            )

        visual_relative_yaw = normalize_angle(
            self.config.visual_image_yaw_sign
            * (observation.trajectory_yaw_rad - baseline.trajectory_yaw_rad)
        )
        yaw_disagreement = abs(
            normalize_angle(visual_relative_yaw - odometry_pose.yaw_rad)
        )
        visual_evidence["body_relative_yaw_rad"] = visual_relative_yaw
        visual_evidence["yaw_disagreement_rad"] = yaw_disagreement
        if yaw_disagreement > self.config.maximum_visual_yaw_disagreement_rad:
            with self._lock:
                self._visual_conflict_count += 1
                conflict_count = self._visual_conflict_count
            visual_evidence["state"] = "conflict"
            visual_evidence["consecutive_conflicts"] = conflict_count
            if conflict_count >= self.config.maximum_consecutive_visual_conflicts:
                return self._unavailable(
                    "Go2 and visual yaw disagree persistently",
                    odometry=odometry_evidence,
                    visual=visual_evidence,
                    uncertainty=uncertainty,
                )
            return self._trusted_odometry(
                odometry_pose, odometry_evidence, uncertainty, visual=visual_evidence
            )

        with self._lock:
            self._visual_conflict_count = 0
        weight = self.config.visual_yaw_weight * observation.motion_quality
        fused_yaw = normalize_angle(
            odometry_pose.yaw_rad
            + weight * normalize_angle(visual_relative_yaw - odometry_pose.yaw_rad)
        )
        fused = Pose2D(odometry_pose.x_m, odometry_pose.y_m, fused_yaw)
        uncertainty["heading_sigma_rad"] = (
            self.config.base_heading_sigma_rad * (1.0 - 0.5 * weight)
        )
        visual_evidence["state"] = "fused"
        visual_evidence["yaw_weight"] = weight
        return HomeEstimate(
            state=HomeEstimateState.TRUSTED,
            pose_from_home=fused,
            source="go2_metric+visual_yaw",
            unavailable_reason=None,
            evidence={
                "odometry": odometry_evidence,
                "visual": visual_evidence,
                "uncertainty": uncertainty,
            },
        )

    def _observe_visual(self) -> tuple[VisualOdometryObservation | None, str | None]:
        if self._visual_adapter is None:
            return None, None
        try:
            return self._visual_adapter.observe_motion(), None
        except Exception as exc:  # noqa: BLE001 - optional evidence must be contained
            return None, f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _trusted_odometry(
        pose: Pose2D,
        odometry: dict[str, object],
        uncertainty: dict[str, object],
        *,
        visual: dict[str, object],
    ) -> HomeEstimate:
        return HomeEstimate(
            state=HomeEstimateState.TRUSTED,
            pose_from_home=pose,
            source="go2_metric",
            unavailable_reason=None,
            evidence={
                "odometry": odometry,
                "visual": visual,
                "uncertainty": uncertainty,
            },
        )

    @staticmethod
    def _unavailable(reason: str, **evidence: object) -> HomeEstimate:
        return HomeEstimate(
            state=HomeEstimateState.UNAVAILABLE,
            pose_from_home=None,
            source=None,
            unavailable_reason=reason,
            evidence=dict(evidence),
        )


def pose_from_home(home: Pose2D, current: Pose2D) -> Pose2D:
    dx = current.x_m - home.x_m
    dy = current.y_m - home.y_m
    cosine = math.cos(home.yaw_rad)
    sine = math.sin(home.yaw_rad)
    return Pose2D(
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        normalize_angle(current.yaw_rad - home.yaw_rad),
    )


def pose_to_world(home: Pose2D, relative: Pose2D) -> Pose2D:
    cosine = math.cos(home.yaw_rad)
    sine = math.sin(home.yaw_rad)
    return Pose2D(
        home.x_m + cosine * relative.x_m - sine * relative.y_m,
        home.y_m + sine * relative.x_m + cosine * relative.y_m,
        normalize_angle(home.yaw_rad + relative.yaw_rad),
    )


def _pose_dict(pose: Pose2D) -> dict[str, float]:
    return {"x_m": pose.x_m, "y_m": pose.y_m, "yaw_rad": pose.yaw_rad}
