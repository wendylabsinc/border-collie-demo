"""Stateful planar sensor fusion for Home-relative robot pose.

The module uses a diagonal-covariance Kalman filter. It deliberately keeps the
interface small and SDK-neutral while owning prediction, innovation gates,
zero-velocity updates, visual direction/yaw fusion, and uncertainty policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .return_home import Pose2D, normalize_angle


@dataclass(frozen=True)
class VisualMotionCue:
    generation: str
    sequence: int
    forward_direction: float | None
    left_direction: float | None
    yaw_delta_rad: float | None
    quality: float

    def __post_init__(self) -> None:
        if not self.generation.strip() or self.sequence < 0:
            raise ValueError("visual motion identity is invalid")
        if not math.isfinite(self.quality) or not 0.0 <= self.quality <= 1.0:
            raise ValueError("visual motion quality must be in [0, 1]")
        optional = (
            self.forward_direction,
            self.left_direction,
            self.yaw_delta_rad,
        )
        if any(value is not None and not math.isfinite(value) for value in optional):
            raise ValueError("visual motion values must be finite")
        one_direction = self.forward_direction is not None
        if one_direction != (self.left_direction is not None):
            raise ValueError("visual direction requires forward and left values")


@dataclass(frozen=True)
class PlanarFusionObservation:
    raw_pose_from_home: Pose2D
    captured_monotonic_s: float
    source_timestamp_s: float | None = None
    velocity_forward_mps: float | None = None
    velocity_left_mps: float | None = None
    imu_yaw_rate_rps: float | None = None
    reported_yaw_rate_rps: float | None = None
    contact_feet: int | None = None
    stationary_stance: bool = False
    visual: VisualMotionCue | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.captured_monotonic_s):
            raise ValueError("fusion capture time must be finite")
        optional = (
            self.source_timestamp_s,
            self.velocity_forward_mps,
            self.velocity_left_mps,
            self.imu_yaw_rate_rps,
            self.reported_yaw_rate_rps,
        )
        if any(value is not None and not math.isfinite(value) for value in optional):
            raise ValueError("fusion observation values must be finite")
        if self.contact_feet is not None and not 0 <= self.contact_feet <= 4:
            raise ValueError("contact feet must be between zero and four")

    @property
    def time_s(self) -> float:
        return (
            self.source_timestamp_s
            if self.source_timestamp_s is not None
            else self.captured_monotonic_s
        )


@dataclass(frozen=True)
class PlanarFusionConfig:
    maximum_sample_gap_s: float = 0.75
    position_process_variance_per_s: float = 0.0025
    velocity_process_variance_per_s: float = 0.04
    heading_process_variance_per_s: float = math.radians(3.0) ** 2
    bias_process_variance_per_s: float = math.radians(0.2) ** 2
    go2_position_measurement_variance: float = 0.04**2
    go2_heading_measurement_variance: float = math.radians(4.0) ** 2
    go2_velocity_measurement_variance: float = 0.10**2
    zero_velocity_measurement_variance: float = 0.015**2
    stationary_bias_measurement_variance: float = math.radians(0.5) ** 2
    visual_position_measurement_variance: float = 0.18**2
    visual_heading_measurement_variance: float = math.radians(8.0) ** 2
    minimum_visual_quality: float = 0.55
    maximum_position_innovation_m: float = 0.75
    maximum_heading_innovation_rad: float = math.radians(35.0)
    maximum_visual_direction_disagreement_rad: float = math.radians(50.0)
    maximum_visual_yaw_disagreement_rad: float = math.radians(25.0)
    maximum_consecutive_metric_rejections: int = 3
    maximum_consecutive_visual_rejections: int = 5
    maximum_position_sigma_m: float = 0.35
    maximum_heading_sigma_rad: float = math.radians(30.0)

    def __post_init__(self) -> None:
        numeric = tuple(
            float(getattr(self, name))
            for name in self.__dataclass_fields__
            if name not in {
                "minimum_visual_quality",
                "maximum_consecutive_metric_rejections",
                "maximum_consecutive_visual_rejections",
            }
        )
        if not all(math.isfinite(value) and value > 0.0 for value in numeric):
            raise ValueError("fusion limits and variances must be finite and positive")
        if not 0.0 <= self.minimum_visual_quality <= 1.0:
            raise ValueError("minimum visual quality must be in [0, 1]")
        if self.maximum_consecutive_metric_rejections < 1:
            raise ValueError("metric rejection limit must be positive")
        if self.maximum_consecutive_visual_rejections < 1:
            raise ValueError("visual rejection limit must be positive")


@dataclass(frozen=True)
class PlanarFusionEstimate:
    trusted: bool
    pose_from_home: Pose2D | None
    velocity_x_mps: float | None
    velocity_y_mps: float | None
    yaw_bias_rps: float | None
    unavailable_reason: str | None
    covariance: dict[str, float]
    evidence: dict[str, object]


class PlanarSensorFusion:
    """Fuse normalized observations into one confidence-bearing estimate."""

    def __init__(self, config: PlanarFusionConfig | None = None) -> None:
        self.config = config or PlanarFusionConfig()
        self._initialized = False
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._yaw_bias = 0.0
        self._var_x = self.config.go2_position_measurement_variance
        self._var_y = self.config.go2_position_measurement_variance
        self._var_yaw = self.config.go2_heading_measurement_variance
        self._var_vx = self.config.go2_velocity_measurement_variance
        self._var_vy = self.config.go2_velocity_measurement_variance
        self._var_bias = self.config.stationary_bias_measurement_variance
        self._last_time_s: float | None = None
        self._last_raw_pose: Pose2D | None = None
        self._last_visual_generation: str | None = None
        self._last_visual_sequence = -1
        self._metric_rejections = 0
        self._visual_rejections = 0

    def reset(self, observation: PlanarFusionObservation) -> PlanarFusionEstimate:
        pose = observation.raw_pose_from_home
        self._x, self._y, self._yaw = pose.x_m, pose.y_m, pose.yaw_rad
        velocity = _body_velocity_in_home(observation, pose.yaw_rad)
        self._vx, self._vy = velocity if velocity is not None else (0.0, 0.0)
        self._yaw_bias = 0.0
        self._var_x = self.config.go2_position_measurement_variance
        self._var_y = self.config.go2_position_measurement_variance
        self._var_yaw = self.config.go2_heading_measurement_variance
        self._var_vx = self.config.go2_velocity_measurement_variance
        self._var_vy = self.config.go2_velocity_measurement_variance
        self._var_bias = self.config.stationary_bias_measurement_variance
        self._last_time_s = observation.time_s
        self._last_raw_pose = pose
        self._last_visual_generation = (
            None if observation.visual is None else observation.visual.generation
        )
        self._last_visual_sequence = (
            -1 if observation.visual is None else observation.visual.sequence
        )
        self._metric_rejections = 0
        self._visual_rejections = 0
        self._initialized = True
        return self._estimate(
            trusted=True,
            reason=None,
            evidence={"state": "reset", "sources": ["go2_pose"]},
        )

    def update(self, observation: PlanarFusionObservation) -> PlanarFusionEstimate:
        if not self._initialized:
            return self.reset(observation)
        assert self._last_time_s is not None and self._last_raw_pose is not None
        dt = observation.time_s - self._last_time_s
        if dt < -1e-6:
            return self._estimate(
                trusted=False,
                reason="fusion source timestamp regressed",
                evidence={"state": "rejected", "dt_s": dt},
            )
        if dt <= 1e-6:
            changed = _pose_distance(
                observation.raw_pose_from_home, self._last_raw_pose
            ) > 1e-6 or abs(
                normalize_angle(
                    observation.raw_pose_from_home.yaw_rad
                    - self._last_raw_pose.yaw_rad
                )
            ) > 1e-6
            if changed:
                return self._estimate(
                    trusted=False,
                    reason="non-advancing timestamp carried a changed Go2 pose",
                    evidence={"state": "rejected_duplicate", "dt_s": dt},
                )
            return self._estimate(
                trusted=True,
                reason=None,
                evidence={"state": "duplicate", "dt_s": dt},
            )
        if dt > self.config.maximum_sample_gap_s:
            if observation.stationary_stance:
                estimate = self.reset(observation)
                return PlanarFusionEstimate(
                    trusted=estimate.trusted,
                    pose_from_home=estimate.pose_from_home,
                    velocity_x_mps=estimate.velocity_x_mps,
                    velocity_y_mps=estimate.velocity_y_mps,
                    yaw_bias_rps=estimate.yaw_bias_rps,
                    unavailable_reason=estimate.unavailable_reason,
                    covariance=estimate.covariance,
                    evidence={
                        "state": "stationary_reseed",
                        "dt_s": dt,
                        "sources": ["go2_pose", "stance_zero_velocity"],
                    },
                )
            return self._estimate(
                trusted=False,
                reason="fusion sample gap exceeded its bound",
                evidence={"state": "stale_gap", "dt_s": dt},
            )

        previous_pose = Pose2D(self._x, self._y, self._yaw)
        previous_raw = self._last_raw_pose
        self._predict(observation, dt)
        evidence: dict[str, object] = {
            "state": "updated",
            "dt_s": dt,
            "sources": ["process_model"],
            "innovations": {},
            "zero_velocity_update": False,
            "visual": {"state": "not_observed"},
        }

        position_innovation = math.hypot(
            observation.raw_pose_from_home.x_m - self._x,
            observation.raw_pose_from_home.y_m - self._y,
        )
        heading_innovation = abs(
            normalize_angle(observation.raw_pose_from_home.yaw_rad - self._yaw)
        )
        evidence["innovations"] = {
            "go2_position_m": position_innovation,
            "go2_heading_rad": heading_innovation,
        }
        if (
            position_innovation > self.config.maximum_position_innovation_m
            or heading_innovation > self.config.maximum_heading_innovation_rad
        ):
            metric_measurement_accepted = False
            self._metric_rejections += 1
            evidence["go2_measurement"] = {
                "state": "rejected",
                "consecutive_rejections": self._metric_rejections,
            }
            if (
                self._metric_rejections
                >= self.config.maximum_consecutive_metric_rejections
            ):
                return self._estimate(
                    trusted=False,
                    reason="Go2 metric pose innovation was rejected persistently",
                    evidence=evidence,
                )
        else:
            metric_measurement_accepted = True
            self._metric_rejections = 0
            self._x, self._var_x, gain_x = _kalman_scalar(
                self._x,
                self._var_x,
                observation.raw_pose_from_home.x_m,
                self.config.go2_position_measurement_variance,
            )
            self._y, self._var_y, gain_y = _kalman_scalar(
                self._y,
                self._var_y,
                observation.raw_pose_from_home.y_m,
                self.config.go2_position_measurement_variance,
            )
            self._yaw, self._var_yaw, gain_yaw = _kalman_angle(
                self._yaw,
                self._var_yaw,
                observation.raw_pose_from_home.yaw_rad,
                self.config.go2_heading_measurement_variance,
            )
            evidence["go2_measurement"] = {
                "state": "fused",
                "position_gain": (gain_x + gain_y) / 2.0,
                "heading_gain": gain_yaw,
            }
            cast_sources = evidence["sources"]
            assert isinstance(cast_sources, list)
            cast_sources.append("go2_pose")

        velocity = _body_velocity_in_home(observation, self._yaw)
        if velocity is not None:
            self._vx, self._var_vx, gain_vx = _kalman_scalar(
                self._vx,
                self._var_vx,
                velocity[0],
                self.config.go2_velocity_measurement_variance,
            )
            self._vy, self._var_vy, gain_vy = _kalman_scalar(
                self._vy,
                self._var_vy,
                velocity[1],
                self.config.go2_velocity_measurement_variance,
            )
            evidence["velocity"] = {
                "state": "fused",
                "gain": (gain_vx + gain_vy) / 2.0,
                "home_x_mps": velocity[0],
                "home_y_mps": velocity[1],
            }
            cast_sources = evidence["sources"]
            assert isinstance(cast_sources, list)
            cast_sources.append("go2_velocity")

        if observation.stationary_stance:
            self._vx, self._var_vx, _ = _kalman_scalar(
                self._vx,
                self._var_vx,
                0.0,
                self.config.zero_velocity_measurement_variance,
            )
            self._vy, self._var_vy, _ = _kalman_scalar(
                self._vy,
                self._var_vy,
                0.0,
                self.config.zero_velocity_measurement_variance,
            )
            if observation.imu_yaw_rate_rps is not None:
                self._yaw_bias, self._var_bias, _ = _kalman_scalar(
                    self._yaw_bias,
                    self._var_bias,
                    observation.imu_yaw_rate_rps,
                    self.config.stationary_bias_measurement_variance,
                )
            evidence["zero_velocity_update"] = True
            cast_sources = evidence["sources"]
            assert isinstance(cast_sources, list)
            cast_sources.append("stance_zero_velocity")

        self._fuse_visual(
            observation.visual,
            previous_pose=previous_pose,
            previous_raw=previous_raw,
            current_raw=observation.raw_pose_from_home,
            metric_translation_accepted=metric_measurement_accepted,
            evidence=evidence,
        )
        if (
            self._visual_rejections
            >= self.config.maximum_consecutive_visual_rejections
        ):
            return self._estimate(
                trusted=False,
                reason="qualified visual motion disagreed persistently",
                evidence=evidence,
            )
        self._last_time_s = observation.time_s
        self._last_raw_pose = observation.raw_pose_from_home

        position_sigma = math.sqrt(max(self._var_x, self._var_y))
        heading_sigma = math.sqrt(self._var_yaw)
        if (
            position_sigma > self.config.maximum_position_sigma_m
            or heading_sigma > self.config.maximum_heading_sigma_rad
        ):
            return self._estimate(
                trusted=False,
                reason="fused pose uncertainty exceeded its bound",
                evidence=evidence,
            )
        return self._estimate(trusted=True, reason=None, evidence=evidence)

    def _predict(self, observation: PlanarFusionObservation, dt: float) -> None:
        yaw_rate = (
            observation.imu_yaw_rate_rps
            if observation.imu_yaw_rate_rps is not None
            else observation.reported_yaw_rate_rps
        )
        measured_velocity = _body_velocity_in_home(observation, self._yaw)
        predict_vx, predict_vy = (
            measured_velocity if measured_velocity is not None else (self._vx, self._vy)
        )
        self._x += predict_vx * dt
        self._y += predict_vy * dt
        if yaw_rate is not None:
            self._yaw = normalize_angle(self._yaw + (yaw_rate - self._yaw_bias) * dt)
        self._var_x += (
            dt * dt * self._var_vx
            + self.config.position_process_variance_per_s * dt
        )
        self._var_y += (
            dt * dt * self._var_vy
            + self.config.position_process_variance_per_s * dt
        )
        self._var_vx += self.config.velocity_process_variance_per_s * dt
        self._var_vy += self.config.velocity_process_variance_per_s * dt
        self._var_yaw += (
            dt * dt * self._var_bias
            + self.config.heading_process_variance_per_s * dt
        )
        self._var_bias += self.config.bias_process_variance_per_s * dt

    def _fuse_visual(
        self,
        visual: VisualMotionCue | None,
        *,
        previous_pose: Pose2D,
        previous_raw: Pose2D,
        current_raw: Pose2D,
        metric_translation_accepted: bool,
        evidence: dict[str, object],
    ) -> None:
        if visual is None:
            return
        if visual.generation != self._last_visual_generation:
            self._last_visual_generation = visual.generation
            self._last_visual_sequence = visual.sequence
            self._visual_rejections = 0
            evidence["visual"] = {"state": "generation_reset"}
            return
        if visual.sequence <= self._last_visual_sequence:
            evidence["visual"] = {"state": "duplicate"}
            return
        self._last_visual_sequence = visual.sequence
        if visual.quality < self.config.minimum_visual_quality:
            evidence["visual"] = {
                "state": "low_quality",
                "quality": visual.quality,
            }
            return

        visual_evidence: dict[str, object] = {
            "state": "qualified",
            "quality": visual.quality,
            "direction": "unavailable",
            "yaw": "unavailable",
        }
        if (
            visual.forward_direction is not None
            and visual.left_direction is not None
        ):
            if not metric_translation_accepted:
                visual_evidence["direction"] = "metric_scale_rejected"
            else:
                norm = math.hypot(visual.forward_direction, visual.left_direction)
                raw_dx = current_raw.x_m - previous_raw.x_m
                raw_dy = current_raw.y_m - previous_raw.y_m
                metric_distance = math.hypot(raw_dx, raw_dy)
            if (
                metric_translation_accepted
                and norm > 1e-6
                and metric_distance > 0.01
            ):
                body_forward = visual.forward_direction / norm
                body_left = visual.left_direction / norm
                cosine = math.cos(previous_pose.yaw_rad)
                sine = math.sin(previous_pose.yaw_rad)
                visual_dx = cosine * body_forward - sine * body_left
                visual_dy = sine * body_forward + cosine * body_left
                raw_direction = math.atan2(raw_dy, raw_dx)
                visual_direction = math.atan2(visual_dy, visual_dx)
                disagreement = abs(normalize_angle(visual_direction - raw_direction))
                visual_evidence["direction_disagreement_rad"] = disagreement
                if (
                    disagreement
                    <= self.config.maximum_visual_direction_disagreement_rad
                ):
                    variance = (
                        self.config.visual_position_measurement_variance
                        / max(visual.quality, 0.1)
                    )
                    visual_x = previous_pose.x_m + metric_distance * visual_dx
                    visual_y = previous_pose.y_m + metric_distance * visual_dy
                    self._x, self._var_x, _ = _kalman_scalar(
                        self._x, self._var_x, visual_x, variance
                    )
                    self._y, self._var_y, _ = _kalman_scalar(
                        self._y, self._var_y, visual_y, variance
                    )
                    visual_evidence["direction"] = "fused"
                else:
                    visual_evidence["direction"] = "rejected"

        if visual.yaw_delta_rad is not None:
            visual_yaw = normalize_angle(previous_pose.yaw_rad + visual.yaw_delta_rad)
            disagreement = abs(normalize_angle(visual_yaw - self._yaw))
            visual_evidence["yaw_disagreement_rad"] = disagreement
            if disagreement <= self.config.maximum_visual_yaw_disagreement_rad:
                variance = (
                    self.config.visual_heading_measurement_variance
                    / max(visual.quality, 0.1)
                )
                self._yaw, self._var_yaw, _ = _kalman_angle(
                    self._yaw, self._var_yaw, visual_yaw, variance
                )
                visual_evidence["yaw"] = "fused"
            else:
                visual_evidence["yaw"] = "rejected"
        evidence["visual"] = visual_evidence
        sources = evidence["sources"]
        assert isinstance(sources, list)
        states = {
            visual_evidence.get("direction"),
            visual_evidence.get("yaw"),
        }
        if "fused" in states:
            self._visual_rejections = 0
            sources.append("visual_motion")
        elif "rejected" in states:
            self._visual_rejections += 1
        visual_evidence["consecutive_rejections"] = self._visual_rejections

    def _estimate(
        self,
        *,
        trusted: bool,
        reason: str | None,
        evidence: dict[str, object],
    ) -> PlanarFusionEstimate:
        covariance = {
            "x_variance_m2": self._var_x,
            "y_variance_m2": self._var_y,
            "heading_variance_rad2": self._var_yaw,
            "velocity_x_variance_m2ps2": self._var_vx,
            "velocity_y_variance_m2ps2": self._var_vy,
            "yaw_bias_variance_rps2": self._var_bias,
            "position_sigma_m": math.sqrt(max(self._var_x, self._var_y)),
            "heading_sigma_rad": math.sqrt(self._var_yaw),
        }
        return PlanarFusionEstimate(
            trusted=trusted,
            pose_from_home=(Pose2D(self._x, self._y, self._yaw) if trusted else None),
            velocity_x_mps=self._vx if trusted else None,
            velocity_y_mps=self._vy if trusted else None,
            yaw_bias_rps=self._yaw_bias if trusted else None,
            unavailable_reason=reason,
            covariance=covariance,
            evidence=evidence,
        )


def _body_velocity_in_home(
    observation: PlanarFusionObservation,
    yaw_rad: float,
) -> tuple[float, float] | None:
    if (
        observation.velocity_forward_mps is None
        or observation.velocity_left_mps is None
    ):
        return None
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return (
        cosine * observation.velocity_forward_mps
        - sine * observation.velocity_left_mps,
        sine * observation.velocity_forward_mps
        + cosine * observation.velocity_left_mps,
    )


def _kalman_scalar(
    estimate: float,
    variance: float,
    measurement: float,
    measurement_variance: float,
) -> tuple[float, float, float]:
    gain = variance / (variance + measurement_variance)
    updated = estimate + gain * (measurement - estimate)
    return updated, max(1e-12, (1.0 - gain) * variance), gain


def _kalman_angle(
    estimate: float,
    variance: float,
    measurement: float,
    measurement_variance: float,
) -> tuple[float, float, float]:
    gain = variance / (variance + measurement_variance)
    updated = normalize_angle(
        estimate + gain * normalize_angle(measurement - estimate)
    )
    return updated, max(1e-12, (1.0 - gain) * variance), gain


def _pose_distance(left: Pose2D, right: Pose2D) -> float:
    return math.hypot(left.x_m - right.x_m, left.y_m - right.y_m)
