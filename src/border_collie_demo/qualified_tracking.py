"""Temporal fruit qualification with explicit motion recommendations.

The perception sidecar intentionally publishes its raw best candidate.  This
module is the motion-policy seam: it turns a sequence of normalized perception
statuses into one conservative recommendation without allowing any single raw
frame to authorize forward movement.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from .fruits import FruitPolicy, fruit_policy


class MotionRecommendation(str, Enum):
    SEARCH = "search"
    ALIGN = "align"
    APPROACH = "approach"
    SLOW = "slow"
    ARRIVAL = "arrival"
    STOP = "stop"


@dataclass(frozen=True)
class QualifiedTrackingConfig:
    target_fruit: str
    acquisition_confidence: float
    tracking_confidence: float
    acquisition_confirmations: int
    center_tolerance_ratio: float
    center_confirmations: int
    near_bottom_ratio: float
    near_center_ratio: float
    near_confirmations: int
    sight_loss_grace_s: float
    maximum_detection_age_s: float = 0.250
    close_bottom_ratio: float = 0.70
    maximum_center_delta_ratio: float = 0.20
    stationary_recenter_error_ratio: float = 0.40
    maximum_vertical_retreat_ratio: float = 0.08
    maximum_area_retreat_fraction: float = 0.35
    slow_speed_scale: float = 0.30

    @classmethod
    def for_fruit(
        cls,
        target_fruit: str,
        *,
        acquisition_confirmations: int,
        center_tolerance_ratio: float,
        center_confirmations: int,
        near_bottom_ratio: float,
        near_center_ratio: float,
        near_confirmations: int,
        sight_loss_grace_s: float,
        slow_speed_scale: float,
        minimum_tracking_confidence: float | None = None,
    ) -> QualifiedTrackingConfig:
        policy: FruitPolicy = fruit_policy(target_fruit)
        configured_floor = (
            policy.close_range_tracking_confidence
            if minimum_tracking_confidence is None
            else float(minimum_tracking_confidence)
        )
        if not math.isfinite(configured_floor) or not 0.0 <= configured_floor <= 1.0:
            raise ValueError(
                "minimum_tracking_confidence must be finite and between zero and one"
            )
        return cls(
            target_fruit=target_fruit.casefold().strip(),
            acquisition_confidence=policy.acquisition_confidence,
            # Runtime configuration may tighten a policy, never weaken the
            # fruit-specific floor qualified by tests and recorded evidence.
            tracking_confidence=max(
                policy.close_range_tracking_confidence,
                configured_floor,
            ),
            acquisition_confirmations=acquisition_confirmations,
            center_tolerance_ratio=center_tolerance_ratio,
            center_confirmations=center_confirmations,
            near_bottom_ratio=near_bottom_ratio,
            near_center_ratio=near_center_ratio,
            near_confirmations=near_confirmations,
            sight_loss_grace_s=sight_loss_grace_s,
            slow_speed_scale=slow_speed_scale,
        )

    def __post_init__(self) -> None:
        if not self.target_fruit:
            raise ValueError("target_fruit must be non-empty")
        ratios = (
            self.acquisition_confidence,
            self.tracking_confidence,
            self.center_tolerance_ratio,
            self.near_bottom_ratio,
            self.near_center_ratio,
            self.close_bottom_ratio,
            self.maximum_center_delta_ratio,
            self.stationary_recenter_error_ratio,
            self.maximum_vertical_retreat_ratio,
            self.maximum_area_retreat_fraction,
            self.slow_speed_scale,
        )
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in ratios):
            raise ValueError("tracking ratios must be finite and between zero and one")
        if (
            min(
                self.acquisition_confirmations,
                self.center_confirmations,
                self.near_confirmations,
            )
            < 1
        ):
            raise ValueError("tracking confirmation counts must be positive")
        if self.stationary_recenter_error_ratio <= self.center_tolerance_ratio:
            raise ValueError(
                "stationary recenter error must exceed initial center tolerance"
            )
        if (
            not math.isfinite(self.sight_loss_grace_s)
            or self.sight_loss_grace_s <= 0.0
            or not math.isfinite(self.maximum_detection_age_s)
            or self.maximum_detection_age_s <= 0.0
        ):
            raise ValueError("tracking time limits must be finite and positive")


@dataclass(frozen=True)
class TrackDecision:
    recommendation: MotionRecommendation
    reason: str
    forward_scale: float
    horizontal_error: float
    evidence: Mapping[str, object]

    @property
    def arrival_confirmed(self) -> bool:
        return self.recommendation is MotionRecommendation.ARRIVAL


@dataclass(frozen=True)
class _Observation:
    confidence: float | None
    center_x: float
    center_y: float
    bottom: float
    area: float | None
    source_pts: int | None


class QualifiedFruitTracker:
    """Fuse raw observations behind one deterministic motion-policy interface."""

    def __init__(
        self,
        config: QualifiedTrackingConfig,
        *,
        clock=time.monotonic,
    ) -> None:
        self.config = config
        self._clock = clock
        self._acquisition_samples = 0
        self._centered_samples = 0
        self._near_samples = 0
        self._qualified_samples = 0
        self._close_samples = 0
        self._weak_samples = 0
        self._stale_samples = 0
        self._duplicate_samples = 0
        self._identity_resets = 0
        self._discontinuity_stops = 0
        self._bbox_growth_samples = 0
        self._track_acquired = False
        self._initial_centered = False
        self._approach_authorized = False
        self._stationary_recenter_samples = 0
        self._last_observation: _Observation | None = None
        self._last_source_pts: int | None = None
        self._last_qualified_at: float | None = None
        self._minimum_tracking_confidence: float | None = None
        self._maximum_bottom_ratio = 0.0
        self._maximum_bbox_area_ratio = 0.0
        self._arrival_mode: str | None = None

    def observe(
        self,
        status: Mapping[str, object],
        *,
        now_s: float | None = None,
    ) -> TrackDecision:
        """Return the only motion recommendation authorized by this history."""
        now = self._clock() if now_s is None else float(now_s)
        if not math.isfinite(now):
            raise ValueError("observation time must be finite")
        if status.get("camera_healthy") is not True:
            self._invalidate_close_loss()
            return self._decision(MotionRecommendation.STOP, "camera_unhealthy")

        detection = status.get("detection")
        if not isinstance(detection, Mapping):
            return self._handle_loss(now, "detection_missing")
        label = str(detection.get("label") or "").casefold().strip()
        if label != self.config.target_fruit:
            if label:
                self._identity_resets += 1
                self._reset_track()
                return self._decision(
                    MotionRecommendation.STOP,
                    "target_identity_changed",
                )
            return self._handle_loss(now, "detection_missing")

        age_s = _finite_number(detection.get("age_s"))
        freshness_attested = status.get("target_ready") is True
        if age_s is None and not freshness_attested:
            self._stale_samples += 1
            self._invalidate_close_loss()
            return self._decision(MotionRecommendation.STOP, "detection_age_missing")
        if age_s is not None and (
            age_s < 0.0 or age_s > self.config.maximum_detection_age_s
        ):
            self._stale_samples += 1
            self._invalidate_close_loss()
            return self._decision(MotionRecommendation.STOP, "detection_stale")

        generation = status.get("generation")
        detection_generation = detection.get("generation")
        if (
            isinstance(generation, str)
            and generation
            and detection_generation is not None
            and detection_generation != generation
        ):
            self._stale_samples += 1
            self._invalidate_close_loss()
            return self._decision(MotionRecommendation.STOP, "generation_mismatch")

        observation = self._parse_observation(detection)
        if observation is None:
            self._invalidate_close_loss()
            return self._decision(MotionRecommendation.STOP, "geometry_invalid")
        if (
            observation.source_pts is not None
            and observation.source_pts == self._last_source_pts
        ):
            self._duplicate_samples += 1
            return self._decision(
                MotionRecommendation.STOP, "duplicate_detection_frame"
            )

        confidence = observation.confidence
        acquisition_qualified = (
            freshness_attested
            if confidence is None
            else confidence >= self.config.acquisition_confidence
        )
        tracking_qualified = (
            freshness_attested
            if confidence is None
            else confidence >= self.config.tracking_confidence
        )

        if not self._track_acquired:
            if not acquisition_qualified:
                self._weak_samples += 1
                self._acquisition_samples = 0
                self._centered_samples = 0
                return self._decision(MotionRecommendation.SEARCH, "target_unqualified")
            self._accept_observation(observation, now)
            self._acquisition_samples += 1
            self._update_centering(observation)
            if self._acquisition_samples < self.config.acquisition_confirmations:
                if self._approach_authorized:
                    return self._decision(
                        MotionRecommendation.STOP,
                        "confirming_target_reacquisition",
                        observation,
                    )
                return self._decision(
                    MotionRecommendation.ALIGN,
                    "confirming_target_identity",
                    observation,
                )
            self._track_acquired = True
            if self._approach_authorized:
                self._initial_centered = True
            if self._centered_samples >= self.config.center_confirmations:
                self._initial_centered = True
            if not self._initial_centered:
                return self._decision(
                    MotionRecommendation.ALIGN,
                    "centering_acquired_target",
                    observation,
                )
            return self._recommend_visible(observation, now)

        if not tracking_qualified:
            self._weak_samples += 1
            return self._handle_weak_close_observation(observation, now)
        if not self._continuous_with_last(observation):
            self._discontinuity_stops += 1
            self._reset_track(preserve_approach_authorization=True)
            return self._decision(MotionRecommendation.STOP, "track_discontinuous")

        self._accept_observation(observation, now)
        if not self._initial_centered:
            self._update_centering(observation)
            if self._centered_samples < self.config.center_confirmations:
                return self._decision(
                    MotionRecommendation.ALIGN,
                    "centering_acquired_target",
                    observation,
                )
            self._initial_centered = True
        return self._recommend_visible(observation, now)

    def _recommend_visible(
        self,
        observation: _Observation,
        now: float,
    ) -> TrackDecision:
        horizontal_error = observation.center_x - 0.5
        if (
            self._approach_authorized
            and abs(horizontal_error) > self.config.stationary_recenter_error_ratio
        ):
            self._stationary_recenter_samples += 1
            return self._decision(
                MotionRecommendation.ALIGN,
                "large_tracking_error",
                observation,
            )

        near = (
            observation.bottom >= self.config.near_bottom_ratio
            and observation.center_y >= self.config.near_center_ratio
        )
        self._near_samples = self._near_samples + 1 if near else 0
        if self._near_samples >= self.config.near_confirmations:
            self._arrival_mode = "visible_geometry"
            return self._decision(
                MotionRecommendation.ARRIVAL,
                "qualified_visible_arrival",
                observation,
            )

        slow_bottom = max(
            self.config.close_bottom_ratio,
            self.config.near_bottom_ratio - 0.10,
        )
        if observation.bottom >= slow_bottom or self._close_samples >= 2:
            self._approach_authorized = True
            return self._decision(
                MotionRecommendation.SLOW,
                "close_range_track",
                observation,
                forward_scale=self.config.slow_speed_scale,
            )
        self._approach_authorized = True
        return self._decision(
            MotionRecommendation.APPROACH,
            "qualified_track",
            observation,
            forward_scale=1.0,
        )

    def _handle_weak_close_observation(
        self,
        observation: _Observation,
        now: float,
    ) -> TrackDecision:
        if self._close_loss_is_arrival(now, corroborating=observation):
            self._arrival_mode = "confidence_collapse_at_close_range"
            return self._decision(
                MotionRecommendation.ARRIVAL,
                "qualified_close_track_confidence_collapsed",
                observation,
            )
        self._invalidate_close_loss()
        return self._decision(MotionRecommendation.STOP, "tracking_confidence_low")

    def _handle_loss(self, now: float, reason: str) -> TrackDecision:
        if self._close_loss_is_arrival(now):
            self._arrival_mode = "sight_lost_at_close_range"
            return self._decision(
                MotionRecommendation.ARRIVAL,
                "qualified_close_track_lost",
            )
        recommendation = (
            MotionRecommendation.STOP
            if self._track_acquired
            else MotionRecommendation.SEARCH
        )
        return self._decision(recommendation, reason)

    def _close_loss_is_arrival(
        self,
        now: float,
        *,
        corroborating: _Observation | None = None,
    ) -> bool:
        previous = self._last_observation
        if (
            not self._track_acquired
            or previous is None
            or self._last_qualified_at is None
            or now - self._last_qualified_at > self.config.sight_loss_grace_s
            or self._close_samples < 2
        ):
            return False
        close_enough = (
            previous.bottom >= self.config.near_bottom_ratio - 0.05
            and previous.center_y >= self.config.near_center_ratio - 0.05
        )
        if not close_enough:
            return False
        if corroborating is None:
            return True
        return (
            self._continuous_with_last(corroborating)
            and corroborating.bottom >= previous.bottom
            and (
                corroborating.area is None
                or previous.area is None
                or corroborating.area >= previous.area
            )
        )

    def _accept_observation(self, observation: _Observation, now: float) -> None:
        previous = self._last_observation
        if (
            previous is not None
            and observation.area is not None
            and previous.area is not None
            and observation.area > previous.area
        ):
            self._bbox_growth_samples += 1
        if observation.bottom >= self.config.close_bottom_ratio:
            self._close_samples += 1
        self._qualified_samples += 1
        if observation.confidence is not None:
            self._minimum_tracking_confidence = (
                observation.confidence
                if self._minimum_tracking_confidence is None
                else min(self._minimum_tracking_confidence, observation.confidence)
            )
        self._maximum_bottom_ratio = max(
            self._maximum_bottom_ratio,
            observation.bottom,
        )
        if observation.area is not None:
            self._maximum_bbox_area_ratio = max(
                self._maximum_bbox_area_ratio,
                observation.area,
            )
        self._last_observation = observation
        self._last_source_pts = observation.source_pts
        self._last_qualified_at = now

    def _continuous_with_last(self, observation: _Observation) -> bool:
        previous = self._last_observation
        if previous is None:
            return True
        if (
            abs(observation.center_x - previous.center_x)
            > self.config.maximum_center_delta_ratio
        ):
            return False
        if (
            observation.center_y
            < previous.center_y - self.config.maximum_vertical_retreat_ratio
        ):
            return False
        if (
            observation.bottom
            < previous.bottom - self.config.maximum_vertical_retreat_ratio
        ):
            return False
        return not (
            observation.area is not None
            and previous.area is not None
            and observation.area
            < previous.area * (1.0 - self.config.maximum_area_retreat_fraction)
        )

    def _update_centering(self, observation: _Observation) -> None:
        if abs(observation.center_x - 0.5) <= self.config.center_tolerance_ratio:
            self._centered_samples += 1
        else:
            self._centered_samples = 0

    def _parse_observation(
        self, detection: Mapping[str, object]
    ) -> _Observation | None:
        center_x = _finite_number(detection.get("center_x_ratio"))
        center_y = _finite_number(detection.get("center_y_ratio"))
        bottom = _finite_number(detection.get("bottom_ratio"))
        if (
            center_x is None
            or center_y is None
            or bottom is None
            or not all(0.0 <= value <= 1.0 for value in (center_x, center_y, bottom))
            or bottom < center_y
        ):
            return None
        area = _finite_number(detection.get("bbox_area_ratio"))
        if area is not None and not 0.0 < area <= 1.0:
            return None
        confidence = _finite_number(detection.get("confidence"))
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            return None
        raw_source_pts = detection.get("source_pts")
        source_pts = (
            raw_source_pts
            if isinstance(raw_source_pts, int) and not isinstance(raw_source_pts, bool)
            else None
        )
        return _Observation(
            confidence=confidence,
            center_x=center_x,
            center_y=center_y,
            bottom=bottom,
            area=area,
            source_pts=source_pts,
        )

    def _reset_track(self, *, preserve_approach_authorization: bool = False) -> None:
        approach_authorized = (
            self._approach_authorized if preserve_approach_authorization else False
        )
        self._acquisition_samples = 0
        self._centered_samples = 0
        self._near_samples = 0
        self._close_samples = 0
        self._track_acquired = False
        self._approach_authorized = approach_authorized
        self._initial_centered = approach_authorized
        self._last_observation = None
        self._last_source_pts = None
        self._last_qualified_at = None

    def _invalidate_close_loss(self) -> None:
        """Prevent stale or ambiguous evidence from becoming a later Arrival."""
        self._last_qualified_at = None
        self._near_samples = 0

    def _decision(
        self,
        recommendation: MotionRecommendation,
        reason: str,
        observation: _Observation | None = None,
        *,
        forward_scale: float = 0.0,
    ) -> TrackDecision:
        horizontal_error = 0.0 if observation is None else observation.center_x - 0.5
        evidence: dict[str, object] = {
            "tracking_recommendation": recommendation.value,
            "tracking_reason": reason,
            "track_acquired": self._track_acquired,
            "initial_centered": self._initial_centered,
            "approach_authorized": self._approach_authorized,
            "stationary_recenter_samples": self._stationary_recenter_samples,
            "stationary_recenter_error_ratio": (
                self.config.stationary_recenter_error_ratio
            ),
            "acquisition_samples": self._acquisition_samples,
            "qualified_samples": self._qualified_samples,
            "close_range_samples": self._close_samples,
            "near_samples": self._near_samples,
            "weak_samples": self._weak_samples,
            "stale_samples": self._stale_samples,
            "duplicate_samples": self._duplicate_samples,
            "identity_resets": self._identity_resets,
            "discontinuity_stops": self._discontinuity_stops,
            "bbox_growth_samples": self._bbox_growth_samples,
            "minimum_observed_tracking_confidence": self._minimum_tracking_confidence,
            "maximum_bottom_ratio": self._maximum_bottom_ratio,
            "maximum_bbox_area_ratio": self._maximum_bbox_area_ratio,
            "arrival_mode": self._arrival_mode,
            "acquisition_confidence": self.config.acquisition_confidence,
            "close_range_tracking_confidence": self.config.tracking_confidence,
            "detection_maximum_age_s": self.config.maximum_detection_age_s,
        }
        return TrackDecision(
            recommendation=recommendation,
            reason=reason,
            forward_scale=forward_scale,
            horizontal_error=horizontal_error,
            evidence=evidence,
        )


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None
