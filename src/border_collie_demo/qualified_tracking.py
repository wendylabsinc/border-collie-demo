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
    HOLD = "hold"
    ARRIVAL = "arrival"
    STOP = "stop"


SEARCH_QUALIFICATION_MINIMUM_DETECTIONS = 5


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
    center_filter_alpha: float = 0.70
    moving_steering_enter_ratio: float = 0.25
    moving_steering_exit_ratio: float = 0.12
    moving_steering_enter_confirmations: int = 2
    stationary_recenter_error_ratio: float = 0.40
    stationary_recenter_confirmations: int = 2
    close_handoff_center_ratio: float = 0.08
    close_handoff_center_confirmations: int = 3
    final_approach_latch_enabled: bool = False
    final_approach_loss_confirmations: int = 2
    close_recenter_enter_ratio: float = 0.12
    close_recenter_confirmations: int = 2
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
        final_approach_latch_enabled: bool = False,
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
            final_approach_latch_enabled=final_approach_latch_enabled,
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
            self.center_filter_alpha,
            self.moving_steering_enter_ratio,
            self.moving_steering_exit_ratio,
            self.stationary_recenter_error_ratio,
            self.close_handoff_center_ratio,
            self.close_recenter_enter_ratio,
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
                self.moving_steering_enter_confirmations,
                self.stationary_recenter_confirmations,
                self.close_handoff_center_confirmations,
                self.final_approach_loss_confirmations,
                self.close_recenter_confirmations,
            )
            < 1
        ):
            raise ValueError("tracking confirmation counts must be positive")
        if self.stationary_recenter_error_ratio <= self.center_tolerance_ratio:
            raise ValueError(
                "stationary recenter error must exceed initial center tolerance"
            )
        if not (
            self.moving_steering_exit_ratio
            < self.moving_steering_enter_ratio
            < self.stationary_recenter_error_ratio
        ):
            raise ValueError(
                "steering thresholds must satisfy exit < enter < stationary recenter"
            )
        if self.close_recenter_enter_ratio <= self.close_handoff_center_ratio:
            raise ValueError(
                "close recenter entry must exceed the handoff center corridor"
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
class SearchQualificationHandoff:
    """Narrow authority to carry a qualified search identity into approach.

    The handoff is evidence, not a motion command.  A tracker must match it to
    a current fresh observation and independently earn initial centering before
    it may recommend translation.
    """

    search_qualified: bool
    target_fruit: str
    generation: str
    source_pts: int
    source_time_base: str
    qualified_monotonic_s: float
    stable_detections: int
    confidence: float
    center_x_ratio: float
    center_y_ratio: float
    bottom_ratio: float
    bbox_area_ratio: float

    def __post_init__(self) -> None:
        if not self.target_fruit.casefold().strip():
            raise ValueError("handoff target_fruit must be non-empty")
        if not self.generation.strip() or not self.source_time_base.strip():
            raise ValueError("handoff generation and time base must be non-empty")
        if (
            isinstance(self.source_pts, bool)
            or not isinstance(self.source_pts, int)
            or self.source_pts < 0
        ):
            raise ValueError("handoff source PTS must be a non-negative integer")
        if (
            isinstance(self.stable_detections, bool)
            or not isinstance(self.stable_detections, int)
            or self.stable_detections < 1
        ):
            raise ValueError("handoff stable detections must be positive")
        if not math.isfinite(self.qualified_monotonic_s):
            raise ValueError("handoff qualification time must be finite")
        ratios = (
            self.confidence,
            self.center_x_ratio,
            self.center_y_ratio,
            self.bottom_ratio,
        )
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in ratios):
            raise ValueError("handoff confidence and geometry must be normalized")
        if self.bottom_ratio < self.center_y_ratio:
            raise ValueError("handoff bottom must not precede its vertical center")
        if not (
            math.isfinite(self.bbox_area_ratio) and 0.0 < self.bbox_area_ratio <= 1.0
        ):
            raise ValueError("handoff area must be normalized")

    def to_evidence(self) -> dict[str, object]:
        return {
            "search_qualified": self.search_qualified,
            "target_fruit": self.target_fruit,
            "generation": self.generation,
            "source_pts": self.source_pts,
            "source_time_base": self.source_time_base,
            "qualified_monotonic_s": self.qualified_monotonic_s,
            "stable_detections": self.stable_detections,
            "confidence": self.confidence,
            "center_x_ratio": self.center_x_ratio,
            "center_y_ratio": self.center_y_ratio,
            "bottom_ratio": self.bottom_ratio,
            "bbox_area_ratio": self.bbox_area_ratio,
        }

    @classmethod
    def from_evidence(
        cls,
        evidence: Mapping[str, object],
    ) -> SearchQualificationHandoff:
        return cls(
            search_qualified=evidence.get("search_qualified") is True,
            target_fruit=str(evidence.get("target_fruit") or ""),
            generation=str(evidence.get("generation") or ""),
            source_pts=_required_int(evidence.get("source_pts"), "source_pts"),
            source_time_base=str(evidence.get("source_time_base") or ""),
            qualified_monotonic_s=_required_number(
                evidence.get("qualified_monotonic_s"),
                "qualified_monotonic_s",
            ),
            stable_detections=_required_int(
                evidence.get("stable_detections"),
                "stable_detections",
            ),
            confidence=_required_number(evidence.get("confidence"), "confidence"),
            center_x_ratio=_required_number(
                evidence.get("center_x_ratio"),
                "center_x_ratio",
            ),
            center_y_ratio=_required_number(
                evidence.get("center_y_ratio"),
                "center_y_ratio",
            ),
            bottom_ratio=_required_number(
                evidence.get("bottom_ratio"),
                "bottom_ratio",
            ),
            bbox_area_ratio=_required_number(
                evidence.get("bbox_area_ratio"),
                "bbox_area_ratio",
            ),
        )


def search_handoff_from_status(
    status: Mapping[str, object],
    target_fruit: str,
    *,
    qualified_monotonic_s: float,
) -> SearchQualificationHandoff | None:
    """Capture only a complete, motion-qualified search observation."""
    target = target_fruit.casefold().strip()
    detection = status.get("detection")
    source = status.get("source")
    if not (
        target
        and status.get("camera_healthy") is True
        and status.get("target_ready") is True
        and isinstance(detection, Mapping)
        and isinstance(source, Mapping)
        and str(detection.get("label") or "").casefold().strip() == target
    ):
        return None
    generation = status.get("generation")
    detection_generation = detection.get("generation")
    source_time_base = source.get("time_base")
    detection_time_base = detection.get("source_time_base")
    source_pts = detection.get("source_pts")
    current_source_pts = source.get("pts")
    stable_detections = detection.get("consecutive_detections")
    if (
        not isinstance(generation, str)
        or not generation
        or detection_generation != generation
        or not isinstance(source_time_base, str)
        or not source_time_base
        or detection_time_base != source_time_base
        or not isinstance(source_pts, int)
        or isinstance(source_pts, bool)
        or not isinstance(current_source_pts, int)
        or isinstance(current_source_pts, bool)
        or source_pts > current_source_pts
        or not isinstance(stable_detections, int)
        or isinstance(stable_detections, bool)
    ):
        return None
    try:
        handoff = SearchQualificationHandoff(
            search_qualified=True,
            target_fruit=target,
            generation=generation,
            source_pts=source_pts,
            source_time_base=source_time_base,
            qualified_monotonic_s=float(qualified_monotonic_s),
            stable_detections=stable_detections,
            confidence=_required_number(detection.get("confidence"), "confidence"),
            center_x_ratio=_required_number(
                detection.get("center_x_ratio"),
                "center_x_ratio",
            ),
            center_y_ratio=_required_number(
                detection.get("center_y_ratio"),
                "center_y_ratio",
            ),
            bottom_ratio=_required_number(
                detection.get("bottom_ratio"),
                "bottom_ratio",
            ),
            bbox_area_ratio=_required_number(
                detection.get("bbox_area_ratio"),
                "bbox_area_ratio",
            ),
        )
    except (TypeError, ValueError):
        return None
    policy = fruit_policy(target)
    if (
        handoff.confidence < policy.acquisition_confidence
        or handoff.stable_detections < SEARCH_QUALIFICATION_MINIMUM_DETECTIONS
    ):
        return None
    return handoff


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
        search_handoff: SearchQualificationHandoff | None = None,
        clock=time.monotonic,
    ) -> None:
        self.config = config
        self._clock = clock
        self._acquisition_samples = 0
        self._centered_samples = 0
        self._near_samples = 0
        self._qualified_samples = 0
        self._close_samples = 0
        self._close_handoff_centered_samples = 0
        self._close_recenter_samples = 0
        self._close_recenter_active = False
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
        self._extreme_error_samples = 0
        self._steering_outside_samples = 0
        self._steering_active = False
        self._filtered_center_x: float | None = None
        self._pending_center_jump_samples = 0
        self._pending_center_jump_pts: int | None = None
        self._last_observation: _Observation | None = None
        self._last_source_pts: int | None = None
        self._last_qualified_at: float | None = None
        self._minimum_tracking_confidence: float | None = None
        self._maximum_bottom_ratio = 0.0
        self._maximum_bbox_area_ratio = 0.0
        self._arrival_mode: str | None = None
        self._final_approach_latched_at: float | None = None
        self._final_approach_authority_at: float | None = None
        self._final_approach_generation: str | None = None
        self._final_approach_last_evidence_pts: int | None = None
        self._final_approach_loss_samples = 0
        self._final_approach_cancelled_reason: str | None = None
        self._final_approach_completed = False
        self._current_generation: str | None = None
        self._search_handoff = search_handoff
        self._search_handoff_accepted = False
        self._search_handoff_rejection_reason: str | None = None

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
            self._reject_search_handoff("camera_unhealthy")
            self._cancel_final_approach("camera_unhealthy")
            self._invalidate_close_loss()
            return self._decision(MotionRecommendation.STOP, "camera_unhealthy")

        if self._final_approach_completed:
            return self._decision(MotionRecommendation.STOP, "arrival_already_confirmed")
        if self._final_approach_cancelled_reason is not None:
            return self._decision(
                MotionRecommendation.STOP,
                self._final_approach_cancelled_reason,
            )

        generation = status.get("generation")
        if isinstance(generation, str) and generation:
            self._current_generation = generation
        if self._final_approach_latched_at is not None and (
            not isinstance(generation, str)
            or not generation
            or generation != self._final_approach_generation
        ):
            self._cancel_final_approach("final_approach_generation_changed")
            return self._decision(
                MotionRecommendation.STOP,
                "final_approach_generation_changed",
            )
        if self._final_approach_expired(now):
            self._cancel_final_approach("final_approach_expired")
            return self._decision(MotionRecommendation.STOP, "final_approach_expired")

        detection = status.get("detection")
        if not isinstance(detection, Mapping):
            self._reject_search_handoff("detection_missing")
            if self._final_approach_latched_at is not None:
                return self._confirm_final_approach_loss(status, now)
            return self._handle_loss(now, "detection_missing")
        label = str(detection.get("label") or "").casefold().strip()
        if label != self.config.target_fruit:
            if label:
                self._reject_search_handoff("target_mismatch")
                self._cancel_final_approach("target_identity_changed")
                self._identity_resets += 1
                self._reset_track()
                return self._decision(
                    MotionRecommendation.STOP,
                    "target_identity_changed",
                )
            if self._final_approach_latched_at is not None:
                return self._confirm_final_approach_loss(status, now)
            self._reject_search_handoff("detection_missing")
            return self._handle_loss(now, "detection_missing")

        age_s = _finite_number(detection.get("age_s"))
        freshness_attested = status.get("target_ready") is True
        if age_s is None:
            self._reject_search_handoff("detection_age_missing")
            self._stale_samples += 1
            self._note_final_approach_frame(detection.get("source_pts"))
            self._invalidate_close_loss()
            return self._decision(MotionRecommendation.STOP, "detection_age_missing")
        if age_s is not None and (
            age_s < 0.0 or age_s > self.config.maximum_detection_age_s
        ):
            self._reject_search_handoff("detection_stale")
            self._stale_samples += 1
            self._note_final_approach_frame(detection.get("source_pts"))
            self._invalidate_close_loss()
            return self._decision(MotionRecommendation.STOP, "detection_stale")

        detection_generation = detection.get("generation")
        if (
            isinstance(generation, str)
            and generation
            and detection_generation is not None
            and detection_generation != generation
        ):
            self._reject_search_handoff("generation_mismatch")
            self._stale_samples += 1
            self._cancel_final_approach("generation_mismatch")
            self._invalidate_close_loss()
            return self._decision(MotionRecommendation.STOP, "generation_mismatch")

        observation = self._parse_observation(detection)
        if observation is None:
            self._reject_search_handoff("geometry_invalid")
            self._cancel_final_approach("geometry_invalid")
            self._invalidate_close_loss()
            return self._decision(MotionRecommendation.STOP, "geometry_invalid")
        duplicate = (
            observation.source_pts is not None
            and observation.source_pts == self._last_source_pts
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

        if self._final_approach_latched_at is not None:
            return self._observe_final_approach(
                observation,
                now,
                tracking_qualified=tracking_qualified,
            )

        if not self._track_acquired:
            if self._search_handoff is not None:
                accepted, rejection_reason = self._consume_search_handoff(
                    status,
                    detection,
                    observation,
                    now,
                    tracking_qualified=tracking_qualified,
                )
                if accepted:
                    self._accept_observation(observation, now)
                    self._track_acquired = True
                    self._update_centering(observation)
                    return self._decision(
                        MotionRecommendation.ALIGN,
                        "confirming_search_handoff_centering",
                        observation,
                        horizontal_error=self._filtered_horizontal_error(observation),
                    )
                self._search_handoff_rejection_reason = rejection_reason
            if not acquisition_qualified:
                self._weak_samples += 1
                self._acquisition_samples = 0
                self._centered_samples = 0
                return self._decision(MotionRecommendation.SEARCH, "target_unqualified")
            if duplicate:
                self._duplicate_samples += 1
                return self._decision(
                    MotionRecommendation.HOLD,
                    "duplicate_detection_frame",
                    observation,
                )
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
                    horizontal_error=self._filtered_horizontal_error(observation),
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
                    horizontal_error=self._filtered_horizontal_error(observation),
                )
            return self._recommend_visible(observation, now)

        if not tracking_qualified:
            self._weak_samples += 1
            if duplicate:
                if self._close_loss_candidate(now) is not None:
                    self._duplicate_samples += 1
                    return self._decision(
                        MotionRecommendation.HOLD,
                        "duplicate_weak_close_frame",
                        observation,
                    )
                self._invalidate_close_loss()
                return self._decision(
                    MotionRecommendation.STOP,
                    "tracking_confidence_low",
                    observation,
                )
            return self._handle_weak_close_observation(observation, now)
        continuity_issue = self._continuity_issue(observation)
        if continuity_issue == "center_jump":
            same_suspect_frame = (
                observation.source_pts is not None
                and observation.source_pts == self._pending_center_jump_pts
            )
            if same_suspect_frame:
                self._duplicate_samples += 1
            else:
                self._pending_center_jump_samples += 1
                self._pending_center_jump_pts = observation.source_pts
            last_authority_fresh = (
                self._last_qualified_at is not None
                and now - self._last_qualified_at
                <= self.config.maximum_detection_age_s
            )
            if self._pending_center_jump_samples < 2 and last_authority_fresh:
                return self._decision(
                    MotionRecommendation.HOLD,
                    "confirming_center_jump",
                    observation,
                )
            self._discontinuity_stops += 1
            self._reset_track(preserve_approach_authorization=True)
            return self._decision(MotionRecommendation.STOP, "track_discontinuous")
        if continuity_issue is not None:
            self._discontinuity_stops += 1
            self._reset_track(preserve_approach_authorization=True)
            return self._decision(MotionRecommendation.STOP, "track_discontinuous")
        self._pending_center_jump_samples = 0
        self._pending_center_jump_pts = None
        if duplicate:
            self._duplicate_samples += 1
            return self._decision(
                MotionRecommendation.HOLD,
                "duplicate_detection_frame",
                observation,
            )

        self._accept_observation(observation, now)
        if not self._initial_centered:
            self._update_centering(observation)
            if self._centered_samples < self.config.center_confirmations:
                return self._decision(
                    MotionRecommendation.ALIGN,
                    "centering_acquired_target",
                    observation,
                    horizontal_error=self._filtered_horizontal_error(observation),
                )
            self._initial_centered = True
        return self._recommend_visible(observation, now)

    def _consume_search_handoff(
        self,
        status: Mapping[str, object],
        detection: Mapping[str, object],
        observation: _Observation,
        now: float,
        *,
        tracking_qualified: bool,
    ) -> tuple[bool, str | None]:
        handoff = self._search_handoff
        self._search_handoff = None
        assert handoff is not None
        target = self.config.target_fruit
        if not handoff.search_qualified:
            return False, "search_not_qualified"
        if handoff.target_fruit.casefold().strip() != target:
            return False, "target_mismatch"
        if handoff.confidence < self.config.acquisition_confidence:
            return False, "handoff_confidence_low"
        if handoff.stable_detections < self.config.acquisition_confirmations:
            return False, "handoff_stability_low"
        age_s = now - handoff.qualified_monotonic_s
        if age_s < 0.0:
            return False, "handoff_time_invalid"
        if age_s > self.config.maximum_detection_age_s:
            return False, "handoff_stale"
        generation = status.get("generation")
        if (
            generation != handoff.generation
            or detection.get("generation") != handoff.generation
        ):
            return False, "generation_mismatch"
        source = status.get("source")
        if not isinstance(source, Mapping):
            return False, "source_evidence_missing"
        source_time_base = source.get("time_base")
        detection_time_base = detection.get("source_time_base")
        if (
            source_time_base != handoff.source_time_base
            or detection_time_base != handoff.source_time_base
        ):
            return False, "time_base_mismatch"
        source_age_s = _finite_number(source.get("age_s"))
        if (
            source_age_s is None
            or source_age_s < 0.0
            or source_age_s > self.config.maximum_detection_age_s
        ):
            return False, "source_stale"
        if (
            observation.source_pts is None
            or observation.source_pts < handoff.source_pts
        ):
            return False, "source_regressed"
        source_pts = source.get("pts")
        if (
            not isinstance(source_pts, int)
            or isinstance(source_pts, bool)
            or observation.source_pts > source_pts
        ):
            return False, "source_identity_invalid"
        if not tracking_qualified:
            return False, "tracking_confidence_low"
        if observation.area is None:
            return False, "geometry_invalid"
        if (
            abs(observation.center_x - 0.5)
            > self.config.moving_steering_enter_ratio
        ):
            return False, "geometry_off_axis"
        if (
            abs(observation.center_x - handoff.center_x_ratio)
            > self.config.maximum_center_delta_ratio
            or observation.center_y
            < handoff.center_y_ratio - self.config.maximum_vertical_retreat_ratio
            or observation.bottom
            < handoff.bottom_ratio - self.config.maximum_vertical_retreat_ratio
            or (
                observation.area is not None
                and observation.area
                < handoff.bbox_area_ratio
                * (1.0 - self.config.maximum_area_retreat_fraction)
            )
        ):
            return False, "geometry_discontinuous"
        self._search_handoff_accepted = True
        self._search_handoff_rejection_reason = None
        return True, None

    def _reject_search_handoff(self, reason: str) -> None:
        if self._search_handoff is None:
            return
        self._search_handoff = None
        self._search_handoff_rejection_reason = reason

    def _recommend_visible(
        self,
        observation: _Observation,
        now: float,
    ) -> TrackDecision:
        horizontal_error = self._filtered_horizontal_error(observation)
        close_geometry = (
            observation.bottom >= self.config.close_bottom_ratio
            or self._close_samples >= 2
        )
        if self.config.target_fruit == "pear" and self._close_recenter_active:
            if abs(horizontal_error) <= self.config.close_handoff_center_ratio:
                self._close_recenter_active = False
                self._close_recenter_samples = 0
            else:
                return self._decision(
                    MotionRecommendation.ALIGN,
                    "close_tracking_recenter",
                    observation,
                    horizontal_error=horizontal_error,
                )
        elif self.config.target_fruit == "pear" and close_geometry:
            self._close_recenter_samples = (
                self._close_recenter_samples + 1
                if abs(observation.center_x - 0.5)
                > self.config.close_recenter_enter_ratio
                else 0
            )
            if (
                self._close_recenter_samples
                >= self.config.close_recenter_confirmations
            ):
                self._close_recenter_active = True
                return self._decision(
                    MotionRecommendation.ALIGN,
                    "close_tracking_recenter",
                    observation,
                    horizontal_error=horizontal_error,
                )
        if self._approach_authorized and (
            self._extreme_error_samples
            >= self.config.stationary_recenter_confirmations
        ):
            self._stationary_recenter_samples += 1
            return self._decision(
                MotionRecommendation.ALIGN,
                "large_tracking_error",
                observation,
                horizontal_error=horizontal_error,
            )

        if self._steering_active:
            if abs(horizontal_error) <= self.config.moving_steering_exit_ratio:
                self._steering_active = False
                self._steering_outside_samples = 0
        else:
            self._steering_outside_samples = (
                self._steering_outside_samples + 1
                if abs(horizontal_error) > self.config.moving_steering_enter_ratio
                else 0
            )
            if (
                self._steering_outside_samples
                >= self.config.moving_steering_enter_confirmations
            ):
                self._steering_active = True

        near = (
            observation.bottom >= self.config.near_bottom_ratio
            and observation.center_y >= self.config.near_center_ratio
        )
        self._near_samples = self._near_samples + 1 if near else 0
        if self._near_samples >= self.config.near_confirmations:
            self._arrival_mode = "visible_geometry"
            if (
                self.config.final_approach_latch_enabled
                and self._current_generation is not None
                and observation.source_pts is not None
            ):
                self._latch_final_approach(observation, now)
            return self._decision(
                MotionRecommendation.ARRIVAL,
                "qualified_visible_arrival",
                observation,
                horizontal_error=horizontal_error,
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
                horizontal_error=(horizontal_error if self._steering_active else 0.0),
            )
        self._approach_authorized = True
        return self._decision(
            MotionRecommendation.APPROACH,
            "qualified_track",
            observation,
            forward_scale=1.0,
            horizontal_error=(horizontal_error if self._steering_active else 0.0),
        )

    def _observe_final_approach(
        self,
        observation: _Observation,
        now: float,
        *,
        tracking_qualified: bool,
    ) -> TrackDecision:
        if observation.source_pts is None:
            return self._decision(
                MotionRecommendation.STOP,
                "final_approach_frame_identity_missing",
            )
        if (
            self._final_approach_last_evidence_pts is not None
            and observation.source_pts <= self._final_approach_last_evidence_pts
        ):
            self._duplicate_samples += 1
            return self._decision(
                MotionRecommendation.STOP,
                "final_approach_frame_not_advancing",
                observation,
            )
        self._final_approach_last_evidence_pts = observation.source_pts

        if (
            abs(observation.center_x - 0.5)
            > self.config.close_recenter_enter_ratio
        ):
            self._cancel_final_approach("target_lost_off_axis")
            return self._decision(
                MotionRecommendation.STOP,
                "target_lost_off_axis",
                observation,
            )
        continuity_issue = self._continuity_issue(observation)
        if self._bottom_edge_area_retreat_is_loss_evidence(
            observation,
            continuity_issue,
        ):
            return self._advance_final_approach_loss(observation)
        if continuity_issue is not None:
            self._cancel_final_approach("final_approach_track_discontinuous")
            return self._decision(
                MotionRecommendation.STOP,
                "final_approach_track_discontinuous",
                observation,
            )
        if not tracking_qualified:
            self._weak_samples += 1
            return self._advance_final_approach_loss(observation)

        self._accept_observation(observation, now)
        self._final_approach_authority_at = now
        self._final_approach_loss_samples = 0
        self._arrival_mode = "visible_geometry"
        return self._decision(
            MotionRecommendation.ARRIVAL,
            "qualified_visible_arrival",
            observation,
            horizontal_error=self._filtered_horizontal_error(observation),
        )

    def _confirm_final_approach_loss(
        self,
        status: Mapping[str, object],
        now: float,
    ) -> TrackDecision:
        source = status.get("source")
        if not isinstance(source, Mapping):
            return self._decision(
                MotionRecommendation.STOP,
                "final_approach_source_evidence_missing",
            )
        source_age_s = _finite_number(source.get("age_s"))
        source_pts = source.get("pts")
        if (
            source_age_s is None
            or source_age_s < 0.0
            or source_age_s > self.config.maximum_detection_age_s
        ):
            self._stale_samples += 1
            self._note_final_approach_frame(source_pts)
            return self._decision(
                MotionRecommendation.STOP,
                "final_approach_source_stale",
            )
        if not isinstance(source_pts, int) or isinstance(source_pts, bool):
            return self._decision(
                MotionRecommendation.STOP,
                "final_approach_frame_identity_missing",
            )
        if (
            self._final_approach_last_evidence_pts is not None
            and source_pts <= self._final_approach_last_evidence_pts
        ):
            self._duplicate_samples += 1
            return self._decision(
                MotionRecommendation.STOP,
                "final_approach_frame_not_advancing",
            )
        self._final_approach_last_evidence_pts = source_pts
        return self._advance_final_approach_loss(None)

    def _advance_final_approach_loss(
        self,
        observation: _Observation | None,
    ) -> TrackDecision:
        self._final_approach_loss_samples += 1
        if (
            self._final_approach_loss_samples
            < self.config.final_approach_loss_confirmations
        ):
            return self._decision(
                MotionRecommendation.STOP,
                "confirming_final_approach_loss",
                observation,
            )
        self._arrival_mode = "final_approach_loss_confirmed"
        self._final_approach_completed = True
        return self._decision(
            MotionRecommendation.ARRIVAL,
            "qualified_final_approach_loss",
            observation,
        )

    def _latch_final_approach(
        self,
        observation: _Observation,
        now: float,
    ) -> None:
        if self._final_approach_latched_at is not None:
            return
        self._final_approach_latched_at = now
        self._final_approach_authority_at = now
        self._final_approach_generation = self._current_generation
        self._final_approach_last_evidence_pts = observation.source_pts
        self._final_approach_loss_samples = 0

    def _final_approach_expired(self, now: float) -> bool:
        return bool(
            self._final_approach_latched_at is not None
            and self._final_approach_authority_at is not None
            and now - self._final_approach_authority_at
            > self.config.sight_loss_grace_s
        )

    def _note_final_approach_frame(self, raw_source_pts: object) -> None:
        if (
            self._final_approach_latched_at is not None
            and isinstance(raw_source_pts, int)
            and not isinstance(raw_source_pts, bool)
            and (
                self._final_approach_last_evidence_pts is None
                or raw_source_pts > self._final_approach_last_evidence_pts
            )
        ):
            self._final_approach_last_evidence_pts = raw_source_pts

    def _cancel_final_approach(self, reason: str) -> None:
        if self._final_approach_latched_at is None:
            return
        self._final_approach_latched_at = None
        self._final_approach_cancelled_reason = reason

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
        if self._close_loss_is_off_axis(now):
            self._arrival_mode = None
            return self._decision(
                MotionRecommendation.STOP,
                "target_lost_off_axis",
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
        if self._close_loss_is_off_axis(now):
            self._arrival_mode = None
            return self._decision(
                MotionRecommendation.STOP,
                "target_lost_off_axis",
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
        previous = self._close_loss_candidate(now)
        if previous is None:
            return False
        if self.config.target_fruit == "pear" and (
            self._close_handoff_centered_samples
            < self.config.close_handoff_center_confirmations
        ):
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

    def _close_loss_is_off_axis(self, now: float) -> bool:
        return (
            self.config.target_fruit == "pear"
            and self._close_loss_candidate(now) is not None
            and self._close_handoff_centered_samples
            < self.config.close_handoff_center_confirmations
        )

    def _close_loss_candidate(self, now: float) -> _Observation | None:
        previous = self._last_observation
        if (
            not self._track_acquired
            or previous is None
            or self._last_qualified_at is None
            or now - self._last_qualified_at > self.config.sight_loss_grace_s
            or self._close_samples < 2
        ):
            return None
        close_enough = (
            previous.bottom >= self.config.near_bottom_ratio - 0.05
            and previous.center_y >= self.config.near_center_ratio - 0.05
        )
        return previous if close_enough else None

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
        self._extreme_error_samples = (
            self._extreme_error_samples + 1
            if abs(observation.center_x - 0.5)
            > self.config.stationary_recenter_error_ratio
            else 0
        )
        self._filtered_center_x = (
            observation.center_x
            if self._filtered_center_x is None
            else self.config.center_filter_alpha * observation.center_x
            + (1.0 - self.config.center_filter_alpha) * self._filtered_center_x
        )
        self._close_handoff_centered_samples = (
            self._close_handoff_centered_samples + 1
            if abs(self._filtered_center_x - 0.5)
            <= self.config.close_handoff_center_ratio
            else 0
        )
        self._last_source_pts = observation.source_pts
        self._last_qualified_at = now

    def _continuity_issue(self, observation: _Observation) -> str | None:
        previous = self._last_observation
        if previous is None:
            return None
        if (
            abs(observation.center_x - previous.center_x)
            > self.config.maximum_center_delta_ratio
        ):
            return "center_jump"
        if (
            observation.center_y
            < previous.center_y - self.config.maximum_vertical_retreat_ratio
        ):
            return "vertical_retreat"
        if (
            observation.bottom
            < previous.bottom - self.config.maximum_vertical_retreat_ratio
        ):
            return "vertical_retreat"
        if (
            observation.area is not None
            and previous.area is not None
            and observation.area
            < previous.area * (1.0 - self.config.maximum_area_retreat_fraction)
        ):
            return "area_retreat"
        return None

    def _continuous_with_last(self, observation: _Observation) -> bool:
        return self._continuity_issue(observation) is None

    def _bottom_edge_area_retreat_is_loss_evidence(
        self,
        observation: _Observation,
        continuity_issue: str | None,
    ) -> bool:
        """Recognize a centered box shrinking as it clips out of the image."""
        previous = self._last_observation
        if continuity_issue != "area_retreat" or previous is None:
            return False
        lower_edge_ratio = max(
            self.config.near_bottom_ratio,
            1.0 - self.config.maximum_vertical_retreat_ratio,
        )
        return bool(
            abs(previous.center_x - 0.5)
            <= self.config.close_handoff_center_ratio
            and abs(observation.center_x - 0.5)
            <= self.config.close_handoff_center_ratio
            and previous.center_y >= self.config.near_center_ratio
            and observation.center_y >= self.config.near_center_ratio
            and previous.bottom >= lower_edge_ratio
            and observation.bottom >= lower_edge_ratio
        )

    def _filtered_horizontal_error(self, observation: _Observation) -> float:
        center_x = (
            observation.center_x
            if self._filtered_center_x is None
            else self._filtered_center_x
        )
        return center_x - 0.5

    def _update_centering(self, observation: _Observation) -> None:
        if (
            abs(self._filtered_horizontal_error(observation))
            <= self.config.center_tolerance_ratio
        ):
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
        self._close_handoff_centered_samples = 0
        self._close_recenter_samples = 0
        self._close_recenter_active = False
        self._track_acquired = False
        self._approach_authorized = approach_authorized
        self._initial_centered = approach_authorized
        self._last_observation = None
        self._last_source_pts = None
        self._last_qualified_at = None
        self._filtered_center_x = None
        self._extreme_error_samples = 0
        self._steering_outside_samples = 0
        self._steering_active = False
        self._pending_center_jump_samples = 0
        self._pending_center_jump_pts = None

    def _invalidate_close_loss(self) -> None:
        """Prevent stale or ambiguous evidence from becoming a later Arrival."""
        self._last_qualified_at = None
        if self._final_approach_latched_at is None:
            self._near_samples = 0

    def _decision(
        self,
        recommendation: MotionRecommendation,
        reason: str,
        observation: _Observation | None = None,
        *,
        forward_scale: float = 0.0,
        horizontal_error: float | None = None,
    ) -> TrackDecision:
        if horizontal_error is None:
            horizontal_error = (
                0.0 if observation is None else observation.center_x - 0.5
            )
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
            "stationary_recenter_confirmations": (
                self.config.stationary_recenter_confirmations
            ),
            "filtered_center_x_ratio": self._filtered_center_x,
            "moving_steering_active": self._steering_active,
            "moving_steering_enter_ratio": self.config.moving_steering_enter_ratio,
            "moving_steering_exit_ratio": self.config.moving_steering_exit_ratio,
            "moving_steering_outside_samples": self._steering_outside_samples,
            "pending_center_jump_samples": self._pending_center_jump_samples,
            "acquisition_samples": self._acquisition_samples,
            "qualified_samples": self._qualified_samples,
            "close_range_samples": self._close_samples,
            "close_handoff_centered_samples": (
                self._close_handoff_centered_samples
            ),
            "close_handoff_center_ratio": self.config.close_handoff_center_ratio,
            "close_handoff_center_confirmations": (
                self.config.close_handoff_center_confirmations
            ),
            "close_recenter_active": self._close_recenter_active,
            "close_recenter_samples": self._close_recenter_samples,
            "close_recenter_enter_ratio": self.config.close_recenter_enter_ratio,
            "close_recenter_confirmations": (
                self.config.close_recenter_confirmations
            ),
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
            "final_approach_latched": self._final_approach_latched_at is not None,
            "final_approach_latched_at_s": self._final_approach_latched_at,
            "final_approach_generation": self._final_approach_generation,
            "final_approach_loss_samples": self._final_approach_loss_samples,
            "final_approach_loss_confirmations": (
                self.config.final_approach_loss_confirmations
            ),
            "final_approach_cancelled_reason": (
                self._final_approach_cancelled_reason
            ),
            "acquisition_confidence": self.config.acquisition_confidence,
            "close_range_tracking_confidence": self.config.tracking_confidence,
            "detection_maximum_age_s": self.config.maximum_detection_age_s,
            "search_handoff_accepted": self._search_handoff_accepted,
            "search_handoff_rejection_reason": (
                self._search_handoff_rejection_reason
            ),
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


def _required_number(value: object, field: str) -> float:
    number = _finite_number(value)
    if number is None:
        raise ValueError(f"handoff {field} must be finite")
    return number


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"handoff {field} must be an integer")
    return value
