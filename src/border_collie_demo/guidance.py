"""One mission-lifetime camera-guidance module for fruit search and Arrival.

The interface is deliberately one method: feed the latest perception status to
``FruitGuidance.observe`` and execute the returned command.  Acquisition,
identity continuity, centering, duplicate-frame cadence, lower-edge closeout,
and the bounded final push remain private implementation state.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from enum import Enum

from .config import env_bool
from .fruits import FruitPolicy, fruit_policy
from .models import VelocityCommand


class GuidancePhase(str, Enum):
    SEARCHING = "searching"
    LOCKED = "locked"
    APPROACHING = "approaching"
    FINAL_PUSH = "final_push"
    ARRIVED = "arrived"
    FAILED = "failed"


class GuidanceAction(str, Enum):
    SEARCH = "search"
    HOLD = "hold"
    ALIGN = "align"
    DRIVE = "drive"
    FINAL_PUSH = "final_push"
    STOP = "stop"
    ARRIVED = "arrived"


@dataclass(frozen=True)
class GuidanceConfig:
    """Runtime-tunable guidance policy with hardware-safety validation."""

    search_yaw_rps: float = 0.50
    search_sweep_rad: float = 2.0 * math.pi
    focus_yaw_rps: float = 0.50
    progressive_focus_yaw_enabled: bool = True
    focus_yaw_step_rps: float = 0.10
    focus_minimum_yaw_rps: float = 0.50
    focus_near_center_ratio: float = 0.20
    focus_missing_grace_s: float = 0.50
    center_tolerance_ratio: float = 0.05
    center_confirmations: int = 3
    approach_forward_mps: float = 1.0
    approach_yaw_rps: float = 0.30
    outer_corridor_ratio: float = 0.20
    recenter_yaw_rps: float = 0.50
    duplicate_hold_s: float = 0.250
    source_maximum_age_s: float = 0.350
    detection_maximum_age_s: float = 0.250
    slow_inference_grace_s: float = 0.50
    near_bottom_ratio: float = 0.90
    disappearance_bottom_ratio: float = 0.80
    near_center_ratio: float = 0.72
    near_confirmations: int = 3
    near_loss_confirmations: int = 2
    near_loss_grace_s: float = 0.75
    final_push_mps: float = 0.60
    final_push_duration_s: float = 1.0

    def __post_init__(self) -> None:
        finite = (
            self.search_yaw_rps,
            self.search_sweep_rad,
            self.focus_yaw_rps,
            self.focus_yaw_step_rps,
            self.focus_minimum_yaw_rps,
            self.focus_near_center_ratio,
            self.focus_missing_grace_s,
            self.center_tolerance_ratio,
            self.approach_forward_mps,
            self.approach_yaw_rps,
            self.outer_corridor_ratio,
            self.recenter_yaw_rps,
            self.duplicate_hold_s,
            self.source_maximum_age_s,
            self.detection_maximum_age_s,
            self.slow_inference_grace_s,
            self.near_bottom_ratio,
            self.disappearance_bottom_ratio,
            self.near_center_ratio,
            self.near_loss_grace_s,
            self.final_push_mps,
            self.final_push_duration_s,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("guidance values must be finite")
        if not 0.50 <= self.search_yaw_rps <= 0.80:
            raise ValueError("search_yaw_rps must stay within 0.50..0.80 rad/s")
        if not 0.0 < self.search_sweep_rad <= 2.0 * math.pi:
            raise ValueError("search_sweep_rad must stay within one revolution")
        if not 0.50 <= self.focus_yaw_rps <= self.search_yaw_rps:
            raise ValueError(
                "focus_yaw_rps must respect the verified 0.50 rad/s floor "
                "and not exceed search yaw"
            )
        if not 0.05 <= self.focus_yaw_step_rps <= 0.20:
            raise ValueError("focus_yaw_step_rps must stay within 0.05..0.20 rad/s")
        if not 0.50 <= self.focus_minimum_yaw_rps <= self.focus_yaw_rps:
            raise ValueError(
                "focus_minimum_yaw_rps must respect the verified 0.50 rad/s "
                "floor and not exceed focus yaw"
            )
        if not self.center_tolerance_ratio < self.focus_near_center_ratio <= 0.35:
            raise ValueError(
                "focus_near_center_ratio must stay above center tolerance "
                "and within 0.35 frame width"
            )
        if not 0.10 <= self.focus_missing_grace_s <= 1.0:
            raise ValueError(
                "focus_missing_grace_s must stay within 0.10..1.0 seconds"
            )
        if not 0.0 < self.center_tolerance_ratio < self.outer_corridor_ratio < 0.5:
            raise ValueError("center and outer corridor ratios are invalid")
        if self.center_confirmations < 1:
            raise ValueError("center_confirmations must be positive")
        if not 0.50 <= self.approach_forward_mps <= 1.0:
            raise ValueError("approach_forward_mps must stay within 0.50..1.0 m/s")
        if not 0.0 < self.approach_yaw_rps <= 0.80:
            raise ValueError("approach_yaw_rps must stay within 0.0..0.80 rad/s")
        if not 0.50 <= self.recenter_yaw_rps <= 0.80:
            raise ValueError("recenter_yaw_rps must stay within 0.50..0.80 rad/s")
        if not 0.0 < self.duplicate_hold_s <= 0.250:
            raise ValueError("duplicate_hold_s must stay within 0.0..0.250 seconds")
        if not 0.0 < self.detection_maximum_age_s <= 0.250:
            raise ValueError(
                "detection_maximum_age_s must stay within 0.0..0.250 seconds"
            )
        if not self.detection_maximum_age_s <= self.slow_inference_grace_s <= 1.0:
            raise ValueError(
                "slow_inference_grace_s must stay between detection freshness "
                "and 1.0 seconds"
            )
        if not 0.0 < self.source_maximum_age_s <= 0.350:
            raise ValueError("source_maximum_age_s must stay within 0.0..0.350 seconds")
        if not 0.0 < self.near_bottom_ratio <= 1.0:
            raise ValueError("near_bottom_ratio must be within 0.0..1.0")
        if not 0.0 < self.disappearance_bottom_ratio < self.near_bottom_ratio:
            raise ValueError(
                "disappearance_bottom_ratio must be positive and below "
                "near_bottom_ratio"
            )
        if not 0.0 < self.near_center_ratio <= 1.0:
            raise ValueError("near_center_ratio must be within 0.0..1.0")
        if self.near_confirmations < 1:
            raise ValueError("near_confirmations must be positive")
        if self.near_loss_confirmations < 2:
            raise ValueError("near_loss_confirmations must be at least two")
        if self.near_loss_grace_s <= 0.0:
            raise ValueError("near_loss_grace_s must be positive")
        if not 0.50 <= self.final_push_mps <= 1.0:
            raise ValueError("final_push_mps must stay within 0.50..1.0 m/s")
        if not 0.0 <= self.final_push_duration_s <= 1.50:
            raise ValueError("final_push_duration_s must stay within 0.0..1.50 seconds")

    @classmethod
    def from_env(cls) -> GuidanceConfig:
        prefix = "BORDER_COLLIE_GUIDANCE_"
        config = cls(
            search_yaw_rps=float(os.environ.get(prefix + "SEARCH_YAW_RPS", "0.50")),
            search_sweep_rad=float(
                os.environ.get(prefix + "SEARCH_SWEEP_RAD", str(2.0 * math.pi))
            ),
            focus_yaw_rps=float(
                os.environ.get(prefix + "FOCUS_YAW_RPS", "0.50")
            ),
            progressive_focus_yaw_enabled=env_bool(
                prefix + "PROGRESSIVE_FOCUS_YAW_ENABLED", True
            ),
            focus_yaw_step_rps=float(
                os.environ.get(prefix + "FOCUS_YAW_STEP_RPS", "0.10")
            ),
            focus_minimum_yaw_rps=float(
                os.environ.get(prefix + "FOCUS_MINIMUM_YAW_RPS", "0.50")
            ),
            focus_near_center_ratio=float(
                os.environ.get(prefix + "FOCUS_NEAR_CENTER_RATIO", "0.20")
            ),
            focus_missing_grace_s=float(
                os.environ.get(prefix + "FOCUS_MISSING_GRACE_S", "0.50")
            ),
            center_tolerance_ratio=float(
                os.environ.get(prefix + "CENTER_TOLERANCE_RATIO", "0.05")
            ),
            center_confirmations=int(
                os.environ.get(prefix + "CENTER_CONFIRMATIONS", "3")
            ),
            approach_forward_mps=float(
                os.environ.get(prefix + "APPROACH_FORWARD_MPS", "1.0")
            ),
            approach_yaw_rps=float(os.environ.get(prefix + "APPROACH_YAW_RPS", "0.30")),
            outer_corridor_ratio=float(
                os.environ.get(prefix + "OUTER_CORRIDOR_RATIO", "0.20")
            ),
            recenter_yaw_rps=float(os.environ.get(prefix + "RECENTER_YAW_RPS", "0.50")),
            duplicate_hold_s=float(
                os.environ.get(prefix + "DUPLICATE_HOLD_S", "0.250")
            ),
            source_maximum_age_s=float(
                os.environ.get(prefix + "SOURCE_MAXIMUM_AGE_S", "0.350")
            ),
            detection_maximum_age_s=float(
                os.environ.get(prefix + "DETECTION_MAXIMUM_AGE_S", "0.250")
            ),
            slow_inference_grace_s=float(
                os.environ.get(prefix + "SLOW_INFERENCE_GRACE_S", "0.50")
            ),
            near_bottom_ratio=float(
                os.environ.get(prefix + "NEAR_BOTTOM_RATIO", "0.90")
            ),
            disappearance_bottom_ratio=float(
                os.environ.get(prefix + "DISAPPEARANCE_BOTTOM_RATIO", "0.80")
            ),
            near_center_ratio=float(
                os.environ.get(prefix + "NEAR_CENTER_RATIO", "0.72")
            ),
            near_confirmations=int(os.environ.get(prefix + "NEAR_CONFIRMATIONS", "3")),
            near_loss_confirmations=int(
                os.environ.get(prefix + "NEAR_LOSS_CONFIRMATIONS", "2")
            ),
            near_loss_grace_s=float(
                os.environ.get(prefix + "NEAR_LOSS_GRACE_S", "0.75")
            ),
            final_push_mps=float(os.environ.get(prefix + "FINAL_PUSH_MPS", "0.60")),
            final_push_duration_s=float(
                os.environ.get(prefix + "FINAL_PUSH_DURATION_S", "1.0")
            ),
        )
        # Global runtime defaults have the same practical bounds as one-run UI
        # tuning. Direct construction remains available to deterministic tests
        # that use a shorter synthetic clock.
        if 0.0 < config.final_push_duration_s < 0.10:
            raise ValueError(
                "final push duration must be 0 (disabled) or at least 0.10 seconds"
            )
        return config


@dataclass(frozen=True)
class GuidanceDecision:
    phase: GuidancePhase
    action: GuidanceAction
    command: VelocityCommand
    reason: str
    centered_fresh_samples: int
    near_fresh_samples: int
    near_loss_samples: int
    arrival_eligible: bool
    frame_advanced: bool
    terminal: bool = False
    arrival_confirmed: bool = False
    camera_failure: bool = False
    focus_active: bool = False
    focus_direction: int = 0
    focus_grace_remaining_s: float | None = None


@dataclass(frozen=True)
class _Observation:
    generation: str
    source_pts: int
    source_time_base: str
    label: str | None
    confidence: float | None
    center_x: float | None
    center_y: float | None
    bottom: float | None


class FruitGuidance:
    """Keep one Target Fruit identity from search through bounded Arrival."""

    def __init__(
        self,
        target_fruit: str,
        *,
        config: GuidanceConfig | None = None,
        policy: FruitPolicy | None = None,
    ) -> None:
        self.target_fruit = target_fruit.casefold().strip()
        self.policy: FruitPolicy = policy or fruit_policy(self.target_fruit)
        self.config = config or GuidanceConfig.from_env()
        self.phase = GuidancePhase.SEARCHING
        self.acquisition_epoch = 0
        self.final_push_count = 0
        self._generation: str | None = None
        self._last_source_pts: int | None = None
        self._last_fresh_at_s: float | None = None
        self._last_decision: GuidanceDecision | None = None
        self._centered_fresh_samples = 0
        self._near_fresh_samples = 0
        self._near_loss_samples = 0
        self._near_latched_at_s: float | None = None
        self._lower_edge_seen_at_s: float | None = None
        self._disappearance_arrival_armed = False
        self._arrival_eligible = False
        self._final_push_started_s: float | None = None
        self._candidate_focus_active = False
        self._focus_active = False
        self._focus_direction = 0
        self._focus_alignment_samples = 0
        self._last_focus_yaw_rps = 0.0
        self._focus_last_qualified_s: float | None = None
        self._focus_near_center_settle_pending = False
        self._last_observed_s: float | None = None
        self._last_trusted_geometry: tuple[float, float, float] | None = None
        self._approach_forward_authorized = False
        self._missing_during_approach = False
        self._stationary_reacquisition_required = False
        self._stationary_reacquisition_candidate: (
            tuple[float, float, float] | None
        ) = None

    def observe(
        self,
        status: dict[str, object],
        *,
        now_s: float,
        allow_forward: bool = False,
    ) -> GuidanceDecision:
        """Return the sole command authorized by the latest camera evidence."""
        if not math.isfinite(now_s):
            return self._fail("invalid_observation_time", camera=True)
        self._last_observed_s = now_s
        if self.phase is GuidancePhase.FAILED:
            return self._stop("guidance_already_failed", terminal=True)
        if self.phase is GuidancePhase.ARRIVED:
            return self._decision(
                GuidanceAction.ARRIVED,
                VelocityCommand(reason="arrival_confirmed"),
                "arrival_confirmed",
                terminal=True,
                arrival_confirmed=True,
            )
        incoming_generation = status.get("generation")
        if (
            self._generation is not None
            and isinstance(incoming_generation, str)
            and incoming_generation != self._generation
        ):
            return self._fail("camera_generation_changed", camera=True)
        parsed, failure = self._parse(status)
        if failure is not None:
            if failure == "detection_stale" and self._slow_inference_wait(status):
                return self._stop("slow_inference_grace")
            return self._fail(failure, camera=failure != "target_identity_changed")
        assert parsed is not None

        if self._generation is None:
            self._generation = parsed.generation
        elif parsed.generation != self._generation:
            return self._fail("camera_generation_changed", camera=True)

        if self._last_source_pts is not None:
            if parsed.source_pts < self._last_source_pts:
                return self._fail("source_frame_regressed", camera=True)
            if parsed.source_pts == self._last_source_pts:
                assert self._last_fresh_at_s is not None
                elapsed = now_s - self._last_fresh_at_s
                if elapsed <= self.config.duplicate_hold_s and self._last_decision:
                    held = replace(
                        self._last_decision,
                        action=GuidanceAction.HOLD,
                        reason="duplicate_frame_bounded_hold",
                        frame_advanced=False,
                    )
                    self._last_decision = held
                    return held
                return self._fail("duplicate_frame_expired", camera=True)

        self._last_source_pts = parsed.source_pts
        self._last_fresh_at_s = now_s

        if self.phase is GuidancePhase.FINAL_PUSH:
            self._arrival_eligible = True
            if parsed.label not in (None, self.target_fruit):
                return self._fail("target_identity_changed", camera=False)
            if parsed.label == self.target_fruit and not self._valid_geometry(parsed):
                return self._fail("invalid_target_geometry", camera=True)
            assert self._final_push_started_s is not None
            if now_s - self._final_push_started_s >= self.config.final_push_duration_s:
                self.phase = GuidancePhase.ARRIVED
                decision = self._decision(
                    GuidanceAction.ARRIVED,
                    VelocityCommand(reason="bounded_final_push_complete"),
                    "bounded_final_push_complete",
                    terminal=True,
                    arrival_confirmed=True,
                )
            else:
                decision = self._decision(
                    GuidanceAction.FINAL_PUSH,
                    VelocityCommand(
                        self.config.final_push_mps,
                        0.0,
                        "fruit_lower_edge_final_push",
                    ),
                    "bounded_final_push_active",
                )
            self._last_decision = decision
            return decision

        if parsed.label is None:
            decision = self._missing_target(now_s, allow_forward=allow_forward)
            self._last_decision = decision
            return decision

        if parsed.label != self.target_fruit:
            if self.acquisition_epoch:
                if self._recent_lower_edge(now_s):
                    decision = self._lower_edge_arrival()
                    self._last_decision = decision
                    return decision
                return self._fail("target_identity_changed", camera=False)
            self._centered_fresh_samples = 0
            focus_was_active = self._focus_active
            self._clear_focus()
            decision = self._search(
                "focus_wrong_label_cancelled"
                if focus_was_active
                else "searching_for_target"
            )
            self._last_decision = decision
            return decision

        if not self._valid_geometry(parsed):
            return self._fail("invalid_target_geometry", camera=True)

        assert parsed.confidence is not None
        assert parsed.center_x is not None
        assert parsed.center_y is not None
        assert parsed.bottom is not None
        horizontal_error = parsed.center_x - 0.5

        if not self.acquisition_epoch:
            decision = self._acquire(parsed.confidence, horizontal_error, now_s)
            self._last_trusted_geometry = (
                parsed.center_x,
                parsed.center_y,
                parsed.bottom,
            )
            self._last_decision = decision
            return decision

        stationary_reacquisition = self._stationary_reacquisition(
            parsed,
            horizontal_error=horizontal_error,
            allow_forward=allow_forward,
        )
        if stationary_reacquisition is not None:
            self._last_decision = stationary_reacquisition
            return stationary_reacquisition
        self._missing_during_approach = False
        self._last_trusted_geometry = (
            parsed.center_x,
            parsed.center_y,
            parsed.bottom,
        )

        # A fresh same-fruit observation in the lower approach corridor arms
        # exactly the immediately following fresh missing frame as Arrival.
        # Any intervening same-fruit frame replaces this decision, so an older
        # close observation cannot authorize a later disappearance.
        self._disappearance_arrival_armed = bool(
            allow_forward
            and parsed.bottom >= self.config.disappearance_bottom_ratio
            and parsed.center_y >= self.config.near_center_ratio
            and abs(horizontal_error) <= self.config.outer_corridor_ratio
        )

        if (
            parsed.bottom >= self.config.near_bottom_ratio
            and parsed.center_y >= self.config.near_center_ratio
        ):
            # Stage reliability deliberately treats the first fresh,
            # geometrically valid lower-edge observation after lock as
            # Arrival. Confidence commonly collapses as a floor-level fruit
            # fills or is clipped by the bottom of the camera frame, so the
            # per-fruit tracking floor does not apply at this boundary.
            self._lower_edge_seen_at_s = now_s
            decision = self._lower_edge_arrival(
                reason="first_qualified_lower_edge_arrival"
            )
            self._last_decision = decision
            return decision

        if parsed.confidence < self.policy.close_range_tracking_confidence:
            self._near_fresh_samples = 0
            decision = self._closeout_loss(
                now_s,
                pending_reason="tracking_confidence_below_floor",
            )
            self._last_decision = decision
            return decision

        centered = (
            abs(horizontal_error) <= self.config.center_tolerance_ratio + 1e-9
        )
        near = (
            centered
            and parsed.bottom >= self.config.near_bottom_ratio
            and parsed.center_y >= self.config.near_center_ratio
        )
        self._near_fresh_samples = self._near_fresh_samples + 1 if near else 0
        if self._near_fresh_samples >= self.config.near_confirmations:
            self._near_latched_at_s = now_s
        self._arrival_eligible = bool(
            self._near_latched_at_s is not None
            and now_s - self._near_latched_at_s <= self.config.near_loss_grace_s
        )
        self._near_loss_samples = 0

        if not allow_forward:
            self.phase = GuidancePhase.LOCKED
            decision = self._decision(
                GuidanceAction.HOLD,
                VelocityCommand(reason="target_locked_waiting_for_approach"),
                "target_locked_waiting_for_approach",
            )
        elif abs(horizontal_error) > self.config.outer_corridor_ratio:
            self.phase = GuidancePhase.APPROACHING
            decision = self._decision(
                GuidanceAction.ALIGN,
                VelocityCommand(
                    0.0,
                    -math.copysign(self.config.recenter_yaw_rps, horizontal_error),
                    "target_outside_outer_corridor",
                ),
                "target_outside_outer_corridor",
            )
        else:
            self.phase = GuidancePhase.APPROACHING
            yaw_rps = (
                0.0
                if centered
                else -math.copysign(self.config.approach_yaw_rps, horizontal_error)
            )
            decision = self._decision(
                GuidanceAction.DRIVE,
                VelocityCommand(
                    self.config.approach_forward_mps,
                    yaw_rps,
                    "approach_target_continuous",
                ),
                "approach_target_continuous",
            )
            self._approach_forward_authorized = True
        self._last_decision = decision
        return decision

    def _parse(
        self,
        status: dict[str, object],
    ) -> tuple[_Observation | None, str | None]:
        if status.get("camera_healthy") is not True:
            return None, "camera_unhealthy"
        generation = status.get("generation")
        source = status.get("source")
        if not isinstance(generation, str) or not generation.strip():
            return None, "camera_generation_missing"
        if not isinstance(source, dict):
            return None, "camera_source_missing"
        source_pts = source.get("pts")
        source_time_base = source.get("time_base")
        source_age = _finite_float(source.get("age_s"))
        if (
            isinstance(source_pts, bool)
            or not isinstance(source_pts, int)
            or source_pts < 0
        ):
            return None, "source_frame_invalid"
        if not isinstance(source_time_base, str) or not source_time_base:
            return None, "source_time_base_invalid"
        if source_age is None or source_age > self.config.source_maximum_age_s:
            return None, "source_frame_stale"

        detection = status.get("detection")
        if detection is None:
            return (
                _Observation(
                    generation,
                    source_pts,
                    source_time_base,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                None,
            )
        if not isinstance(detection, dict):
            return None, "detection_invalid"
        label = detection.get("label")
        if label is None:
            return (
                _Observation(
                    generation,
                    source_pts,
                    source_time_base,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                None,
            )
        if not isinstance(label, str) or not label.strip():
            return None, "detection_label_invalid"
        detection_generation = detection.get("generation")
        detection_pts = detection.get("source_pts")
        detection_time_base = detection.get("source_time_base")
        detection_age = _finite_float(detection.get("age_s"))
        confidence = _finite_float(detection.get("confidence"))
        if detection_generation != generation:
            return None, "detection_generation_mismatch"
        if detection_time_base != source_time_base:
            return None, "detection_time_base_mismatch"
        if (
            isinstance(detection_pts, bool)
            or not isinstance(detection_pts, int)
            or detection_pts < 0
            or detection_pts > source_pts
        ):
            return None, "detection_source_frame_invalid"
        if detection_age is None or detection_age > self.config.detection_maximum_age_s:
            return None, "detection_stale"
        if confidence is None or not 0.0 <= confidence <= 1.0:
            return None, "detection_confidence_invalid"
        return (
            _Observation(
                generation,
                source_pts,
                source_time_base,
                label.casefold().strip(),
                confidence,
                _finite_float(detection.get("center_x_ratio")),
                _finite_float(detection.get("center_y_ratio")),
                _finite_float(detection.get("bottom_ratio")),
            ),
            None,
        )

    def _valid_geometry(self, parsed: _Observation) -> bool:
        return all(
            value is not None and 0.0 <= value <= 1.0
            for value in (parsed.center_x, parsed.center_y, parsed.bottom)
        )

    def _slow_inference_wait(self, status: dict[str, object]) -> bool:
        """Stop without failing for bounded, otherwise-valid inference delay."""
        detection = status.get("detection")
        if not isinstance(detection, dict):
            return False
        age_s = _finite_float(detection.get("age_s"))
        confidence = _finite_float(detection.get("confidence"))
        label = detection.get("label")
        geometry = tuple(
            _finite_float(detection.get(name))
            for name in ("center_x_ratio", "center_y_ratio", "bottom_ratio")
        )
        return bool(
            age_s is not None
            and self.config.detection_maximum_age_s < age_s
            < self.config.slow_inference_grace_s
            and isinstance(label, str)
            and label.casefold().strip() == self.target_fruit
            and confidence is not None
            and 0.0 <= confidence <= 1.0
            and all(value is not None and 0.0 <= value <= 1.0 for value in geometry)
        )

    def _acquire(
        self,
        confidence: float,
        horizontal_error: float,
        now_s: float,
    ) -> GuidanceDecision:
        focus_started = False
        if (
            self.policy.focus_confidence is not None
            and not self._candidate_focus_active
        ):
            if confidence < self.policy.focus_confidence:
                self._centered_fresh_samples = 0
                return self._search("target_below_focus_confidence")
            self._candidate_focus_active = True
            focus_started = True
        if confidence < self.policy.acquisition_confidence:
            self._centered_fresh_samples = 0
            if self._focus_active:
                return self._focus_grace_or_search(
                    now_s,
                    grace_reason="focus_weak_grace",
                    expired_reason="focus_weak_grace_expired",
                )
            if self._candidate_focus_active:
                return self._decision(
                    GuidanceAction.HOLD,
                    VelocityCommand(reason="apple_candidate_focus_below_acquisition"),
                    "apple_candidate_focus_below_acquisition",
                )
            return self._search("target_below_acquisition_confidence")
        centered = (
            abs(horizontal_error) <= self.config.center_tolerance_ratio + 1e-9
        )
        if centered:
            self._focus_near_center_settle_pending = False
        if not centered:
            self._focus_active = True
            if self._focus_near_center_settle_pending:
                self._focus_near_center_settle_pending = False
                self._focus_last_qualified_s = now_s
                return self._decision(
                    GuidanceAction.HOLD,
                    VelocityCommand(reason="focus_near_center_settle"),
                    "focus_near_center_settle",
                )
            direction = -1 if horizontal_error > 0.0 else 1
            near_center = (
                self.config.progressive_focus_yaw_enabled
                and abs(horizontal_error)
                <= self.config.focus_near_center_ratio + 1e-9
            )
            focus_yaw_rps = self._next_focus_yaw_rps(
                direction,
                force_minimum=near_center,
            )
            self._focus_near_center_settle_pending = near_center
            self._focus_last_qualified_s = now_s
            self._centered_fresh_samples = 0
            return self._decision(
                GuidanceAction.ALIGN,
                VelocityCommand(
                    0.0,
                    focus_yaw_rps,
                    (
                        "focus_align_target_near_center"
                        if near_center
                        else "focus_align_target"
                    ),
                ),
                (
                    "focus_align_target_near_center"
                    if near_center
                    else "focus_align_target"
                ),
            )
        self._centered_fresh_samples = (
            self._centered_fresh_samples + 1 if centered else 0
        )
        if focus_started:
            return self._decision(
                GuidanceAction.HOLD,
                VelocityCommand(reason="apple_candidate_focus_started"),
                "apple_candidate_focus_started",
            )
        if self._centered_fresh_samples >= self.config.center_confirmations:
            self.acquisition_epoch = 1
            self.phase = GuidancePhase.LOCKED
            self._clear_focus()
            return self._decision(
                GuidanceAction.HOLD,
                VelocityCommand(reason="target_identity_locked"),
                "target_identity_locked",
            )
        if centered:
            return self._decision(
                GuidanceAction.HOLD,
                VelocityCommand(reason="centered_acquisition_confirmation"),
                "centered_acquisition_confirmation",
            )
        return self._decision(
            GuidanceAction.ALIGN,
            VelocityCommand(
                0.0,
                -math.copysign(self.config.search_yaw_rps, horizontal_error),
                "center_target_during_search",
            ),
            "center_target_during_search",
        )

    def _missing_target(
        self,
        now_s: float,
        *,
        allow_forward: bool,
    ) -> GuidanceDecision:
        if not self.acquisition_epoch:
            self._centered_fresh_samples = 0
            if self._focus_near_center_settle_pending:
                self._focus_near_center_settle_pending = False
                return self._decision(
                    GuidanceAction.HOLD,
                    VelocityCommand(reason="focus_near_center_settle"),
                    "focus_near_center_settle",
                )
            if self._focus_active:
                return self._focus_grace_or_search(
                    now_s,
                    grace_reason="focus_missing_grace",
                    expired_reason="focus_missing_grace_expired",
                )
            if self._candidate_focus_active:
                return self._decision(
                    GuidanceAction.HOLD,
                    VelocityCommand(reason="apple_candidate_focus_missing"),
                    "apple_candidate_focus_missing",
                )
            return self._search("searching_for_target")
        if self._recent_lower_edge(now_s):
            return self._lower_edge_arrival()
        if self._disappearance_arrival_armed:
            self._disappearance_arrival_armed = False
            return self._lower_edge_arrival(
                reason="qualified_lower_edge_disappearance_arrival"
            )
        if allow_forward:
            self._missing_during_approach = True
        return self._closeout_loss(now_s, pending_reason="target_missing_after_lock")

    def _focus_grace_or_search(
        self,
        now_s: float,
        *,
        grace_reason: str,
        expired_reason: str,
    ) -> GuidanceDecision:
        if (
            self._focus_last_qualified_s is not None
            and now_s - self._focus_last_qualified_s
            <= self.config.focus_missing_grace_s
        ):
            return self._decision(
                GuidanceAction.ALIGN,
                VelocityCommand(
                    0.0,
                    self._last_focus_yaw_rps,
                    grace_reason,
                ),
                grace_reason,
            )
        self._clear_focus()
        return self._search(expired_reason)

    def _clear_focus(self) -> None:
        self._focus_active = False
        self._focus_direction = 0
        self._focus_alignment_samples = 0
        self._last_focus_yaw_rps = 0.0
        self._focus_last_qualified_s = None
        self._focus_near_center_settle_pending = False

    def _next_focus_yaw_rps(
        self,
        direction: int,
        *,
        force_minimum: bool = False,
    ) -> float:
        """Return a bounded yaw that slows across agreeing fresh focus passes."""
        if direction != self._focus_direction:
            self._focus_alignment_samples = 0
        self._focus_direction = direction
        self._focus_alignment_samples += 1
        magnitude = self.config.focus_yaw_rps
        if force_minimum:
            magnitude = self.config.focus_minimum_yaw_rps
        elif self.config.progressive_focus_yaw_enabled:
            magnitude = max(
                self.config.focus_minimum_yaw_rps,
                self.config.focus_yaw_rps
                - self.config.focus_yaw_step_rps
                * (self._focus_alignment_samples - 1),
            )
        self._last_focus_yaw_rps = direction * round(magnitude, 10)
        return self._last_focus_yaw_rps

    def _stationary_reacquisition(
        self,
        observation: _Observation,
        *,
        horizontal_error: float,
        allow_forward: bool,
    ) -> GuidanceDecision | None:
        """Reject impossible lower-edge jumps until ordinary geometry agrees.

        Search may hand Approach a locked fruit before Approach sends a forward
        command. If that fruit disappears and then teleports to the lower edge,
        the fresh frame is not stale but its geometry cannot prove Arrival.
        Preserve identity, remain stopped, and require two agreeing ordinary
        observations before restoring forward authority.
        """
        if (
            not allow_forward
            or not self._missing_during_approach
            or self._approach_forward_authorized
        ):
            return None
        assert observation.confidence is not None
        assert observation.center_x is not None
        assert observation.center_y is not None
        assert observation.bottom is not None
        trusted_bottom = (
            self._last_trusted_geometry[2]
            if self._last_trusted_geometry is not None
            else None
        )
        implausible_lower_edge_jump = bool(
            observation.bottom >= self.config.near_bottom_ratio
            and (
                trusted_bottom is None
                or trusted_bottom < self.config.disappearance_bottom_ratio
            )
        )
        if implausible_lower_edge_jump:
            self._stationary_reacquisition_required = True
            self._stationary_reacquisition_candidate = None
            return self._stop("stationary_lower_edge_jump_rejected")
        if not self._stationary_reacquisition_required:
            return None
        if (
            observation.confidence < self.policy.close_range_tracking_confidence
            or abs(horizontal_error) > self.config.outer_corridor_ratio
            or observation.bottom >= self.config.near_bottom_ratio
        ):
            self._stationary_reacquisition_candidate = None
            return self._stop("stationary_reacquisition_unqualified")

        current = (
            observation.center_x,
            observation.center_y,
            observation.bottom,
        )
        previous = self._stationary_reacquisition_candidate
        agrees = bool(
            previous is not None
            and all(
                abs(current_value - previous_value)
                <= self.config.center_tolerance_ratio
                for current_value, previous_value in zip(current, previous, strict=True)
            )
        )
        if not agrees:
            self._stationary_reacquisition_candidate = current
            return self._stop("stationary_reacquisition_confirmation_pending")

        self._stationary_reacquisition_required = False
        self._stationary_reacquisition_candidate = None
        return None

    def _closeout_loss(
        self,
        now_s: float,
        *,
        pending_reason: str,
    ) -> GuidanceDecision:
        if self._recent_lower_edge(now_s):
            return self._lower_edge_arrival()
        if (
            self._near_latched_at_s is not None
            and now_s - self._near_latched_at_s <= self.config.near_loss_grace_s
        ):
            self._arrival_eligible = True
            self._near_loss_samples += 1
            if self._near_loss_samples < self.config.near_loss_confirmations:
                return self._stop("lower_edge_loss_confirmation_pending")
            if self.config.final_push_duration_s == 0.0:
                self.phase = GuidancePhase.ARRIVED
                return self._decision(
                    GuidanceAction.ARRIVED,
                    VelocityCommand(reason="bounded_final_push_disabled"),
                    "bounded_final_push_disabled",
                    terminal=True,
                    arrival_confirmed=True,
                )
            self.phase = GuidancePhase.FINAL_PUSH
            self._final_push_started_s = now_s
            self.final_push_count += 1
            return self._decision(
                GuidanceAction.FINAL_PUSH,
                VelocityCommand(
                    self.config.final_push_mps,
                    0.0,
                    "fruit_lower_edge_final_push",
                ),
                "lower_edge_disappearance_confirmed",
            )
        self._near_fresh_samples = 0
        self._near_loss_samples = 0
        self._arrival_eligible = False
        return self._stop(pending_reason)

    def _recent_lower_edge(self, now_s: float) -> bool:
        return bool(
            self._lower_edge_seen_at_s is not None
            and now_s - self._lower_edge_seen_at_s <= self.config.near_loss_grace_s
        )

    def _lower_edge_arrival(
        self,
        *,
        reason: str = "lower_edge_identity_collapse_arrival",
    ) -> GuidanceDecision:
        self.phase = GuidancePhase.ARRIVED
        self._arrival_eligible = True
        return self._decision(
            GuidanceAction.ARRIVED,
            VelocityCommand(reason=reason),
            reason,
            terminal=True,
            arrival_confirmed=True,
        )

    def _search(self, reason: str) -> GuidanceDecision:
        self.phase = GuidancePhase.SEARCHING
        return self._decision(
            GuidanceAction.SEARCH,
            VelocityCommand(0.0, self.config.search_yaw_rps, reason),
            reason,
        )

    def _fail(self, reason: str, *, camera: bool) -> GuidanceDecision:
        self.phase = GuidancePhase.FAILED
        self._arrival_eligible = False
        self._clear_focus()
        return self._stop(
            reason,
            terminal=True,
            camera_failure=camera,
        )

    def _stop(
        self,
        reason: str,
        *,
        terminal: bool = False,
        camera_failure: bool = False,
    ) -> GuidanceDecision:
        decision = self._decision(
            GuidanceAction.STOP,
            VelocityCommand(reason=reason),
            reason,
            terminal=terminal,
            camera_failure=camera_failure,
        )
        self._last_decision = decision
        return decision

    def _decision(
        self,
        action: GuidanceAction,
        command: VelocityCommand,
        reason: str,
        *,
        terminal: bool = False,
        arrival_confirmed: bool = False,
        camera_failure: bool = False,
    ) -> GuidanceDecision:
        return GuidanceDecision(
            phase=self.phase,
            action=action,
            command=command,
            reason=reason,
            centered_fresh_samples=self._centered_fresh_samples,
            near_fresh_samples=self._near_fresh_samples,
            near_loss_samples=self._near_loss_samples,
            arrival_eligible=self._arrival_eligible,
            frame_advanced=True,
            terminal=terminal,
            arrival_confirmed=arrival_confirmed,
            camera_failure=camera_failure,
            focus_active=self._focus_active,
            focus_direction=self._focus_direction,
            focus_grace_remaining_s=(
                max(
                    0.0,
                    self.config.focus_missing_grace_s
                    - (self._last_observed_s - self._focus_last_qualified_s),
                )
                if self._focus_active
                and self._last_observed_s is not None
                and self._focus_last_qualified_s is not None
                else None
            ),
        )


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None
