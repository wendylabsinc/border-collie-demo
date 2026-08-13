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

    search_yaw_rps: float = 0.40
    search_sweep_rad: float = 2.0 * math.pi
    center_tolerance_ratio: float = 0.08
    center_confirmations: int = 3
    approach_forward_mps: float = 1.0
    approach_yaw_rps: float = 0.30
    outer_corridor_ratio: float = 0.20
    recenter_yaw_rps: float = 0.50
    duplicate_hold_s: float = 0.250
    source_maximum_age_s: float = 0.350
    detection_maximum_age_s: float = 0.250
    near_bottom_ratio: float = 0.90
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
            self.center_tolerance_ratio,
            self.approach_forward_mps,
            self.approach_yaw_rps,
            self.outer_corridor_ratio,
            self.recenter_yaw_rps,
            self.duplicate_hold_s,
            self.source_maximum_age_s,
            self.detection_maximum_age_s,
            self.near_bottom_ratio,
            self.near_center_ratio,
            self.near_loss_grace_s,
            self.final_push_mps,
            self.final_push_duration_s,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("guidance values must be finite")
        if not 0.40 <= self.search_yaw_rps <= 0.80:
            raise ValueError("search_yaw_rps must stay within 0.40..0.80 rad/s")
        if not 0.0 < self.search_sweep_rad <= 2.0 * math.pi:
            raise ValueError("search_sweep_rad must stay within one revolution")
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
        if not 0.0 < self.source_maximum_age_s <= 0.350:
            raise ValueError("source_maximum_age_s must stay within 0.0..0.350 seconds")
        if not 0.0 < self.near_bottom_ratio <= 1.0:
            raise ValueError("near_bottom_ratio must be within 0.0..1.0")
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
            search_yaw_rps=float(os.environ.get(prefix + "SEARCH_YAW_RPS", "0.40")),
            search_sweep_rad=float(
                os.environ.get(prefix + "SEARCH_SWEEP_RAD", str(2.0 * math.pi))
            ),
            center_tolerance_ratio=float(
                os.environ.get(prefix + "CENTER_TOLERANCE_RATIO", "0.08")
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
            near_bottom_ratio=float(
                os.environ.get(prefix + "NEAR_BOTTOM_RATIO", "0.90")
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
        self._arrival_eligible = False
        self._final_push_started_s: float | None = None
        self._candidate_focus_active = False
        self._forward_authorized_once = False

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
            decision = self._missing_target(now_s)
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
            decision = self._search("searching_for_target")
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
            decision = self._acquire(parsed.confidence, horizontal_error)
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

        if (
            parsed.bottom >= self.config.near_bottom_ratio
            and parsed.center_y >= self.config.near_center_ratio
        ):
            # Stage reliability deliberately treats the first fresh,
            # confidence-qualified lower-edge observation after lock as
            # Arrival. Do not wait for additional close frames that commonly
            # disappear as a floor-level fruit is clipped by the camera.
            self._lower_edge_seen_at_s = now_s
            decision = self._lower_edge_arrival(
                reason="first_qualified_lower_edge_arrival"
            )
            self._last_decision = decision
            return decision

        centered = abs(horizontal_error) <= self.config.center_tolerance_ratio
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
            self._forward_authorized_once = True
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

    def _acquire(
        self,
        confidence: float,
        horizontal_error: float,
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
            if self._candidate_focus_active:
                return self._decision(
                    GuidanceAction.HOLD,
                    VelocityCommand(reason="apple_candidate_focus_below_acquisition"),
                    "apple_candidate_focus_below_acquisition",
                )
            return self._search("target_below_acquisition_confidence")
        centered = abs(horizontal_error) <= self.config.center_tolerance_ratio
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

    def _missing_target(self, now_s: float) -> GuidanceDecision:
        if not self.acquisition_epoch:
            self._centered_fresh_samples = 0
            if self._candidate_focus_active:
                return self._decision(
                    GuidanceAction.HOLD,
                    VelocityCommand(reason="apple_candidate_focus_missing"),
                    "apple_candidate_focus_missing",
                )
            return self._search("searching_for_target")
        if self._forward_authorized_once:
            return self._lower_edge_arrival(reason="target_missing_after_approach_arrival")
        if self._recent_lower_edge(now_s):
            return self._lower_edge_arrival()
        return self._closeout_loss(now_s, pending_reason="target_missing_after_lock")

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
        )


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None
