"""Named, attributable Target Fruit search experiments."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

from .fruits import fruit_policy

SEARCH_POLICY_ENV = "BORDER_COLLIE_SEARCH_POLICY"
SEARCH_POLICY_NAMES = ("fast-lock", "slow-sweep", "double-back")
DEFAULT_SEARCH_POLICY = "slow-sweep"
SOURCE_MAXIMUM_AGE_S = 0.350
DETECTION_MAXIMUM_AGE_S = 0.250
INFERENCE_MAXIMUM_S = 0.200


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class SearchLockDecision:
    qualified: bool
    reason: str


@dataclass(frozen=True)
class SearchDirective:
    yaw_rps: float
    reason: str


class SearchGenerationChanged(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchPolicy:
    """One immutable search policy behind the shared experiment interface."""

    name: str
    broad_yaw_rps: float
    broad_sweep_rad: float = 2.0 * math.pi
    broad_timeout_s: float = 30.0
    minimum_consecutive_detections: int = 5
    candidate_mode: str = "baseline"
    candidate_hold_s: float = 0.75
    candidate_loss_grace_s: float = 0.50
    candidate_maximum_age_s: float = 0.25
    double_back_yaw_rps: float = 0.50
    double_back_maximum_angle_rad: float = 0.35
    double_back_maximum_s: float = 0.75
    double_back_maximum_episodes: int = 2
    double_back_episode_budget_s: float = 2.0

    def __post_init__(self) -> None:
        if self.name not in SEARCH_POLICY_NAMES:
            raise ValueError("search policy name must be canonical")
        if self.minimum_consecutive_detections < 1:
            raise ValueError("search confirmation count must be positive")
        if self.candidate_mode not in {"baseline", "double-back"}:
            raise ValueError("candidate mode is invalid")
        positive = (
            self.broad_yaw_rps,
            self.broad_sweep_rad,
            self.broad_timeout_s,
            self.candidate_hold_s,
            self.candidate_loss_grace_s,
            self.candidate_maximum_age_s,
            self.double_back_yaw_rps,
            self.double_back_maximum_angle_rad,
            self.double_back_maximum_s,
            self.double_back_episode_budget_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("search policy limits must be finite and positive")
        if self.broad_sweep_rad > 2.0 * math.pi:
            raise ValueError("search policy sweep cannot exceed one full rotation")
        if self.broad_timeout_s <= self.broad_sweep_rad / self.broad_yaw_rps:
            raise ValueError("search policy timeout must include full-sweep margin")
        if self.double_back_maximum_episodes < 1:
            raise ValueError("double-back episode count must be positive")

    @classmethod
    def named(cls, name: str) -> SearchPolicy:
        normalized = name.strip()
        if normalized == "fast-lock":
            return cls(
                name=normalized,
                broad_yaw_rps=1.0,
                minimum_consecutive_detections=3,
            )
        if normalized == "slow-sweep":
            return cls(name=normalized, broad_yaw_rps=0.5)
        if normalized == "double-back":
            return cls(
                name=normalized,
                broad_yaw_rps=1.0,
                candidate_mode="double-back",
            )
        choices = ", ".join(SEARCH_POLICY_NAMES)
        raise ValueError(f"search policy must be one of: {choices}")

    @classmethod
    def configured(
        cls,
        *,
        cli_name: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> SearchPolicy:
        environment = os.environ if environ is None else environ
        selected = (
            cli_name
            if cli_name is not None
            else environment.get(SEARCH_POLICY_ENV, DEFAULT_SEARCH_POLICY)
        )
        return cls.named(selected)

    def evaluate(
        self,
        observation: Mapping[str, object],
        target_fruit: str,
    ) -> SearchLockDecision:
        """Evaluate a complete observation before fast-lock may shorten count."""
        target = target_fruit.casefold().strip()
        target_config = fruit_policy(target)
        if observation.get("camera_healthy") is not True:
            return SearchLockDecision(False, "camera_unhealthy")
        detection = observation.get("detection")
        source = observation.get("source")
        if not isinstance(detection, Mapping):
            return SearchLockDecision(False, "detection_missing")
        if not isinstance(source, Mapping):
            return SearchLockDecision(False, "source_missing")
        if str(detection.get("label") or "").casefold().strip() != target:
            return SearchLockDecision(False, "wrong_target_fruit")
        confidence = _number(detection.get("confidence"))
        if confidence is None or confidence < target_config.acquisition_confidence:
            return SearchLockDecision(False, "confidence_below_acquisition_floor")
        detection_age_s = _number(detection.get("age_s"))
        if (
            detection_age_s is None
            or detection_age_s < 0.0
            or detection_age_s > DETECTION_MAXIMUM_AGE_S
        ):
            return SearchLockDecision(False, "detection_stale")
        source_age_s = _number(source.get("age_s"))
        if (
            source_age_s is None
            or source_age_s < 0.0
            or source_age_s > SOURCE_MAXIMUM_AGE_S
        ):
            return SearchLockDecision(False, "source_stale")
        generation = observation.get("generation")
        if (
            not isinstance(generation, str)
            or not generation
            or detection.get("generation") != generation
        ):
            return SearchLockDecision(False, "generation_mismatch")
        source_time_base = source.get("time_base")
        if (
            not isinstance(source_time_base, str)
            or not source_time_base
            or detection.get("source_time_base") != source_time_base
        ):
            return SearchLockDecision(False, "timebase_mismatch")
        source_pts = source.get("pts")
        detection_pts = detection.get("source_pts")
        if (
            isinstance(source_pts, bool)
            or not isinstance(source_pts, int)
            or isinstance(detection_pts, bool)
            or not isinstance(detection_pts, int)
            or detection_pts < 0
            or detection_pts > source_pts
        ):
            return SearchLockDecision(False, "source_pts_invalid")
        inference_s = _number(detection.get("inference_s"))
        if inference_s is None or inference_s < 0.0 or inference_s > INFERENCE_MAXIMUM_S:
            return SearchLockDecision(False, "inference_too_slow")
        center_x = _number(detection.get("center_x_ratio"))
        center_y = _number(detection.get("center_y_ratio"))
        bottom = _number(detection.get("bottom_ratio"))
        area = _number(detection.get("bbox_area_ratio"))
        if not (
            center_x is not None
            and 0.0 <= center_x <= 1.0
            and center_y is not None
            and 0.0 <= center_y <= 1.0
            and bottom is not None
            and center_y <= bottom <= 1.0
            and area is not None
            and 0.0 < area <= 1.0
        ):
            return SearchLockDecision(False, "geometry_invalid")
        count = detection.get("consecutive_detections")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < self.minimum_consecutive_detections
        ):
            return SearchLockDecision(False, "insufficient_consecutive_detections")
        return SearchLockDecision(True, "search_lock_qualified")


class DoubleBackSearchController:
    """Bounded candidate revisit state that can authorize yaw only."""

    def __init__(
        self,
        target_fruit: str,
        sweep_yaw_rps: float,
        policy: SearchPolicy,
        *,
        started_at: float,
    ) -> None:
        if policy.candidate_mode != "double-back":
            raise ValueError("double-back controller requires the double-back policy")
        self._target = target_fruit.casefold().strip()
        self._sweep_yaw_rps = float(sweep_yaw_rps)
        self._policy = policy
        self._search_generation: str | None = None
        self._candidate_active = False
        self._candidate_generation: str | None = None
        self._candidate_last_source_pts: int | None = None
        self._candidate_last_seen_at: float | None = None
        self._candidate_hold_until: float | None = None
        self._episode_deadline: float | None = None
        self._cooldown_until = started_at
        self._reversing = False
        self._reverse_started_at: float | None = None
        self._reverse_progress_rad = 0.0
        self._evidence: dict[str, object] = {
            "double_back_episodes": 0,
            "double_back_maximum_episodes": policy.double_back_maximum_episodes,
            "double_back_maximum_angle_rad": policy.double_back_maximum_angle_rad,
            "double_back_maximum_s": policy.double_back_maximum_s,
            "double_back_episode_budget_s": policy.double_back_episode_budget_s,
            "double_back_confidence_threshold": (
                fruit_policy(target_fruit).acquisition_confidence
            ),
        }

    def sample(self, status: Mapping[str, object], *, now_s: float) -> None:
        status_generation = status.get("generation")
        if isinstance(status_generation, str) and status_generation:
            if self._search_generation is None:
                self._search_generation = status_generation
            elif status_generation != self._search_generation:
                raise SearchGenerationChanged(
                    "camera generation changed during target search"
                )
        identity = self._candidate_identity(status)
        episodes = int(self._evidence["double_back_episodes"])
        if (
            identity is None
            or now_s < self._cooldown_until
            or (
                not self._candidate_active
                and episodes >= self._policy.double_back_maximum_episodes
            )
        ):
            return
        generation, source_pts = identity
        if not self._candidate_active:
            self._candidate_active = True
            self._candidate_generation = generation
            self._episode_deadline = now_s + self._policy.double_back_episode_budget_s
            self._candidate_hold_until = now_s + self._policy.candidate_hold_s
            self._evidence["double_back_episodes"] = episodes + 1
            self._evidence["candidate_lock_count"] = (
                int(self._evidence.get("candidate_lock_count", 0)) + 1
            )
        if self._candidate_generation != generation:
            return
        self._candidate_last_source_pts = source_pts
        self._candidate_last_seen_at = now_s
        if self._reversing:
            self._reversing = False
            self._reverse_started_at = None
            self._reverse_progress_rad = 0.0
            assert self._episode_deadline is not None
            self._candidate_hold_until = min(
                now_s + self._policy.candidate_hold_s,
                self._episode_deadline,
            )
            self._evidence["double_back_reacquisitions"] = (
                int(self._evidence.get("double_back_reacquisitions", 0)) + 1
            )

    def command(self, *, now_s: float, measured_yaw_step_rad: float) -> SearchDirective:
        if self._reversing:
            self._reverse_progress_rad += max(0.0, -measured_yaw_step_rad)
        if not self._candidate_active:
            return SearchDirective(self._sweep_yaw_rps, "find_target")
        assert self._episode_deadline is not None
        assert self._candidate_last_seen_at is not None
        assert self._candidate_hold_until is not None
        reverse_expired = bool(
            self._reversing
            and self._reverse_started_at is not None
            and (
                now_s - self._reverse_started_at >= self._policy.double_back_maximum_s
                or self._reverse_progress_rad
                >= self._policy.double_back_maximum_angle_rad
            )
        )
        if now_s >= self._episode_deadline or reverse_expired:
            self._reset_episode(now_s)
            return SearchDirective(self._sweep_yaw_rps, "find_target")
        if (
            not self._reversing
            and now_s >= self._candidate_hold_until
            and now_s - self._candidate_last_seen_at
            > self._policy.candidate_loss_grace_s
        ):
            self._reversing = True
            self._reverse_started_at = now_s
            self._reverse_progress_rad = 0.0
        if self._reversing:
            self._evidence["double_back_command_samples"] = (
                int(self._evidence.get("double_back_command_samples", 0)) + 1
            )
            return SearchDirective(
                -min(self._sweep_yaw_rps, self._policy.double_back_yaw_rps),
                "candidate_double_back",
            )
        self._evidence["candidate_confirmation_dwell_samples"] = (
            int(self._evidence.get("candidate_confirmation_dwell_samples", 0)) + 1
        )
        return SearchDirective(0.0, "candidate_confirmation_dwell")

    def evidence(self) -> dict[str, object]:
        return dict(self._evidence)

    def _candidate_identity(
        self,
        status: Mapping[str, object],
    ) -> tuple[str, int] | None:
        detection = status.get("detection")
        source = status.get("source")
        if not isinstance(detection, Mapping) or not isinstance(source, Mapping):
            return None
        generation = status.get("generation")
        source_pts = detection.get("source_pts")
        current_source_pts = source.get("pts")
        source_time_base = source.get("time_base")
        confidence = _number(detection.get("confidence"))
        age_s = _number(detection.get("age_s"))
        if not (
            isinstance(generation, str)
            and generation
            and detection.get("generation") == generation
            and str(detection.get("label") or "").casefold().strip() == self._target
            and isinstance(source_pts, int)
            and not isinstance(source_pts, bool)
            and isinstance(current_source_pts, int)
            and not isinstance(current_source_pts, bool)
            and source_pts <= current_source_pts
            and isinstance(source_time_base, str)
            and source_time_base
            and detection.get("source_time_base") == source_time_base
            and confidence is not None
            and confidence >= fruit_policy(self._target).acquisition_confidence
            and age_s is not None
            and 0.0 <= age_s <= self._policy.candidate_maximum_age_s
            and (
                self._candidate_last_source_pts is None
                or source_pts > self._candidate_last_source_pts
            )
        ):
            return None
        return generation, source_pts

    def _reset_episode(self, now_s: float) -> None:
        self._candidate_active = False
        self._candidate_generation = None
        self._candidate_last_seen_at = None
        self._candidate_hold_until = None
        self._episode_deadline = None
        self._reversing = False
        self._reverse_started_at = None
        self._reverse_progress_rad = 0.0
        self._cooldown_until = now_s + self._policy.candidate_loss_grace_s
        self._evidence["double_back_bounded_exits"] = (
            int(self._evidence.get("double_back_bounded_exits", 0)) + 1
        )
