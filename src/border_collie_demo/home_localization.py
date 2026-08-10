"""Trusted Home estimates from local odometry and optional absolute evidence.

The module is deliberately SDK-neutral.  Robot and vision integrations adapt
their measurements into the small types below; callers receive one explicit
trusted/unavailable result with enough evidence to explain the decision.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .return_home import Pose2D, normalize_angle


class HomeEstimateState(str, Enum):
    TRUSTED = "trusted"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class AbsoluteHomeObservation:
    """Robot pose in the captured Home coordinate frame.

    ``pose_from_home`` uses Home as (0, 0, 0). Positive x points along the
    captured Home heading, positive y points left, and yaw is relative to the
    captured Home heading. ``captured_monotonic_s`` must use the application
    process's monotonic clock.
    """

    pose_from_home: Pose2D
    captured_monotonic_s: float
    source: str
    reference_id: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.captured_monotonic_s):
            raise ValueError("absolute Home observation time must be finite")
        if not self.source.strip():
            raise ValueError("absolute Home observation source is required")
        if not self.reference_id.strip():
            raise ValueError("absolute Home reference ID is required")


class AbsoluteHomeObservationAdapter(Protocol):
    """Seam implemented by a future fiducial producer."""

    def observe_home(self, home: Pose2D) -> AbsoluteHomeObservation | None: ...


@dataclass(frozen=True)
class HomeLocalizationConfig:
    maximum_odometry_age_s: float = 0.50
    maximum_absolute_age_s: float = 0.50
    maximum_position_disagreement_m: float = 0.30
    maximum_heading_disagreement_rad: float = math.radians(30.0)
    absolute_weight: float = 0.75
    maximum_future_skew_s: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            self.maximum_odometry_age_s,
            self.maximum_absolute_age_s,
            self.maximum_position_disagreement_m,
            self.maximum_heading_disagreement_rad,
            self.maximum_future_skew_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Home localization limits must be finite and positive")
        if (
            not math.isfinite(self.absolute_weight)
            or not 0.0 <= self.absolute_weight <= 1.0
        ):
            raise ValueError("absolute Home weight must be between zero and one")


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


class HomeLocalizer:
    """Produce one fail-closed Home estimate from all available evidence."""

    def __init__(
        self,
        absolute_adapter: AbsoluteHomeObservationAdapter | None = None,
        *,
        config: HomeLocalizationConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._absolute_adapter = absolute_adapter
        self.config = config or HomeLocalizationConfig()
        self._clock = clock

    def describe(self) -> dict[str, object]:
        return {
            "odometry_source": "rt/sportmodestate",
            "absolute_source_configured": self._absolute_adapter is not None,
            "maximum_odometry_age_s": self.config.maximum_odometry_age_s,
            "maximum_absolute_age_s": self.config.maximum_absolute_age_s,
            "maximum_position_disagreement_m": (
                self.config.maximum_position_disagreement_m
            ),
            "maximum_heading_disagreement_rad": (
                self.config.maximum_heading_disagreement_rad
            ),
        }

    def estimate(
        self,
        home: Pose2D,
        current_odometry: Pose2D,
        *,
        odometry_age_s: float,
    ) -> HomeEstimate:
        odometry_pose = _pose_from_home(home, current_odometry)
        odometry_evidence = {
            "age_s": odometry_age_s,
            "pose_from_home": _pose_dict(odometry_pose),
        }
        if (
            not math.isfinite(odometry_age_s)
            or odometry_age_s < 0.0
            or odometry_age_s > self.config.maximum_odometry_age_s
        ):
            return self._unavailable(
                "odometry sample is stale or invalid",
                odometry=odometry_evidence,
                absolute={"state": "not_checked"},
            )

        if self._absolute_adapter is None:
            return self._trusted_odometry(
                odometry_pose,
                odometry_evidence,
                absolute_state="not_configured",
            )

        try:
            absolute = self._absolute_adapter.observe_home(home)
        except Exception as exc:  # noqa: BLE001 - adapter failure must fail closed
            return self._unavailable(
                f"absolute Home adapter failed: {exc}",
                odometry=odometry_evidence,
                absolute={
                    "state": "adapter_failure",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        if absolute is None:
            return self._trusted_odometry(
                odometry_pose,
                odometry_evidence,
                absolute_state="not_observed",
            )

        now = self._clock()
        absolute_age_s = now - absolute.captured_monotonic_s
        absolute_evidence = {
            "state": "observed",
            "source": absolute.source,
            "reference_id": absolute.reference_id,
            "captured_monotonic_s": absolute.captured_monotonic_s,
            "age_s": absolute_age_s,
            "pose_from_home": _pose_dict(absolute.pose_from_home),
        }
        if (
            not math.isfinite(now)
            or not math.isfinite(absolute_age_s)
            or absolute_age_s < -self.config.maximum_future_skew_s
            or absolute_age_s > self.config.maximum_absolute_age_s
        ):
            return self._unavailable(
                "absolute Home observation is stale or has an invalid timestamp",
                odometry=odometry_evidence,
                absolute=absolute_evidence,
            )

        position_disagreement_m = math.hypot(
            absolute.pose_from_home.x_m - odometry_pose.x_m,
            absolute.pose_from_home.y_m - odometry_pose.y_m,
        )
        heading_disagreement_rad = abs(
            normalize_angle(
                absolute.pose_from_home.yaw_rad - odometry_pose.yaw_rad
            )
        )
        disagreement = {
            "position_m": position_disagreement_m,
            "heading_rad": heading_disagreement_rad,
            "maximum_position_m": self.config.maximum_position_disagreement_m,
            "maximum_heading_rad": self.config.maximum_heading_disagreement_rad,
        }
        if (
            position_disagreement_m
            > self.config.maximum_position_disagreement_m
            or heading_disagreement_rad
            > self.config.maximum_heading_disagreement_rad
        ):
            return self._unavailable(
                "odometry and absolute Home observation disagree",
                odometry=odometry_evidence,
                absolute=absolute_evidence,
                disagreement=disagreement,
            )

        weight = self.config.absolute_weight
        fused = Pose2D(
            odometry_pose.x_m * (1.0 - weight)
            + absolute.pose_from_home.x_m * weight,
            odometry_pose.y_m * (1.0 - weight)
            + absolute.pose_from_home.y_m * weight,
            normalize_angle(
                odometry_pose.yaw_rad
                + weight
                * normalize_angle(
                    absolute.pose_from_home.yaw_rad - odometry_pose.yaw_rad
                )
            ),
        )
        return HomeEstimate(
            state=HomeEstimateState.TRUSTED,
            pose_from_home=fused,
            source="odometry+absolute_fiducial",
            unavailable_reason=None,
            evidence={
                "odometry": odometry_evidence,
                "absolute": absolute_evidence,
                "disagreement": disagreement,
                "absolute_weight": weight,
            },
        )

    def _trusted_odometry(
        self,
        pose: Pose2D,
        odometry: dict[str, object],
        *,
        absolute_state: str,
    ) -> HomeEstimate:
        return HomeEstimate(
            state=HomeEstimateState.TRUSTED,
            pose_from_home=pose,
            source="odometry_only",
            unavailable_reason=None,
            evidence={
                "odometry": odometry,
                "absolute": {"state": absolute_state},
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


def _pose_from_home(home: Pose2D, current: Pose2D) -> Pose2D:
    dx = current.x_m - home.x_m
    dy = current.y_m - home.y_m
    cosine = math.cos(home.yaw_rad)
    sine = math.sin(home.yaw_rad)
    return Pose2D(
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        normalize_angle(current.yaw_rad - home.yaw_rad),
    )


def _pose_dict(pose: Pose2D) -> dict[str, float]:
    return {
        "x_m": pose.x_m,
        "y_m": pose.y_m,
        "yaw_rad": pose.yaw_rad,
    }
