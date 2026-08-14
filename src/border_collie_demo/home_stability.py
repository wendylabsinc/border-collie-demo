"""Fresh-sample contract for declaring measured Home position stable."""

from __future__ import annotations

import math
from dataclasses import dataclass


class HomeVerificationError(RuntimeError):
    """Evidence cannot safely participate in Home verification."""


@dataclass(frozen=True)
class HomeStabilityConfig:
    arrival_tolerance_m: float = 0.10
    required_samples: int = 4
    maximum_spread_m: float = 0.03

    def __post_init__(self) -> None:
        if not math.isfinite(self.arrival_tolerance_m) or not (
            0.05 <= self.arrival_tolerance_m <= 0.50
        ):
            raise ValueError("arrival_tolerance_m must stay within 0.05..0.50")
        if isinstance(self.required_samples, bool) or not (
            3 <= self.required_samples <= 5
        ):
            raise ValueError("required_samples must stay within 3..5")
        if not math.isfinite(self.maximum_spread_m) or not (
            0.005 <= self.maximum_spread_m <= 0.05
        ):
            raise ValueError("maximum_spread_m must stay within 0.005..0.05")


@dataclass(frozen=True)
class HomeSample:
    sequence: int
    x_m: float
    y_m: float
    yaw_rad: float
    captured_monotonic_s: float
    age_s: float
    odometry_epoch: str

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or self.sequence < 1:
            raise ValueError("Home sample sequence must be positive")
        values = (
            self.x_m,
            self.y_m,
            self.yaw_rad,
            self.captured_monotonic_s,
            self.age_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Home sample values must be finite")
        if not self.odometry_epoch:
            raise ValueError("Home sample odometry epoch is required")


@dataclass(frozen=True)
class HomeStabilityDecision:
    stable: bool
    reason: str
    sample_count: int
    distance_min_m: float | None
    distance_max_m: float | None
    position_spread_m: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "stable": self.stable,
            "reason": self.reason,
            "sample_count": self.sample_count,
            "distance_min_m": self.distance_min_m,
            "distance_max_m": self.distance_max_m,
            "position_spread_m": self.position_spread_m,
        }


class HomeStabilityWindow:
    """Evaluate one advancing, same-epoch post-stop sample window."""

    def __init__(
        self,
        config: HomeStabilityConfig,
        *,
        expected_odometry_epoch: str,
    ) -> None:
        if not expected_odometry_epoch:
            raise ValueError("expected odometry epoch is required")
        self.config = config
        self.expected_odometry_epoch = expected_odometry_epoch
        self._samples: list[HomeSample] = []

    @property
    def samples(self) -> tuple[HomeSample, ...]:
        return tuple(self._samples)

    def observe(self, sample: HomeSample) -> HomeStabilityDecision:
        if sample.odometry_epoch != self.expected_odometry_epoch:
            raise HomeVerificationError("Home odometry epoch changed")
        if sample.age_s < 0.0:
            raise HomeVerificationError("Home sample age is invalid")
        if self._samples and (
            sample.captured_monotonic_s
            <= self._samples[-1].captured_monotonic_s
        ):
            raise HomeVerificationError("Home pose timestamp regressed or repeated")
        self._samples.append(sample)
        if len(self._samples) > self.config.required_samples:
            self._samples.pop(0)

        distances = [math.hypot(item.x_m, item.y_m) for item in self._samples]
        spread = _maximum_position_spread(self._samples)
        outside = any(
            distance > self.config.arrival_tolerance_m for distance in distances
        )
        if outside:
            reason = "outside_home_gate"
        elif spread > self.config.maximum_spread_m:
            reason = "position_spread_exceeded"
        elif len(self._samples) < self.config.required_samples:
            reason = "collecting_stable_samples"
        else:
            reason = "stable_inside_home_gate"
        return HomeStabilityDecision(
            stable=reason == "stable_inside_home_gate",
            reason=reason,
            sample_count=len(self._samples),
            distance_min_m=min(distances),
            distance_max_m=max(distances),
            position_spread_m=spread,
        )


def _maximum_position_spread(samples: list[HomeSample]) -> float:
    spread = 0.0
    for index, left in enumerate(samples):
        for right in samples[index + 1 :]:
            spread = max(spread, math.hypot(left.x_m - right.x_m, left.y_m - right.y_m))
    return spread
