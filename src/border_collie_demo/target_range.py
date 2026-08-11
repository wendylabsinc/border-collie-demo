"""Confidence-bearing metric Arrival from associated Go2 range evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class MetricArrivalAction(str, Enum):
    ADVANCE = "advance"
    BRAKE = "brake"
    ARRIVAL = "arrival"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RangeCalibration:
    """Qualified transformation from a raw directional range to clearance."""

    forward_index: int
    sensor_to_front_envelope_m: float
    sensor_latency_s: float
    braking_distance_m: float
    noise_m: float
    target_clearance_m: float = 0.15
    tolerance_m: float = 0.05
    maximum_age_s: float = 0.25
    association_center_ratio: float = 0.15

    def __post_init__(self) -> None:
        if self.forward_index < 0:
            raise ValueError("forward range index must be non-negative")
        nonnegative = (
            self.sensor_to_front_envelope_m,
            self.sensor_latency_s,
            self.braking_distance_m,
            self.noise_m,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in nonnegative):
            raise ValueError(
                "range offsets and uncertainty must be finite and non-negative"
            )
        positive = (
            self.target_clearance_m,
            self.tolerance_m,
            self.maximum_age_s,
            self.association_center_ratio,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("range gates must be finite and positive")
        if self.noise_m > self.tolerance_m / 2.0:
            raise ValueError(
                "range noise is too large for the requested clearance tolerance"
            )


@dataclass(frozen=True)
class RangeObservation:
    ranges_m: tuple[float, ...] | None
    age_s: float | None
    pear_center_error_ratio: float | None
    visual_evidence_fresh: bool
    robot_stopped: bool | None
    close_speed_mps: float
    associated_range_m: float | None = None
    association_confidence: float | None = None
    association_valid: bool = False
    association_mode: str = "directional"
    range_source: str = "directional"
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class MetricArrivalDecision:
    action: MetricArrivalAction
    reason: str
    clearance_m: float | None
    brake_trigger_m: float
    evidence: Mapping[str, object]


class MetricArrivalGate:
    """Associate fresh centered vision with calibrated forward clearance."""

    def __init__(self, calibration: RangeCalibration) -> None:
        self.calibration = calibration

    def observe(self, observation: RangeObservation) -> MetricArrivalDecision:
        trigger = self._brake_trigger(observation.close_speed_mps)
        unavailable = self._unavailable_reason(observation)
        if unavailable is not None:
            return self._decision(
                MetricArrivalAction.UNAVAILABLE,
                unavailable,
                None,
                trigger,
            )
        raw_range_m = (
            float(observation.associated_range_m)
            if observation.associated_range_m is not None
            else float(observation.ranges_m[self.calibration.forward_index])
        )
        clearance_m = raw_range_m - self.calibration.sensor_to_front_envelope_m
        lower = self.calibration.target_clearance_m - self.calibration.tolerance_m
        upper = self.calibration.target_clearance_m + self.calibration.tolerance_m

        if observation.robot_stopped:
            if lower <= clearance_m <= upper:
                return self._decision(
                    MetricArrivalAction.ARRIVAL,
                    "stopped_clearance_confirmed",
                    clearance_m,
                    trigger,
                )
            if clearance_m < lower:
                return self._decision(
                    MetricArrivalAction.UNAVAILABLE,
                    "clearance_overshoot",
                    clearance_m,
                    trigger,
                )
            return self._decision(
                MetricArrivalAction.ADVANCE,
                "stopped_short_of_arrival",
                clearance_m,
                trigger,
            )

        action = (
            MetricArrivalAction.BRAKE
            if clearance_m <= trigger
            else MetricArrivalAction.ADVANCE
        )
        return self._decision(
            action,
            "predicted_stop_gate"
            if action is MetricArrivalAction.BRAKE
            else "clearance_open",
            clearance_m,
            trigger,
        )

    def describe(self) -> dict[str, object]:
        return {
            "configured": True,
            "forward_index": self.calibration.forward_index,
            "sensor_to_front_envelope_m": (self.calibration.sensor_to_front_envelope_m),
            "sensor_latency_s": self.calibration.sensor_latency_s,
            "braking_distance_m": self.calibration.braking_distance_m,
            "noise_m": self.calibration.noise_m,
            "target_clearance_m": self.calibration.target_clearance_m,
            "tolerance_m": self.calibration.tolerance_m,
            "maximum_age_s": self.calibration.maximum_age_s,
        }

    def _unavailable_reason(self, observation: RangeObservation) -> str | None:
        if observation.unavailable_reason:
            return observation.unavailable_reason
        if observation.robot_stopped is None:
            return "stopped_state_unavailable"
        using_associated_range = observation.associated_range_m is not None
        if using_associated_range:
            if not observation.association_valid:
                return "pear_range_not_associated"
        else:
            if not observation.visual_evidence_fresh:
                return "pear_visual_evidence_unavailable"
            center_error = observation.pear_center_error_ratio
            if (
                center_error is None
                or not math.isfinite(center_error)
                or abs(center_error) > self.calibration.association_center_ratio
            ):
                return "pear_range_not_associated"
        age_s = observation.age_s
        if age_s is None or not math.isfinite(age_s) or age_s < 0.0:
            return "forward_range_age_unavailable"
        if age_s > self.calibration.maximum_age_s:
            return "forward_range_stale"
        if using_associated_range:
            confidence = observation.association_confidence
            if confidence is None or confidence < 0.50:
                return "pear_range_association_confidence_low"
            value = observation.associated_range_m
        else:
            ranges = observation.ranges_m
            if ranges is None or self.calibration.forward_index >= len(ranges):
                return "forward_range_unavailable"
            value = ranges[self.calibration.forward_index]
        if not math.isfinite(value) or value <= 0.0:
            return "forward_range_unavailable"
        if (
            not math.isfinite(observation.close_speed_mps)
            or observation.close_speed_mps <= 0.0
        ):
            return "close_speed_invalid"
        return None

    def _brake_trigger(self, close_speed_mps: float) -> float:
        return (
            self.calibration.target_clearance_m
            + self.calibration.sensor_latency_s * close_speed_mps
            + self.calibration.braking_distance_m
            + 2.0 * self.calibration.noise_m
        )

    def _decision(
        self,
        action: MetricArrivalAction,
        reason: str,
        clearance_m: float | None,
        trigger: float,
    ) -> MetricArrivalDecision:
        return MetricArrivalDecision(
            action=action,
            reason=reason,
            clearance_m=clearance_m,
            brake_trigger_m=trigger,
            evidence={
                **self.describe(),
                "metric_arrival_action": action.value,
                "metric_arrival_reason": reason,
                "front_clearance_m": clearance_m,
                "brake_trigger_m": trigger,
                "clearance_tolerance_m": self.calibration.tolerance_m,
            },
        )
