"""Pure high/low evaluation logic with no ROS2 or clock dependency."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Band(str, Enum):
    CRITICAL_LOW = "critical_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL_HIGH = "critical_high"
    INVALID = "invalid"


@dataclass(frozen=True)
class Evaluation:
    band: Band
    value: float
    unit: str
    usable: bool
    payload: dict[str, Any]


@dataclass(frozen=True)
class TemperatureThresholds:
    critical_low_c: float = -10.0
    low_c: float = 0.0
    high_c: float = 60.0
    critical_high_c: float = 75.0

    def __post_init__(self) -> None:
        if not (
            self.critical_low_c < self.low_c < self.high_c < self.critical_high_c
        ):
            raise ValueError("temperature thresholds must be strictly increasing")


@dataclass(frozen=True)
class BatteryThresholds:
    critical_low_fraction: float = 0.10
    low_fraction: float = 0.20
    high_fraction: float = 0.95
    critical_high_fraction: float = 1.01

    def __post_init__(self) -> None:
        if not (
            0 <= self.critical_low_fraction
            < self.low_fraction
            < self.high_fraction
            < self.critical_high_fraction
        ):
            raise ValueError("battery thresholds must be strictly increasing and non-negative")


def _band(
    value: float,
    *,
    critical_low: float,
    low: float,
    high: float,
    critical_high: float,
) -> Band:
    if not math.isfinite(value):
        return Band.INVALID
    if value <= critical_low:
        return Band.CRITICAL_LOW
    if value < low:
        return Band.LOW
    if value >= critical_high:
        return Band.CRITICAL_HIGH
    if value > high:
        return Band.HIGH
    return Band.NORMAL


def evaluate_temperature(
    temperature_c: float,
    variance: float | None = None,
    *,
    thresholds: TemperatureThresholds = TemperatureThresholds(),
) -> Evaluation:
    band = _band(
        temperature_c,
        critical_low=thresholds.critical_low_c,
        low=thresholds.low_c,
        high=thresholds.high_c,
        critical_high=thresholds.critical_high_c,
    )
    payload: dict[str, Any] = {
        "temperature_c": temperature_c,
        "band": band.value,
        "thresholds_c": {
            "critical_low": thresholds.critical_low_c,
            "low": thresholds.low_c,
            "high": thresholds.high_c,
            "critical_high": thresholds.critical_high_c,
        },
    }
    if variance is not None and math.isfinite(variance):
        payload["variance"] = variance
    return Evaluation(
        band=band,
        value=temperature_c,
        unit="celsius",
        usable=band is not Band.INVALID,
        payload=payload,
    )


def evaluate_battery(
    percentage: float,
    *,
    voltage_v: float | None = None,
    current_a: float | None = None,
    charging: bool | None = None,
    thresholds: BatteryThresholds = BatteryThresholds(),
) -> Evaluation:
    """Evaluate BatteryState percentage, whose ROS2 unit is the fraction 0..1."""
    band = _band(
        percentage,
        critical_low=thresholds.critical_low_fraction,
        low=thresholds.low_fraction,
        high=thresholds.high_fraction,
        critical_high=thresholds.critical_high_fraction,
    )
    payload: dict[str, Any] = {
        "percentage": percentage,
        "percent": percentage * 100 if math.isfinite(percentage) else percentage,
        "band": band.value,
        "thresholds_fraction": {
            "critical_low": thresholds.critical_low_fraction,
            "low": thresholds.low_fraction,
            "high": thresholds.high_fraction,
            "critical_high": thresholds.critical_high_fraction,
        },
    }
    if voltage_v is not None and math.isfinite(voltage_v):
        payload["voltage_v"] = voltage_v
    if current_a is not None and math.isfinite(current_a):
        payload["current_a"] = current_a
    if charging is not None:
        payload["charging"] = charging
    return Evaluation(
        band=band,
        value=percentage,
        unit="fraction",
        usable=band is not Band.INVALID and 0 <= percentage <= 1.05,
        payload=payload,
    )

