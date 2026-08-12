import math

import pytest

from robotkit.perception.high_low.logic import (
    Band,
    BatteryThresholds,
    TemperatureThresholds,
    evaluate_battery,
    evaluate_temperature,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-20, Band.CRITICAL_LOW),
        (-1, Band.LOW),
        (20, Band.NORMAL),
        (61, Band.HIGH),
        (80, Band.CRITICAL_HIGH),
        (math.nan, Band.INVALID),
    ],
)
def test_temperature_bands(value, expected):
    assert evaluate_temperature(value).band is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.05, Band.CRITICAL_LOW),
        (0.15, Band.LOW),
        (0.50, Band.NORMAL),
        (0.98, Band.HIGH),
        (1.02, Band.CRITICAL_HIGH),
    ],
)
def test_battery_bands_use_ros_fraction(value, expected):
    evaluation = evaluate_battery(value, voltage_v=24.1, current_a=-2.0)
    assert evaluation.band is expected
    assert evaluation.payload["percent"] == value * 100
    assert evaluation.payload["voltage_v"] == 24.1


def test_invalid_battery_is_not_usable():
    evaluation = evaluate_battery(math.nan)
    assert evaluation.band is Band.INVALID
    assert evaluation.usable is False


def test_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        TemperatureThresholds(low_c=70, high_c=60)
    with pytest.raises(ValueError):
        BatteryThresholds(low_fraction=0.99, high_fraction=0.9)
