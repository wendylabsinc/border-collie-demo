from __future__ import annotations

import pytest

from border_collie_demo.home_stability import (
    HomeSample,
    HomeStabilityConfig,
    HomeStabilityWindow,
    HomeVerificationError,
)


def sample(
    sequence: int,
    distance_m: float,
    *,
    captured_monotonic_s: float | None = None,
    epoch: str = "odom-a",
    age_s: float = 0.01,
) -> HomeSample:
    return HomeSample(
        sequence=sequence,
        x_m=distance_m,
        y_m=0.0,
        yaw_rad=0.0,
        captured_monotonic_s=(
            float(sequence) / 20.0
            if captured_monotonic_s is None
            else captured_monotonic_s
        ),
        age_s=age_s,
        odometry_epoch=epoch,
    )


@pytest.mark.parametrize(
    "distances",
    [
        # 3465b41f: the controller accepted 0.0577 m, then the next fresh
        # post-stop sample was 0.113 m roughly 160 ms later.
        (0.0577, 0.1130, 0.1120, 0.1140),
        # 7fc8ef15: the same defect at 0.0255 m then 0.124 m.
        (0.0255, 0.1240, 0.1220, 0.1230),
    ],
)
def test_physical_inside_then_outside_replays_never_complete_home(
    distances: tuple[float, ...],
) -> None:
    window = HomeStabilityWindow(
        HomeStabilityConfig(
            arrival_tolerance_m=0.10,
            required_samples=4,
            maximum_spread_m=0.03,
        ),
        expected_odometry_epoch="odom-a",
    )

    decisions = [window.observe(sample(index, distance)) for index, distance in enumerate(distances, 1)]

    assert all(decision.stable is False for decision in decisions)
    assert decisions[-1].reason == "outside_home_gate"


def test_four_advancing_close_samples_complete_home_with_bounded_spread() -> None:
    window = HomeStabilityWindow(
        HomeStabilityConfig(
            arrival_tolerance_m=0.10,
            required_samples=4,
            maximum_spread_m=0.03,
        ),
        expected_odometry_epoch="odom-a",
    )

    decision = None
    for index, distance in enumerate((0.061, 0.064, 0.060, 0.063), 1):
        decision = window.observe(sample(index, distance))

    assert decision is not None
    assert decision.stable is True
    assert decision.reason == "stable_inside_home_gate"
    assert decision.sample_count == 4
    assert decision.distance_min_m == pytest.approx(0.060)
    assert decision.distance_max_m == pytest.approx(0.064)
    assert decision.position_spread_m == pytest.approx(0.004)


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (sample(2, 0.06, captured_monotonic_s=0.05), "regressed"),
        (sample(2, 0.06, epoch="odom-b"), "epoch"),
        (sample(2, 0.06, age_s=-0.01), "age"),
    ],
)
def test_regressed_epoch_changed_and_invalid_age_evidence_fail_closed(
    bad: HomeSample,
    message: str,
) -> None:
    window = HomeStabilityWindow(
        HomeStabilityConfig(
            arrival_tolerance_m=0.10,
            required_samples=4,
            maximum_spread_m=0.03,
        ),
        expected_odometry_epoch="odom-a",
    )
    window.observe(sample(1, 0.06, captured_monotonic_s=0.10))

    with pytest.raises(HomeVerificationError, match=message):
        window.observe(bad)


def test_large_position_spread_does_not_complete_even_when_every_sample_is_inside() -> None:
    window = HomeStabilityWindow(
        HomeStabilityConfig(
            arrival_tolerance_m=0.10,
            required_samples=4,
            maximum_spread_m=0.03,
        ),
        expected_odometry_epoch="odom-a",
    )

    decision = None
    for index, distance in enumerate((0.020, 0.091, 0.022, 0.090), 1):
        decision = window.observe(sample(index, distance))

    assert decision is not None
    assert decision.stable is False
    assert decision.reason == "position_spread_exceeded"
    assert decision.position_spread_m == pytest.approx(0.071)
