from __future__ import annotations

import pytest

from border_collie_demo.target_range import (
    MetricArrivalAction,
    MetricArrivalGate,
    RangeCalibration,
    RangeObservation,
)


def calibration() -> RangeCalibration:
    return RangeCalibration(
        forward_index=1,
        sensor_to_front_envelope_m=0.20,
        sensor_latency_s=0.10,
        braking_distance_m=0.04,
        noise_m=0.01,
        target_clearance_m=0.15,
        tolerance_m=0.05,
    )


def observed(
    raw_range_m: float | None,
    *,
    age_s: float = 0.02,
    center_error: float = 0.03,
    stopped: bool = False,
) -> RangeObservation:
    ranges = (0.0, raw_range_m or 0.0, 0.0, 0.0)
    return RangeObservation(
        ranges_m=ranges,
        age_s=age_s,
        pear_center_error_ratio=center_error,
        visual_evidence_fresh=True,
        robot_stopped=stopped,
        close_speed_mps=0.55,
    )


def test_metric_gate_brakes_early_using_latency_and_stopping_distance() -> None:
    gate = MetricArrivalGate(calibration())

    advance = gate.observe(observed(0.55))
    brake = gate.observe(observed(0.45))

    assert advance.action is MetricArrivalAction.ADVANCE
    assert advance.clearance_m == pytest.approx(0.35)
    assert brake.action is MetricArrivalAction.BRAKE
    assert brake.clearance_m == pytest.approx(0.25)
    assert brake.brake_trigger_m == pytest.approx(0.265)


def test_arrival_requires_stopped_fresh_range_within_15_plus_or_minus_5_cm() -> None:
    gate = MetricArrivalGate(calibration())

    moving = gate.observe(observed(0.35, stopped=False))
    stopped = gate.observe(observed(0.35, stopped=True))

    assert moving.action is MetricArrivalAction.BRAKE
    assert stopped.action is MetricArrivalAction.ARRIVAL
    assert stopped.clearance_m == pytest.approx(0.15)
    assert stopped.evidence["clearance_tolerance_m"] == 0.05


@pytest.mark.parametrize(
    ("sample", "reason"),
    [
        (observed(None), "forward_range_unavailable"),
        (observed(0.35, age_s=0.251), "forward_range_stale"),
        (observed(0.35, center_error=0.20), "pear_range_not_associated"),
    ],
)
def test_unavailable_or_unassociated_range_fails_closed(
    sample: RangeObservation,
    reason: str,
) -> None:
    decision = MetricArrivalGate(calibration()).observe(sample)

    assert decision.action is MetricArrivalAction.UNAVAILABLE
    assert decision.reason == reason


def test_stopped_clearance_outside_tolerance_cannot_arrive() -> None:
    decision = MetricArrivalGate(calibration()).observe(observed(0.28, stopped=True))

    assert decision.action is MetricArrivalAction.UNAVAILABLE
    assert decision.reason == "clearance_overshoot"
    assert decision.clearance_m == pytest.approx(0.08)


def test_projected_lidar_clearance_requires_confident_bbox_association() -> None:
    gate = MetricArrivalGate(
        RangeCalibration(
            forward_index=0,
            sensor_to_front_envelope_m=0.0,
            sensor_latency_s=0.10,
            braking_distance_m=0.04,
            noise_m=0.01,
        )
    )
    confident = observed(None)
    confident = RangeObservation(
        **{
            **confident.__dict__,
            "ranges_m": None,
            "associated_range_m": 0.35,
            "association_confidence": 0.75,
            "association_valid": True,
            "association_mode": "lidar_handoff",
            "range_source": "lidar_temporal_pear_handoff",
        }
    )
    weak = RangeObservation(**{**confident.__dict__, "association_confidence": 0.49})

    assert gate.observe(confident).action is MetricArrivalAction.ADVANCE
    assert gate.observe(weak).reason == "pear_range_association_confidence_low"


def test_temporal_lidar_handoff_does_not_require_current_visual_evidence() -> None:
    gate = MetricArrivalGate(calibration())
    handoff = RangeObservation(
        ranges_m=None,
        age_s=0.05,
        pear_center_error_ratio=None,
        visual_evidence_fresh=False,
        robot_stopped=False,
        close_speed_mps=0.55,
        associated_range_m=0.70,
        association_confidence=0.80,
        association_valid=True,
        association_mode="lidar_handoff",
        range_source="lidar_temporal_pear_handoff",
    )
    unassociated = RangeObservation(**{**handoff.__dict__, "association_valid": False})

    assert gate.observe(handoff).action is MetricArrivalAction.ADVANCE
    assert gate.observe(unassociated).reason == "pear_range_not_associated"


def test_18_inch_target_accepts_only_16_to_20_inches_when_stopped() -> None:
    gate = MetricArrivalGate(
        RangeCalibration(
            forward_index=0,
            sensor_to_front_envelope_m=0.0,
            sensor_latency_s=0.20,
            braking_distance_m=0.02,
            noise_m=0.02,
            target_clearance_m=0.4572,
            tolerance_m=0.0508,
            maximum_age_s=0.30,
        )
    )

    for inches in (16.0, 18.0, 20.0):
        sample = RangeObservation(
            ranges_m=None,
            age_s=0.01,
            pear_center_error_ratio=None,
            visual_evidence_fresh=False,
            robot_stopped=True,
            close_speed_mps=0.55,
            associated_range_m=inches * 0.0254,
            association_confidence=0.75,
            association_valid=True,
            association_mode="lidar_handoff",
        )
        assert gate.observe(sample).action is MetricArrivalAction.ARRIVAL
