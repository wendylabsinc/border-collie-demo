from __future__ import annotations

import pytest

from border_collie_demo.qualified_track import (
    QualifiedTrackGate,
    TrackIdentityChanged,
)


def gate() -> QualifiedTrackGate:
    return QualifiedTrackGate(
        target_fruit="pear",
        acquisition_confidence=0.55,
        acquisition_confirmations=3,
        close_range_confidence=0.25,
        close_range_minimum_bottom_ratio=0.70,
        maximum_center_delta_ratio=0.20,
        maximum_vertical_retreat_ratio=0.08,
    )


def sample(confidence: float, bottom: float, *, ready: bool = False) -> dict:
    return {
        "target_ready": ready,
        "detection": {
            "label": "pear",
            "confidence": confidence,
            "center_x_ratio": 0.5,
            "center_y_ratio": bottom - 0.15,
            "bottom_ratio": bottom,
        },
    }


def test_track_requires_temporal_acquisition_then_allows_close_continuation() -> None:
    tracker = gate()
    assert tracker.observe(sample(0.60, 0.72)).ready is False
    assert tracker.observe(sample(0.60, 0.76)).ready is False
    assert tracker.observe(sample(0.60, 0.80)).ready is True

    continued = tracker.observe(sample(0.30, 0.86))

    assert continued.ready is True
    assert continued.close_range_continuation is True
    assert tracker.close_range_continuation_samples == 1
    assert tracker.minimum_observed_confidence == pytest.approx(0.30)


def test_weak_phantom_never_acquires_a_track() -> None:
    tracker = gate()
    assert all(
        tracker.observe(sample(0.01, 0.95)).ready is False for _ in range(10)
    )


def test_discontinuous_close_detection_is_rejected() -> None:
    tracker = gate()
    for bottom in (0.72, 0.76, 0.80):
        tracker.observe(sample(0.60, bottom))
    discontinuous = sample(0.30, 0.60)
    discontinuous["detection"]["center_x_ratio"] = 0.9

    assert tracker.observe(discontinuous).ready is False


def test_ready_wrong_label_fails_identity_contract() -> None:
    tracker = gate()
    with pytest.raises(TrackIdentityChanged, match="changed identity"):
        tracker.observe(
            {
                "target_ready": True,
                "detection": {
                    "label": "apple",
                    "confidence": 0.9,
                    "center_x_ratio": 0.5,
                    "center_y_ratio": 0.5,
                    "bottom_ratio": 0.7,
                },
            }
        )
