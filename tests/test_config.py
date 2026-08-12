from __future__ import annotations

import pytest

from border_collie_demo.config import HardwareConfig


def test_tracking_confidence_window_is_configured_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BORDER_COLLIE_TRACKING_CONFIDENCE_WINDOW_FRAMES", "10")

    config = HardwareConfig.from_env()

    assert config.tracking_confidence_window_frames == 10


def test_tracking_confidence_window_rejects_values_outside_documented_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BORDER_COLLIE_TRACKING_CONFIDENCE_WINDOW_FRAMES", "121")

    with pytest.raises(
        ValueError,
        match="tracking_confidence_window_frames must be between 1 and 120",
    ):
        HardwareConfig.from_env()


def test_approach_motion_profile_is_configured_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BORDER_COLLIE_APPROACH_FORWARD_MPS", "1.0")
    monkeypatch.setenv("BORDER_COLLIE_CLOSE_RANGE_MPS", "1.0")
    monkeypatch.setenv("BORDER_COLLIE_FINAL_PUSH_MPS", "0.6")
    monkeypatch.setenv("BORDER_COLLIE_FINAL_PUSH_DURATION_S", "1.25")

    config = HardwareConfig.from_env()

    assert config.approach_forward_mps == 1.0
    assert config.close_range_mps == 1.0
    assert config.final_push_mps == 0.6
    assert config.final_push_duration_s == 1.25


def test_approach_motion_profile_cannot_bypass_hardware_envelope() -> None:
    with pytest.raises(ValueError, match="below the configured movement minimum"):
        HardwareConfig(close_range_mps=0.54)
    with pytest.raises(ValueError, match="exceeds the configured motion limit"):
        HardwareConfig(approach_forward_mps=1.01)
