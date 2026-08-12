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
