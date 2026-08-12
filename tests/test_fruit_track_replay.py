from __future__ import annotations

from border_collie_demo.fruit_track_replay import FruitTrackReplay
from border_collie_demo.persistent_fruit_tracker import PersistentFruitTracker


def status(*, pts: int, confidence: float | None, center_x: float = 0.5):
    detection = (
        None
        if confidence is None
        else {
            "label": "apple",
            "confidence": confidence,
            "bbox_xyxy": [400, 300, 600, 600],
            "center_x_ratio": center_x,
            "center_y_ratio": 0.625,
            "bottom_ratio": 0.75,
            "bbox_area_ratio": 0.0625,
            "generation": "camera-1",
            "source_pts": pts,
            "source_time_base": "1/90000",
            "age_s": 0.01,
            "route": "full_frame",
        }
    )
    return {
        "camera_healthy": True,
        "generation": "camera-1",
        "source": {"pts": pts, "time_base": "1/90000", "age_s": 0.01},
        "observations": {"full_frame": detection, "crop": None},
    }


def event(phase: str, pts: int, confidence: float | None, *, center_x: float = 0.5):
    return {
        "kind": "perception_sample",
        "recorded_monotonic_s": pts / 10,
        "payload": {
            "phase": phase,
            "target_fruit": "apple",
            **status(pts=pts, confidence=confidence, center_x=center_x),
        },
    }


def test_replay_keeps_one_identity_epoch_across_phase_transition() -> None:
    replay = FruitTrackReplay(
        PersistentFruitTracker.for_fruit("apple", acquisition_confirmations=2)
    )

    result = replay.replay(
        [
            event("turn_to_fruit", 1, 0.70),
            event("turn_to_fruit", 2, 0.70),
            event("approach_fruit", 3, 0.30),
            event("approach_fruit", 4, 0.65),
            event("approach_fruit", 5, 0.65, center_x=0.78),
            event("approach_fruit", 6, 0.65),
        ]
    )

    assert result.phase_identity_resets == 0
    assert result.acquisition_epochs == 1
    assert result.degraded_recoveries == 1
    assert result.off_axis_identity_preserved == 1
    assert result.degraded_arrival_advances == 0


def test_replay_confirms_persistent_loss_inside_250_ms() -> None:
    replay = FruitTrackReplay(
        PersistentFruitTracker.for_fruit("apple", acquisition_confirmations=2)
    )

    result = replay.replay(
        [
            event("find_fruit", 1, 0.70),
            event("find_fruit", 2, 0.70),
            event("approach_fruit", 3, None),
            event("approach_fruit", 4, None),
        ]
    )

    assert result.confirmed_losses == 1
    assert result.maximum_confirmed_loss_s <= 0.250
    assert result.degraded_arrival_advances == 0
