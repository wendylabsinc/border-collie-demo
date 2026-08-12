from __future__ import annotations

from border_collie_demo.persistent_fruit_tracker import (
    FruitTrackState,
    PersistentFruitTracker,
    PersistentFruitTrackerConfig,
)


def tracker(*, confirmations: int = 2) -> PersistentFruitTracker:
    return PersistentFruitTracker(
        PersistentFruitTrackerConfig(
            target_fruit="apple",
            acquisition_confidence=0.50,
            maintenance_confidence=0.40,
            acquisition_confirmations=confirmations,
            maximum_evidence_age_s=0.250,
            degraded_grace_s=0.250,
            center_corridor_ratio=0.20,
        )
    )


def observation(
    *,
    pts: int,
    confidence: float | None = 0.70,
    center_x: float = 0.50,
    label: str = "apple",
    generation: str = "camera-1",
    age_s: float = 0.01,
    crop_confidence: float | None = None,
) -> dict[str, object]:
    detection = (
        None
        if confidence is None
        else {
            "label": label,
            "confidence": confidence,
            "bbox_xyxy": [400, 300, 600, 600],
            "center_x_ratio": center_x,
            "center_y_ratio": 0.625,
            "bottom_ratio": 0.75,
            "bbox_area_ratio": 0.0625,
            "generation": generation,
            "source_pts": pts,
            "source_time_base": "1/90000",
            "age_s": age_s,
            "route": "full_frame",
        }
    )
    crop = (
        None
        if crop_confidence is None
        else {
            "label": label,
            "confidence": crop_confidence,
            "bbox_xyxy": [410, 305, 605, 600],
            "center_x_ratio": center_x,
            "center_y_ratio": 0.628,
            "bottom_ratio": 0.75,
            "bbox_area_ratio": 0.060,
            "generation": generation,
            "source_pts": pts,
            "source_time_base": "1/90000",
            "age_s": age_s,
            "route": "crop_confirmation",
        }
    )
    return {
        "camera_healthy": True,
        "generation": generation,
        "source": {
            "pts": pts,
            "time_base": "1/90000",
            "age_s": age_s,
        },
        "observations": {"full_frame": detection, "crop": crop},
    }


def lock(target: PersistentFruitTracker) -> None:
    target.observe(observation(pts=1), now_s=0.00)
    report = target.observe(observation(pts=2), now_s=0.10)
    assert report.state is FruitTrackState.LOCKED


def test_one_weak_frame_degrades_without_reacquiring_and_recovers() -> None:
    target = tracker()
    lock(target)

    degraded = target.observe(observation(pts=3, confidence=0.30), now_s=0.20)
    recovered = target.observe(observation(pts=4, confidence=0.65), now_s=0.24)

    assert degraded.state is FruitTrackState.DEGRADED
    assert degraded.same_identity is True
    assert degraded.motion_authorized is False
    assert degraded.arrival_eligible is False
    assert degraded.reason == "weak_full_frame_observation"
    assert recovered.state is FruitTrackState.LOCKED
    assert recovered.same_identity is True
    assert recovered.acquisition_epoch == degraded.acquisition_epoch


def test_second_fresh_miss_confirms_loss_inside_250_ms() -> None:
    target = tracker()
    lock(target)

    first = target.observe(observation(pts=3, confidence=None), now_s=0.20)
    second = target.observe(observation(pts=4, confidence=None), now_s=0.24)

    assert first.state is FruitTrackState.DEGRADED
    assert second.state is FruitTrackState.LOST
    assert second.motion_authorized is False
    assert second.reason == "confirmed_full_frame_loss"
    assert second.degraded_elapsed_s <= 0.250


def test_off_axis_track_preserves_identity_but_disables_forward_motion() -> None:
    target = tracker()
    lock(target)

    off_axis = target.observe(observation(pts=3, center_x=0.76), now_s=0.20)
    centered = target.observe(observation(pts=4, center_x=0.58), now_s=0.30)

    assert off_axis.state is FruitTrackState.LOCKED_OFF_AXIS
    assert off_axis.same_identity is True
    assert off_axis.motion_authorized is False
    assert off_axis.alignment_authorized is True
    assert centered.state is FruitTrackState.LOCKED
    assert centered.acquisition_epoch == off_axis.acquisition_epoch


def test_crop_miss_cannot_erase_full_frame_identity() -> None:
    target = tracker()
    lock(target)

    report = target.observe(
        observation(pts=3, confidence=0.62, crop_confidence=None),
        now_s=0.20,
    )

    assert report.state is FruitTrackState.LOCKED
    assert report.identity_route == "full_frame"
    assert report.motion_authorized is True


def test_crop_can_refine_confidence_but_not_supply_identity_alone() -> None:
    target = tracker()

    no_full_frame = target.observe(
        observation(pts=1, confidence=None, crop_confidence=0.90),
        now_s=0.00,
    )
    target.observe(observation(pts=2, confidence=0.55, crop_confidence=0.85), now_s=0.10)
    refined = target.observe(
        observation(pts=3, confidence=0.55, crop_confidence=0.85),
        now_s=0.20,
    )

    assert no_full_frame.state is FruitTrackState.UNSEEN
    assert no_full_frame.same_identity is False
    assert refined.state is FruitTrackState.LOCKED
    assert refined.identity_route == "full_frame"
    assert refined.geometry_route == "crop_confirmation"
    assert refined.filtered_confidence > 0.55


def test_duplicate_and_degraded_frames_never_advance_arrival() -> None:
    target = tracker()
    lock(target)

    locked = target.observe(observation(pts=3), now_s=0.20)
    duplicate = target.observe(observation(pts=3), now_s=0.22)
    degraded = target.observe(observation(pts=4, confidence=0.30), now_s=0.24)

    assert locked.arrival_counter > 0
    assert duplicate.arrival_counter == locked.arrival_counter
    assert duplicate.arrival_eligible is False
    assert degraded.arrival_counter == locked.arrival_counter
    assert degraded.arrival_eligible is False


def test_hard_safety_faults_stop_immediately() -> None:
    cases = (
        ({**observation(pts=3), "camera_healthy": False}, "camera_unhealthy"),
        (observation(pts=3, label="pear"), "confirmed_wrong_identity"),
        (observation(pts=3, generation="camera-2"), "generation_changed"),
        (observation(pts=3, age_s=0.30), "stale_full_frame_observation"),
    )

    for status, reason in cases:
        target = tracker()
        lock(target)
        report = target.observe(status, now_s=0.20)
        assert report.state is FruitTrackState.LOST
        assert report.motion_authorized is False
        assert report.reason == reason
