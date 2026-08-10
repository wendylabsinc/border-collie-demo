from __future__ import annotations

from border_collie_demo.qualified_tracking import (
    MotionRecommendation,
    QualifiedFruitTracker,
    QualifiedTrackingConfig,
)


def tracker(*, grace_s: float = 0.75) -> QualifiedFruitTracker:
    return QualifiedFruitTracker(
        QualifiedTrackingConfig.for_fruit(
            "pear",
            acquisition_confirmations=3,
            center_tolerance_ratio=0.08,
            center_confirmations=3,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            sight_loss_grace_s=grace_s,
            slow_speed_scale=0.30,
        )
    )


def test_runtime_configuration_can_tighten_but_not_weaken_fruit_policy() -> None:
    common = {
        "target_fruit": "pear",
        "acquisition_confirmations": 3,
        "center_tolerance_ratio": 0.08,
        "center_confirmations": 3,
        "near_bottom_ratio": 0.86,
        "near_center_ratio": 0.72,
        "near_confirmations": 3,
        "sight_loss_grace_s": 0.75,
        "slow_speed_scale": 0.30,
    }

    weaker = QualifiedTrackingConfig.for_fruit(
        **common,
        minimum_tracking_confidence=0.10,
    )
    stricter = QualifiedTrackingConfig.for_fruit(
        **common,
        minimum_tracking_confidence=0.70,
    )

    assert weaker.tracking_confidence == 0.55
    assert stricter.tracking_confidence == 0.70


def observation(
    *,
    label: str = "pear",
    confidence: float = 0.80,
    center_x: float = 0.50,
    center_y: float = 0.50,
    bottom: float = 0.65,
    area: float = 0.04,
    age_s: float = 0.01,
    source_pts: int,
) -> dict[str, object]:
    return {
        "camera_healthy": True,
        "target_ready": confidence >= 0.65 and label == "pear",
        "generation": "camera-1",
        "detection": {
            "label": label,
            "generation": "camera-1",
            "confidence": confidence,
            "center_x_ratio": center_x,
            "center_y_ratio": center_y,
            "bottom_ratio": bottom,
            "bbox_area_ratio": area,
            "age_s": age_s,
            "source_pts": source_pts,
        },
    }


def acquire(target: QualifiedFruitTracker, *, start_pts: int = 1) -> int:
    for pts in range(start_pts, start_pts + 3):
        decision = target.observe(observation(source_pts=pts), now_s=pts / 10)
    assert decision.recommendation is MotionRecommendation.APPROACH
    return start_pts + 3


def test_weak_phantom_never_authorizes_motion_or_acquires_a_track() -> None:
    target = tracker()

    decisions = [
        target.observe(
            observation(confidence=0.01, source_pts=pts),
            now_s=pts / 10,
        )
        for pts in range(1, 8)
    ]

    assert {decision.recommendation for decision in decisions} == {
        MotionRecommendation.SEARCH
    }
    assert decisions[-1].evidence["track_acquired"] is False
    assert decisions[-1].evidence["weak_samples"] == 7

    inconsistent_ready = target.observe(
        {
            **observation(confidence=0.01, source_pts=8),
            "target_ready": True,
        },
        now_s=0.8,
    )
    assert inconsistent_ready.recommendation is MotionRecommendation.SEARCH
    assert inconsistent_ready.evidence["track_acquired"] is False


def test_pear_close_range_confidence_collapse_confirms_no_motion_arrival() -> None:
    target = tracker()
    pts = acquire(target)
    target.observe(
        observation(
            confidence=0.72,
            center_y=0.69,
            bottom=0.74,
            area=0.07,
            source_pts=pts,
        ),
        now_s=0.4,
    )
    target.observe(
        observation(
            confidence=0.64,
            center_y=0.75,
            bottom=0.83,
            area=0.12,
            source_pts=pts + 1,
        ),
        now_s=0.5,
    )

    decision = target.observe(
        observation(
            confidence=0.27,
            center_y=0.91,
            bottom=0.997,
            area=0.24,
            source_pts=pts + 2,
        ),
        now_s=0.6,
    )

    assert decision.recommendation is MotionRecommendation.ARRIVAL
    assert decision.forward_scale == 0.0
    assert decision.reason == "qualified_close_track_confidence_collapsed"
    assert decision.evidence["arrival_mode"] == ("confidence_collapse_at_close_range")
    assert decision.evidence["close_range_tracking_confidence"] == 0.55


def test_stale_detection_stops_without_reusing_close_range_evidence() -> None:
    target = tracker()
    pts = acquire(target)
    target.observe(
        observation(
            center_y=0.69,
            bottom=0.74,
            area=0.07,
            source_pts=pts,
        ),
        now_s=0.4,
    )
    target.observe(
        observation(
            center_y=0.75,
            bottom=0.83,
            area=0.12,
            source_pts=pts + 1,
        ),
        now_s=0.5,
    )

    decision = target.observe(
        observation(
            center_y=0.78,
            bottom=0.89,
            area=0.15,
            age_s=0.40,
            source_pts=pts + 2,
        ),
        now_s=0.6,
    )

    assert decision.recommendation is MotionRecommendation.STOP
    assert decision.reason == "detection_stale"
    assert decision.evidence["arrival_mode"] is None

    missing_after_stale = target.observe(
        {"camera_healthy": True, "target_ready": False},
        now_s=0.61,
    )
    assert missing_after_stale.recommendation is MotionRecommendation.STOP
    assert missing_after_stale.evidence["arrival_mode"] is None


def test_wrong_label_stops_and_resets_identity_history() -> None:
    target = tracker()
    pts = acquire(target)

    changed = target.observe(
        observation(
            label="apple",
            confidence=0.90,
            source_pts=pts,
        ),
        now_s=0.4,
    )
    restarted = target.observe(
        observation(source_pts=pts + 1),
        now_s=0.5,
    )

    assert changed.recommendation is MotionRecommendation.STOP
    assert changed.reason == "target_identity_changed"
    assert changed.evidence["track_acquired"] is False
    assert changed.evidence["identity_resets"] == 1
    assert restarted.recommendation is MotionRecommendation.ALIGN
    assert restarted.evidence["acquisition_samples"] == 1


def test_same_detection_frame_cannot_satisfy_temporal_confirmations() -> None:
    target = tracker()

    first = target.observe(observation(source_pts=1), now_s=0.1)
    second = target.observe(observation(source_pts=1), now_s=0.2)
    third = target.observe(observation(source_pts=1), now_s=0.3)

    assert first.recommendation is MotionRecommendation.ALIGN
    assert second.recommendation is MotionRecommendation.STOP
    assert third.recommendation is MotionRecommendation.STOP
    assert third.evidence["track_acquired"] is False
    assert third.evidence["duplicate_samples"] == 2


def test_sight_lost_close_arrival_is_bounded_by_time_and_geometry() -> None:
    close_target = tracker(grace_s=0.75)
    pts = acquire(close_target)
    close_target.observe(
        observation(
            center_y=0.69,
            bottom=0.74,
            area=0.07,
            source_pts=pts,
        ),
        now_s=0.4,
    )
    close_target.observe(
        observation(
            center_y=0.75,
            bottom=0.83,
            area=0.12,
            source_pts=pts + 1,
        ),
        now_s=0.5,
    )

    arrived = close_target.observe(
        {"camera_healthy": True, "target_ready": False},
        now_s=0.6,
    )
    assert arrived.recommendation is MotionRecommendation.ARRIVAL
    assert arrived.reason == "qualified_close_track_lost"

    expired_target = tracker(grace_s=0.10)
    pts = acquire(expired_target)
    expired_target.observe(
        observation(
            center_y=0.69,
            bottom=0.74,
            area=0.07,
            source_pts=pts,
        ),
        now_s=0.4,
    )
    expired_target.observe(
        observation(
            center_y=0.75,
            bottom=0.83,
            area=0.12,
            source_pts=pts + 1,
        ),
        now_s=0.5,
    )
    stopped = expired_target.observe(
        {"camera_healthy": True, "target_ready": False},
        now_s=0.7,
    )
    assert stopped.recommendation is MotionRecommendation.STOP
    assert stopped.evidence["arrival_mode"] is None


def test_bbox_retreat_breaks_continuity_and_stops_forward_motion() -> None:
    target = tracker()
    pts = acquire(target)

    decision = target.observe(
        observation(
            center_y=0.49,
            bottom=0.64,
            area=0.01,
            source_pts=pts,
        ),
        now_s=0.4,
    )

    assert decision.recommendation is MotionRecommendation.STOP
    assert decision.reason == "track_discontinuous"
    assert decision.evidence["discontinuity_stops"] == 1
    assert decision.evidence["track_acquired"] is False


def test_moderate_reacquisition_error_resumes_with_moving_correction() -> None:
    target = tracker()
    pts = acquire(target)

    stopped = target.observe(
        observation(
            center_y=0.49,
            bottom=0.64,
            area=0.01,
            source_pts=pts,
        ),
        now_s=0.4,
    )
    reacquiring = [
        target.observe(
            observation(
                center_x=0.64,
                center_y=0.50,
                bottom=0.65,
                area=0.04,
                source_pts=pts + offset,
            ),
            now_s=0.4 + offset / 10,
        )
        for offset in range(1, 4)
    ]

    assert stopped.recommendation is MotionRecommendation.STOP
    assert [decision.recommendation for decision in reacquiring] == [
        MotionRecommendation.STOP,
        MotionRecommendation.STOP,
        MotionRecommendation.APPROACH,
    ]
    assert reacquiring[-1].horizontal_error == 0.14
    assert reacquiring[-1].evidence["approach_authorized"] is True
    assert reacquiring[-1].evidence["initial_centered"] is True
    assert reacquiring[-1].evidence["stationary_recenter_samples"] == 0


def test_large_reacquisition_error_requires_stationary_recenter() -> None:
    target = tracker()
    pts = acquire(target)
    target.observe(
        observation(
            center_y=0.49,
            bottom=0.64,
            area=0.01,
            source_pts=pts,
        ),
        now_s=0.4,
    )

    decisions = [
        target.observe(
            observation(
                center_x=0.95,
                center_y=0.50,
                bottom=0.65,
                area=0.04,
                source_pts=pts + offset,
            ),
            now_s=0.4 + offset / 10,
        )
        for offset in range(1, 4)
    ]

    assert [decision.recommendation for decision in decisions] == [
        MotionRecommendation.STOP,
        MotionRecommendation.STOP,
        MotionRecommendation.ALIGN,
    ]
    assert decisions[-1].reason == "large_tracking_error"
    assert decisions[-1].forward_scale == 0.0
    assert decisions[-1].evidence["stationary_recenter_samples"] == 1
    assert decisions[-1].evidence["stationary_recenter_error_ratio"] == 0.40
