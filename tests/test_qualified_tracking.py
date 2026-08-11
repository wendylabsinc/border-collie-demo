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
            final_approach_latch_enabled=True,
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


def missing_observation(*, source_pts: int, age_s: float = 0.01) -> dict[str, object]:
    return {
        "camera_healthy": True,
        "target_ready": False,
        "generation": "camera-1",
        "source": {"pts": source_pts, "age_s": age_s},
        "detection": {},
    }


def test_final_approach_latch_survives_weak_and_stale_frames_until_fresh_loss() -> None:
    """Replay the closeout shape from run bce852b0 without a timeout stall."""
    target = tracker()
    pts = acquire(target)

    visible = [
        target.observe(
            observation(
                center_x=0.52,
                center_y=0.76,
                bottom=0.91,
                area=0.12,
                source_pts=pts + offset,
            ),
            now_s=0.4 + offset / 10,
        )
        for offset in range(3)
    ]
    weak = target.observe(
        observation(
            confidence=0.27,
            center_x=0.52,
            center_y=0.91,
            bottom=0.997,
            area=0.24,
            source_pts=pts + 3,
        ),
        now_s=0.7,
    )
    stale = target.observe(
        observation(
            confidence=0.27,
            center_x=0.52,
            center_y=0.91,
            bottom=0.997,
            area=0.24,
            age_s=0.40,
            source_pts=pts + 4,
        ),
        now_s=0.8,
    )
    arrived = target.observe(
        missing_observation(source_pts=pts + 5),
        now_s=0.9,
    )
    repeated = target.observe(
        missing_observation(source_pts=pts + 6),
        now_s=0.95,
    )

    assert visible[-1].reason == "qualified_visible_arrival"
    assert visible[-1].evidence["final_approach_latched"] is True
    assert weak.recommendation is MotionRecommendation.STOP
    assert weak.evidence["final_approach_loss_samples"] == 1
    assert stale.recommendation is MotionRecommendation.STOP
    assert stale.evidence["final_approach_loss_samples"] == 1
    assert arrived.recommendation is MotionRecommendation.ARRIVAL
    assert arrived.reason == "qualified_final_approach_loss"
    assert arrived.evidence["arrival_mode"] == "final_approach_loss_confirmed"
    assert repeated.recommendation is MotionRecommendation.STOP
    assert repeated.reason == "arrival_already_confirmed"


def test_bottom_clipped_area_retreat_counts_as_final_approach_loss() -> None:
    """Replay the decisive closeout samples from run acfdb493."""
    target = tracker()
    for source_pts in (276000, 276060, 276180):
        target.observe(
            observation(
                center_x=0.54,
                area=0.005,
                source_pts=source_pts,
            ),
            now_s=0.50,
        )
    for source_pts, confidence, center_x, center_y, bottom, area in (
        (276360, 0.821043848991394, 0.541015625, 0.9125, 0.9680555555555556, 0.005815972222222222),
        (276540, 0.7732153534889221, 0.5453125, 0.9361111111111111, 0.9972222222222222, 0.006493055555555555),
        (276660, 0.7592233419418335, 0.5546875, 0.9430555555555555, 1.0, 0.00640625),
    ):
        target.observe(
            observation(
                confidence=confidence,
                center_x=center_x,
                center_y=center_y,
                bottom=bottom,
                area=area,
                source_pts=source_pts,
            ),
            now_s=0.60,
        )

    last_full_box = target.observe(
        observation(
            confidence=0.644224226474762,
            center_x=0.56015625,
            center_y=0.9597222222222223,
            bottom=1.0,
            area=0.0050347222222222225,
            source_pts=276780,
        ),
        now_s=0.65,
    )
    clipped = target.observe(
        observation(
            confidence=0.8311417698860168,
            center_x=0.57109375,
            center_y=0.9729166666666667,
            bottom=1.0,
            area=0.0022851562500000003,
            source_pts=276960,
        ),
        now_s=0.70,
    )
    later_weak_bottom_edge = target.observe(
        observation(
            confidence=0.1436695158481598,
            center_x=0.57109375,
            center_y=0.975,
            bottom=0.9958333333333333,
            area=0.0017578125,
            source_pts=277320,
        ),
        now_s=0.75,
    )

    assert last_full_box.reason == "qualified_visible_arrival"
    assert clipped.recommendation is MotionRecommendation.STOP
    assert clipped.reason == "confirming_final_approach_loss"
    assert clipped.evidence["final_approach_loss_samples"] == 1
    assert later_weak_bottom_edge.recommendation is MotionRecommendation.ARRIVAL
    assert later_weak_bottom_edge.reason == "qualified_final_approach_loss"
    assert later_weak_bottom_edge.evidence["final_approach_loss_samples"] == 2


def test_bottom_clip_loss_needs_two_fresh_advancing_samples() -> None:
    target = tracker()
    pts = latch_bottom_edge(target)

    first_loss = target.observe(
        observation(
            center_x=0.52,
            center_y=0.93,
            bottom=1.0,
            area=0.05,
            source_pts=pts,
        ),
        now_s=0.7,
    )
    duplicate = target.observe(
        observation(
            center_x=0.52,
            center_y=0.93,
            bottom=1.0,
            area=0.04,
            source_pts=pts,
        ),
        now_s=0.75,
    )
    stale = target.observe(
        observation(
            confidence=0.20,
            center_x=0.52,
            center_y=0.94,
            bottom=1.0,
            area=0.03,
            age_s=0.40,
            source_pts=pts + 1,
        ),
        now_s=0.80,
    )

    assert first_loss.recommendation is MotionRecommendation.STOP
    assert first_loss.evidence["final_approach_loss_samples"] == 1
    assert duplicate.reason == "final_approach_frame_not_advancing"
    assert duplicate.evidence["final_approach_loss_samples"] == 1
    assert stale.reason == "detection_stale"
    assert stale.evidence["final_approach_loss_samples"] == 1


def test_area_retreat_away_from_centered_lower_edge_cancels_final_approach() -> None:
    cases = (
        {"center_x": 0.52, "center_y": 0.75, "bottom": 0.88},
        {"center_x": 0.60, "center_y": 0.93, "bottom": 1.0},
        {"center_x": 0.52, "center_y": 0.82, "bottom": 1.0},
    )

    for case in cases:
        target = tracker()
        pts = latch_bottom_edge(target)
        decision = target.observe(
            observation(
                area=0.05,
                source_pts=pts,
                **case,
            ),
            now_s=0.7,
        )

        assert decision.recommendation is MotionRecommendation.STOP
        assert decision.reason == "final_approach_track_discontinuous"
        assert decision.evidence["final_approach_latched"] is False
        assert decision.evidence["final_approach_loss_samples"] == 0


def test_bottom_clip_loss_authority_cancels_on_identity_generation_or_camera() -> None:
    status_mutations = (
        {"detection": {"label": "apple"}},
        {"generation": "camera-2"},
        {"camera_healthy": False},
    )

    for mutation in status_mutations:
        target = tracker()
        pts = latch_bottom_edge(target)
        first_loss = target.observe(
            observation(
                center_x=0.52,
                center_y=0.93,
                bottom=1.0,
                area=0.05,
                source_pts=pts,
            ),
            now_s=0.7,
        )
        status = observation(
            center_x=0.52,
            center_y=0.94,
            bottom=1.0,
            area=0.04,
            source_pts=pts + 1,
        )
        if "detection" in mutation:
            status["detection"].update(mutation["detection"])  # type: ignore[union-attr]
        else:
            status.update(mutation)
        cancelled = target.observe(status, now_s=0.75)

        assert first_loss.evidence["final_approach_loss_samples"] == 1
        assert cancelled.recommendation is MotionRecommendation.STOP
        assert cancelled.evidence["final_approach_latched"] is False
        assert cancelled.evidence["final_approach_cancelled_reason"] is not None


def latch_final_approach(target: QualifiedFruitTracker) -> int:
    pts = acquire(target)
    for offset in range(3):
        decision = target.observe(
            observation(
                center_x=0.51,
                center_y=0.76,
                bottom=0.91,
                area=0.12,
                source_pts=pts + offset,
            ),
            now_s=0.4 + offset / 10,
        )
    assert decision.reason == "qualified_visible_arrival"
    return pts + 3


def latch_bottom_edge(target: QualifiedFruitTracker) -> int:
    pts = latch_final_approach(target)
    decision = target.observe(
        observation(
            center_x=0.52,
            center_y=0.93,
            bottom=1.0,
            area=0.12,
            source_pts=pts,
        ),
        now_s=0.65,
    )
    assert decision.reason == "qualified_visible_arrival"
    return pts + 1


def test_stale_or_frozen_evidence_never_confirms_final_approach_loss() -> None:
    target = tracker()
    pts = latch_final_approach(target)

    stale = target.observe(
        observation(
            center_y=0.91,
            bottom=0.997,
            age_s=0.40,
            source_pts=pts,
        ),
        now_s=0.7,
    )
    frozen = target.observe(
        missing_observation(source_pts=pts),
        now_s=0.8,
    )
    expired = target.observe(
        missing_observation(source_pts=pts + 1),
        now_s=1.4,
    )

    assert stale.recommendation is MotionRecommendation.STOP
    assert stale.evidence["final_approach_loss_samples"] == 0
    assert frozen.recommendation is MotionRecommendation.STOP
    assert frozen.reason == "final_approach_frame_not_advancing"
    assert frozen.evidence["final_approach_loss_samples"] == 0
    assert expired.recommendation is MotionRecommendation.STOP
    assert expired.reason == "final_approach_expired"


def test_final_approach_wrong_label_or_generation_cancels_push_authority() -> None:
    wrong_label_target = tracker()
    pts = latch_final_approach(wrong_label_target)
    wrong_label = wrong_label_target.observe(
        observation(label="apple", confidence=0.90, source_pts=pts),
        now_s=0.7,
    )
    after_wrong_label = wrong_label_target.observe(
        missing_observation(source_pts=pts + 1),
        now_s=0.8,
    )

    wrong_generation_target = tracker()
    pts = latch_final_approach(wrong_generation_target)
    wrong_generation = wrong_generation_target.observe(
        {
            **missing_observation(source_pts=pts),
            "generation": "camera-2",
        },
        now_s=0.7,
    )
    after_wrong_generation = wrong_generation_target.observe(
        missing_observation(source_pts=pts + 1),
        now_s=0.8,
    )

    assert wrong_label.reason == "target_identity_changed"
    assert after_wrong_label.recommendation is MotionRecommendation.STOP
    assert after_wrong_label.evidence["final_approach_cancelled_reason"] == (
        "target_identity_changed"
    )
    assert wrong_generation.reason == "final_approach_generation_changed"
    assert after_wrong_generation.recommendation is MotionRecommendation.STOP
    assert after_wrong_generation.evidence["final_approach_cancelled_reason"] == (
        "final_approach_generation_changed"
    )


def test_final_approach_off_axis_invalid_or_unhealthy_evidence_never_arrives() -> None:
    off_axis_target = tracker()
    pts = latch_final_approach(off_axis_target)
    off_axis = off_axis_target.observe(
        observation(
            center_x=0.70,
            center_y=0.91,
            bottom=0.997,
            area=0.24,
            source_pts=pts,
        ),
        now_s=0.7,
    )

    invalid_target = tracker()
    pts = latch_final_approach(invalid_target)
    invalid_status = observation(source_pts=pts)
    invalid_status["detection"]["bottom_ratio"] = None  # type: ignore[index]
    invalid = invalid_target.observe(invalid_status, now_s=0.7)

    unhealthy_target = tracker()
    pts = latch_final_approach(unhealthy_target)
    unhealthy = unhealthy_target.observe(
        {
            **missing_observation(source_pts=pts),
            "camera_healthy": False,
        },
        now_s=0.7,
    )

    for decision in (off_axis, invalid, unhealthy):
        assert decision.recommendation is MotionRecommendation.STOP
        assert decision.evidence["final_approach_latched"] is False
        assert decision.evidence["final_approach_cancelled_reason"] is not None


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


def test_duplicate_weak_close_frame_holds_before_fresh_collapse_arrives() -> None:
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

    duplicate = target.observe(
        observation(
            confidence=0.27,
            center_y=0.75,
            bottom=0.83,
            area=0.12,
            source_pts=pts + 1,
        ),
        now_s=0.55,
    )
    arrived = target.observe(
        observation(
            confidence=0.27,
            center_y=0.91,
            bottom=0.997,
            area=0.24,
            source_pts=pts + 2,
        ),
        now_s=0.6,
    )

    assert duplicate.recommendation is MotionRecommendation.HOLD
    assert duplicate.reason == "duplicate_weak_close_frame"
    assert arrived.recommendation is MotionRecommendation.ARRIVAL
    assert arrived.reason == "qualified_close_track_confidence_collapsed"


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
    assert second.recommendation is MotionRecommendation.HOLD
    assert third.recommendation is MotionRecommendation.HOLD
    assert third.evidence["track_acquired"] is False
    assert third.evidence["acquisition_samples"] == 1
    assert third.evidence["qualified_samples"] == 1
    assert third.evidence["duplicate_samples"] == 2


def test_duplicate_frame_loses_hold_authority_when_it_becomes_stale() -> None:
    target = tracker()

    first = target.observe(observation(source_pts=1), now_s=0.1)
    stale = target.observe(
        observation(age_s=0.251, source_pts=1),
        now_s=0.35,
    )

    assert first.recommendation is MotionRecommendation.ALIGN
    assert stale.recommendation is MotionRecommendation.STOP
    assert stale.reason == "detection_stale"
    assert stale.evidence["duplicate_samples"] == 0
    assert stale.evidence["stale_samples"] == 1

    missing_age_status = observation(source_pts=1)
    del missing_age_status["detection"]["age_s"]  # type: ignore[index]
    missing = target.observe(missing_age_status, now_s=0.36)
    assert missing.recommendation is MotionRecommendation.STOP
    assert missing.reason == "detection_age_missing"


def test_unsafe_duplicate_frame_stops_instead_of_holding() -> None:
    target = tracker()
    target.observe(observation(source_pts=1), now_s=0.1)

    weak = target.observe(
        observation(confidence=0.01, source_pts=1),
        now_s=0.2,
    )
    generation_mismatch = target.observe(
        {
            **observation(source_pts=1),
            "generation": "camera-2",
        },
        now_s=0.21,
    )

    assert weak.recommendation is MotionRecommendation.SEARCH
    assert weak.reason == "target_unqualified"
    assert weak.forward_scale == 0.0
    assert generation_mismatch.recommendation is MotionRecommendation.STOP
    assert generation_mismatch.reason == "generation_mismatch"
    assert generation_mismatch.forward_scale == 0.0


def test_single_center_jump_holds_prior_authority_until_confirmed() -> None:
    target = tracker()
    acquire(target)

    discontinuous = target.observe(
        observation(center_x=0.90, source_pts=4),
        now_s=0.4,
    )

    assert discontinuous.recommendation is MotionRecommendation.HOLD
    assert discontinuous.reason == "confirming_center_jump"
    assert discontinuous.evidence["pending_center_jump_samples"] == 1


def test_two_distinct_center_jumps_stop_and_reset_the_track() -> None:
    target = tracker()
    pts = acquire(target)

    first = target.observe(
        observation(center_x=0.90, source_pts=pts),
        now_s=0.4,
    )
    confirmed = target.observe(
        observation(center_x=0.88, source_pts=pts + 1),
        now_s=0.5,
    )

    assert first.recommendation is MotionRecommendation.HOLD
    assert confirmed.recommendation is MotionRecommendation.STOP
    assert confirmed.reason == "track_discontinuous"
    assert confirmed.evidence["discontinuity_stops"] == 1
    assert confirmed.evidence["track_acquired"] is False


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


def test_off_axis_close_loss_cannot_authorize_lidar_handoff() -> None:
    target = tracker()
    pts = acquire(target)
    target.observe(
        observation(
            center_x=0.64,
            center_y=0.69,
            bottom=0.74,
            area=0.07,
            source_pts=pts,
        ),
        now_s=0.4,
    )
    target.observe(
        observation(
            center_x=0.66,
            center_y=0.75,
            bottom=0.83,
            area=0.12,
            source_pts=pts + 1,
        ),
        now_s=0.5,
    )

    lost = target.observe(
        {"camera_healthy": True, "target_ready": False},
        now_s=0.6,
    )

    assert lost.recommendation is MotionRecommendation.STOP
    assert lost.reason == "target_lost_off_axis"
    assert lost.evidence["close_handoff_centered_samples"] == 0
    assert lost.evidence["arrival_mode"] is None


def test_two_fresh_close_off_center_samples_stop_forward_to_recenter() -> None:
    target = tracker()
    pts = acquire(target)

    first = target.observe(
        observation(
            center_x=0.64,
            center_y=0.69,
            bottom=0.74,
            area=0.07,
            source_pts=pts,
        ),
        now_s=0.4,
    )
    second = target.observe(
        observation(
            center_x=0.66,
            center_y=0.75,
            bottom=0.83,
            area=0.12,
            source_pts=pts + 1,
        ),
        now_s=0.5,
    )

    assert first.recommendation is MotionRecommendation.APPROACH
    assert second.recommendation is MotionRecommendation.ALIGN
    assert second.reason == "close_tracking_recenter"
    assert second.forward_scale == 0.0
    assert second.evidence["close_recenter_active"] is True


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
    assert reacquiring[-1].horizontal_error == 0.0
    assert reacquiring[-1].evidence["moving_steering_active"] is False
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
