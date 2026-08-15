from __future__ import annotations

import pytest

from border_collie_demo.guidance import (
    FruitGuidance,
    GuidanceAction,
    GuidanceConfig,
    GuidancePhase,
)


def observation(
    *,
    pts: int,
    now_s: float,
    label: str | None = "pear",
    confidence: float = 0.80,
    center_x: float = 0.50,
    center_y: float = 0.55,
    bottom: float = 0.65,
    generation: str = "camera-1",
    camera_healthy: bool = True,
) -> dict[str, object]:
    detection: dict[str, object] | None
    if label is None:
        detection = None
    else:
        detection = {
            "label": label,
            "confidence": confidence,
            "generation": generation,
            "source_pts": pts,
            "source_time_base": "1/90000",
            "age_s": 0.02,
            "center_x_ratio": center_x,
            "center_y_ratio": center_y,
            "bottom_ratio": bottom,
        }
    return {
        "camera_healthy": camera_healthy,
        "generation": generation,
        "source": {
            "pts": pts,
            "time_base": "1/90000",
            "age_s": 0.02,
        },
        "detection": detection,
        "detail": "fresh camera evidence" if camera_healthy else "camera stale",
        "sampled_at_s": now_s,
    }


def test_search_identity_is_kept_when_approach_is_enabled() -> None:
    guidance = FruitGuidance("pear")

    first = guidance.observe(observation(pts=1, now_s=0.00), now_s=0.00)
    second = guidance.observe(observation(pts=2, now_s=0.15), now_s=0.15)
    locked = guidance.observe(observation(pts=3, now_s=0.30), now_s=0.30)

    assert first.action is GuidanceAction.HOLD
    assert second.action is GuidanceAction.HOLD
    assert locked.phase is GuidancePhase.LOCKED
    assert locked.command.forward_mps == 0.0
    assert locked.centered_fresh_samples == 3
    assert guidance.acquisition_epoch == 1

    moving = guidance.observe(
        observation(pts=4, now_s=0.45, center_x=0.61),
        now_s=0.45,
        allow_forward=True,
    )

    assert moving.phase is GuidancePhase.APPROACHING
    assert moving.action is GuidanceAction.DRIVE
    assert moving.command.forward_mps == 1.0
    assert moving.command.yaw_rps < 0.0
    assert guidance.acquisition_epoch == 1


def test_unselected_all_fruit_observations_cannot_lock_or_authorize_motion() -> None:
    guidance = FruitGuidance("pear")
    status = observation(pts=1, now_s=0.0, label=None)
    status["observations"] = {
        "apple": {
            "label": "apple",
            "confidence": 0.99,
            "center_x_ratio": 0.5,
            "source_pts": 1,
            "source_time_base": "1/90000",
            "generation": "camera-1",
        },
        "banana": {
            "label": "banana",
            "confidence": 0.99,
            "center_x_ratio": 0.5,
            "source_pts": 1,
            "source_time_base": "1/90000",
            "generation": "camera-1",
        },
    }

    decision = guidance.observe(status, now_s=0.0, allow_forward=True)

    assert decision.phase is GuidancePhase.SEARCHING
    assert decision.action is GuidanceAction.SEARCH
    assert decision.command.forward_mps == 0.0
    assert decision.centered_fresh_samples == 0

@pytest.mark.parametrize("source_fps", [4, 5, 8])
def test_duplicate_frames_hold_authorized_motion_without_advancing_counters(
    source_fps: int,
) -> None:
    guidance = FruitGuidance("pear")
    source_period = 1.0 / source_fps
    now_s = 0.0
    pts = 0
    fresh_decisions = 0
    duplicate_decisions = 0

    # A 10 Hz controller consumes a 4-8 FPS camera without injecting zero-speed
    # commands between fresh frames.
    for tick in range(20):
        now_s = tick * 0.1
        expected_pts = int(now_s / source_period)
        if expected_pts != pts or tick == 0:
            pts = expected_pts
            decision = guidance.observe(
                observation(pts=pts, now_s=now_s),
                now_s=now_s,
                allow_forward=True,
            )
            fresh_decisions += 1
        else:
            before = decision.centered_fresh_samples
            decision = guidance.observe(
                observation(pts=pts, now_s=now_s),
                now_s=now_s,
                allow_forward=True,
            )
            duplicate_decisions += 1
            assert decision.action is GuidanceAction.HOLD
            assert decision.centered_fresh_samples == before

        if guidance.phase is GuidancePhase.APPROACHING:
            assert decision.command.forward_mps == 1.0

    assert fresh_decisions >= 8
    assert duplicate_decisions >= 4
    assert guidance.acquisition_epoch == 1


def test_duplicate_hold_expires_at_250_ms_and_stops() -> None:
    config = GuidanceConfig(duplicate_hold_s=0.25)
    guidance = FruitGuidance("pear", config=config)
    for pts, now_s in ((1, 0.00), (2, 0.10), (3, 0.20), (4, 0.30)):
        decision = guidance.observe(
            observation(pts=pts, now_s=now_s),
            now_s=now_s,
            allow_forward=True,
        )
    assert decision.command.forward_mps == 1.0

    held = guidance.observe(
        observation(pts=4, now_s=0.50),
        now_s=0.50,
        allow_forward=True,
    )
    expired = guidance.observe(
        observation(pts=4, now_s=0.56),
        now_s=0.56,
        allow_forward=True,
    )

    assert held.action is GuidanceAction.HOLD
    assert held.command.forward_mps == 1.0
    assert expired.action is GuidanceAction.STOP
    assert expired.terminal is True
    assert expired.reason == "duplicate_frame_expired"


def test_outer_corridor_removes_forward_authority_and_recenters() -> None:
    guidance = FruitGuidance("apple")
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(
            observation(
                pts=pts,
                now_s=now_s,
                label="apple",
                confidence=0.75,
            ),
            now_s=now_s,
        )

    decision = guidance.observe(
        observation(
            pts=4,
            now_s=0.3,
            label="apple",
            confidence=0.20,
            center_x=0.75,
        ),
        now_s=0.3,
        allow_forward=True,
    )

    assert decision.action is GuidanceAction.ALIGN
    assert decision.command.forward_mps == 0.0
    assert decision.command.yaw_rps < 0.0
    assert decision.reason == "target_outside_outer_corridor"


@pytest.mark.parametrize(
    ("fruit", "below", "accepted"),
    [
        ("banana", 0.19, 0.20),
        ("pear", 0.64, 0.65),
    ],
)
def test_acquisition_uses_the_existing_per_fruit_confidence_policy(
    fruit: str,
    below: float,
    accepted: float,
) -> None:
    guidance = FruitGuidance(fruit)

    rejected = guidance.observe(
        observation(pts=1, now_s=0.0, label=fruit, confidence=below),
        now_s=0.0,
    )
    first = guidance.observe(
        observation(pts=2, now_s=0.1, label=fruit, confidence=accepted),
        now_s=0.1,
    )

    assert rejected.action is GuidanceAction.SEARCH
    assert rejected.centered_fresh_samples == 0
    assert first.centered_fresh_samples == 1
    assert guidance.policy.acquisition_confidence == accepted


def test_physical_banana_search_replay_retains_fine_focus_through_brief_misses() -> None:
    """Replay 689e9005: do not resume the broad sweep between sightings."""
    guidance = FruitGuidance(
        "banana",
        config=GuidanceConfig(
            focus_yaw_rps=0.50,
            focus_missing_grace_s=0.50,
        ),
    )

    focused = guidance.observe(
        observation(
            pts=93,
            now_s=10.7699,
            label="banana",
            confidence=0.5893887,
            center_x=0.02265625,
        ),
        now_s=10.7699,
    )
    missing = guidance.observe(
        observation(pts=94, now_s=10.98, label=None),
        now_s=10.98,
    )

    assert focused.action is GuidanceAction.ALIGN
    assert focused.command.yaw_rps == 0.50
    assert focused.reason == "focus_align_target"
    assert focused.focus_active is True
    assert focused.focus_direction == 1
    assert missing.action is GuidanceAction.ALIGN
    assert missing.command.yaw_rps == 0.50
    assert missing.reason == "focus_missing_grace"
    assert missing.centered_fresh_samples == 0
    assert missing.focus_active is True

    expired = guidance.observe(
        observation(pts=95, now_s=11.27, label=None),
        now_s=11.27,
    )
    assert expired.action is GuidanceAction.SEARCH
    assert expired.command.yaw_rps == 0.50
    assert expired.reason == "focus_missing_grace_expired"
    assert expired.focus_active is False


def test_progressive_focus_yaw_slows_on_each_fresh_qualified_pass() -> None:
    guidance = FruitGuidance(
        "pear",
        config=GuidanceConfig(
            search_yaw_rps=0.80,
            focus_yaw_rps=0.80,
            focus_yaw_step_rps=0.10,
            focus_minimum_yaw_rps=0.50,
            focus_near_center_ratio=0.055,
        ),
    )

    passes = [
        guidance.observe(
            observation(
                pts=pts,
                now_s=now_s,
                confidence=0.80,
                center_x=center_x,
            ),
            now_s=now_s,
        )
        for pts, now_s, center_x in (
            (1, 0.0, 0.64),
            (2, 0.1, 0.60),
            (3, 0.2, 0.56),
            (4, 0.3, 0.56),
            (5, 0.4, 0.56),
        )
    ]

    assert [decision.action for decision in passes] == [
        GuidanceAction.ALIGN,
        GuidanceAction.ALIGN,
        GuidanceAction.ALIGN,
        GuidanceAction.ALIGN,
        GuidanceAction.ALIGN,
    ]
    assert [decision.command.yaw_rps for decision in passes] == [
        -0.80,
        -0.70,
        -0.60,
        -0.50,
        -0.50,
    ]
    assert all(decision.command.forward_mps == 0.0 for decision in passes)

    first_centered = guidance.observe(
        observation(pts=6, now_s=0.5, confidence=0.80, center_x=0.55),
        now_s=0.5,
    )
    assert first_centered.action is GuidanceAction.HOLD
    assert first_centered.command.yaw_rps == 0.0
    assert first_centered.centered_fresh_samples == 1


def test_near_center_focus_uses_minimum_yaw_then_waits_for_fresh_settle() -> None:
    guidance = FruitGuidance(
        "pear",
        config=GuidanceConfig(
            search_yaw_rps=0.80,
            focus_yaw_rps=0.80,
            focus_minimum_yaw_rps=0.50,
            focus_near_center_ratio=0.20,
        ),
    )

    corrective_pulse = guidance.observe(
        observation(pts=1, now_s=0.0, center_x=0.68),
        now_s=0.0,
    )
    settle = guidance.observe(
        observation(pts=2, now_s=0.1, center_x=0.85),
        now_s=0.1,
    )
    far_correction = guidance.observe(
        observation(pts=3, now_s=0.2, center_x=0.84),
        now_s=0.2,
    )

    assert corrective_pulse.action is GuidanceAction.ALIGN
    assert corrective_pulse.command.yaw_rps == -0.50
    assert corrective_pulse.reason == "focus_align_target_near_center"
    assert settle.action is GuidanceAction.HOLD
    assert settle.command.yaw_rps == 0.0
    assert settle.reason == "focus_near_center_settle"
    assert far_correction.action is GuidanceAction.ALIGN
    assert far_correction.command.yaw_rps == -0.70
    assert all(
        decision.command.forward_mps == 0.0
        for decision in (corrective_pulse, settle, far_correction)
    )


def test_progressive_focus_yaw_resets_to_full_rate_after_crossing_center() -> None:
    guidance = FruitGuidance("pear")
    guidance.observe(
        observation(pts=1, now_s=0.0, center_x=0.64),
        now_s=0.0,
    )
    guidance.observe(
        observation(pts=2, now_s=0.1, center_x=0.60),
        now_s=0.1,
    )

    crossed = guidance.observe(
        observation(pts=3, now_s=0.2, center_x=0.44),
        now_s=0.2,
    )

    assert crossed.action is GuidanceAction.ALIGN
    assert crossed.command.yaw_rps == 0.50


def test_legacy_focus_profile_is_a_runtime_rollback() -> None:
    guidance = FruitGuidance(
        "pear",
        config=GuidanceConfig(
            progressive_focus_yaw_enabled=False,
            center_tolerance_ratio=0.08,
        ),
    )

    first = guidance.observe(
        observation(pts=1, now_s=0.0, center_x=0.64),
        now_s=0.0,
    )
    second = guidance.observe(
        observation(pts=2, now_s=0.1, center_x=0.60),
        now_s=0.1,
    )
    centered = guidance.observe(
        observation(pts=3, now_s=0.2, center_x=0.575),
        now_s=0.2,
    )

    assert first.command.yaw_rps == -0.50
    assert second.command.yaw_rps == -0.50
    assert centered.action is GuidanceAction.HOLD
    assert centered.centered_fresh_samples == 1


def test_apple_high_confidence_candidate_holds_then_sustained_tracking_locks() -> None:
    guidance = FruitGuidance("apple")

    sweeping = guidance.observe(
        observation(pts=1, now_s=0.0, label="apple", confidence=0.39),
        now_s=0.0,
    )
    focused = guidance.observe(
        observation(pts=2, now_s=0.1, label="apple", confidence=0.40),
        now_s=0.1,
    )
    confirming = guidance.observe(
        observation(pts=3, now_s=0.2, label="apple", confidence=0.41),
        now_s=0.2,
    )
    locked = guidance.observe(
        observation(pts=4, now_s=0.3, label="apple", confidence=0.40),
        now_s=0.3,
    )

    assert sweeping.action is GuidanceAction.SEARCH
    assert sweeping.command.yaw_rps == 0.50
    assert focused.action is GuidanceAction.HOLD
    assert focused.command.forward_mps == 0.0
    assert focused.command.yaw_rps == 0.0
    assert focused.reason == "apple_candidate_focus_started"
    assert confirming.action is GuidanceAction.HOLD
    assert locked.phase is GuidancePhase.LOCKED
    assert locked.reason == "target_identity_locked"
    assert guidance.acquisition_epoch == 1


def test_apple_focus_resets_confirmation_below_40_without_resuming_sweep() -> None:
    guidance = FruitGuidance("apple")

    guidance.observe(
        observation(pts=1, now_s=0.0, label="apple", confidence=0.55),
        now_s=0.0,
    )
    guidance.observe(
        observation(pts=2, now_s=0.1, label="apple", confidence=0.42),
        now_s=0.1,
    )
    weak = guidance.observe(
        observation(pts=3, now_s=0.2, label="apple", confidence=0.39),
        now_s=0.2,
    )

    assert weak.action is GuidanceAction.HOLD
    assert weak.command.forward_mps == 0.0
    assert weak.command.yaw_rps == 0.0
    assert weak.reason == "apple_candidate_focus_below_acquisition"
    assert weak.centered_fresh_samples == 0
    assert guidance.acquisition_epoch == 0


def test_apple_focus_holds_through_one_missing_observation() -> None:
    guidance = FruitGuidance("apple")
    guidance.observe(
        observation(pts=1, now_s=0.0, label="apple", confidence=0.55),
        now_s=0.0,
    )

    missing = guidance.observe(
        observation(pts=2, now_s=0.1, label=None),
        now_s=0.1,
    )

    assert missing.action is GuidanceAction.HOLD
    assert missing.command.forward_mps == 0.0
    assert missing.command.yaw_rps == 0.0
    assert missing.reason == "apple_candidate_focus_missing"
    assert guidance.acquisition_epoch == 0


def test_apple_focus_and_acquisition_thresholds_are_runtime_tunable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BORDER_COLLIE_APPLE_FOCUS_CONFIDENCE", "0.56")
    monkeypatch.setenv("BORDER_COLLIE_APPLE_ACQUISITION_CONFIDENCE", "0.44")
    guidance = FruitGuidance("apple")

    still_sweeping = guidance.observe(
        observation(pts=1, now_s=0.0, label="apple", confidence=0.55),
        now_s=0.0,
    )
    focused = guidance.observe(
        observation(pts=2, now_s=0.1, label="apple", confidence=0.56),
        now_s=0.1,
    )

    assert guidance.policy.focus_confidence == 0.56
    assert guidance.policy.acquisition_confidence == 0.44
    assert still_sweeping.action is GuidanceAction.SEARCH
    assert focused.action is GuidanceAction.HOLD


def test_lower_edge_disappearance_arrives_stopped_without_final_push() -> None:
    config = GuidanceConfig(final_push_mps=0.6, final_push_duration_s=1.0)
    guidance = FruitGuidance("banana", config=config)
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(observation(pts=pts, now_s=now_s, label="banana"), now_s=now_s)

    near = guidance.observe(
        observation(
            pts=4,
            now_s=0.3,
            label="banana",
            center_y=0.80,
            bottom=0.92,
        ),
        now_s=0.3,
        allow_forward=True,
    )
    assert near.action is GuidanceAction.ARRIVED
    assert near.reason == "first_qualified_lower_edge_arrival"

    pending = guidance.observe(
        observation(pts=7, now_s=0.6, label=None),
        now_s=0.6,
        allow_forward=True,
    )
    still_arrived = guidance.observe(
        observation(pts=8, now_s=0.7, label=None),
        now_s=0.7,
        allow_forward=True,
    )

    assert pending.action is GuidanceAction.ARRIVED
    assert pending.terminal is True
    assert pending.command.forward_mps == 0.0
    assert pending.command.yaw_rps == 0.0
    assert still_arrived.action is GuidanceAction.ARRIVED
    assert guidance.final_push_count == 0


@pytest.mark.parametrize(
    ("fruit", "weak_confidence"),
    [("apple", 0.03), ("banana", 0.10), ("pear", 0.46)],
)
def test_any_fruit_lower_edge_identity_collapse_arrives_without_three_samples(
    fruit: str,
    weak_confidence: float,
) -> None:
    """Replay the Apple/Pear closeout failures as the shared stage contract."""
    guidance = FruitGuidance(fruit)
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(observation(pts=pts, now_s=now_s, label=fruit), now_s=now_s)

    lower_edge = guidance.observe(
        observation(
            pts=4,
            now_s=0.3,
            label=fruit,
            confidence=0.763481,
            center_x=0.5875,
            center_y=0.84375,
            bottom=0.9027778,
        ),
        now_s=0.3,
        allow_forward=True,
    )
    still_arrived = guidance.observe(
        observation(
            pts=5,
            now_s=0.4,
            label=fruit,
            confidence=weak_confidence,
            center_x=0.610156,
            center_y=0.918056,
            bottom=0.969444,
        ),
        now_s=0.4,
        allow_forward=True,
    )
    assert lower_edge.action is GuidanceAction.ARRIVED
    assert lower_edge.near_fresh_samples == 0
    assert lower_edge.reason == "first_qualified_lower_edge_arrival"
    assert lower_edge.arrival_confirmed is True
    assert lower_edge.command.forward_mps == 0.0
    assert lower_edge.command.yaw_rps == 0.0
    assert still_arrived.action is GuidanceAction.ARRIVED
    assert guidance.final_push_count == 0


def test_disappearance_after_forward_approach_stops_without_lower_edge_frame() -> None:
    guidance = FruitGuidance("pear")
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(observation(pts=pts, now_s=now_s), now_s=now_s)

    guidance.observe(
        observation(
            pts=4,
            now_s=0.3,
            confidence=0.80,
            center_x=0.5734375,
            center_y=0.60,
            bottom=0.70,
        ),
        now_s=0.3,
        allow_forward=True,
    )
    missing = guidance.observe(
        observation(pts=5, now_s=0.4, label=None),
        now_s=0.4,
        allow_forward=True,
    )

    assert missing.action is GuidanceAction.STOP
    assert missing.reason == "target_missing_after_lock"
    assert missing.arrival_confirmed is False
    assert missing.command.forward_mps == 0.0
    assert missing.command.yaw_rps == 0.0


def test_banana_disappearance_immediately_after_80_percent_bottom_arrives() -> None:
    """Replay run 0022e9c0's final 85.1% -> missing frame transition."""
    guidance = FruitGuidance(
        "banana",
        config=GuidanceConfig(
            near_bottom_ratio=0.90,
            disappearance_bottom_ratio=0.80,
        ),
    )
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(
            observation(pts=pts, now_s=now_s, label="banana"),
            now_s=now_s,
        )

    approaching = guidance.observe(
        observation(
            pts=909600,
            now_s=0.3,
            label="banana",
            confidence=0.6629605293273926,
            center_x=0.50,
            center_y=0.8263888888888888,
            bottom=0.8513888888888889,
        ),
        now_s=0.3,
        allow_forward=True,
    )
    arrived = guidance.observe(
        observation(pts=909720, now_s=0.434, label=None),
        now_s=0.434,
        allow_forward=True,
    )

    assert approaching.action is GuidanceAction.DRIVE
    assert approaching.arrival_confirmed is False
    assert arrived.action is GuidanceAction.ARRIVED
    assert arrived.reason == "qualified_lower_edge_disappearance_arrival"
    assert arrived.arrival_confirmed is True
    assert arrived.command.forward_mps == 0.0
    assert arrived.command.yaw_rps == 0.0


def test_disappearance_uses_only_the_immediately_previous_qualified_frame() -> None:
    guidance = FruitGuidance(
        "banana",
        config=GuidanceConfig(
            near_bottom_ratio=0.90,
            disappearance_bottom_ratio=0.80,
        ),
    )
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(
            observation(pts=pts, now_s=now_s, label="banana"),
            now_s=now_s,
        )

    guidance.observe(
        observation(
            pts=4,
            now_s=0.3,
            label="banana",
            center_y=0.82,
            bottom=0.85,
        ),
        now_s=0.3,
        allow_forward=True,
    )
    guidance.observe(
        observation(
            pts=5,
            now_s=0.4,
            label="banana",
            center_y=0.78,
            bottom=0.79,
        ),
        now_s=0.4,
        allow_forward=True,
    )
    missing = guidance.observe(
        observation(pts=6, now_s=0.5, label=None),
        now_s=0.5,
        allow_forward=True,
    )

    assert missing.action is GuidanceAction.STOP
    assert missing.reason == "target_missing_after_lock"
    assert missing.arrival_confirmed is False


def test_disappearance_before_forward_approach_does_not_assume_arrival() -> None:
    guidance = FruitGuidance("pear")
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(observation(pts=pts, now_s=now_s), now_s=now_s)

    missing = guidance.observe(
        observation(pts=4, now_s=0.3, label=None),
        now_s=0.3,
        allow_forward=True,
    )

    assert missing.action is GuidanceAction.STOP
    assert missing.reason == "target_missing_after_lock"
    assert missing.arrival_confirmed is False


def test_stationary_bottom_jump_is_rejected_then_real_banana_resumes() -> None:
    """Replay run 331cb167: lock, 55 misses, false bottom box, real target."""
    guidance = FruitGuidance("banana")
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(
            observation(
                pts=pts,
                now_s=now_s,
                label="banana",
                confidence=0.72,
                center_x=0.45,
                center_y=0.60,
                bottom=0.70,
            ),
            now_s=now_s,
        )

    for pts in range(4, 59):
        missing = guidance.observe(
            observation(pts=pts, now_s=pts / 10, label=None),
            now_s=pts / 10,
            allow_forward=True,
        )
        assert missing.action is GuidanceAction.STOP
        assert missing.terminal is False

    false_bottom = guidance.observe(
        observation(
            pts=59,
            now_s=5.9,
            label="banana",
            confidence=0.5610339641571045,
            center_x=0.48828125,
            center_y=0.9666666666666667,
            bottom=0.9986111111111111,
        ),
        now_s=5.9,
        allow_forward=True,
    )
    real_first = guidance.observe(
        observation(
            pts=60,
            now_s=6.0,
            label="banana",
            confidence=0.70,
            center_x=0.49,
            center_y=0.68,
            bottom=0.76,
        ),
        now_s=6.0,
        allow_forward=True,
    )
    real_second = guidance.observe(
        observation(
            pts=61,
            now_s=6.1,
            label="banana",
            confidence=0.73,
            center_x=0.50,
            center_y=0.69,
            bottom=0.77,
        ),
        now_s=6.1,
        allow_forward=True,
    )

    assert false_bottom.action is GuidanceAction.STOP
    assert false_bottom.reason == "stationary_lower_edge_jump_rejected"
    assert false_bottom.arrival_confirmed is False
    assert false_bottom.command.forward_mps == 0.0
    assert false_bottom.command.yaw_rps == 0.0
    assert real_first.action is GuidanceAction.STOP
    assert real_first.reason == "stationary_reacquisition_confirmation_pending"
    assert real_second.action is GuidanceAction.DRIVE
    assert real_second.command.forward_mps == 1.0
    assert guidance.acquisition_epoch == 1


def test_disappearance_after_sub_disappearance_threshold_frame_stops() -> None:
    guidance = FruitGuidance("banana")
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(
            observation(pts=pts, now_s=now_s, label="banana"),
            now_s=now_s,
        )

    moving = guidance.observe(
        observation(
            pts=4,
            now_s=0.3,
            label="banana",
            confidence=0.70,
            center_x=0.50,
            center_y=0.72,
            bottom=0.799,
        ),
        now_s=0.3,
        allow_forward=True,
    )
    missing = guidance.observe(
        observation(pts=5, now_s=0.4, label=None),
        now_s=0.4,
        allow_forward=True,
    )

    assert moving.action is GuidanceAction.DRIVE
    assert missing.action is GuidanceAction.STOP
    assert missing.reason == "target_missing_after_lock"
    assert missing.arrival_confirmed is False


def test_zero_duration_disables_final_push_and_arrival_remains_stopped() -> None:
    guidance = FruitGuidance(
        "banana",
        config=GuidanceConfig(final_push_mps=0.60, final_push_duration_s=0.0),
    )
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(observation(pts=pts, now_s=now_s, label="banana"), now_s=now_s)
    arrived = guidance.observe(
        observation(
            pts=4,
            now_s=0.3,
            label="banana",
            center_y=0.80,
            bottom=0.92,
        ),
        now_s=0.3,
        allow_forward=True,
    )
    still_arrived = guidance.observe(
        observation(pts=7, now_s=0.6, label=None),
        now_s=0.6,
        allow_forward=True,
    )
    assert arrived.action is GuidanceAction.ARRIVED
    assert arrived.reason == "first_qualified_lower_edge_arrival"
    assert arrived.command.forward_mps == 0.0
    assert arrived.arrival_confirmed is True
    assert still_arrived.action is GuidanceAction.ARRIVED
    assert guidance.final_push_count == 0


def test_detection_past_slow_inference_grace_fails_without_motion_authority() -> None:
    guidance = FruitGuidance("pear")
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2), (4, 0.3)):
        moving = guidance.observe(
            observation(pts=pts, now_s=now_s),
            now_s=now_s,
            allow_forward=True,
        )
    assert moving.command.forward_mps == 1.0

    stale = observation(pts=5, now_s=0.4)
    stale["detection"]["age_s"] = 0.50
    stopped = guidance.observe(stale, now_s=0.4, allow_forward=True)

    assert stopped.action is GuidanceAction.STOP
    assert stopped.command.forward_mps == 0.0
    assert stopped.terminal is True
    assert stopped.reason == "detection_stale"


def test_physical_banana_slow_inference_replay_stops_then_resumes_fresh_motion() -> None:
    """Replay af45a566: 267.8 ms evidence is safe to stop, not fail."""
    guidance = FruitGuidance(
        "banana",
        config=GuidanceConfig(slow_inference_grace_s=0.50),
    )
    for pts, now_s in ((101, 0.0), (102, 0.1), (103, 0.2), (104, 0.3)):
        moving = guidance.observe(
            observation(
                pts=pts,
                now_s=now_s,
                label="banana",
                confidence=0.788,
            ),
            now_s=now_s,
            allow_forward=True,
        )
    assert moving.action is GuidanceAction.DRIVE
    before = moving.near_fresh_samples

    delayed = observation(
        pts=105,
        now_s=0.4,
        label="banana",
        confidence=0.788,
    )
    delayed["detection"]["source_pts"] = 103
    delayed["detection"]["age_s"] = 0.2678
    waiting = guidance.observe(delayed, now_s=0.4, allow_forward=True)

    assert waiting.action is GuidanceAction.STOP
    assert waiting.command.forward_mps == 0.0
    assert waiting.command.yaw_rps == 0.0
    assert waiting.reason == "slow_inference_grace"
    assert waiting.terminal is False
    assert waiting.near_fresh_samples == before

    resumed = guidance.observe(
        observation(
            pts=106,
            now_s=0.5,
            label="banana",
            confidence=0.781,
        ),
        now_s=0.5,
        allow_forward=True,
    )
    assert resumed.action is GuidanceAction.DRIVE
    assert resumed.command.forward_mps == 1.0


def test_slow_inference_grace_expires_terminally_at_configured_deadline() -> None:
    guidance = FruitGuidance(
        "banana",
        config=GuidanceConfig(slow_inference_grace_s=0.50),
    )
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(
            observation(pts=pts, now_s=now_s, label="banana"),
            now_s=now_s,
        )

    stale = observation(pts=4, now_s=0.3, label="banana")
    stale["detection"]["source_pts"] = 2
    stale["detection"]["age_s"] = 0.50
    failed = guidance.observe(stale, now_s=0.3, allow_forward=True)

    assert failed.action is GuidanceAction.STOP
    assert failed.terminal is True
    assert failed.camera_failure is True
    assert failed.reason == "detection_stale"


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda value: value.update(camera_healthy=False), "camera_unhealthy"),
        (
            lambda value: value.update(generation="camera-2"),
            "camera_generation_changed",
        ),
        (
            lambda value: value["detection"].update(label="banana"),
            "target_identity_changed",
        ),
        (
            lambda value: value["detection"].update(center_x_ratio=1.5),
            "invalid_target_geometry",
        ),
    ],
)
def test_locked_track_safety_replays_stop_immediately(mutate, reason: str) -> None:
    guidance = FruitGuidance("pear")
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(observation(pts=pts, now_s=now_s), now_s=now_s)

    unsafe = observation(pts=4, now_s=0.3)
    mutate(unsafe)
    decision = guidance.observe(unsafe, now_s=0.3, allow_forward=True)

    assert decision.action is GuidanceAction.STOP
    assert decision.command.forward_mps == 0.0
    assert decision.command.yaw_rps == 0.0
    assert decision.terminal is True
    assert decision.reason == reason


def test_guidance_env_defaults_match_the_canonical_deployment_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in tuple(__import__("os").environ):
        if name.startswith("BORDER_COLLIE_GUIDANCE_"):
            monkeypatch.delenv(name)

    config = GuidanceConfig.from_env()

    assert config.search_yaw_rps == 0.5
    assert config.focus_yaw_rps == 0.5
    assert config.progressive_focus_yaw_enabled is True
    assert config.focus_yaw_step_rps == 0.10
    assert config.focus_minimum_yaw_rps == 0.50
    assert config.focus_missing_grace_s == 0.5
    assert config.center_tolerance_ratio == 0.05
    assert config.center_confirmations == 3
    assert config.approach_forward_mps == 1.0
    assert config.outer_corridor_ratio == 0.20
    assert config.detection_maximum_age_s == 0.25
    assert config.slow_inference_grace_s == 0.5
    assert config.final_push_mps == 0.6
    assert config.final_push_duration_s == 1.0
    assert config.near_loss_confirmations == 2


def test_search_yaw_accepts_bounded_physical_qualification_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BORDER_COLLIE_GUIDANCE_SEARCH_YAW_RPS", "0.50")

    config = GuidanceConfig.from_env()

    assert config.search_yaw_rps == 0.50


def test_one_weak_close_frame_declares_stopped_arrival() -> None:
    guidance = FruitGuidance("pear")
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(observation(pts=pts, now_s=now_s), now_s=now_s)
    arrived = guidance.observe(
        observation(pts=4, now_s=0.3, center_y=0.8, bottom=0.92),
        now_s=0.3,
        allow_forward=True,
    )

    pending = guidance.observe(
        observation(pts=7, now_s=0.6, confidence=0.40, center_y=0.8, bottom=0.94),
        now_s=0.6,
        allow_forward=True,
    )
    recovered = guidance.observe(
        observation(pts=8, now_s=0.7, confidence=0.70, center_y=0.8, bottom=0.94),
        now_s=0.7,
        allow_forward=True,
    )

    assert arrived.action is GuidanceAction.ARRIVED
    assert arrived.reason == "first_qualified_lower_edge_arrival"
    assert pending.action is GuidanceAction.ARRIVED
    assert pending.command.forward_mps == 0.0
    assert guidance.final_push_count == 0
    assert recovered.action is GuidanceAction.ARRIVED


@pytest.mark.parametrize("fruit", ["apple", "banana", "pear"])
def test_first_lower_edge_frame_ignores_tracking_confidence_after_lock(
    fruit: str,
) -> None:
    """Replay Pear run 7e7afe7a at the shared fruit-guidance seam."""
    guidance = FruitGuidance(fruit)
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(
            observation(pts=pts, now_s=now_s, label=fruit),
            now_s=now_s,
        )

    arrived = guidance.observe(
        observation(
            pts=4,
            now_s=0.3,
            label=fruit,
            confidence=0.4707366,
            center_x=0.6046875,
            center_y=0.8833333,
            bottom=0.9444444,
        ),
        now_s=0.3,
        allow_forward=True,
    )

    assert arrived.action is GuidanceAction.ARRIVED
    assert arrived.reason == "first_qualified_lower_edge_arrival"
    assert arrived.arrival_confirmed is True
    assert arrived.command.forward_mps == 0.0
    assert arrived.command.yaw_rps == 0.0


def test_low_confidence_stop_preserves_the_exact_decision_reason() -> None:
    guidance = FruitGuidance("pear")
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(observation(pts=pts, now_s=now_s), now_s=now_s)

    stopped = guidance.observe(
        observation(pts=4, now_s=0.3, confidence=0.54),
        now_s=0.3,
        allow_forward=True,
    )

    assert stopped.action is GuidanceAction.STOP
    assert stopped.reason == "tracking_confidence_below_floor"
    assert stopped.command.to_dict() == {
        "forward_mps": 0.0,
        "yaw_rps": 0.0,
        "reason": "tracking_confidence_below_floor",
    }
    assert stopped.near_fresh_samples == 0


def test_missing_target_stop_cannot_advance_arrival_counters() -> None:
    guidance = FruitGuidance("pear")
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(observation(pts=pts, now_s=now_s), now_s=now_s)

    stopped = guidance.observe(
        observation(pts=4, now_s=0.3, label=None),
        now_s=0.3,
        allow_forward=True,
    )

    assert stopped.action is GuidanceAction.STOP
    assert stopped.reason == "target_missing_after_lock"
    assert stopped.command.reason == "target_missing_after_lock"
    assert stopped.centered_fresh_samples == 3
    assert stopped.near_fresh_samples == 0
    assert stopped.near_loss_samples == 0
    assert stopped.arrival_eligible is False
