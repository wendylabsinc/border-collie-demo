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


def test_apple_high_confidence_candidate_holds_then_sustained_tracking_locks() -> None:
    guidance = FruitGuidance("apple")

    sweeping = guidance.observe(
        observation(pts=1, now_s=0.0, label="apple", confidence=0.49),
        now_s=0.0,
    )
    focused = guidance.observe(
        observation(pts=2, now_s=0.1, label="apple", confidence=0.50),
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
    assert sweeping.command.yaw_rps == 0.40
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
            center_x=0.50,
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


def test_disappearance_after_sub_threshold_bottom_frame_stops() -> None:
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
            bottom=0.899,
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


def test_stale_detection_replay_stops_without_reusing_motion_authority() -> None:
    guidance = FruitGuidance("pear")
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2), (4, 0.3)):
        moving = guidance.observe(
            observation(pts=pts, now_s=now_s),
            now_s=now_s,
            allow_forward=True,
        )
    assert moving.command.forward_mps == 1.0

    stale = observation(pts=5, now_s=0.4)
    stale["detection"]["age_s"] = 0.251
    stopped = guidance.observe(stale, now_s=0.4, allow_forward=True)

    assert stopped.action is GuidanceAction.STOP
    assert stopped.command.forward_mps == 0.0
    assert stopped.terminal is True
    assert stopped.reason == "detection_stale"


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


def test_guidance_env_defaults_match_the_physically_proven_base_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in tuple(__import__("os").environ):
        if name.startswith("BORDER_COLLIE_GUIDANCE_"):
            monkeypatch.delenv(name)

    config = GuidanceConfig.from_env()

    assert config.search_yaw_rps == 0.4
    assert config.center_tolerance_ratio == 0.08
    assert config.center_confirmations == 3
    assert config.approach_forward_mps == 1.0
    assert config.outer_corridor_ratio == 0.20
    assert config.final_push_mps == 0.6
    assert config.final_push_duration_s == 1.0
    assert config.near_loss_confirmations == 2


def test_search_yaw_accepts_bounded_physical_qualification_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BORDER_COLLIE_GUIDANCE_SEARCH_YAW_RPS", "0.40")

    config = GuidanceConfig.from_env()

    assert config.search_yaw_rps == 0.40


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
