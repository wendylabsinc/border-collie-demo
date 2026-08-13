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
        ("apple", 0.69, 0.70),
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


def test_lower_edge_disappearance_allows_exactly_one_bounded_final_push() -> None:
    config = GuidanceConfig(final_push_mps=0.6, final_push_duration_s=1.0)
    guidance = FruitGuidance("pear", config=config)
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(observation(pts=pts, now_s=now_s), now_s=now_s)

    for pts, now_s in ((4, 0.3), (5, 0.4), (6, 0.5)):
        near = guidance.observe(
            observation(
                pts=pts,
                now_s=now_s,
                center_y=0.80,
                bottom=0.92,
            ),
            now_s=now_s,
            allow_forward=True,
        )
    assert near.near_fresh_samples == 3

    pending = guidance.observe(
        observation(pts=7, now_s=0.6, label=None),
        now_s=0.6,
        allow_forward=True,
    )
    push = guidance.observe(
        observation(pts=8, now_s=0.7, label=None),
        now_s=0.7,
        allow_forward=True,
    )
    during = guidance.observe(
        observation(pts=9, now_s=1.1, label=None),
        now_s=1.1,
        allow_forward=True,
    )
    arrived = guidance.observe(
        observation(pts=10, now_s=1.7, label=None),
        now_s=1.7,
        allow_forward=True,
    )
    still_arrived = guidance.observe(
        observation(pts=11, now_s=1.8, label=None),
        now_s=1.8,
        allow_forward=True,
    )

    assert pending.action is GuidanceAction.STOP
    assert pending.terminal is False
    assert push.action is GuidanceAction.FINAL_PUSH
    assert push.command.forward_mps == 0.6
    assert during.action is GuidanceAction.FINAL_PUSH
    assert arrived.action is GuidanceAction.ARRIVED
    assert arrived.command.forward_mps == 0.0
    assert still_arrived.action is GuidanceAction.ARRIVED
    assert guidance.final_push_count == 1


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

    assert config.search_yaw_rps == 0.5
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


def test_one_weak_close_frame_does_not_start_final_push() -> None:
    guidance = FruitGuidance("pear")
    for pts, now_s in ((1, 0.0), (2, 0.1), (3, 0.2)):
        guidance.observe(observation(pts=pts, now_s=now_s), now_s=now_s)
    for pts, now_s in ((4, 0.3), (5, 0.4), (6, 0.5)):
        guidance.observe(
            observation(pts=pts, now_s=now_s, center_y=0.8, bottom=0.92),
            now_s=now_s,
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

    assert pending.action is GuidanceAction.STOP
    assert pending.reason == "lower_edge_loss_confirmation_pending"
    assert guidance.final_push_count == 0
    assert recovered.action is GuidanceAction.DRIVE
