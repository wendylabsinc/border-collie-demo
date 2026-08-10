from __future__ import annotations

import pytest

from media.service_supervision import (
    ServiceState,
    ServiceSupervisionConfig,
    ServiceSupervisor,
)


def config(**changes) -> ServiceSupervisionConfig:
    values = {
        "stable_frame_count": 3,
        "frame_stall_timeout_s": 0.5,
        "restart_budget": 3,
        "initial_backoff_s": 0.25,
        "maximum_backoff_s": 1.0,
    }
    values.update(changes)
    return ServiceSupervisionConfig(**values)


def test_two_data_channel_startup_failures_back_off_then_recover() -> None:
    supervisor = ServiceSupervisor(config())

    assert supervisor.begin_attempt(now_s=0.0) is True
    supervisor.session_started("generation-1", now_s=0.0)
    supervisor.session_failed(
        "DataChannelTimeoutError: data channel did not open",
        now_s=0.0,
    )
    assert supervisor.status(now_s=0.0)["next_retry_in_s"] == 0.25
    assert supervisor.begin_attempt(now_s=0.24) is False

    assert supervisor.begin_attempt(now_s=0.25) is True
    supervisor.session_started("generation-2", now_s=0.25)
    supervisor.session_failed(
        "DataChannelTimeoutError: data channel did not open",
        now_s=0.25,
    )
    assert supervisor.status(now_s=0.25)["next_retry_in_s"] == 0.5

    assert supervisor.begin_attempt(now_s=0.75) is True
    supervisor.session_started("generation-3", now_s=0.75)
    for index in range(3):
        supervisor.note_frame(
            "generation-3",
            100 + index,
            now_s=0.80 + index * 0.05,
        )

    status = supervisor.status(now_s=0.90)
    assert status["state"] == "ready"
    assert status["ready"] is True
    assert status["attempts"] == 3
    assert status["total_restarts"] == 2
    assert status["failures_since_ready"] == 0


def test_ready_session_degrades_and_requests_cleanup_on_frame_stall() -> None:
    supervisor = ServiceSupervisor(config(stable_frame_count=2))
    supervisor.begin_attempt(now_s=1.0)
    supervisor.session_started("generation-1", now_s=1.0)
    supervisor.note_frame("generation-1", 10, now_s=1.10)
    supervisor.note_frame("generation-1", 11, now_s=1.20)

    assert supervisor.check_health(now_s=1.69) is True
    assert supervisor.check_health(now_s=1.71) is False
    status = supervisor.status(now_s=1.71)
    assert status["state"] == "degraded"
    assert status["ready"] is False
    assert status["restart_required"] is True
    assert status["last_error"] == "media camera frame progress stalled"
    for pts in range(12, 20):
        assert supervisor.note_frame("generation-1", pts, now_s=1.72) is False
    assert supervisor.state is ServiceState.DEGRADED
    assert supervisor.ready is False


def test_generation_change_must_earn_readiness_again_and_ignores_stale_frames() -> None:
    supervisor = ServiceSupervisor(config(stable_frame_count=2))
    supervisor.begin_attempt(now_s=0.0)
    supervisor.session_started("generation-1", now_s=0.0)
    supervisor.note_frame("generation-1", 1, now_s=0.1)
    supervisor.note_frame("generation-1", 2, now_s=0.2)
    assert supervisor.ready is True

    supervisor.session_started("generation-2", now_s=0.3)
    assert supervisor.state is ServiceState.DEGRADED
    assert supervisor.note_frame("generation-1", 3, now_s=0.4) is False
    assert supervisor.status(now_s=0.4)["stable_frames"] == 0

    supervisor.note_frame("generation-2", 1, now_s=0.5)
    supervisor.note_frame("generation-2", 2, now_s=0.6)
    assert supervisor.ready is True
    assert supervisor.status(now_s=0.6)["generation"] == "generation-2"


def test_restart_budget_exhaustion_is_explicitly_terminal() -> None:
    supervisor = ServiceSupervisor(config(restart_budget=2))

    for attempt in range(3):
        assert supervisor.begin_attempt(now_s=float(attempt)) is True
        supervisor.session_started(f"generation-{attempt}", now_s=float(attempt))
        supervisor.session_failed("DataChannelTimeoutError", now_s=float(attempt))

    status = supervisor.status(now_s=3.0)
    assert status["state"] == "failed"
    assert status["ready"] is False
    assert status["restart_required"] is False
    assert status["failures_since_ready"] == 3
    assert supervisor.begin_attempt(now_s=100.0) is False


def test_healthy_steady_state_stays_ready_without_spending_restart_budget() -> None:
    supervisor = ServiceSupervisor(config(stable_frame_count=2))
    supervisor.begin_attempt(now_s=5.0)
    supervisor.session_started("generation-1", now_s=5.0)
    supervisor.note_frame("generation-1", 100, now_s=5.1)
    supervisor.note_frame("generation-1", 101, now_s=5.2)

    for pts, now in ((102, 5.4), (103, 5.6), (104, 5.8)):
        assert supervisor.check_health(now_s=now - 0.01) is True
        supervisor.note_frame("generation-1", pts, now_s=now)

    status = supervisor.status(now_s=6.0)
    assert status["state"] == "ready"
    assert status["total_restarts"] == 0
    assert status["last_error"] is None


def test_supervision_config_rejects_unbounded_or_invalid_values() -> None:
    with pytest.raises(ValueError, match="stable frame"):
        config(stable_frame_count=0)
    with pytest.raises(ValueError, match="restart budget"):
        config(restart_budget=-1)
    with pytest.raises(ValueError, match="maximum backoff"):
        config(initial_backoff_s=2.0, maximum_backoff_s=1.0)
