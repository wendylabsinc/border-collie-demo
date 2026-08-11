from __future__ import annotations

import pytest

from border_collie_demo.models import VelocityCommand
from border_collie_demo.motion_guardian import (
    MotionAuthority,
    MotionAuthorityError,
    MotionGuardian,
)


def authority(*, forward: bool = True, ttl_s: float = 1.0) -> MotionAuthority:
    return MotionAuthority(
        run_id="run-123",
        epoch="epoch-7",
        operation="approach_target" if forward else "measured_turn",
        allow_forward=forward,
        maximum_forward_mps=1.0,
        maximum_yaw_rps=0.8,
        ttl_s=ttl_s,
    )


def test_permit_is_fenced_to_one_authority_generation() -> None:
    guardian = MotionGuardian()
    first = guardian.acquire(authority())
    guardian.release(first)
    second = guardian.acquire(authority())

    with pytest.raises(MotionAuthorityError, match="stale"):
        guardian.authorize(first, VelocityCommand(0.5, 0.0))

    assert guardian.authorize(second, VelocityCommand(0.5, 0.0)).forward_mps == 0.5


def test_turn_only_authority_rejects_forward_motion() -> None:
    guardian = MotionGuardian()
    permit = guardian.acquire(authority(forward=False))

    with pytest.raises(MotionAuthorityError, match="not authorized"):
        guardian.authorize(permit, VelocityCommand(0.01, 0.0))

    assert guardian.authorize(permit, VelocityCommand(0.0, 0.4)).yaw_rps == 0.4


def test_expired_authority_is_revoked() -> None:
    now = [10.0]
    guardian = MotionGuardian(clock=lambda: now[0])
    permit = guardian.acquire(authority(ttl_s=0.5))
    now[0] = 10.6

    with pytest.raises(MotionAuthorityError, match="expired"):
        guardian.authorize(permit, VelocityCommand(0.0, 0.1))

    assert guardian.status()["active"] is False
    assert guardian.status()["trip_reason"] == "motion authority expired"


def test_trip_revokes_the_current_permit_without_exposing_it_in_status() -> None:
    guardian = MotionGuardian()
    permit = guardian.acquire(authority())
    guardian.trip("emergency stop")

    with pytest.raises(MotionAuthorityError, match="stale"):
        guardian.authorize(permit, VelocityCommand())
    assert "permit" not in guardian.status()
    assert guardian.status()["trip_reason"] == "emergency stop"
