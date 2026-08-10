"""SDK-neutral fencing and expiry for motion authority."""

from __future__ import annotations

import math
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass

from .models import VelocityCommand


class MotionAuthorityError(RuntimeError):
    pass


class MotionPermitMismatch(MotionAuthorityError):
    pass


class MotionAuthorityExpired(MotionAuthorityError):
    pass


@dataclass(frozen=True)
class MotionAuthority:
    run_id: str
    epoch: str
    operation: str
    allow_forward: bool
    maximum_forward_mps: float
    maximum_yaw_rps: float
    ttl_s: float

    def __post_init__(self) -> None:
        for name in ("run_id", "epoch", "operation"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        for name in ("maximum_forward_mps", "maximum_yaw_rps", "ttl_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


class MotionGuardian:
    """Issue one fenced permit and validate every command against it."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._generation = 0
        self._permit: str | None = None
        self._authority: MotionAuthority | None = None
        self._expires_at: float | None = None
        self._trip_reason: str | None = None

    def acquire(self, authority: MotionAuthority) -> str:
        if self._permit is not None:
            raise MotionAuthorityError("motion authority is already active")
        self._generation += 1
        self._authority = authority
        self._permit = f"{self._generation}.{secrets.token_urlsafe(32)}"
        self._expires_at = self._clock() + authority.ttl_s
        self._trip_reason = None
        return self._permit

    def authorize(self, permit: str, command: VelocityCommand) -> VelocityCommand:
        authority = self._require_permit(permit)
        forward = float(command.forward_mps)
        yaw = float(command.yaw_rps)
        if not math.isfinite(forward) or forward < 0.0:
            raise ValueError("forward speed must be finite and non-negative")
        if not math.isfinite(yaw):
            raise ValueError("yaw speed must be finite")
        if forward > 0.0 and not authority.allow_forward:
            raise MotionAuthorityError(
                f"forward motion is not authorized for {authority.operation}"
            )
        if forward > authority.maximum_forward_mps:
            raise ValueError("forward speed exceeds the configured limit for authority")
        if abs(yaw) > authority.maximum_yaw_rps:
            raise ValueError("yaw speed exceeds the configured limit for authority")
        self._expires_at = self._clock() + authority.ttl_s
        return VelocityCommand(forward, yaw, command.reason)

    def release(self, permit: str) -> None:
        self._require_permit(permit, allow_expired=True)
        self._clear()

    def trip(self, reason: str) -> None:
        self._generation += 1
        self._trip_reason = str(reason)
        self._permit = None
        self._authority = None
        self._expires_at = None

    def status(self) -> dict[str, object]:
        authority = self._authority
        expires_in = (
            None
            if self._expires_at is None
            else max(0.0, self._expires_at - self._clock())
        )
        return {
            "generation": self._generation,
            "active": self._permit is not None,
            "run_id": None if authority is None else authority.run_id,
            "epoch": None if authority is None else authority.epoch,
            "operation": None if authority is None else authority.operation,
            "allow_forward": None if authority is None else authority.allow_forward,
            "expires_in_s": expires_in,
            "trip_reason": self._trip_reason,
        }

    def _require_permit(
        self,
        permit: str,
        *,
        allow_expired: bool = False,
    ) -> MotionAuthority:
        if (
            self._permit is None
            or not isinstance(permit, str)
            or not secrets.compare_digest(permit, self._permit)
            or self._authority is None
        ):
            raise MotionPermitMismatch("motion permit is stale or does not match")
        assert self._expires_at is not None
        if not allow_expired and self._clock() > self._expires_at:
            self.trip("motion authority expired")
            raise MotionAuthorityExpired("motion authority expired")
        return self._authority

    def _clear(self) -> None:
        self._generation += 1
        self._permit = None
        self._authority = None
        self._expires_at = None
