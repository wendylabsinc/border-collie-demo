"""SDK-neutral connection supervision for the media sidecar.

The state machine deliberately knows nothing about Unitree or WebRTC.  The
sidecar adapter owns those details and reports session starts, advancing
frames, and failures here.  Readiness is earned only by one stable frame
generation and is revoked before a reconnect is attempted.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class ServiceState(str, Enum):
    DEGRADED = "degraded"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class ServiceSupervisionConfig:
    stable_frame_count: int = 10
    frame_stall_timeout_s: float = 0.75
    restart_budget: int = 5
    initial_backoff_s: float = 0.5
    maximum_backoff_s: float = 8.0

    def __post_init__(self) -> None:
        if self.stable_frame_count < 1:
            raise ValueError("stable frame count must be positive")
        if self.restart_budget < 0:
            raise ValueError("restart budget must be non-negative")
        for name in (
            "frame_stall_timeout_s",
            "initial_backoff_s",
            "maximum_backoff_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_backoff_s < self.initial_backoff_s:
            raise ValueError("maximum backoff must not be below initial backoff")


class ServiceSupervisor:
    """Track one recoverable service without owning its SDK resources."""

    def __init__(
        self,
        config: ServiceSupervisionConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or ServiceSupervisionConfig()
        self._clock = clock
        self._state = ServiceState.DEGRADED
        self._generation: str | None = None
        self._attempts = 0
        self._total_restarts = 0
        self._failures_since_ready = 0
        self._stable_frames = 0
        self._last_pts: int | None = None
        self._session_started_s: float | None = None
        self._last_frame_s: float | None = None
        self._next_retry_s: float | None = None
        self._last_error: str | None = "media session has not connected"
        self._restart_required = False

    @property
    def state(self) -> ServiceState:
        return self._state

    @property
    def ready(self) -> bool:
        return self._state is ServiceState.READY

    @property
    def restart_required(self) -> bool:
        return self._restart_required

    @property
    def next_retry_s(self) -> float | None:
        return self._next_retry_s

    def begin_attempt(self, *, now_s: float | None = None) -> bool:
        now = self._now(now_s)
        if self._state is ServiceState.FAILED:
            return False
        if self._next_retry_s is not None and now < self._next_retry_s:
            return False
        self._attempts += 1
        self._state = ServiceState.DEGRADED
        # Fence callbacks from the old session before cleanup can yield.
        self._generation = None
        self._stable_frames = 0
        self._last_pts = None
        self._session_started_s = None
        self._last_frame_s = None
        self._restart_required = False
        self._next_retry_s = None
        self._last_error = "media session connection is in progress"
        return True

    def session_started(
        self,
        generation: str,
        *,
        now_s: float | None = None,
    ) -> None:
        normalized = str(generation).strip()
        if not normalized:
            raise ValueError("media session generation must be non-empty")
        now = self._now(now_s)
        self._generation = normalized
        self._state = ServiceState.DEGRADED
        self._stable_frames = 0
        self._last_pts = None
        self._session_started_s = now
        self._last_frame_s = None
        self._restart_required = False
        self._last_error = "waiting for a stable advancing frame generation"

    def note_frame(
        self,
        generation: str,
        pts: int,
        *,
        now_s: float | None = None,
    ) -> bool:
        if self._state is ServiceState.FAILED or self._restart_required:
            return False
        if generation != self._generation:
            # Delayed callbacks from a cleaned-up session must not revive it.
            return False
        if isinstance(pts, bool) or not isinstance(pts, int):
            raise TypeError("frame PTS must be an integer")
        now = self._now(now_s)
        advancing = self._last_pts is None or pts > self._last_pts
        timely = self._last_frame_s is None or (
            0.0 < now - self._last_frame_s <= self.config.frame_stall_timeout_s
        )
        self._stable_frames = self._stable_frames + 1 if advancing and timely else 1
        self._last_pts = pts
        self._last_frame_s = now
        if self._stable_frames >= self.config.stable_frame_count:
            self._state = ServiceState.READY
            self._failures_since_ready = 0
            self._last_error = None
        else:
            self._state = ServiceState.DEGRADED
            self._last_error = (
                "waiting for a stable advancing frame generation "
                f"({self._stable_frames}/{self.config.stable_frame_count})"
            )
        return self.ready

    def check_health(self, *, now_s: float | None = None) -> bool:
        """Revoke readiness and request cleanup when frame progress stalls."""
        if self._state is ServiceState.FAILED or self._restart_required:
            return False
        now = self._now(now_s)
        reference = self._last_frame_s or self._session_started_s
        if reference is None or now - reference <= self.config.frame_stall_timeout_s:
            return self.ready
        if self._last_frame_s is None:
            error = "media session produced no camera frames before the stall timeout"
        else:
            error = "media camera frame progress stalled"
        self.session_failed(error, now_s=now)
        return False

    def session_failed(
        self,
        error: object,
        *,
        now_s: float | None = None,
    ) -> None:
        now = self._now(now_s)
        self._stable_frames = 0
        self._last_pts = None
        self._session_started_s = None
        self._last_frame_s = None
        self._failures_since_ready += 1
        self._last_error = str(error).strip() or type(error).__name__
        if self._failures_since_ready > self.config.restart_budget:
            self._state = ServiceState.FAILED
            self._restart_required = False
            self._next_retry_s = None
            return
        delay_s = min(
            self.config.maximum_backoff_s,
            self.config.initial_backoff_s
            * (2 ** max(0, self._failures_since_ready - 1)),
        )
        self._state = ServiceState.DEGRADED
        self._restart_required = True
        self._total_restarts += 1
        self._next_retry_s = now + delay_s

    def status(self, *, now_s: float | None = None) -> dict[str, object]:
        now = self._now(now_s)
        return {
            "state": self._state.value,
            "ready": self.ready,
            "generation": self._generation,
            "stable_frames": self._stable_frames,
            "required_stable_frames": self.config.stable_frame_count,
            "attempts": self._attempts,
            "total_restarts": self._total_restarts,
            "failures_since_ready": self._failures_since_ready,
            "restart_budget": self.config.restart_budget,
            "restart_required": self._restart_required,
            "last_frame_age_s": (
                None if self._last_frame_s is None else max(0.0, now - self._last_frame_s)
            ),
            "next_retry_in_s": (
                None
                if self._next_retry_s is None
                else max(0.0, self._next_retry_s - now)
            ),
            "last_error": self._last_error,
        }

    def _now(self, supplied: float | None) -> float:
        now = self._clock() if supplied is None else float(supplied)
        if not math.isfinite(now):
            raise ValueError("supervision time must be finite")
        return now
