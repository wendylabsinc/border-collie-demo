"""Fail-closed physical-controller takeover from the Go2 low-state stream."""

from __future__ import annotations

import asyncio
import math
import os
import struct
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Any

from .models import RemoteInput

REMOTE_TOPIC = "rt/lowstate"

_BUTTONS_1 = (
    "R1",
    "L1",
    "Start",
    "Select",
    "R2",
    "L2",
    "F1",
    "F3",
)
_BUTTONS_2 = (
    "A",
    "B",
    "X",
    "Y",
    "Up",
    "Right",
    "Down",
    "Left",
)


class RemoteInputUnavailable(RuntimeError):
    """Raised when the controller-monitoring stream is not trustworthy."""


@dataclass(frozen=True)
class RemoteInputConfig:
    axis_deadzone: float = 0.15
    active_confirmations: int = 2
    maximum_sample_age_s: float = 0.50
    startup_grace_s: float = 1.0
    freshness_poll_s: float = 0.05

    def __post_init__(self) -> None:
        if not math.isfinite(self.axis_deadzone) or not 0.0 < self.axis_deadzone < 1.0:
            raise ValueError("axis_deadzone must be finite and between zero and one")
        if self.active_confirmations < 1:
            raise ValueError("active_confirmations must be positive")
        for name in (
            "maximum_sample_age_s",
            "startup_grace_s",
            "freshness_poll_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def from_env(cls) -> RemoteInputConfig:
        return cls(
            axis_deadzone=float(
                os.environ.get("BORDER_COLLIE_REMOTE_AXIS_DEADZONE", "0.15")
            ),
            active_confirmations=int(
                os.environ.get("BORDER_COLLIE_REMOTE_CONFIRMATIONS", "2")
            ),
            maximum_sample_age_s=float(
                os.environ.get("BORDER_COLLIE_REMOTE_MAX_AGE_S", "0.50")
            ),
            startup_grace_s=float(
                os.environ.get("BORDER_COLLIE_REMOTE_STARTUP_GRACE_S", "1.0")
            ),
            freshness_poll_s=float(
                os.environ.get("BORDER_COLLIE_REMOTE_POLL_S", "0.05")
            ),
        )


def decode_wireless_remote(
    wireless_remote: Sequence[int],
    *,
    axis_deadzone: float,
) -> str | None:
    """Return a stable control name for one valid active Go2 remote sample."""

    if len(wireless_remote) < 24:
        raise ValueError("wireless_remote must contain at least 24 bytes")
    try:
        payload = bytes(wireless_remote[:24])
    except (TypeError, ValueError) as exc:
        raise ValueError("wireless_remote contains invalid bytes") from exc

    lx = struct.unpack_from("<f", payload, 4)[0]
    rx = struct.unpack_from("<f", payload, 8)[0]
    ry = struct.unpack_from("<f", payload, 12)[0]
    ly = struct.unpack_from("<f", payload, 20)[0]
    axes = (lx, ly, rx, ry)
    if any(not math.isfinite(value) or abs(value) > 1.25 for value in axes):
        raise ValueError("wireless_remote contains an invalid stick value")

    controls: list[str] = []
    buttons = [
        name
        for byte_value, names in (
            (payload[2], _BUTTONS_1),
            (payload[3], _BUTTONS_2),
        )
        for bit, name in enumerate(names)
        if byte_value & (1 << bit)
    ]
    if buttons:
        controls.append("buttons:" + "+".join(buttons))
    if max(abs(lx), abs(ly)) >= axis_deadzone:
        controls.append("left_stick")
    if max(abs(rx), abs(ry)) >= axis_deadzone:
        controls.append("right_stick")
    return "+".join(controls) or None


class Go2RemoteInput:
    """Watch controller bytes embedded in fresh Go2 ``LowState_`` samples."""

    def __init__(self, config: RemoteInputConfig | None = None) -> None:
        self.config = config or RemoteInputConfig()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[RemoteInput] | None = None
        self._subscriber: Any | None = None
        self._watching = False
        self._closed = False
        self._error: str | None = None
        self._last_sample_at: float | None = None
        self._sample_count = 0
        self._invalid_sample_count = 0
        self._candidate_control: str | None = None
        self._candidate_confirmations = 0
        self._takeover_emitted = False

    def observe(
        self,
        wireless_remote: Sequence[int],
        *,
        received_monotonic_s: float | None = None,
    ) -> RemoteInput | None:
        """Consume one sample; exposed separately for deterministic replay tests."""

        received_at = monotonic() if received_monotonic_s is None else received_monotonic_s
        try:
            control = decode_wireless_remote(
                wireless_remote,
                axis_deadzone=self.config.axis_deadzone,
            )
        except ValueError as exc:
            self._invalid_sample_count += 1
            self._error = str(exc)
            self._candidate_control = None
            self._candidate_confirmations = 0
            return None

        self._last_sample_at = received_at
        self._sample_count += 1
        self._error = None
        if control is None:
            self._candidate_control = None
            self._candidate_confirmations = 0
            return None
        if self._takeover_emitted:
            return None
        if control == self._candidate_control:
            self._candidate_confirmations += 1
        else:
            self._candidate_control = control
            self._candidate_confirmations = 1
        if self._candidate_confirmations < self.config.active_confirmations:
            return None

        self._takeover_emitted = True
        return RemoteInput(
            source="unitree_remote",
            control=control,
            received_monotonic_s=received_at,
        )

    async def watch(
        self,
        handler: Callable[[RemoteInput], Awaitable[None]],
    ) -> None:
        """Subscribe until cancelled; stream loss raises and must stop autonomy."""

        if self._watching:
            raise RemoteInputUnavailable("remote input watcher is already running")
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=1)
        started_at = monotonic()
        try:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

            self._subscriber = ChannelSubscriber(REMOTE_TOPIC, LowState_)
            self._subscriber.Init(self._on_low_state, 1)
            self._watching = True
            self._closed = False
            while True:
                try:
                    remote_input = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=self.config.freshness_poll_s,
                    )
                except TimeoutError:
                    now = monotonic()
                    if self._last_sample_at is None:
                        if now - started_at > self.config.startup_grace_s:
                            raise RemoteInputUnavailable(
                                f"no {REMOTE_TOPIC} controller-monitor samples"
                            )
                    elif now - self._last_sample_at > self.config.maximum_sample_age_s:
                        raise RemoteInputUnavailable(
                            f"{REMOTE_TOPIC} controller-monitor samples became stale"
                        )
                    continue
                await handler(remote_input)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._error = str(exc)
            raise
        finally:
            self._watching = False
            subscriber = self._subscriber
            self._subscriber = None
            if subscriber is not None:
                subscriber.Close()
            self._closed = True

    def status(self, *, now: float | None = None) -> dict[str, object]:
        checked_at = monotonic() if now is None else now
        age = (
            None
            if self._last_sample_at is None
            else max(0.0, checked_at - self._last_sample_at)
        )
        ready = bool(
            self._last_sample_at is not None
            and age is not None
            and age <= self.config.maximum_sample_age_s
            and self._error is None
            and not self._closed
        )
        return {
            "ready": ready,
            "topic": REMOTE_TOPIC,
            "age_s": age,
            "error": self._error,
            "watching": self._watching,
            "sample_count": self._sample_count,
            "invalid_sample_count": self._invalid_sample_count,
            "axis_deadzone": self.config.axis_deadzone,
            "active_confirmations": self.config.active_confirmations,
            "takeover_emitted": self._takeover_emitted,
        }

    def _on_low_state(self, message: Any) -> None:
        remote_input = self.observe(message.wireless_remote)
        loop = self._loop
        if remote_input is None or loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._enqueue, remote_input)

    def _enqueue(self, remote_input: RemoteInput) -> None:
        queue = self._queue
        if queue is None or queue.full():
            return
        queue.put_nowait(remote_input)
