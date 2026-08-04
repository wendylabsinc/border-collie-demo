"""Fresh robot-local pose samples from Unitree SportModeState."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

from .models import Pose


@dataclass(frozen=True)
class PoseStatus:
    pose: Pose | None
    age_s: float | None
    healthy: bool
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "healthy": self.healthy,
            "age_s": None if self.age_s is None else round(self.age_s, 3),
            "error": self.error,
            "pose": None
            if self.pose is None
            else {
                "x_m": round(self.pose.x_m, 4),
                "y_m": round(self.pose.y_m, 4),
                "yaw_rad": round(self.pose.yaw_rad, 4),
                "captured_monotonic_s": self.pose.captured_monotonic_s,
            },
        }


class Go2PoseProvider:
    def __init__(self, *, maximum_age_s: float = 0.50) -> None:
        if not math.isfinite(maximum_age_s) or maximum_age_s <= 0.0:
            raise ValueError("maximum_age_s must be finite and positive")
        self.maximum_age_s = float(maximum_age_s)
        self._lock = Lock()
        self._pose: Pose | None = None
        self._error: str | None = "waiting for rt/sportmodestate"
        self._subscriber = None

    def start(self) -> None:
        if self._subscriber is not None:
            return
        try:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

            subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
            subscriber.Init(self._on_state, 1)
            self._subscriber = subscriber
        except Exception as exc:  # noqa: BLE001 - Unitree import/init is untyped
            with self._lock:
                self._error = f"pose subscriber failed: {exc}"

    def close(self) -> None:
        subscriber, self._subscriber = self._subscriber, None
        close_error: str | None = None
        if subscriber is not None:
            try:
                subscriber.Close()
            except Exception as exc:  # noqa: BLE001 - Unitree close is untyped
                close_error = f"pose subscriber close failed: {exc}"
        with self._lock:
            self._pose = None
            self._error = close_error or "pose provider closed"

    def status(self) -> PoseStatus:
        now = time.monotonic()
        with self._lock:
            pose = self._pose
            error = self._error
        age = None if pose is None else max(0.0, now - pose.captured_monotonic_s)
        healthy = bool(
            pose is not None and age is not None and age <= self.maximum_age_s
        )
        return PoseStatus(
            pose,
            age,
            healthy,
            None if healthy else error or "pose sample is stale",
        )

    def _on_state(self, message: Any) -> None:
        try:
            yaw_rad = normalize_angle(float(message.imu_state.rpy[2]))
            position = message.position
            x_m = float(position[0])
            y_m = float(position[1])
            if not all(math.isfinite(value) for value in (x_m, y_m, yaw_rad)):
                raise ValueError("non-finite local pose")
        except Exception as exc:  # noqa: BLE001 - DDS message types are dynamic
            with self._lock:
                self._error = f"invalid pose sample: {exc}"
            return
        with self._lock:
            self._pose = Pose(x_m, y_m, yaw_rad, time.monotonic())
            self._error = None


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))
