"""Bounded LiDAR handoff for a visually qualified close pear track."""

from __future__ import annotations

import math
import os
import statistics
import struct
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class PearLidarHandoffConfig:
    """Qualified body-frame geometry and temporal limits for pear handoff."""

    enabled: bool = False
    front_envelope_x_m: float = 0.30
    minimum_body_x_m: float = 0.35
    maximum_body_x_m: float = 1.50
    lateral_half_width_m: float = 0.10
    minimum_body_z_m: float = -0.50
    maximum_body_z_m: float = -0.08
    range_bin_width_m: float = 0.05
    minimum_cluster_points: int = 3
    minimum_cluster_height_m: float = 0.065
    maximum_cluster_height_m: float = 0.20
    maximum_age_s: float = 0.30
    maximum_range_jump_m: float = 0.18
    visual_association_window_s: float = 0.80
    visual_association_confirmations: int = 2
    maximum_handoff_s: float = 2.0
    topic: str = "rt/utlidar/cloud_base"

    def __post_init__(self) -> None:
        values = (
            self.front_envelope_x_m,
            self.minimum_body_x_m,
            self.maximum_body_x_m,
            self.lateral_half_width_m,
            self.minimum_body_z_m,
            self.maximum_body_z_m,
            self.range_bin_width_m,
            self.minimum_cluster_height_m,
            self.maximum_cluster_height_m,
            self.maximum_age_s,
            self.maximum_range_jump_m,
            self.visual_association_window_s,
            self.maximum_handoff_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("LiDAR handoff calibration must be finite")
        if not 0.0 <= self.front_envelope_x_m < self.minimum_body_x_m:
            raise ValueError("front envelope must precede the LiDAR search corridor")
        if not self.minimum_body_x_m < self.maximum_body_x_m:
            raise ValueError("LiDAR forward corridor is empty")
        if not self.minimum_body_z_m < self.maximum_body_z_m:
            raise ValueError("LiDAR height slab is empty")
        if not 0.0 < self.minimum_cluster_height_m < self.maximum_cluster_height_m:
            raise ValueError("LiDAR cluster height limits are invalid")
        if any(
            value <= 0.0
            for value in (
                self.lateral_half_width_m,
                self.range_bin_width_m,
                self.maximum_age_s,
                self.maximum_range_jump_m,
                self.visual_association_window_s,
                self.maximum_handoff_s,
            )
        ):
            raise ValueError("LiDAR handoff limits must be positive")
        if self.minimum_cluster_points < 3:
            raise ValueError("pear association requires at least three LiDAR points")
        if self.visual_association_confirmations < 2:
            raise ValueError("pear handoff requires at least two visual associations")
        if not self.topic.startswith("rt/"):
            raise ValueError("LiDAR topic must use the Unitree rt/ DDS name")

    @classmethod
    def from_env(cls) -> PearLidarHandoffConfig:
        return cls(
            enabled=os.environ.get("BORDER_COLLIE_LIDAR_HANDOFF_ENABLED", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            front_envelope_x_m=float(
                os.environ.get("BORDER_COLLIE_LIDAR_FRONT_ENVELOPE_M", "0.30")
            ),
            maximum_age_s=float(
                os.environ.get("BORDER_COLLIE_LIDAR_MAXIMUM_AGE_S", "0.30")
            ),
            maximum_handoff_s=float(
                os.environ.get("BORDER_COLLIE_LIDAR_MAXIMUM_HANDOFF_S", "2.0")
            ),
        )


@dataclass(frozen=True)
class PearCluster:
    body_x_m: float
    body_y_m: float
    height_m: float
    points: int
    confidence: float


@dataclass(frozen=True)
class LidarHandoffObservation:
    available: bool
    reason: str
    front_clearance_m: float | None
    age_s: float | None
    confidence: float
    association_valid: bool
    association_mode: str
    handoff_active: bool
    cluster_points: int = 0
    body_x_m: float | None = None
    body_y_m: float | None = None
    source: str = "lidar_temporal_pear_handoff"


def detect_centered_pear_cluster(
    points_body_xyz: Iterable[Sequence[float]],
    calibration: PearLidarHandoffConfig,
) -> tuple[PearCluster | None, str]:
    """Find one pear-sized vertical return in the centered body corridor."""
    bins: dict[int, list[tuple[float, float, float]]] = {}
    for point in points_body_xyz:
        if len(point) < 3:
            continue
        x_m, y_m, z_m = (float(point[index]) for index in range(3))
        if not all(math.isfinite(value) for value in (x_m, y_m, z_m)):
            continue
        if not calibration.minimum_body_x_m <= x_m <= calibration.maximum_body_x_m:
            continue
        if abs(y_m) > calibration.lateral_half_width_m:
            continue
        if not calibration.minimum_body_z_m <= z_m <= calibration.maximum_body_z_m:
            continue
        index = int(
            (x_m - calibration.minimum_body_x_m) / calibration.range_bin_width_m
        )
        bins.setdefault(index, []).append((x_m, y_m, z_m))

    candidate_windows: list[tuple[float, set[tuple[float, float, float]]]] = []
    for index in sorted(bins):
        points = [
            point
            for nearby in (index - 1, index, index + 1)
            for point in bins.get(nearby, ())
        ]
        if len(points) < calibration.minimum_cluster_points:
            continue
        height = max(point[2] for point in points) - min(point[2] for point in points)
        if (
            calibration.minimum_cluster_height_m
            <= height
            <= calibration.maximum_cluster_height_m
        ):
            candidate_windows.append(
                (statistics.median(point[0] for point in points), set(points))
            )

    groups: list[set[tuple[float, float, float]]] = []
    previous_center: float | None = None
    for center, points in candidate_windows:
        if (
            previous_center is None
            or center - previous_center > calibration.range_bin_width_m * 2.1
        ):
            groups.append(set())
        groups[-1].update(points)
        previous_center = center
    if not groups:
        return None, "pear_lidar_cluster_unavailable"
    if len(groups) != 1:
        return None, "pear_lidar_cluster_ambiguous"

    points = tuple(groups[0])
    height = max(point[2] for point in points) - min(point[2] for point in points)
    confidence = 0.5 * min(1.0, len(points) / 6.0) + 0.5 * min(1.0, height / 0.10)
    return (
        PearCluster(
            body_x_m=statistics.median(point[0] for point in points),
            body_y_m=statistics.mean(point[1] for point in points),
            height_m=height,
            points=len(points),
            confidence=confidence,
        ),
        "pear_lidar_cluster_centered",
    )


class PearLidarHandoffProvider:
    """Own cloud parsing, visual association, and the short LiDAR handoff token."""

    def __init__(
        self,
        calibration: PearLidarHandoffConfig,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        subscriber_factory: Callable[[Callable[[Any], None]], Any] | None = None,
    ) -> None:
        self.calibration = calibration
        self._monotonic = monotonic
        self._subscriber_factory = subscriber_factory
        self._lock = Lock()
        self._subscriber: Any = None
        self._frame_sequence = 0
        self._captured_at: float | None = None
        self._cluster: PearCluster | None = None
        self._cluster_reason = "waiting for body-frame LiDAR"
        self._frame_id: str | None = None
        self._last_observed_sequence = -1
        self._last_cluster: PearCluster | None = None
        self._last_cluster_at: float | None = None
        self._visual_hits: deque[float] = deque()
        self._visual_pending_started_at: float | None = None
        self._association_armed_at: float | None = None
        self._handoff_started_at: float | None = None

    def start(self) -> None:
        if self._subscriber is not None or not self.calibration.enabled:
            return
        try:
            if self._subscriber_factory is not None:
                self._subscriber = self._subscriber_factory(self.ingest)
                return
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

            subscriber = ChannelSubscriber(self.calibration.topic, PointCloud2_)
            subscriber.Init(self.ingest, 1)
            self._subscriber = subscriber
        except Exception as exc:  # noqa: BLE001 - Unitree DDS is dynamically typed
            with self._lock:
                self._cluster_reason = f"LiDAR subscriber failed: {exc}"

    def close(self) -> None:
        subscriber, self._subscriber = self._subscriber, None
        if subscriber is not None and hasattr(subscriber, "Close"):
            try:
                subscriber.Close()
            except Exception as exc:  # noqa: BLE001 - close is best effort
                with self._lock:
                    self._cluster_reason = f"LiDAR subscriber close failed: {exc}"

    def ingest(self, message: Any) -> None:
        try:
            frame_id = str(message.header.frame_id)
            if frame_id != "base_link":
                raise ValueError(f"expected base_link cloud, got {frame_id!r}")
            points = _parse_pointcloud2(message)
            cluster, reason = detect_centered_pear_cluster(points, self.calibration)
        except Exception as exc:  # noqa: BLE001 - DDS message is dynamic
            cluster, reason, frame_id = None, f"invalid body-frame LiDAR: {exc}", None
        with self._lock:
            self._frame_sequence += 1
            self._captured_at = self._monotonic()
            self._cluster = cluster
            self._cluster_reason = reason
            self._frame_id = frame_id

    def observe(
        self,
        *,
        visual_close_authorized: bool,
        visual_center_error_ratio: float | None,
        allow_handoff: bool,
    ) -> LidarHandoffObservation:
        now = self._monotonic()
        with self._lock:
            sequence = self._frame_sequence
            captured_at = self._captured_at
            cluster = self._cluster
            reason = self._cluster_reason
        if not self.calibration.enabled:
            return self._unavailable("pear_lidar_handoff_disabled")
        if captured_at is None:
            return self._unavailable(reason)
        cloud_age = max(0.0, now - captured_at)
        if cloud_age > self.calibration.maximum_age_s:
            self._disarm()
            return self._unavailable("pear_lidar_cloud_stale", age_s=cloud_age)

        centered_visual = bool(
            visual_close_authorized
            and visual_center_error_ratio is not None
            and math.isfinite(visual_center_error_ratio)
            and abs(visual_center_error_ratio) <= 0.12
        )
        if not centered_visual and not allow_handoff:
            self._disarm()
            return self._unavailable(
                "pear_lidar_handoff_not_authorized", age_s=cloud_age
            )
        if centered_visual and self._visual_pending_started_at is None:
            self._visual_pending_started_at = now
        if (
            allow_handoff
            and self._handoff_started_at is not None
            and now - self._handoff_started_at > self.calibration.maximum_handoff_s
        ):
            self._disarm()
            return self._unavailable("pear_lidar_handoff_expired", age_s=cloud_age)

        new_frame = sequence != self._last_observed_sequence
        if new_frame:
            self._last_observed_sequence = sequence
            if cluster is not None:
                if (
                    self._last_cluster is not None
                    and abs(cluster.body_x_m - self._last_cluster.body_x_m)
                    > self.calibration.maximum_range_jump_m
                ):
                    self._disarm()
                    return self._unavailable(
                        "pear_lidar_range_discontinuous", age_s=cloud_age
                    )
                self._last_cluster = cluster
                self._last_cluster_at = captured_at
                if centered_visual:
                    self._visual_hits.append(captured_at)
            elif reason == "pear_lidar_cluster_ambiguous":
                self._disarm()
                return self._unavailable(reason, age_s=cloud_age)

        while (
            self._visual_hits
            and now - self._visual_hits[0]
            > self.calibration.visual_association_window_s
        ):
            self._visual_hits.popleft()
        if len(self._visual_hits) >= self.calibration.visual_association_confirmations:
            self._association_armed_at = self._visual_hits[-1]

        if centered_visual and self._association_armed_at is None:
            assert self._visual_pending_started_at is not None
            if (
                now - self._visual_pending_started_at
                > self.calibration.visual_association_window_s
            ):
                self._disarm()
                return self._unavailable(
                    "pear_lidar_visual_association_failed", age_s=cloud_age
                )
            return self._unavailable(
                "pear_lidar_visual_association_pending", age_s=cloud_age
            )

        if (
            centered_visual
            and self._last_cluster is not None
            and self._last_cluster_at is not None
        ):
            return self._available(
                self._last_cluster,
                age_s=max(0.0, now - self._last_cluster_at),
                mode="lidar_visual_association",
                handoff_active=self._association_armed_at is not None,
                reason="pear_lidar_handoff_armed",
            )

        if self._association_armed_at is None:
            return self._unavailable("pear_lidar_handoff_not_armed", age_s=cloud_age)
        if self._handoff_started_at is None:
            self._handoff_started_at = now
        if now - self._handoff_started_at > self.calibration.maximum_handoff_s:
            self._disarm()
            return self._unavailable("pear_lidar_handoff_expired", age_s=cloud_age)
        if self._last_cluster is None or self._last_cluster_at is None:
            return self._unavailable("pear_lidar_cluster_unavailable", age_s=cloud_age)
        candidate_age = max(0.0, now - self._last_cluster_at)
        if candidate_age > self.calibration.maximum_age_s:
            self._disarm()
            return self._unavailable("pear_lidar_cluster_lost", age_s=candidate_age)
        return self._available(
            self._last_cluster,
            age_s=candidate_age,
            mode="lidar_handoff",
            handoff_active=True,
            reason=(
                "pear_lidar_handoff_tracking"
                if cluster is not None
                else "pear_lidar_handoff_bounded_hold"
            ),
        )

    def status(self) -> dict[str, object]:
        now = self._monotonic()
        with self._lock:
            captured_at = self._captured_at
            frame_id = self._frame_id
            reason = self._cluster_reason
        age_s = None if captured_at is None else max(0.0, now - captured_at)
        fresh = age_s is not None and age_s <= self.calibration.maximum_age_s
        return {
            "configured": self.calibration.enabled,
            "ready": bool(self.calibration.enabled and fresh),
            "source": "lidar_temporal_pear_handoff",
            "topic": self.calibration.topic,
            "frame_id": frame_id,
            "age_s": age_s,
            "fresh": fresh,
            "cluster_state": reason,
            "association_armed": self._association_armed_at is not None,
            "handoff_active": self._handoff_started_at is not None,
            "front_envelope_x_m": self.calibration.front_envelope_x_m,
            "maximum_handoff_s": self.calibration.maximum_handoff_s,
        }

    def _available(
        self,
        cluster: PearCluster,
        *,
        age_s: float,
        mode: str,
        handoff_active: bool,
        reason: str,
    ) -> LidarHandoffObservation:
        return LidarHandoffObservation(
            available=True,
            reason=reason,
            front_clearance_m=cluster.body_x_m - self.calibration.front_envelope_x_m,
            age_s=age_s,
            confidence=cluster.confidence,
            association_valid=True,
            association_mode=mode,
            handoff_active=handoff_active,
            cluster_points=cluster.points,
            body_x_m=cluster.body_x_m,
            body_y_m=cluster.body_y_m,
        )

    def _unavailable(
        self, reason: str, *, age_s: float | None = None
    ) -> LidarHandoffObservation:
        return LidarHandoffObservation(
            available=False,
            reason=reason,
            front_clearance_m=None,
            age_s=age_s,
            confidence=0.0,
            association_valid=False,
            association_mode="unavailable",
            handoff_active=False,
        )

    def _disarm(self) -> None:
        self._visual_hits.clear()
        self._visual_pending_started_at = None
        self._association_armed_at = None
        self._handoff_started_at = None
        self._last_cluster = None
        self._last_cluster_at = None


def _parse_pointcloud2(message: Any) -> tuple[tuple[float, float, float], ...]:
    if bool(getattr(message, "is_bigendian", False)):
        raise ValueError("big-endian PointCloud2 is unsupported")
    width, height, point_step = (
        int(message.width),
        int(message.height),
        int(message.point_step),
    )
    if width <= 0 or height <= 0 or point_step <= 0:
        raise ValueError("PointCloud2 shape is empty")
    fields = {str(field.name): field for field in message.fields}
    if any(name not in fields for name in ("x", "y", "z")):
        raise ValueError("PointCloud2 is missing x/y/z")
    if any(int(fields[name].datatype) != 7 for name in ("x", "y", "z")):
        raise ValueError("PointCloud2 x/y/z must be FLOAT32")
    raw = bytes(message.data)
    count = min(width * height, len(raw) // point_step)
    offsets = tuple(int(fields[name].offset) for name in ("x", "y", "z"))
    if count <= 0 or any(offset < 0 or offset + 4 > point_step for offset in offsets):
        raise ValueError("PointCloud2 payload or field offsets are invalid")
    return tuple(
        tuple(
            struct.unpack_from("<f", raw, index * point_step + offset)[0]
            for offset in offsets
        )
        for index in range(count)
    )
