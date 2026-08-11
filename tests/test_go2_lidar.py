from __future__ import annotations

import struct
from dataclasses import dataclass

import pytest

from border_collie_demo.go2_lidar import (
    PearLidarHandoffConfig,
    PearLidarHandoffProvider,
    detect_centered_pear_cluster,
)


def config() -> PearLidarHandoffConfig:
    return PearLidarHandoffConfig(enabled=True)


def pear_points(
    x_m: float = 0.824, y_m: float = 0.0
) -> tuple[tuple[float, float, float], ...]:
    return (
        (x_m - 0.006, y_m, -0.18),
        (x_m, y_m + 0.01, -0.14),
        (x_m + 0.006, y_m - 0.01, -0.09),
        (x_m + 0.01, y_m, -0.11),
    )


def test_detects_measured_centered_pear_and_maps_front_clearance() -> None:
    cluster, reason = detect_centered_pear_cluster(pear_points(), config())

    assert reason == "pear_lidar_cluster_centered"
    assert cluster is not None
    assert cluster.body_x_m == pytest.approx(0.827, abs=0.01)
    assert cluster.body_x_m - config().front_envelope_x_m == pytest.approx(
        0.524, abs=0.01
    )
    assert cluster.confidence >= 0.50


def test_floor_only_and_lateral_objects_are_not_pear_clusters() -> None:
    floor = tuple((0.8 + index * 0.004, 0.0, -0.18) for index in range(8))
    lateral = pear_points(y_m=0.18)

    assert detect_centered_pear_cluster(floor, config())[0] is None
    assert detect_centered_pear_cluster(lateral, config())[0] is None


def test_multiple_vertical_clusters_are_ambiguous() -> None:
    cluster, reason = detect_centered_pear_cluster(
        (*pear_points(0.80), *pear_points(1.10)), config()
    )

    assert cluster is None
    assert reason == "pear_lidar_cluster_ambiguous"


@dataclass
class Field:
    name: str
    offset: int
    datatype: int = 7


@dataclass
class Header:
    frame_id: str = "base_link"


@dataclass
class PointCloud:
    points: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        self.width = len(self.points)
        self.height = 1
        self.point_step = 12
        self.fields = (Field("x", 0), Field("y", 4), Field("z", 8))
        self.data = b"".join(struct.pack("<fff", *point) for point in self.points)
        self.header = Header()
        self.is_bigendian = False


def provider(now: list[float]) -> PearLidarHandoffProvider:
    result = PearLidarHandoffProvider(
        config(),
        monotonic=lambda: now[0],
        subscriber_factory=lambda _callback: object(),
    )
    result.start()
    return result


def observe_visible(subject: PearLidarHandoffProvider):
    return subject.observe(
        visual_close_authorized=True,
        visual_center_error_ratio=0.02,
        allow_handoff=False,
    )


def test_two_fresh_visual_associations_arm_handoff_then_camera_may_disappear() -> None:
    now = [10.0]
    subject = provider(now)
    subject.ingest(PointCloud(pear_points(1.08)))
    first = observe_visible(subject)
    now[0] += 0.20
    subject.ingest(PointCloud(pear_points(1.00)))
    second = observe_visible(subject)
    now[0] += 0.10
    lost = subject.observe(
        visual_close_authorized=False,
        visual_center_error_ratio=None,
        allow_handoff=True,
    )

    assert first.available is False
    assert first.reason == "pear_lidar_visual_association_pending"
    assert first.handoff_active is False
    assert second.handoff_active is True
    assert lost.available is True
    assert lost.association_mode == "lidar_handoff"
    assert lost.front_clearance_m == pytest.approx(0.70, abs=0.02)


def test_one_missing_cloud_holds_but_persistent_loss_fails_closed() -> None:
    now = [10.0]
    subject = provider(now)
    for x_m in (1.08, 1.00):
        subject.ingest(PointCloud(pear_points(x_m)))
        observe_visible(subject)
        now[0] += 0.20
    subject.ingest(PointCloud(((0.8, 0.0, -0.18),)))
    held = subject.observe(
        visual_close_authorized=False,
        visual_center_error_ratio=None,
        allow_handoff=True,
    )
    now[0] += 0.31
    lost = subject.observe(
        visual_close_authorized=False,
        visual_center_error_ratio=None,
        allow_handoff=True,
    )

    assert held.available is True
    assert held.reason == "pear_lidar_handoff_bounded_hold"
    assert lost.available is False
    assert lost.reason in {"pear_lidar_cloud_stale", "pear_lidar_cluster_lost"}


def test_discontinuous_or_ambiguous_handoff_disarms_immediately() -> None:
    now = [10.0]
    subject = provider(now)
    for x_m in (1.08, 1.00):
        subject.ingest(PointCloud(pear_points(x_m)))
        observe_visible(subject)
        now[0] += 0.20
    subject.ingest(PointCloud((*pear_points(0.95), *pear_points(1.25))))
    result = subject.observe(
        visual_close_authorized=False,
        visual_center_error_ratio=None,
        allow_handoff=True,
    )

    assert result.available is False
    assert result.reason == "pear_lidar_cluster_ambiguous"
    assert subject.status()["association_armed"] is False


def test_handoff_expires_even_with_fresh_clouds() -> None:
    now = [10.0]
    subject = provider(now)
    for x_m in (1.08, 1.00):
        subject.ingest(PointCloud(pear_points(x_m)))
        observe_visible(subject)
        now[0] += 0.20
    subject.observe(
        visual_close_authorized=False,
        visual_center_error_ratio=None,
        allow_handoff=True,
    )
    now[0] += 2.01
    subject.ingest(PointCloud(pear_points(0.80)))
    expired = subject.observe(
        visual_close_authorized=False,
        visual_center_error_ratio=None,
        allow_handoff=True,
    )

    assert expired.available is False
    assert expired.reason == "pear_lidar_handoff_expired"
