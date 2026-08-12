from __future__ import annotations

import math

import pytest

from robotkit.perception.lidar_voxel import (
    Pose2D,
    ProducerIdentity,
    ProximityConfig,
    SearchConfig,
    SparseVoxelMap,
    VoxelConfig,
    angular_proximity,
    estimate_pose,
    interpret_scan,
    voxelize,
)
from tests.conftest import NOW


def _synthetic_room(width: float = 5.0, height: float = 3.0) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for step in range(51):
        x = width * step / 50
        points.extend(((x, 0.0, 0.5), (x, height, 0.5)))
    for step in range(31):
        y = height * step / 30
        points.extend(((0.0, y, 0.5), (width, y, 0.5)))
    # An asymmetric interior feature removes rectangular-room aliases.
    points.extend((1.0, 0.8 + step * 0.04, 0.5) for step in range(16))
    return points


def _to_robot_frame(
    points: list[tuple[float, float, float]], pose: Pose2D
) -> list[tuple[float, float, float]]:
    cosine = math.cos(-pose.yaw_rad)
    sine = math.sin(-pose.yaw_rad)
    result = []
    for x, y, z in points:
        dx = x - pose.x_m
        dy = y - pose.y_m
        result.append((cosine * dx - sine * dy, sine * dx + cosine * dy, z))
    return result


def test_voxelization_is_deterministic_filtered_and_bounded():
    points = [
        (0.21, 0.21, 0.21),
        (0.22, 0.22, 0.22),
        (0.61, 0.21, 0.21),
        (0.81, 0.21, 0.21),
        (float("nan"), 0.0, 0.0),
        (100.0, 0.0, 0.0),
    ]
    config = VoxelConfig(
        resolution_m=0.2,
        min_range_m=0.1,
        max_range_m=10,
        min_z_m=-1,
        max_z_m=2,
        max_voxels=2,
    )

    forward = voxelize(points, config)
    reverse = voxelize(list(reversed(points)), config)

    assert forward == reverse
    assert forward.input_point_count == 6
    assert forward.accepted_point_count == 4
    assert forward.occupied_voxel_count == 3
    assert forward.omitted_voxel_count == 1
    assert [(cell.index, cell.point_count) for cell in forward.cells] == [
        ((1, 1, 1), 2),
        ((3, 1, 1), 1),
    ]


def test_sparse_payload_round_trips_without_hidden_map_state():
    original = voxelize(
        _synthetic_room(),
        VoxelConfig(resolution_m=0.2, min_range_m=0, max_range_m=20),
    )
    restored = SparseVoxelMap.from_payload(original.to_payload())
    assert restored == original
    assert len(restored.to_payload()["cells"]) <= 512


def test_proximity_sectors_are_deterministic_and_nearest_range_wins():
    config = ProximityConfig(
        sector_width_rad=math.pi / 2,
        min_bearing_rad=-math.pi,
        max_bearing_rad=math.pi,
        min_range_m=0.1,
        max_range_m=5.0,
        min_z_m=-0.5,
        max_z_m=1.0,
        max_sectors=4,
    )
    points = [
        (2.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -2.0, 0.0),
        (0.0, 0.0, 0.0),  # below minimum range
        (1.0, 0.0, 2.0),  # outside height filter
        (float("nan"), 0.0, 0.0),
    ]

    proximity = angular_proximity(points, config)
    reversed_proximity = angular_proximity(list(reversed(points)), config)

    assert proximity == reversed_proximity
    assert len(proximity.sectors) == 4
    assert proximity.input_point_count == 7
    assert proximity.accepted_point_count == 4
    assert proximity.populated_sector_count == 3
    # Zero bearing belongs to [0, pi/2), and nearest of 0.5m/2m wins.
    assert proximity.sectors[2].nearest_distance_m == pytest.approx(0.5)
    assert proximity.sectors[2].point_count == 2
    assert proximity.sectors[0].nearest_distance_m is None


def test_proximity_configuration_caps_payload():
    with pytest.raises(ValueError, match="exceed max_sectors"):
        ProximityConfig(sector_width_rad=0.01, max_sectors=72)

    configured = ProximityConfig(
        sector_width_rad=0.25,
        min_bearing_rad=-1.0,
        max_bearing_rad=1.0,
        max_sectors=8,
    )
    result = angular_proximity([], configured)
    payload = result.to_payload()
    assert len(payload["sectors"]) == 8
    assert all(sector["nearest_distance_m"] is None for sector in payload["sectors"])
    assert payload["sector_boundary_convention"].startswith("min inclusive")


def test_bounded_scan_match_corrects_odometry_in_synthetic_room():
    room = _synthetic_room()
    map_config = VoxelConfig(
        resolution_m=0.1,
        min_range_m=0,
        max_range_m=30,
        max_voxels=1000,
    )
    reference = voxelize(room, map_config)
    actual = Pose2D(2.0, 1.0, 0.10)
    scan = _to_robot_frame(room, actual)
    odometry = Pose2D(2.2, 0.8, 0.0)

    estimate = estimate_pose(
        scan,
        reference,
        odometry,
        SearchConfig(
            translation_radius_m=0.3,
            translation_step_m=0.1,
            yaw_radius_rad=0.15,
            yaw_step_rad=0.05,
            max_points=1000,
        ),
    )

    assert estimate.pose.x_m == pytest.approx(actual.x_m, abs=0.05)
    assert estimate.pose.y_m == pytest.approx(actual.y_m, abs=0.05)
    assert estimate.pose.yaw_rad == pytest.approx(actual.yaw_rad, abs=0.025)
    assert estimate.score > 0.9
    assert estimate.source == "voxel_scan_match"


def test_interpret_scan_publishes_small_map_and_position_observations():
    room = _synthetic_room()
    identity = ProducerIdentity("go2-lidar-voxel", "blue-1", 4)
    observations, sparse_map = interpret_scan(
        room,
        observed_at=NOW,
        identity=identity,
        message_key="42.000000001",
        odometry_pose=Pose2D(0.0, 0.0, 0.0),
        voxel_config=VoxelConfig(
            resolution_m=0.2, min_range_m=0, max_range_m=20, max_voxels=40
        ),
    )

    assert [observation.stream for observation in observations] == [
        "lidar.proximity",
        "lidar.room_map",
        "localization.pose",
    ]
    proximity_observation = observations[0]
    assert proximity_observation.frame_id == "base_link"
    assert proximity_observation.ttl_seconds == pytest.approx(0.75)
    assert (
        proximity_observation.payload["representation"]
        == "angular_proximity_sectors_v1"
    )
    assert len(proximity_observation.payload["sectors"]) == 36
    map_observation = observations[1]
    assert map_observation.frame_id == "map"
    assert map_observation.payload["representation"] == "sparse_occupied_voxels_v1"
    assert len(map_observation.payload["cells"]) == 40
    assert sparse_map.omitted_voxel_count > 0
    assert observations[2].payload["source"] == "odometry"
    assert observations[2].deployment_generation == 4


def test_no_pose_keeps_map_local_and_does_not_invent_localization():
    observations, _ = interpret_scan(
        [(1.0, 0.0, 0.0)],
        observed_at=NOW,
        identity=ProducerIdentity("lidar", "test"),
        message_key="one",
        scan_frame_id="utlidar_lidar",
    )
    assert len(observations) == 2
    assert [item.stream for item in observations] == [
        "lidar.proximity",
        "lidar.room_map",
    ]
    assert all(item.frame_id == "utlidar_lidar" for item in observations)
