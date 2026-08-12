"""Observation creation for the LIDAR B component."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from robotkit.contracts import Observation

from .core import (
    AngularProximity,
    Point3D,
    Pose2D,
    ProximityConfig,
    SearchConfig,
    SparseVoxelMap,
    VoxelConfig,
    angular_proximity,
    estimate_pose,
    transform_points,
    voxelize,
)


@dataclass(frozen=True)
class ProducerIdentity:
    producer_id: str
    instance_id: str
    deployment_generation: int = 0


def interpret_scan(
    scan_points: Sequence[Point3D],
    *,
    observed_at: datetime,
    identity: ProducerIdentity,
    message_key: str,
    odometry_pose: Pose2D | None = None,
    reference_map: SparseVoxelMap | None = None,
    scan_frame_id: str = "base_link",
    map_frame_id: str = "map",
    voxel_config: VoxelConfig = VoxelConfig(),
    search_config: SearchConfig = SearchConfig(),
    proximity_config: ProximityConfig | None = ProximityConfig(),
) -> tuple[list[Observation], SparseVoxelMap]:
    """Produce proximity, a sparse room map, and when possible a planar pose.

    A scan match is bounded around ``odometry_pose`` when a prior map exists.
    Otherwise odometry is passed through explicitly as the localization source.
    With no pose, the map remains in the scan frame and no position is invented.
    ``scan_points`` must be expressed in ``scan_frame_id``; target-bearing fusion
    requires that frame to be ``base_link`` (the default).
    """
    pose = odometry_pose
    estimate_payload: dict[str, Any] | None = None
    if reference_map is not None and odometry_pose is not None:
        estimate = estimate_pose(scan_points, reference_map, odometry_pose, search_config)
        pose = estimate.pose
        estimate_payload = {
            "x_m": round(pose.x_m, 6),
            "y_m": round(pose.y_m, 6),
            "yaw_rad": round(pose.yaw_rad, 6),
            "source": estimate.source,
            "match_score": round(estimate.score, 6),
            "matched_points": estimate.matched_points,
            "evaluated_points": estimate.evaluated_points,
            "candidates_evaluated": estimate.candidates_evaluated,
            "bounded_by_odometry": True,
        }
    elif pose is not None:
        estimate_payload = {
            "x_m": round(pose.x_m, 6),
            "y_m": round(pose.y_m, 6),
            "yaw_rad": round(pose.yaw_rad, 6),
            "source": "odometry",
            "match_score": None,
            "bounded_by_odometry": False,
        }

    map_points = transform_points(scan_points, pose) if pose is not None else list(scan_points)
    sensor_origin = (pose.x_m, pose.y_m, 0.0) if pose is not None else (0.0, 0.0, 0.0)
    sparse_map = voxelize(map_points, voxel_config, sensor_origin=sensor_origin)
    observations: list[Observation] = []
    if proximity_config is not None:
        proximity = angular_proximity(scan_points, proximity_config)
        observations.append(
            Observation(
                idempotency_key=f"{identity.instance_id}:lidar.proximity:{message_key}",
                producer_id=identity.producer_id,
                instance_id=identity.instance_id,
                deployment_generation=identity.deployment_generation,
                stream="lidar.proximity",
                observation_type="obstacles.angular_proximity",
                observed_at=observed_at,
                frame_id=scan_frame_id,
                confidence=_proximity_confidence(proximity),
                ttl_seconds=proximity_config.ttl_seconds,
                payload=proximity.to_payload(),
            )
        )
    observations.append(
        Observation(
            idempotency_key=f"{identity.instance_id}:lidar.room_map:{message_key}",
            producer_id=identity.producer_id,
            instance_id=identity.instance_id,
            deployment_generation=identity.deployment_generation,
            stream="lidar.room_map",
            observation_type="room.voxel_map.sparse",
            observed_at=observed_at,
            frame_id=map_frame_id if pose is not None else scan_frame_id,
            confidence=_map_confidence(sparse_map),
            ttl_seconds=10.0,
            payload=sparse_map.to_payload(),
        )
    )
    if estimate_payload is not None:
        confidence = (
            float(estimate_payload["match_score"])
            if estimate_payload["match_score"] is not None
            else 0.75
        )
        observations.append(
            Observation(
                idempotency_key=f"{identity.instance_id}:localization.pose:{message_key}",
                producer_id=identity.producer_id,
                instance_id=identity.instance_id,
                deployment_generation=identity.deployment_generation,
                stream="localization.pose",
                observation_type="robot.pose.2d",
                observed_at=observed_at,
                frame_id=map_frame_id,
                confidence=max(0.0, min(1.0, confidence)),
                ttl_seconds=2.0,
                payload=estimate_payload,
            )
        )
    return observations, sparse_map


def _map_confidence(voxel_map: SparseVoxelMap) -> float:
    if voxel_map.accepted_point_count == 0:
        return 0.0
    retained = len(voxel_map.cells)
    coverage = retained / max(1, voxel_map.occupied_voxel_count)
    density = min(1.0, voxel_map.accepted_point_count / 100.0)
    return round(coverage * density, 6)


def _proximity_confidence(proximity: AngularProximity) -> float:
    if proximity.input_point_count == 0:
        return 0.0
    valid_ratio = proximity.accepted_point_count / proximity.input_point_count
    density = min(1.0, proximity.accepted_point_count / 20.0)
    return round(valid_ratio * density, 6)
