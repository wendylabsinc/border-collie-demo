"""Sparse LIDAR room mapping and bounded planar positioning."""

from .core import (
    AngularProximity,
    Point3D,
    Pose2D,
    PoseEstimate,
    ProximityConfig,
    ProximitySector,
    SearchConfig,
    SparseVoxelMap,
    VoxelCell,
    VoxelConfig,
    angular_proximity,
    estimate_pose,
    sensor_points_to_base,
    transform_points,
    voxelize,
)
from .producer import ProducerIdentity, interpret_proximity, interpret_scan

__all__ = [
    "Point3D",
    "AngularProximity",
    "Pose2D",
    "PoseEstimate",
    "ProximityConfig",
    "ProximitySector",
    "ProducerIdentity",
    "SearchConfig",
    "SparseVoxelMap",
    "VoxelCell",
    "VoxelConfig",
    "angular_proximity",
    "estimate_pose",
    "interpret_proximity",
    "interpret_scan",
    "sensor_points_to_base",
    "transform_points",
    "voxelize",
]
