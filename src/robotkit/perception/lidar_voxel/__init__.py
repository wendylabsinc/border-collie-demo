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
    transform_points,
    voxelize,
)
from .producer import ProducerIdentity, interpret_scan

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
    "interpret_scan",
    "transform_points",
    "voxelize",
]
