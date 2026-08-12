"""ROS2 runtime adapter for the LIDAR voxel producer.

All ROS imports live inside :func:`main`, so importing and testing the package
does not require a ROS installation.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any

from robotkit.client import WorldStateClient
from robotkit.runtime import (
    configure_logging,
    deployment_generation,
    instance_id,
    world_state_url,
)

from .core import (
    Pose2D,
    ProximityConfig,
    SearchConfig,
    SparseVoxelMap,
    VoxelConfig,
    sensor_points_to_base,
)
from .producer import ProducerIdentity, interpret_scan


def main() -> None:
    """Subscribe to PointCloud2/Odometry and publish interpretations to A."""
    # ROS images supply these packages.  Keeping imports here preserves a small,
    # ROS-free base package for unit tests and non-ROS replay jobs.
    try:
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import PointCloud2
        from sensor_msgs_py import point_cloud2
    except ImportError as error:  # pragma: no cover - exercised in ROS image
        raise SystemExit(
            "ROS2 runtime requires rclpy, sensor_msgs, sensor_msgs_py, and nav_msgs"
        ) from error

    configure_logging()
    client = WorldStateClient(world_state_url())
    deployment = instance_id()
    identity = ProducerIdentity(
        producer_id=os.getenv("ROBOTKIT_PRODUCER_ID", "go2-lidar-voxel"),
        instance_id=deployment,
        deployment_generation=deployment_generation(),
    )
    voxel_config = VoxelConfig(
        resolution_m=_env_float("LIDAR_VOXEL_RESOLUTION_M", 0.20),
        min_range_m=_env_float("LIDAR_MIN_RANGE_M", 0.10),
        max_range_m=_env_float("LIDAR_MAX_RANGE_M", 20.0),
        min_z_m=_env_float("LIDAR_MIN_Z_M", -1.0),
        max_z_m=_env_float("LIDAR_MAX_Z_M", 3.0),
        min_points_per_voxel=_env_int("LIDAR_MIN_POINTS_PER_VOXEL", 1),
        max_voxels=_env_int("LIDAR_MAX_VOXELS", 512),
    )
    search_config = SearchConfig(
        translation_radius_m=_env_float("LIDAR_SEARCH_RADIUS_M", 0.60),
        translation_step_m=_env_float("LIDAR_SEARCH_STEP_M", 0.10),
        yaw_radius_rad=_env_float("LIDAR_YAW_RADIUS_RAD", 0.30),
        yaw_step_rad=_env_float("LIDAR_YAW_STEP_RAD", 0.05),
        max_points=_env_int("LIDAR_MATCH_MAX_POINTS", 1200),
    )
    proximity_config = ProximityConfig(
        sector_width_rad=_env_float(
            "LIDAR_PROXIMITY_SECTOR_WIDTH_RAD", math.pi / 18.0
        ),
        min_bearing_rad=_env_float("LIDAR_PROXIMITY_MIN_BEARING_RAD", -math.pi),
        max_bearing_rad=_env_float("LIDAR_PROXIMITY_MAX_BEARING_RAD", math.pi),
        min_range_m=_env_float("LIDAR_PROXIMITY_MIN_RANGE_M", 0.10),
        max_range_m=_env_float("LIDAR_PROXIMITY_MAX_RANGE_M", 10.0),
        min_z_m=_env_float("LIDAR_PROXIMITY_MIN_Z_M", -0.20),
        max_z_m=_env_float("LIDAR_PROXIMITY_MAX_Z_M", 0.28),
        max_sectors=_env_int("LIDAR_PROXIMITY_MAX_SECTORS", 72),
        ttl_seconds=_env_float("LIDAR_PROXIMITY_TTL_SECONDS", 0.75),
    )
    initial_reference = _load_reference_map(client)

    class LidarVoxelNode(Node):
        def __init__(self) -> None:
            super().__init__("robotkit_lidar_voxel")
            self._odometry: Pose2D | None = None
            self._reference_map = initial_reference
            cloud_topic = os.getenv("LIDAR_POINT_CLOUD_TOPIC", "/utlidar/cloud")
            odometry_topic = os.getenv("LIDAR_ODOMETRY_TOPIC", "/utlidar/robot_odom")
            self.create_subscription(
                Odometry, odometry_topic, self._on_odometry, qos_profile_sensor_data
            )
            self.create_subscription(
                PointCloud2, cloud_topic, self._on_cloud, qos_profile_sensor_data
            )
            self.get_logger().info(
                f"listening on {cloud_topic} with odometry {odometry_topic}"
            )

        def _on_odometry(self, message: Any) -> None:
            position = message.pose.pose.position
            orientation = message.pose.pose.orientation
            yaw = math.atan2(
                2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
                1.0 - 2.0 * (orientation.y**2 + orientation.z**2),
            )
            self._odometry = Pose2D(float(position.x), float(position.y), yaw)

        def _on_cloud(self, message: Any) -> None:
            points = [
                (float(row[0]), float(row[1]), float(row[2]))
                for row in point_cloud2.read_points(
                    message, field_names=("x", "y", "z"), skip_nans=True
                )
            ]
            incoming_frame = message.header.frame_id or "lidar"
            points_frame = os.getenv("LIDAR_POINTS_FRAME", "sensor").casefold()
            if points_frame == "sensor":
                points = sensor_points_to_base(
                    points,
                    pitch_rad=_env_float(
                        "LIDAR_MOUNT_PITCH_RAD", math.radians(13.0)
                    ),
                    x_offset_m=_env_float("LIDAR_MOUNT_X_M", 0.16143),
                    y_offset_m=_env_float("LIDAR_MOUNT_Y_M", 0.0),
                    z_offset_m=_env_float("LIDAR_MOUNT_Z_M", 0.12262),
                )
                scan_frame_id = "base_link"
            elif points_frame == "base_link":
                scan_frame_id = "base_link"
            else:
                raise ValueError(
                    "LIDAR_POINTS_FRAME must be 'sensor' or 'base_link', got "
                    f"{points_frame!r} for ROS frame {incoming_frame!r}"
                )
            observed_at, message_key = _ros_stamp(message.header.stamp)
            observations, current_map = interpret_scan(
                points,
                observed_at=observed_at,
                identity=identity,
                message_key=message_key,
                odometry_pose=self._odometry,
                reference_map=self._reference_map,
                scan_frame_id=scan_frame_id,
                map_frame_id=os.getenv("LIDAR_MAP_FRAME", "map"),
                voxel_config=voxel_config,
                search_config=search_config,
                proximity_config=proximity_config,
            )
            for observation in observations:
                client.publish_observation(observation)
            # Disposable cache: it can be reconstructed from A after a restart.
            self._reference_map = current_map

    rclpy.init()
    node = LidarVoxelNode()
    try:
        rclpy.spin(node)
    finally:  # pragma: no cover - exercised in ROS image
        node.destroy_node()
        client.close()
        rclpy.shutdown()


def _load_reference_map(client: WorldStateClient) -> SparseVoxelMap | None:
    """Restore the latest persisted LIDAR map; failure means odometry fallback."""
    try:
        candidates = [
            observation
            for observation in client.snapshot().observations
            if observation.stream == "lidar.room_map"
        ]
        if not candidates:
            return None
        latest = max(candidates, key=lambda observation: observation.revision)
        return SparseVoxelMap.from_payload(latest.payload)
    except Exception:
        # A malformed/old map must not prevent fresh perception from starting.
        return None


def _ros_stamp(stamp: Any) -> tuple[datetime, str]:
    """Receipt time drives freshness; the robot stamp drives idempotency."""
    seconds = int(stamp.sec)
    nanoseconds = int(stamp.nanosec)
    return (
        datetime.now(timezone.utc),
        f"{seconds}.{nanoseconds:09d}",
    )


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


if __name__ == "__main__":
    main()
