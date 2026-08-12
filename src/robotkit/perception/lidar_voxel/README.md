# LIDAR voxel perception producer

This B component consumes ROS2 `sensor_msgs/PointCloud2` and optional
`nav_msgs/Odometry`, then publishes these versioned observations to A:

- `lidar.proximity`: fixed, bounded angular sectors in `base_link`, with the
  nearest planar range in each sector. The Go2 runtime applies the documented
  MID-360 13-degree mount rotation and body-centre translation to sensor-frame
  clouds; inputs explicitly configured as `base_link` are not transformed.
  Missing ranges are `null` and must never be interpreted as free space.
- `lidar.room_map`: a bounded sparse list of occupied voxels, never a full point
  cloud. `LIDAR_MAX_VOXELS` defaults to 512.
- `localization.pose`: planar `x/y/yaw`, either odometry or a deterministic voxel
  scan match bounded around odometry.

The scan matcher is intentionally small and auditable. It searches only the
configured `x/y/yaw` window around odometry and scores occupied-voxel overlap.
It is useful for local drift correction, but it is **not** loop-closing SLAM and
cannot disambiguate geometrically symmetric rooms. On restart, the runtime tries
to recover its reference map from A; otherwise it safely falls back to odometry.

## Proximity contract

The `lidar.proximity` observation has type `obstacles.angular_proximity`, a
short default TTL of 0.75 seconds, and this payload:

```json
{
  "representation": "angular_proximity_sectors_v1",
  "angle_convention": "bearing=atan2(y,x); positive is left",
  "distance_convention": "planar sqrt(x^2+y^2)",
  "sector_boundary_convention": "min inclusive, max exclusive; final sector max inclusive",
  "sector_width_rad": 0.174532925,
  "min_bearing_rad": -3.141592654,
  "max_bearing_rad": 3.141592654,
  "min_range_m": 0.1,
  "max_range_m": 10.0,
  "sectors": [
    {
      "index": 0,
      "bearing_min_rad": -3.141592654,
      "bearing_center_rad": -3.054326191,
      "bearing_max_rad": -2.967059728,
      "nearest_distance_m": null,
      "point_count": 0
    }
  ],
  "input_point_count": 1000,
  "accepted_point_count": 800,
  "populated_sector_count": 30
}
```

A controller selects the sector containing a normalized YOLO bearing. It must
stop if the observation is absent/stale, its frame is not `base_link`, or the
selected sector's distance is `null`. The fixed sector array is capped by
`LIDAR_PROXIMITY_MAX_SECTORS` (72 by default).

Run in a ROS2 environment with RobotKit installed:

```sh
WORLD_STATE_URL=http://world-state:8080 \
LIDAR_POINT_CLOUD_TOPIC=/utlidar/cloud \
LIDAR_ODOMETRY_TOPIC=/utlidar/robot_odom \
python -m robotkit.perception.lidar_voxel
```

Proximity settings use the `LIDAR_PROXIMITY_*` environment prefix, including
`SECTOR_WIDTH_RAD`, `MIN_BEARING_RAD`, `MAX_BEARING_RAD`, `MIN_RANGE_M`,
`MAX_RANGE_M`, `MIN_Z_M`, `MAX_Z_M`, `MAX_SECTORS`, and `TTL_SECONDS`.

Set `LIDAR_POINTS_FRAME=sensor` (the Go2 default) or `base_link`. Sensor mode
uses `LIDAR_MOUNT_PITCH_RAD`, `LIDAR_MOUNT_X_M`, `LIDAR_MOUNT_Y_M`, and
`LIDAR_MOUNT_Z_M`. Freshness is based on companion-computer receipt time while
the ROS header stamp remains the immutable message identity; this prevents a
robot/host wall-clock offset from making every live scan stale on arrival.

The core modules have no ROS or numpy dependency. Unit tests use synthetic room
scans and run with the normal RobotKit test environment.
