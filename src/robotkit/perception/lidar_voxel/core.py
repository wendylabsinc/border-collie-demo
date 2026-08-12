"""Deterministic point-cloud voxelization and bounded 2-D scan matching.

The algorithms deliberately use only the Python standard library.  A ROS2
runtime may therefore be audited and tested without ROS, numpy, or a GPU.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

Point3D = tuple[float, float, float]
VoxelIndex = tuple[int, int, int]


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class VoxelConfig:
    resolution_m: float = 0.20
    min_range_m: float = 0.10
    max_range_m: float = 20.0
    min_z_m: float = -1.0
    max_z_m: float = 3.0
    min_points_per_voxel: int = 1
    max_voxels: int = 512

    def __post_init__(self) -> None:
        if not math.isfinite(self.resolution_m) or self.resolution_m <= 0:
            raise ValueError("resolution_m must be finite and positive")
        if self.min_range_m < 0 or self.max_range_m <= self.min_range_m:
            raise ValueError("range bounds are invalid")
        if self.max_z_m <= self.min_z_m:
            raise ValueError("z bounds are invalid")
        if self.min_points_per_voxel < 1 or self.max_voxels < 1:
            raise ValueError("voxel limits must be positive")


@dataclass(frozen=True)
class ProximityConfig:
    """Bounds for the base-link angular proximity reduction."""

    sector_width_rad: float = math.pi / 18.0
    min_bearing_rad: float = -math.pi
    max_bearing_rad: float = math.pi
    min_range_m: float = 0.10
    max_range_m: float = 10.0
    min_z_m: float = -0.30
    max_z_m: float = 1.50
    max_sectors: int = 72
    ttl_seconds: float = 0.75

    def __post_init__(self) -> None:
        values = (
            self.sector_width_rad,
            self.min_bearing_rad,
            self.max_bearing_rad,
            self.min_range_m,
            self.max_range_m,
            self.min_z_m,
            self.max_z_m,
            self.ttl_seconds,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("proximity bounds must be finite")
        if self.sector_width_rad <= 0:
            raise ValueError("sector_width_rad must be positive")
        if not (
            -math.pi <= self.min_bearing_rad < self.max_bearing_rad <= math.pi
        ):
            raise ValueError("bearing bounds must be ordered within [-pi, pi]")
        if self.min_range_m < 0 or self.max_range_m <= self.min_range_m:
            raise ValueError("range bounds are invalid")
        if self.max_z_m <= self.min_z_m:
            raise ValueError("z bounds are invalid")
        if self.max_sectors < 1:
            raise ValueError("max_sectors must be positive")
        if self.ttl_seconds <= 0 or self.ttl_seconds > 86_400:
            raise ValueError("ttl_seconds is invalid")
        sector_count = math.ceil(
            (self.max_bearing_rad - self.min_bearing_rad) / self.sector_width_rad
            - 1e-12
        )
        if sector_count > self.max_sectors:
            raise ValueError("angular range and width exceed max_sectors")


@dataclass(frozen=True)
class ProximitySector:
    index: int
    bearing_min_rad: float
    bearing_center_rad: float
    bearing_max_rad: float
    nearest_distance_m: float | None
    point_count: int


@dataclass(frozen=True)
class AngularProximity:
    sector_width_rad: float
    min_bearing_rad: float
    max_bearing_rad: float
    min_range_m: float
    max_range_m: float
    sectors: tuple[ProximitySector, ...]
    input_point_count: int
    accepted_point_count: int

    @property
    def populated_sector_count(self) -> int:
        return sum(sector.nearest_distance_m is not None for sector in self.sectors)

    def to_payload(self) -> dict[str, object]:
        return {
            "representation": "angular_proximity_sectors_v1",
            "angle_convention": "bearing=atan2(y,x); positive is left",
            "distance_convention": "planar sqrt(x^2+y^2)",
            "sector_boundary_convention": (
                "min inclusive, max exclusive; final sector max inclusive"
            ),
            "sector_width_rad": round(self.sector_width_rad, 9),
            "min_bearing_rad": round(self.min_bearing_rad, 9),
            "max_bearing_rad": round(self.max_bearing_rad, 9),
            "min_range_m": self.min_range_m,
            "max_range_m": self.max_range_m,
            "sectors": [
                {
                    "index": sector.index,
                    "bearing_min_rad": round(sector.bearing_min_rad, 9),
                    "bearing_center_rad": round(sector.bearing_center_rad, 9),
                    "bearing_max_rad": round(sector.bearing_max_rad, 9),
                    "nearest_distance_m": (
                        round(sector.nearest_distance_m, 6)
                        if sector.nearest_distance_m is not None
                        else None
                    ),
                    "point_count": sector.point_count,
                }
                for sector in self.sectors
            ],
            "input_point_count": self.input_point_count,
            "accepted_point_count": self.accepted_point_count,
            "populated_sector_count": self.populated_sector_count,
        }


@dataclass(frozen=True)
class VoxelCell:
    index: VoxelIndex
    point_count: int


@dataclass(frozen=True)
class SparseVoxelMap:
    resolution_m: float
    cells: tuple[VoxelCell, ...]
    input_point_count: int
    accepted_point_count: int
    occupied_voxel_count: int
    omitted_voxel_count: int
    bounds_m: tuple[float, float, float, float, float, float] | None

    @property
    def occupied_indices(self) -> frozenset[VoxelIndex]:
        return frozenset(cell.index for cell in self.cells)

    def to_payload(self) -> dict[str, object]:
        bounds = None
        if self.bounds_m is not None:
            bounds = {
                "min_x_m": self.bounds_m[0],
                "max_x_m": self.bounds_m[1],
                "min_y_m": self.bounds_m[2],
                "max_y_m": self.bounds_m[3],
                "min_z_m": self.bounds_m[4],
                "max_z_m": self.bounds_m[5],
            }
        return {
            "representation": "sparse_occupied_voxels_v1",
            "resolution_m": self.resolution_m,
            "index_convention": (
                "floor(coordinate_m / resolution_m) with 1e-9 boundary epsilon"
            ),
            "cells": [
                {"index": list(cell.index), "point_count": cell.point_count}
                for cell in self.cells
            ],
            "input_point_count": self.input_point_count,
            "accepted_point_count": self.accepted_point_count,
            "occupied_voxel_count": self.occupied_voxel_count,
            "omitted_voxel_count": self.omitted_voxel_count,
            "bounds_m": bounds,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "SparseVoxelMap":
        if payload.get("representation") != "sparse_occupied_voxels_v1":
            raise ValueError("unsupported voxel representation")
        resolution = float(payload["resolution_m"])
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution_m must be finite and positive")
        raw_cells = payload.get("cells")
        if not isinstance(raw_cells, list):
            raise ValueError("cells must be a list")
        cells: list[VoxelCell] = []
        for raw in raw_cells:
            if not isinstance(raw, Mapping):
                raise ValueError("invalid voxel cell")
            raw_index = raw.get("index")
            if not isinstance(raw_index, (list, tuple)) or len(raw_index) != 3:
                raise ValueError("voxel index must contain three integers")
            cells.append(
                VoxelCell(
                    tuple(int(value) for value in raw_index),
                    int(raw.get("point_count", 1)),
                )
            )
        raw_bounds = payload.get("bounds_m")
        bounds = None
        if isinstance(raw_bounds, Mapping):
            bounds = tuple(
                float(raw_bounds[key])
                for key in (
                    "min_x_m",
                    "max_x_m",
                    "min_y_m",
                    "max_y_m",
                    "min_z_m",
                    "max_z_m",
                )
            )
        return cls(
            resolution_m=resolution,
            cells=tuple(sorted(cells, key=lambda cell: cell.index)),
            input_point_count=int(payload.get("input_point_count", 0)),
            accepted_point_count=int(payload.get("accepted_point_count", 0)),
            occupied_voxel_count=int(payload.get("occupied_voxel_count", len(cells))),
            omitted_voxel_count=int(payload.get("omitted_voxel_count", 0)),
            bounds_m=bounds,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class SearchConfig:
    translation_radius_m: float = 0.60
    translation_step_m: float = 0.10
    yaw_radius_rad: float = 0.30
    yaw_step_rad: float = 0.05
    max_points: int = 1200

    def __post_init__(self) -> None:
        if self.translation_radius_m < 0 or self.yaw_radius_rad < 0:
            raise ValueError("search radii cannot be negative")
        if self.translation_step_m <= 0 or self.yaw_step_rad <= 0:
            raise ValueError("search steps must be positive")
        if self.max_points < 1:
            raise ValueError("max_points must be positive")


@dataclass(frozen=True)
class PoseEstimate:
    pose: Pose2D
    score: float
    matched_points: int
    evaluated_points: int
    candidates_evaluated: int
    source: str


def transform_points(points: Iterable[Point3D], pose: Pose2D) -> list[Point3D]:
    """Transform points from the robot frame into a map frame."""
    cosine = math.cos(pose.yaw_rad)
    sine = math.sin(pose.yaw_rad)
    return [
        (
            pose.x_m + cosine * x - sine * y,
            pose.y_m + sine * x + cosine * y,
            z,
        )
        for x, y, z in points
    ]


def angular_proximity(
    points: Iterable[Point3D],
    config: ProximityConfig = ProximityConfig(),
) -> AngularProximity:
    """Reduce base-link points to fixed, bounded nearest-range sectors.

    The full configured sector list is emitted. An unobserved sector has a null
    ``nearest_distance_m`` rather than implying free space. Results are invariant
    to input point order.
    """
    count = _sector_count(config)
    nearest: list[float | None] = [None] * count
    point_counts = [0] * count
    input_count = 0
    accepted_count = 0
    for raw_point in points:
        input_count += 1
        if len(raw_point) != 3:
            continue
        x, y, z = (float(value) for value in raw_point)
        if not all(math.isfinite(value) for value in (x, y, z)):
            continue
        distance = math.hypot(x, y)
        bearing = math.atan2(y, x)
        if not (
            config.min_range_m <= distance <= config.max_range_m
            and config.min_z_m <= z <= config.max_z_m
            and config.min_bearing_rad <= bearing <= config.max_bearing_rad
        ):
            continue
        index = min(
            count - 1,
            int(math.floor((bearing - config.min_bearing_rad) / config.sector_width_rad)),
        )
        accepted_count += 1
        point_counts[index] += 1
        if nearest[index] is None or distance < nearest[index]:
            nearest[index] = distance

    sectors = []
    for index in range(count):
        bearing_min = config.min_bearing_rad + index * config.sector_width_rad
        bearing_max = min(
            config.max_bearing_rad, bearing_min + config.sector_width_rad
        )
        sectors.append(
            ProximitySector(
                index=index,
                bearing_min_rad=bearing_min,
                bearing_center_rad=(bearing_min + bearing_max) / 2.0,
                bearing_max_rad=bearing_max,
                nearest_distance_m=nearest[index],
                point_count=point_counts[index],
            )
        )
    return AngularProximity(
        sector_width_rad=config.sector_width_rad,
        min_bearing_rad=config.min_bearing_rad,
        max_bearing_rad=config.max_bearing_rad,
        min_range_m=config.min_range_m,
        max_range_m=config.max_range_m,
        sectors=tuple(sectors),
        input_point_count=input_count,
        accepted_point_count=accepted_count,
    )


def voxelize(
    points: Iterable[Point3D],
    config: VoxelConfig = VoxelConfig(),
    *,
    sensor_origin: Point3D = (0.0, 0.0, 0.0),
) -> SparseVoxelMap:
    """Convert finite, in-range points to a bounded sparse occupancy summary.

    When more cells exist than ``max_voxels``, the densest cells are retained;
    ties are resolved lexicographically.  The serialized result is always sorted
    lexicographically, making output independent of input order.
    """
    counts: Counter[VoxelIndex] = Counter()
    input_count = 0
    accepted_count = 0
    ox, oy, oz = sensor_origin
    resolution = config.resolution_m
    for raw_point in points:
        input_count += 1
        if len(raw_point) != 3:
            continue
        x, y, z = (float(value) for value in raw_point)
        if not all(math.isfinite(value) for value in (x, y, z)):
            continue
        relative_z = z - oz
        distance = math.sqrt((x - ox) ** 2 + (y - oy) ** 2 + relative_z**2)
        if not (
            config.min_range_m <= distance <= config.max_range_m
            and config.min_z_m <= relative_z <= config.max_z_m
        ):
            continue
        accepted_count += 1
        counts[
            (
                _voxel_coordinate(x, resolution),
                _voxel_coordinate(y, resolution),
                _voxel_coordinate(z, resolution),
            )
        ] += 1

    eligible = [
        VoxelCell(index=index, point_count=count)
        for index, count in counts.items()
        if count >= config.min_points_per_voxel
    ]
    eligible.sort(key=lambda cell: (-cell.point_count, cell.index))
    retained = sorted(eligible[: config.max_voxels], key=lambda cell: cell.index)
    bounds = _bounds_from_cells(retained, resolution)
    return SparseVoxelMap(
        resolution_m=resolution,
        cells=tuple(retained),
        input_point_count=input_count,
        accepted_point_count=accepted_count,
        occupied_voxel_count=len(eligible),
        omitted_voxel_count=max(0, len(eligible) - len(retained)),
        bounds_m=bounds,
    )


def estimate_pose(
    scan_points: Sequence[Point3D],
    reference_map: SparseVoxelMap,
    initial_pose: Pose2D,
    config: SearchConfig = SearchConfig(),
) -> PoseEstimate:
    """Match a scan to occupied reference voxels near an odometry prior.

    This intentionally bounded exhaustive search estimates planar ``x/y/yaw``;
    it is not SLAM and cannot resolve aliases in symmetric rooms.  Candidate
    ties prefer the smallest correction from odometry, then stable numeric order.
    """
    points = _finite_subsample(scan_points, config.max_points)
    if not points or not reference_map.cells:
        return PoseEstimate(initial_pose, 0.0, 0, len(points), 0, "odometry_fallback")

    resolution = reference_map.resolution_m
    occupied = reference_map.occupied_indices
    x_offsets = _symmetric_offsets(
        config.translation_radius_m, config.translation_step_m
    )
    y_offsets = x_offsets
    yaw_offsets = _symmetric_offsets(config.yaw_radius_rad, config.yaw_step_rad)

    best_rank: tuple[int, float, float, float, float] | None = None
    best_pose = initial_pose
    best_matches = 0
    candidates = 0
    for yaw_offset in yaw_offsets:
        yaw = _normalize_angle(initial_pose.yaw_rad + yaw_offset)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        rotated = [(cosine * x - sine * y, sine * x + cosine * y, z) for x, y, z in points]
        for x_offset in x_offsets:
            x_position = initial_pose.x_m + x_offset
            for y_offset in y_offsets:
                y_position = initial_pose.y_m + y_offset
                candidates += 1
                matches = sum(
                    (
                        _voxel_coordinate(x_position + x, resolution),
                        _voxel_coordinate(y_position + y, resolution),
                        _voxel_coordinate(z, resolution),
                    )
                    in occupied
                    for x, y, z in rotated
                )
                correction = x_offset**2 + y_offset**2 + yaw_offset**2
                # max() rank: matches first, then smallest correction and stable
                # preference for negative-to-positive numeric order.
                rank = (matches, -correction, -yaw_offset, -x_offset, -y_offset)
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_matches = matches
                    best_pose = Pose2D(x_position, y_position, yaw)

    return PoseEstimate(
        pose=best_pose,
        score=best_matches / len(points),
        matched_points=best_matches,
        evaluated_points=len(points),
        candidates_evaluated=candidates,
        source="voxel_scan_match",
    )


def _finite_subsample(points: Sequence[Point3D], limit: int) -> list[Point3D]:
    finite = [
        (float(x), float(y), float(z))
        for x, y, z in points
        if all(math.isfinite(float(value)) for value in (x, y, z))
    ]
    if len(finite) <= limit:
        return finite
    # Even sampling preserves the scan's angular coverage and is deterministic.
    return [finite[index * len(finite) // limit] for index in range(limit)]


def _symmetric_offsets(radius: float, step: float) -> tuple[float, ...]:
    count = int(math.floor(radius / step + 1e-9))
    return tuple(round(index * step, 12) for index in range(-count, count + 1))


def _sector_count(config: ProximityConfig) -> int:
    span = config.max_bearing_rad - config.min_bearing_rad
    return int(math.ceil(span / config.sector_width_rad - 1e-12))


def _normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _voxel_coordinate(coordinate: float, resolution: float) -> int:
    # Matrix transforms commonly turn an exact boundary such as 1.0 into
    # 0.9999999999999999. A scale-relative epsilon keeps equivalent geometry in
    # the same cell without materially changing non-boundary coordinates.
    return math.floor(coordinate / resolution + 1e-9)


def _bounds_from_cells(
    cells: Sequence[VoxelCell], resolution: float
) -> tuple[float, float, float, float, float, float] | None:
    if not cells:
        return None
    xs = [cell.index[0] for cell in cells]
    ys = [cell.index[1] for cell in cells]
    zs = [cell.index[2] for cell in cells]
    return (
        min(xs) * resolution,
        (max(xs) + 1) * resolution,
        min(ys) * resolution,
        (max(ys) + 1) * resolution,
        min(zs) * resolution,
        (max(zs) + 1) * resolution,
    )
