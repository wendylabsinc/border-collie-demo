from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from media.visual_odometry import SparseVisualOdometry, VisualOdometryConfig


def textured_frame(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    for _ in range(500):
        x = int(rng.integers(8, 632))
        y = int(rng.integers(8, 352))
        radius = int(rng.integers(1, 4))
        color = tuple(int(value) for value in rng.integers(80, 255, size=3))
        cv2.circle(frame, (x, y), radius, color, -1)
    return frame


def shifted(frame: np.ndarray, x_px: float, angle_deg: float = 0.0) -> np.ndarray:
    transform = cv2.getRotationMatrix2D((320.0, 180.0), angle_deg, 1.0)
    transform[0, 2] += x_px
    return cv2.warpAffine(frame, transform, (640, 360))


def config(**changes: object) -> VisualOdometryConfig:
    values: dict[str, object] = {
        "minimum_interval_s": 0.01,
        "maximum_interval_s": 0.50,
        "processing_budget_s": 1.0,
        "processing_width_px": 424,
        "maximum_features": 500,
        "minimum_tracks": 20,
        "maximum_keyframes": 8,
        "keyframe_interval": 1,
        "loop_exclusion_frames": 3,
        "minimum_loop_matches": 20,
    }
    values.update(changes)
    return VisualOdometryConfig(**values)  # type: ignore[arg-type]


def test_sparse_visual_odometry_tracks_motion_without_gpu() -> None:
    odometry = SparseVisualOdometry(config())
    odometry.reset("camera-1")
    base = textured_frame()

    odometry.observe(
        base,
        captured_monotonic_s=10.0,
        source_pts=1,
        generation="camera-1",
    )
    odometry.observe(
        shifted(base, 6.0, 1.5),
        captured_monotonic_s=10.1,
        source_pts=2,
        generation="camera-1",
    )

    status = odometry.status()
    assert status["state"] == "tracking"
    assert status["uses_gpu"] is False
    assert status["opencv_threads"] == 1
    assert status["backend"] == "opencv-cpu-sparse"
    motion = status["latest_motion"]
    assert isinstance(motion, dict)
    assert motion["inliers"] >= 20
    assert abs(float(motion["image_dx_px"])) > 0.5
    assert abs(float(motion["image_yaw_rad"])) > 0.01
    assert status["resource"]["processing_width_px"] == 424


def test_rate_limit_and_generation_fence_discard_work_before_opencv() -> None:
    odometry = SparseVisualOdometry(config(minimum_interval_s=0.10))
    odometry.reset("camera-1")
    frame = textured_frame()

    odometry.observe(
        frame,
        captured_monotonic_s=10.0,
        source_pts=1,
        generation="camera-1",
    )
    odometry.observe(
        frame,
        captured_monotonic_s=10.02,
        source_pts=2,
        generation="camera-1",
    )
    odometry.observe(
        frame,
        captured_monotonic_s=11.0,
        source_pts=3,
        generation="old-camera",
    )

    status = odometry.status()
    assert status["frame_sequence"] == 1
    assert status["resource"]["skipped_frames"] == 1


def test_processing_over_budget_adaptively_reduces_frame_rate() -> None:
    timestamps = iter((0.0, 0.10))
    odometry = SparseVisualOdometry(
        config(processing_budget_s=0.03),
        clock=lambda: next(timestamps),
    )
    odometry.reset("camera-1")

    odometry.observe(
        textured_frame(),
        captured_monotonic_s=10.0,
        source_pts=1,
        generation="camera-1",
    )

    resource = odometry.status()["resource"]
    assert resource["target_interval_s"] > 0.01
    assert resource["maximum_processing_s"] == 0.10


def test_natural_scene_revisit_is_geometrically_verified_as_loop_closure() -> None:
    odometry = SparseVisualOdometry(config())
    odometry.reset("camera-1")
    base = textured_frame()
    frames = [
        base,
        shifted(base, 4.0),
        shifted(base, 8.0),
        shifted(base, 12.0),
        shifted(base, 8.0),
        shifted(base, 4.0),
        base,
    ]
    for index, frame in enumerate(frames, start=1):
        odometry.observe(
            frame,
            captured_monotonic_s=10.0 + index * 0.1,
            source_pts=index,
            generation="camera-1",
        )

    closure = odometry.status()["loop_closure"]
    assert isinstance(closure, dict)
    assert closure["current_frame_sequence"] == len(frames)
    assert closure["reference_frame_sequence"] <= 3
    assert closure["inliers"] >= 20


def test_reset_discards_trajectory_and_old_scene_memory() -> None:
    odometry = SparseVisualOdometry(config())
    odometry.reset("camera-1")
    frame = textured_frame()
    odometry.observe(
        frame,
        captured_monotonic_s=10.0,
        source_pts=1,
        generation="camera-1",
    )
    odometry.observe(
        shifted(frame, 5.0),
        captured_monotonic_s=10.1,
        source_pts=2,
        generation="camera-1",
    )

    odometry.reset("camera-2")
    status = odometry.status()

    assert status["generation"] == "camera-2"
    assert status["frame_sequence"] == 0
    assert status["latest_motion"] is None
    assert status["loop_closure"] is None
    assert status["resource"]["keyframes"] == 0
