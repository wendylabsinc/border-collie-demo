"""Bounded CPU-only sparse visual odometry and natural-scene loop closure.

The module deliberately reports image-space motion rather than pretending that
a monocular camera can produce metric translation without calibration.  The
mission process combines this independent direction/yaw evidence with the Go2's
metric motion state.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class VisualOdometryConfig:
    enabled: bool = True
    minimum_interval_s: float = 0.10
    maximum_interval_s: float = 0.50
    processing_budget_s: float = 0.030
    processing_width_px: int = 424
    maximum_features: int = 800
    minimum_tracks: int = 30
    maximum_keyframes: int = 48
    keyframe_interval: int = 10
    loop_exclusion_frames: int = 30
    minimum_loop_matches: int = 35

    def __post_init__(self) -> None:
        positive_floats = (
            self.minimum_interval_s,
            self.maximum_interval_s,
            self.processing_budget_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive_floats):
            raise ValueError("visual odometry timing values must be finite and positive")
        if self.minimum_interval_s > self.maximum_interval_s:
            raise ValueError("minimum visual interval exceeds maximum interval")
        positive_ints = (
            self.processing_width_px,
            self.maximum_features,
            self.minimum_tracks,
            self.maximum_keyframes,
            self.keyframe_interval,
            self.loop_exclusion_frames,
            self.minimum_loop_matches,
        )
        if any(value < 1 for value in positive_ints):
            raise ValueError("visual odometry count limits must be positive")
        if self.minimum_tracks > self.maximum_features:
            raise ValueError("minimum tracks exceeds maximum features")

    @classmethod
    def from_env(cls, env: dict[str, str]) -> VisualOdometryConfig:
        enabled = env.get("VISUAL_ODOMETRY_ENABLED", "1").strip().casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return cls(
            enabled=enabled,
            minimum_interval_s=float(env.get("VISUAL_ODOMETRY_INTERVAL_S", "0.10")),
            maximum_interval_s=float(
                env.get("VISUAL_ODOMETRY_MAX_INTERVAL_S", "0.50")
            ),
            processing_budget_s=float(
                env.get("VISUAL_ODOMETRY_PROCESSING_BUDGET_S", "0.030")
            ),
            processing_width_px=int(
                env.get("VISUAL_ODOMETRY_PROCESSING_WIDTH_PX", "424")
            ),
            maximum_features=int(env.get("VISUAL_ODOMETRY_MAX_FEATURES", "800")),
            minimum_tracks=int(env.get("VISUAL_ODOMETRY_MIN_TRACKS", "30")),
            maximum_keyframes=int(
                env.get("VISUAL_ODOMETRY_MAX_KEYFRAMES", "48")
            ),
            keyframe_interval=int(
                env.get("VISUAL_ODOMETRY_KEYFRAME_INTERVAL", "10")
            ),
            loop_exclusion_frames=int(
                env.get("VISUAL_ODOMETRY_LOOP_EXCLUSION_FRAMES", "30")
            ),
            minimum_loop_matches=int(
                env.get("VISUAL_ODOMETRY_MIN_LOOP_MATCHES", "35")
            ),
        )


@dataclass(frozen=True)
class _Keyframe:
    frame_sequence: int
    descriptors: Any
    keypoints_xy: Any
    trajectory_x_px: float
    trajectory_y_px: float
    trajectory_yaw_rad: float


class SparseVisualOdometry:
    """Observe frames through one small, thread-safe status interface."""

    def __init__(
        self,
        config: VisualOdometryConfig | None = None,
        *,
        clock: Any = time.monotonic,
    ) -> None:
        self.config = config or VisualOdometryConfig()
        self._clock = clock
        self._lock = Lock()
        self._generation: str | None = None
        self._last_processed_s: float | None = None
        self._previous_gray: Any | None = None
        self._previous_points: Any | None = None
        self._frame_sequence = 0
        self._motion_sequence = 0
        self._trajectory_x_px = 0.0
        self._trajectory_y_px = 0.0
        self._trajectory_yaw_rad = 0.0
        self._current_interval_s = self.config.minimum_interval_s
        self._under_budget_streak = 0
        self._processed_frames = 0
        self._skipped_frames = 0
        self._rejected_frames = 0
        self._total_processing_s = 0.0
        self._maximum_processing_s = 0.0
        self._latest_motion: dict[str, object] | None = None
        self._latest_loop: dict[str, object] | None = None
        self._error: str | None = None
        self._opencv_configured = False
        self._keyframes: deque[_Keyframe] = deque(
            maxlen=self.config.maximum_keyframes
        )

    def reset(self, generation: str) -> None:
        if not generation.strip():
            raise ValueError("visual odometry generation is required")
        with self._lock:
            self._generation = generation
            self._last_processed_s = None
            self._previous_gray = None
            self._previous_points = None
            self._frame_sequence = 0
            self._motion_sequence = 0
            self._trajectory_x_px = 0.0
            self._trajectory_y_px = 0.0
            self._trajectory_yaw_rad = 0.0
            self._current_interval_s = self.config.minimum_interval_s
            self._under_budget_streak = 0
            self._processed_frames = 0
            self._skipped_frames = 0
            self._rejected_frames = 0
            self._total_processing_s = 0.0
            self._maximum_processing_s = 0.0
            self._latest_motion = None
            self._latest_loop = None
            self._error = None
            self._keyframes.clear()

    def observe(
        self,
        bgr: Any,
        *,
        captured_monotonic_s: float,
        source_pts: int,
        generation: str,
    ) -> None:
        """Process at most one bounded-rate frame; stale generations are fenced."""
        if not self.config.enabled:
            return
        with self._lock:
            if generation != self._generation:
                return
            if (
                self._last_processed_s is not None
                and captured_monotonic_s - self._last_processed_s
                < self._current_interval_s
            ):
                self._skipped_frames += 1
                return
            self._last_processed_s = captured_monotonic_s

        started = self._clock()
        try:
            gray = self._prepare_gray(bgr)
            result = self._track(gray)
            elapsed_s = max(0.0, self._clock() - started)
            with self._lock:
                if generation != self._generation:
                    return
                self._frame_sequence += 1
                self._processed_frames += 1
                self._total_processing_s += elapsed_s
                self._maximum_processing_s = max(self._maximum_processing_s, elapsed_s)
                self._adapt_interval(elapsed_s)
                self._apply_result(
                    gray,
                    result,
                    captured_monotonic_s=captured_monotonic_s,
                    source_pts=source_pts,
                    processing_s=elapsed_s,
                )
                self._error = None
        except Exception as exc:  # noqa: BLE001 - OpenCV errors must degrade only VO
            elapsed_s = max(0.0, self._clock() - started)
            with self._lock:
                if generation == self._generation:
                    self._processed_frames += 1
                    self._rejected_frames += 1
                    self._total_processing_s += elapsed_s
                    self._maximum_processing_s = max(
                        self._maximum_processing_s, elapsed_s
                    )
                    self._adapt_interval(elapsed_s)
                    self._previous_gray = None
                    self._previous_points = None
                    self._error = f"{type(exc).__name__}: {exc}"

    def status(self) -> dict[str, object]:
        with self._lock:
            average_s = (
                None
                if self._processed_frames == 0
                else self._total_processing_s / self._processed_frames
            )
            return {
                "enabled": self.config.enabled,
                "backend": "opencv-cpu-sparse",
                "uses_gpu": False,
                "opencv_threads": 1,
                "generation": self._generation,
                "state": (
                    "disabled"
                    if not self.config.enabled
                    else "tracking"
                    if self._latest_motion is not None
                    else "initializing"
                ),
                "frame_sequence": self._frame_sequence,
                "motion_sequence": self._motion_sequence,
                "latest_motion": (
                    None if self._latest_motion is None else dict(self._latest_motion)
                ),
                "trajectory_image_space": {
                    "x_px": self._trajectory_x_px,
                    "y_px": self._trajectory_y_px,
                    "yaw_rad": self._trajectory_yaw_rad,
                },
                "loop_closure": (
                    None if self._latest_loop is None else dict(self._latest_loop)
                ),
                "resource": {
                    "processing_width_px": self.config.processing_width_px,
                    "target_interval_s": self._current_interval_s,
                    "processing_budget_s": self.config.processing_budget_s,
                    "processed_frames": self._processed_frames,
                    "skipped_frames": self._skipped_frames,
                    "rejected_frames": self._rejected_frames,
                    "average_processing_s": average_s,
                    "maximum_processing_s": self._maximum_processing_s,
                    "keyframes": len(self._keyframes),
                    "maximum_keyframes": self.config.maximum_keyframes,
                },
                "error": self._error,
            }

    def _prepare_gray(self, bgr: Any) -> Any:
        import cv2

        if not self._opencv_configured:
            cv2.setNumThreads(1)
            if hasattr(cv2, "ocl"):
                cv2.ocl.setUseOpenCL(False)
            self._opencv_configured = True
        height, width = bgr.shape[:2]
        if width < 2 or height < 2:
            raise ValueError("camera frame is too small")
        target_width = min(width, self.config.processing_width_px)
        target_height = max(2, round(height * target_width / width))
        resized = cv2.resize(
            bgr,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    def _track(self, gray: Any) -> dict[str, Any]:
        import cv2
        import numpy as np

        previous_gray = self._previous_gray
        previous_points = self._previous_points
        if previous_gray is None or previous_points is None:
            return {"state": "bootstrap", "points": self._detect_points(gray)}
        current_points, status, _errors = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            gray,
            previous_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if current_points is None or status is None:
            return {"state": "lost", "points": self._detect_points(gray)}
        mask = status.reshape(-1) == 1
        before = previous_points.reshape(-1, 2)[mask]
        after = current_points.reshape(-1, 2)[mask]
        if len(before) < self.config.minimum_tracks:
            return {
                "state": "lost",
                "tracked": len(before),
                "points": self._detect_points(gray),
            }
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            before,
            after,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            maxIters=1000,
            confidence=0.995,
        )
        if affine is None or inlier_mask is None:
            return {
                "state": "lost",
                "tracked": len(before),
                "points": self._detect_points(gray),
            }
        inliers = int(np.count_nonzero(inlier_mask))
        if inliers < self.config.minimum_tracks:
            return {
                "state": "lost",
                "tracked": len(before),
                "inliers": inliers,
                "points": self._detect_points(gray),
            }
        # The image transform is the inverse sign of camera motion. Keep the
        # raw convention explicit; the mission estimator owns body-frame use.
        yaw_rad = math.atan2(float(affine[1, 0]), float(affine[0, 0]))
        return {
            "state": "tracked",
            "tracked": len(before),
            "inliers": inliers,
            "dx_px": float(affine[0, 2]),
            "dy_px": float(affine[1, 2]),
            "yaw_rad": yaw_rad,
            "quality": min(1.0, inliers / max(len(before), self.config.minimum_tracks)),
            "points": after.reshape(-1, 1, 2).astype("float32"),
        }

    def _detect_points(self, gray: Any) -> Any:
        import cv2

        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.maximum_features,
            qualityLevel=0.01,
            minDistance=7.0,
            blockSize=7,
        )

    def _apply_result(
        self,
        gray: Any,
        result: dict[str, Any],
        *,
        captured_monotonic_s: float,
        source_pts: int,
        processing_s: float,
    ) -> None:
        points = result.get("points")
        self._previous_gray = gray
        self._previous_points = points
        if result.get("state") != "tracked":
            self._rejected_frames += 1
            return
        dx_px = float(result["dx_px"])
        dy_px = float(result["dy_px"])
        yaw_rad = float(result["yaw_rad"])
        cosine = math.cos(self._trajectory_yaw_rad)
        sine = math.sin(self._trajectory_yaw_rad)
        self._trajectory_x_px += cosine * dx_px - sine * dy_px
        self._trajectory_y_px += sine * dx_px + cosine * dy_px
        self._trajectory_yaw_rad = _normalize_angle(
            self._trajectory_yaw_rad + yaw_rad
        )
        self._motion_sequence += 1
        self._latest_motion = {
            "sequence": self._motion_sequence,
            "frame_sequence": self._frame_sequence,
            "source_pts": source_pts,
            "captured_monotonic_s": captured_monotonic_s,
            "image_dx_px": dx_px,
            "image_dy_px": dy_px,
            "image_yaw_rad": yaw_rad,
            "tracked_features": int(result["tracked"]),
            "inliers": int(result["inliers"]),
            "quality": float(result["quality"]),
            "processing_s": processing_s,
        }
        if self._frame_sequence % self.config.keyframe_interval == 0:
            self._consider_keyframe(gray)

    def _consider_keyframe(self, gray: Any) -> None:
        import cv2
        import numpy as np

        orb = cv2.ORB_create(nfeatures=min(500, self.config.maximum_features))
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < self.config.minimum_loop_matches:
            return
        points = np.asarray([point.pt for point in keypoints], dtype="float32")
        best: dict[str, object] | None = None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        for candidate in self._keyframes:
            if (
                self._frame_sequence - candidate.frame_sequence
                < self.config.loop_exclusion_frames
            ):
                continue
            pairs = matcher.knnMatch(candidate.descriptors, descriptors, k=2)
            good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
            if len(good) < self.config.minimum_loop_matches:
                continue
            before = np.asarray(
                [candidate.keypoints_xy[match.queryIdx] for match in good],
                dtype="float32",
            )
            after = np.asarray(
                [points[match.trainIdx] for match in good],
                dtype="float32",
            )
            affine, inlier_mask = cv2.estimateAffinePartial2D(
                before,
                after,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.5,
                maxIters=1000,
                confidence=0.995,
            )
            if affine is None or inlier_mask is None:
                continue
            inliers = int(np.count_nonzero(inlier_mask))
            if inliers < self.config.minimum_loop_matches:
                continue
            quality = inliers / max(len(good), self.config.minimum_loop_matches)
            closure = {
                "reference_frame_sequence": candidate.frame_sequence,
                "current_frame_sequence": self._frame_sequence,
                "matches": len(good),
                "inliers": inliers,
                "quality": min(1.0, quality),
                "relative_image_dx_px": float(affine[0, 2]),
                "relative_image_dy_px": float(affine[1, 2]),
                "relative_image_yaw_rad": math.atan2(
                    float(affine[1, 0]), float(affine[0, 0])
                ),
                "reference_trajectory_image_space": {
                    "x_px": candidate.trajectory_x_px,
                    "y_px": candidate.trajectory_y_px,
                    "yaw_rad": candidate.trajectory_yaw_rad,
                },
            }
            if best is None or float(closure["quality"]) > float(best["quality"]):
                best = closure
        self._latest_loop = best
        self._keyframes.append(
            _Keyframe(
                frame_sequence=self._frame_sequence,
                descriptors=descriptors,
                keypoints_xy=points,
                trajectory_x_px=self._trajectory_x_px,
                trajectory_y_px=self._trajectory_y_px,
                trajectory_yaw_rad=self._trajectory_yaw_rad,
            )
        )

    def _adapt_interval(self, elapsed_s: float) -> None:
        if elapsed_s > self.config.processing_budget_s:
            self._current_interval_s = min(
                self.config.maximum_interval_s,
                max(self.config.minimum_interval_s, self._current_interval_s * 1.5),
            )
            self._under_budget_streak = 0
            return
        self._under_budget_streak += 1
        if self._under_budget_streak >= 10:
            self._current_interval_s = max(
                self.config.minimum_interval_s,
                self._current_interval_s / 1.25,
            )
            self._under_budget_streak = 0


def _normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))
