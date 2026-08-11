import asyncio
import math

import pytest

from border_collie_demo.go2_pose import Go2MotionEvidence
from border_collie_demo.home_localization import (
    HomeEstimateState,
    HomeLocalizationConfig,
    HomeLocalizer,
    VisualOdometryObservation,
)
from border_collie_demo.recovery import FailedRunHomeRecovery
from border_collie_demo.return_home import Pose2D
from border_collie_demo.run_results import RunResultStore


class FixedVisualAdapter:
    def __init__(self, observation: VisualOdometryObservation | None) -> None:
        self.observation = observation

    def observe_motion(self) -> VisualOdometryObservation | None:
        return self.observation


def observation(
    *,
    generation: str = "camera-1",
    frame_sequence: int = 10,
    motion_sequence: int = 8,
    captured_monotonic_s: float = 99.8,
    x_px: float = 0.0,
    y_px: float = 0.0,
    yaw_rad: float = 0.0,
    quality: float = 0.8,
    body_forward: float | None = None,
    body_left: float | None = None,
    body_yaw: float | None = None,
    motion_geometry: str | None = None,
) -> VisualOdometryObservation:
    return VisualOdometryObservation(
        generation=generation,
        frame_sequence=frame_sequence,
        motion_sequence=motion_sequence,
        captured_monotonic_s=captured_monotonic_s,
        trajectory_x_px=x_px,
        trajectory_y_px=y_px,
        trajectory_yaw_rad=yaw_rad,
        motion_quality=quality,
        tracked_features=100,
        inliers=80,
        body_forward_direction=body_forward,
        body_left_direction=body_left,
        body_yaw_delta_rad=body_yaw,
        motion_geometry=motion_geometry,
    )


def motion(
    source_time_s: float,
    *,
    forward_mps: float = 0.0,
    left_mps: float = 0.0,
    yaw_rate_rps: float = 0.0,
    stationary: bool = False,
) -> Go2MotionEvidence:
    return Go2MotionEvidence(
        source_timestamp_s=source_time_s,
        velocity_x_mps=forward_mps,
        velocity_y_mps=left_mps,
        yaw_rate_rps=yaw_rate_rps,
        imu_yaw_rate_rps=yaw_rate_rps,
        mode=1,
        gait_type=0,
        obstacle_ranges_m=None,
        foot_force=(20.0, 20.0, 20.0, 20.0),
        contact_feet=4,
        stationary_stance=stationary,
    )


def test_go2_only_preserves_metric_home_pose_and_exposes_uncertainty() -> None:
    localizer = HomeLocalizer(clock=lambda: 100.0)

    estimate = localizer.estimate(
        Pose2D(2.0, 3.0, math.pi / 2.0),
        Pose2D(2.0, 3.12, math.pi / 2.0),
        odometry_age_s=0.04,
    )

    assert estimate.state is HomeEstimateState.TRUSTED
    assert estimate.source == "go2_metric"
    assert estimate.pose_from_home is not None
    assert estimate.pose_from_home.x_m == pytest.approx(0.12)
    assert estimate.pose_from_home.y_m == pytest.approx(0.0)
    assert estimate.home_distance_m == pytest.approx(0.12)
    assert estimate.evidence["visual"] == {"state": "not_configured"}
    assert estimate.evidence["uncertainty"]["position_sigma_m"] > 0.03


def test_fresh_visual_motion_fuses_yaw_but_does_not_invent_metric_translation() -> None:
    adapter = FixedVisualAdapter(observation())
    localizer = HomeLocalizer(adapter, clock=lambda: 100.0)
    assert localizer.capture_home()["state"] == "captured"
    adapter.observation = observation(
        frame_sequence=20,
        motion_sequence=18,
        x_px=48.0,
        yaw_rad=-0.8,
    )

    estimate = localizer.estimate(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(1.25, -0.25, 1.0),
        odometry_age_s=0.03,
    )

    assert estimate.trusted is True
    assert estimate.source == "go2_metric+visual_yaw"
    assert estimate.pose_from_home is not None
    assert estimate.pose_from_home.x_m == pytest.approx(1.25)
    assert estimate.pose_from_home.y_m == pytest.approx(-0.25)
    assert 0.8 < estimate.pose_from_home.yaw_rad < 1.0
    assert estimate.evidence["visual"]["state"] == "fused"
    assert estimate.evidence["visual"]["trajectory_image_space"]["x_px"] == 48.0


def test_stale_or_restarted_visual_evidence_falls_back_to_fresh_go2_metric() -> None:
    adapter = FixedVisualAdapter(observation())
    localizer = HomeLocalizer(adapter, clock=lambda: 100.0)
    localizer.capture_home()
    adapter.observation = observation(
        generation="camera-2",
        captured_monotonic_s=98.0,
        yaw_rad=-1.0,
    )

    estimate = localizer.estimate(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.4, 0.0, 0.5),
        odometry_age_s=0.02,
    )

    assert estimate.trusted is True
    assert estimate.source == "go2_metric"
    assert estimate.evidence["visual"]["state"] == "generation_changed"


def test_persistent_qualified_visual_yaw_conflict_fails_closed() -> None:
    adapter = FixedVisualAdapter(observation())
    localizer = HomeLocalizer(
        adapter,
        config=HomeLocalizationConfig(
            maximum_visual_yaw_disagreement_rad=math.radians(10.0),
            maximum_consecutive_visual_conflicts=3,
        ),
        clock=lambda: 100.0,
    )
    localizer.capture_home()
    adapter.observation = observation(frame_sequence=20, yaw_rad=-1.2)

    first = localizer.estimate(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.2, 0.0, 0.0),
        odometry_age_s=0.02,
    )
    second = localizer.estimate(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.2, 0.0, 0.0),
        odometry_age_s=0.02,
    )
    third = localizer.estimate(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.2, 0.0, 0.0),
        odometry_age_s=0.02,
    )

    assert first.trusted is True and second.trusted is True
    assert third.state is HomeEstimateState.UNAVAILABLE
    assert "persistently" in str(third.unavailable_reason)
    assert third.evidence["visual"]["consecutive_conflicts"] == 3


def test_stateful_fusion_combines_go2_kinematics_imu_stance_and_visual_motion() -> None:
    adapter = FixedVisualAdapter(
        observation(
            captured_monotonic_s=99.8,
            body_forward=1.0,
            body_left=0.0,
            body_yaw=0.0,
            motion_geometry="essential_matrix_scale_free",
        )
    )
    localizer = HomeLocalizer(adapter, clock=lambda: 100.0)
    captured = localizer.capture_home(
        Pose2D(0.0, 0.0, 0.0),
        captured_monotonic_s=10.0,
        motion=motion(10.0, stationary=True),
    )
    adapter.observation = observation(
        frame_sequence=20,
        motion_sequence=9,
        captured_monotonic_s=99.9,
        body_forward=1.0,
        body_left=0.0,
        body_yaw=0.02,
        motion_geometry="essential_matrix_scale_free",
    )

    estimate = localizer.estimate(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.10, 0.01, 0.02),
        odometry_age_s=0.02,
        captured_monotonic_s=10.1,
        motion=motion(10.1, forward_mps=1.0, yaw_rate_rps=0.2),
    )

    assert captured["sensor_fusion"] == "initialized"
    assert estimate.trusted is True
    assert estimate.source == "planar_sensor_fusion"
    assert estimate.evidence["fusion"]["visual"]["direction"] == "fused"
    assert "go2_velocity" in estimate.evidence["fusion"]["sources"]
    assert estimate.evidence["fusion"]["covariance"]["position_sigma_m"] < 0.1


def test_metric_home_gate_uses_farther_raw_or_filtered_distance() -> None:
    localizer = HomeLocalizer(clock=lambda: 100.0)
    localizer.capture_home(
        Pose2D(0.0, 0.0, 0.0),
        captured_monotonic_s=10.0,
        motion=motion(10.0, stationary=True),
    )

    estimate = localizer.estimate(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.12, 0.0, 0.0),
        odometry_age_s=0.02,
        captured_monotonic_s=10.1,
        motion=motion(10.1),
    )

    assert estimate.trusted is True
    assert estimate.home_distance_m == pytest.approx(0.12)
    assert estimate.to_dict()["filtered_home_distance_m"] <= 0.12
    assert estimate.evidence["fusion"]["metric_gate_policy"] == (
        "max(raw_go2, filtered)"
    )


class CorrectedHomeRecoveryHardware:
    def status(self) -> dict[str, object]:
        return {
            "configured": True,
            "autonomy_enabled": True,
            "connected": True,
            "fault": None,
            "active_operation": None,
            "pose": {
                "healthy": True,
                "age_s": 0.02,
                "pose": {"x_m": 0.095, "y_m": 0.0, "yaw_rad": 0.0},
            },
            "motion": {"initialized": True, "armed": False, "fault": None},
        }

    def estimate_home(self, home: dict[str, object]) -> dict[str, object]:
        return {
            "state": "trusted",
            "trusted": True,
            "source": "go2_metric+visual_yaw",
            "unavailable_reason": None,
            "home_distance_m": 0.095,
            "heading_error_rad": -0.015,
            "pose_from_home": {"x_m": 0.095, "y_m": 0.0, "yaw_rad": 0.015},
            "evidence": {"visual": {"state": "fused"}},
        }

    def start_motion_trace(self, phase: str) -> None:
        raise AssertionError("recovery inside the Home gate must not arm motion")

    def motion_trace(self) -> list[dict[str, object]]:
        return []

    async def turn_toward_home(self, home, **options):
        raise AssertionError("recovery inside the Home gate must not turn")

    async def return_home(self, home, **options):
        raise AssertionError("recovery inside the Home gate must not translate")

    async def emergency_stop(self) -> list[str]:
        return []


def test_final_recovery_verification_uses_trusted_fused_point_one_meter_gate(
    tmp_path,
) -> None:
    results = RunResultStore(tmp_path)
    run = results.start_run(target_fruit="pear", activation_source="test")
    results.record_home(
        run["run_id"],
        {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
    )
    results.seal(
        run["run_id"],
        phase="failed",
        outcome="FAILED",
        reason="ARRIVAL_FAILURE",
        message="test failure",
        final_safety_state="DISARMED_CONFIRMED",
    )
    recovery = FailedRunHomeRecovery(CorrectedHomeRecoveryHardware(), results)
    recovery.validate(results.get(run["run_id"]))
    attempt = results.start_recovery(run["run_id"], confirmation="confirmed")

    completed = asyncio.run(recovery.run(run["run_id"], attempt["recovery_id"]))

    assert completed["outcome"] == "COMPLETED"
    assert completed["reason"] == "HOME_POSITION_ALREADY_RECOVERED"
    assert completed["final_evidence"]["home_distance_m"] == pytest.approx(0.095)
    assert completed["final_evidence"]["home_localization"]["source"] == (
        "go2_metric+visual_yaw"
    )
