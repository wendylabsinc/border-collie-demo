import math

import pytest

from border_collie_demo.pose_fusion import (
    PlanarFusionConfig,
    PlanarFusionObservation,
    PlanarSensorFusion,
    VisualMotionCue,
)
from border_collie_demo.return_home import Pose2D


def observation(
    time_s: float,
    *,
    x_m: float = 0.0,
    y_m: float = 0.0,
    yaw_rad: float = 0.0,
    forward_mps: float | None = None,
    left_mps: float | None = None,
    yaw_rate_rps: float | None = None,
    stationary: bool = False,
    visual: VisualMotionCue | None = None,
) -> PlanarFusionObservation:
    return PlanarFusionObservation(
        raw_pose_from_home=Pose2D(x_m, y_m, yaw_rad),
        captured_monotonic_s=time_s,
        velocity_forward_mps=forward_mps,
        velocity_left_mps=left_mps,
        imu_yaw_rate_rps=yaw_rate_rps,
        contact_feet=4 if stationary else 2,
        stationary_stance=stationary,
        visual=visual,
    )


def visual(
    sequence: int,
    *,
    forward: float | None = 1.0,
    left: float | None = 0.0,
    yaw_delta: float | None = 0.0,
    quality: float = 0.9,
    generation: str = "camera-1",
) -> VisualMotionCue:
    return VisualMotionCue(
        generation=generation,
        sequence=sequence,
        forward_direction=forward,
        left_direction=left,
        yaw_delta_rad=yaw_delta,
        quality=quality,
    )


def test_fuses_metric_pose_velocity_and_imu_prediction() -> None:
    fusion = PlanarSensorFusion()
    fusion.reset(observation(1.0, forward_mps=1.0, left_mps=0.0))

    estimate = fusion.update(
        observation(
            1.1,
            x_m=0.1,
            yaw_rad=0.02,
            forward_mps=1.0,
            left_mps=0.0,
            yaw_rate_rps=0.2,
        )
    )

    assert estimate.trusted is True
    assert estimate.pose_from_home is not None
    assert estimate.pose_from_home.x_m == pytest.approx(0.1, abs=0.01)
    assert estimate.pose_from_home.yaw_rad == pytest.approx(0.02, abs=0.01)
    assert estimate.velocity_x_mps == pytest.approx(1.0, abs=0.05)
    assert "go2_velocity" in estimate.evidence["sources"]
    assert estimate.covariance["position_sigma_m"] < 0.1


def test_stationary_foot_contact_applies_zero_velocity_and_gyro_bias_update() -> None:
    fusion = PlanarSensorFusion()
    fusion.reset(observation(1.0, forward_mps=0.5, left_mps=0.0))

    estimate = fusion.update(
        observation(
            1.1,
            x_m=0.0,
            forward_mps=0.01,
            left_mps=0.0,
            yaw_rate_rps=0.04,
            stationary=True,
        )
    )

    assert estimate.trusted is True
    assert estimate.evidence["zero_velocity_update"] is True
    assert abs(float(estimate.velocity_x_mps)) < 0.02
    assert estimate.yaw_bias_rps is not None
    assert estimate.yaw_bias_rps > 0.0
    assert "stance_zero_velocity" in estimate.evidence["sources"]


def test_visual_direction_and_yaw_are_fused_using_go2_distance_as_scale() -> None:
    fusion = PlanarSensorFusion()
    fusion.reset(observation(1.0, visual=visual(1)))

    estimate = fusion.update(
        observation(
            1.1,
            x_m=0.10,
            y_m=0.02,
            yaw_rad=0.03,
            forward_mps=1.0,
            left_mps=0.0,
            visual=visual(2, forward=1.0, left=0.0, yaw_delta=0.02),
        )
    )

    assert estimate.trusted is True
    assert estimate.evidence["visual"]["direction"] == "fused"
    assert estimate.evidence["visual"]["yaw"] == "fused"
    assert "visual_motion" in estimate.evidence["sources"]
    assert estimate.pose_from_home is not None
    assert estimate.pose_from_home.x_m > 0.07


def test_visual_direction_conflict_is_rejected_without_overriding_metric_pose() -> None:
    fusion = PlanarSensorFusion()
    fusion.reset(observation(1.0, visual=visual(1)))

    estimate = fusion.update(
        observation(
            1.1,
            x_m=0.10,
            forward_mps=1.0,
            left_mps=0.0,
            visual=visual(2, forward=-1.0, left=0.0, yaw_delta=None),
        )
    )

    assert estimate.trusted is True
    assert estimate.evidence["visual"]["direction"] == "rejected"
    assert "visual_motion" not in estimate.evidence["sources"]
    assert estimate.pose_from_home is not None
    assert estimate.pose_from_home.x_m > 0.07


def test_persistent_qualified_visual_conflict_fails_closed() -> None:
    fusion = PlanarSensorFusion(
        PlanarFusionConfig(maximum_consecutive_visual_rejections=3)
    )
    fusion.reset(observation(1.0, visual=visual(1)))

    estimates = [
        fusion.update(
            observation(
                1.0 + sequence * 0.1,
                x_m=sequence * 0.1,
                forward_mps=1.0,
                left_mps=0.0,
                visual=visual(
                    sequence + 1,
                    forward=-1.0,
                    left=0.0,
                    yaw_delta=None,
                ),
            )
        )
        for sequence in range(1, 4)
    ]

    assert estimates[0].trusted is True and estimates[1].trusted is True
    assert estimates[2].trusted is False
    assert "visual motion" in str(estimates[2].unavailable_reason)
    assert estimates[2].evidence["visual"]["consecutive_rejections"] == 3


def test_changed_pose_with_duplicate_timestamp_fails_closed() -> None:
    fusion = PlanarSensorFusion()
    fusion.reset(observation(1.0))

    estimate = fusion.update(observation(1.0, x_m=0.2))

    assert estimate.trusted is False
    assert estimate.pose_from_home is None
    assert "non-advancing" in str(estimate.unavailable_reason)


def test_persistent_metric_pose_jumps_fail_closed() -> None:
    fusion = PlanarSensorFusion(
        PlanarFusionConfig(
            maximum_position_innovation_m=0.20,
            maximum_consecutive_metric_rejections=3,
        )
    )
    fusion.reset(observation(1.0))

    first = fusion.update(observation(1.1, x_m=1.0))
    second = fusion.update(observation(1.2, x_m=2.0))
    third = fusion.update(observation(1.3, x_m=3.0))

    assert first.trusted is True and second.trusted is True
    assert third.trusted is False
    assert "persistently" in str(third.unavailable_reason)
    assert third.evidence["go2_measurement"]["consecutive_rejections"] == 3


def test_rejected_metric_jump_cannot_scale_visual_translation() -> None:
    fusion = PlanarSensorFusion(
        PlanarFusionConfig(maximum_position_innovation_m=0.20)
    )
    fusion.reset(observation(1.0, visual=visual(1)))

    estimate = fusion.update(
        observation(
            1.1,
            x_m=2.0,
            visual=visual(2, forward=1.0, left=0.0, yaw_delta=None),
        )
    )

    assert estimate.trusted is True
    assert estimate.pose_from_home is not None
    assert estimate.pose_from_home.x_m == pytest.approx(0.0)
    assert estimate.evidence["visual"]["direction"] == "metric_scale_rejected"
    assert "visual_motion" not in estimate.evidence["sources"]


def test_sample_gap_and_heading_uncertainty_are_bounded() -> None:
    fusion = PlanarSensorFusion()
    initial = fusion.reset(observation(1.0))
    assert initial.covariance["heading_sigma_rad"] == pytest.approx(
        math.radians(4.0)
    )

    estimate = fusion.update(observation(2.0))

    assert estimate.trusted is False
    assert "gap" in str(estimate.unavailable_reason)


def test_long_gap_can_reseed_only_from_a_stationary_stance() -> None:
    fusion = PlanarSensorFusion()
    fusion.reset(observation(1.0, forward_mps=0.5, left_mps=0.0))

    estimate = fusion.update(
        observation(
            2.0,
            x_m=0.2,
            forward_mps=0.0,
            left_mps=0.0,
            stationary=True,
        )
    )

    assert estimate.trusted is True
    assert estimate.evidence["state"] == "stationary_reseed"
    assert estimate.velocity_x_mps == 0.0
