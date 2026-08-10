import math
import time
from types import SimpleNamespace

from border_collie_demo.go2_pose import Go2PoseProvider


def test_pose_provider_accepts_fresh_finite_sport_mode_state() -> None:
    provider = Go2PoseProvider(maximum_age_s=0.50)
    message = SimpleNamespace(
        imu_state=SimpleNamespace(rpy=[0.0, 0.0, 3.5]),
        position=[1.25, -0.50, 0.0],
    )

    provider._on_state(message)
    status = provider.status()

    assert status.healthy is True
    assert status.pose is not None
    assert status.pose.x_m == 1.25
    assert status.pose.y_m == -0.50
    assert -math.pi <= status.pose.yaw_rad <= math.pi


def test_pose_provider_preserves_robot_timestamp_and_motion_evidence() -> None:
    provider = Go2PoseProvider(maximum_age_s=0.50)
    provider._on_state(
        SimpleNamespace(
            stamp=SimpleNamespace(sec=12, nanosec=250_000_000),
            imu_state=SimpleNamespace(
                rpy=[0.0, 0.0, 0.2],
                gyroscope=[0.0, 0.0, 0.04],
            ),
            position=[1.0, 2.0, 0.0],
            velocity=[0.4, -0.1, 0.0],
            yaw_speed=0.3,
            mode=2,
            gait_type=1,
            foot_force=[20.0, 18.0, 22.0, 19.0],
        )
    )

    motion = provider.status().motion

    assert motion is not None
    assert motion.source_timestamp_s == 12.25
    assert motion.velocity_x_mps == 0.4
    assert motion.velocity_y_mps == -0.1
    assert motion.yaw_rate_rps == 0.3
    assert motion.imu_yaw_rate_rps == 0.04
    assert motion.mode == 2
    assert motion.gait_type == 1
    assert motion.contact_feet == 4
    assert motion.stationary_stance is False


def test_pose_provider_qualifies_stationary_stance_for_zero_velocity_updates() -> None:
    provider = Go2PoseProvider(maximum_age_s=0.50)
    provider._on_state(
        SimpleNamespace(
            imu_state=SimpleNamespace(
                rpy=[0.0, 0.0, 0.0],
                gyroscope=[0.0, 0.0, 0.02],
            ),
            position=[0.0, 0.0, 0.0],
            velocity=[0.01, -0.01, 0.0],
            yaw_speed=0.02,
            foot_force=[20.0, 18.0, 0.0, 19.0],
        )
    )

    motion = provider.status().motion

    assert motion is not None
    assert motion.contact_feet == 3
    assert motion.stationary_stance is True


def test_pose_provider_rejects_regressing_robot_timestamp() -> None:
    provider = Go2PoseProvider(maximum_age_s=0.50)
    common = {
        "imu_state": SimpleNamespace(rpy=[0.0, 0.0, 0.0]),
        "position": [0.0, 0.0, 0.0],
    }
    provider._on_state(
        SimpleNamespace(**common, stamp=SimpleNamespace(sec=10, nanosec=0))
    )
    provider._on_state(
        SimpleNamespace(**common, stamp=SimpleNamespace(sec=9, nanosec=0))
    )

    status = provider.status()

    assert status.healthy is True
    assert "regressed" in str(status.sample_rejection)


def test_pose_provider_rejects_invalid_and_stale_samples() -> None:
    provider = Go2PoseProvider(maximum_age_s=0.01)
    provider._on_state(
        SimpleNamespace(
            imu_state=SimpleNamespace(rpy=[0.0, 0.0, float("nan")]),
            position=[0.0, 0.0, 0.0],
        )
    )
    assert provider.status().healthy is False
    assert "non-finite" in str(provider.status().error)

    provider._on_state(
        SimpleNamespace(
            imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.0]),
            position=[0.0, 0.0, 0.0],
        )
    )
    time.sleep(0.02)
    assert provider.status().healthy is False
