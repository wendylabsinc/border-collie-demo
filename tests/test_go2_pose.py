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
