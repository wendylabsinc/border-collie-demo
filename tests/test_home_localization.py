import asyncio
import math

import pytest

from border_collie_demo.home_localization import (
    AbsoluteHomeObservation,
    HomeEstimateState,
    HomeLocalizationConfig,
    HomeLocalizer,
)
from border_collie_demo.recovery import FailedRunHomeRecovery
from border_collie_demo.return_home import Pose2D
from border_collie_demo.run_results import RunResultStore


class FixedAbsoluteAdapter:
    def __init__(self, observation: AbsoluteHomeObservation | None) -> None:
        self.observation = observation

    def observe_home(self, home: Pose2D) -> AbsoluteHomeObservation | None:
        return self.observation


def observation(
    pose: Pose2D,
    *,
    captured_monotonic_s: float = 99.8,
) -> AbsoluteHomeObservation:
    return AbsoluteHomeObservation(
        pose_from_home=pose,
        captured_monotonic_s=captured_monotonic_s,
        source="apriltag-camera-adapter",
        reference_id="home-tag-7",
    )


def test_odometry_only_preserves_existing_home_distance_and_exposes_absence() -> None:
    localizer = HomeLocalizer(clock=lambda: 100.0)

    estimate = localizer.estimate(
        Pose2D(2.0, 3.0, math.pi / 2.0),
        Pose2D(2.0, 3.12, math.pi / 2.0),
        odometry_age_s=0.04,
    )

    assert estimate.state is HomeEstimateState.TRUSTED
    assert estimate.source == "odometry_only"
    assert estimate.pose_from_home is not None
    assert estimate.pose_from_home.x_m == pytest.approx(0.12)
    assert estimate.pose_from_home.y_m == pytest.approx(0.0)
    assert estimate.pose_from_home.yaw_rad == pytest.approx(0.0)
    assert estimate.home_distance_m == pytest.approx(0.12)
    assert estimate.evidence["absolute"] == {"state": "not_configured"}


def test_fresh_fiducial_corrects_an_agreeing_odometry_estimate() -> None:
    localizer = HomeLocalizer(
        FixedAbsoluteAdapter(observation(Pose2D(0.08, 0.0, 0.02))),
        clock=lambda: 100.0,
    )

    estimate = localizer.estimate(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.14, 0.0, 0.0),
        odometry_age_s=0.03,
    )

    assert estimate.trusted is True
    assert estimate.source == "odometry+absolute_fiducial"
    assert estimate.home_distance_m == pytest.approx(0.095)
    assert estimate.pose_from_home is not None
    assert estimate.pose_from_home.yaw_rad == pytest.approx(0.015)
    assert estimate.evidence["disagreement"]["position_m"] == pytest.approx(0.06)


def test_stale_fiducial_makes_home_unavailable_instead_of_falling_back() -> None:
    localizer = HomeLocalizer(
        FixedAbsoluteAdapter(
            observation(Pose2D(0.08, 0.0, 0.0), captured_monotonic_s=98.0)
        ),
        clock=lambda: 100.0,
    )

    estimate = localizer.estimate(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.09, 0.0, 0.0),
        odometry_age_s=0.02,
    )

    assert estimate.state is HomeEstimateState.UNAVAILABLE
    assert estimate.home_distance_m is None
    assert "stale" in str(estimate.unavailable_reason)
    assert estimate.evidence["absolute"]["age_s"] == pytest.approx(2.0)


def test_fiducial_disagreement_fails_closed_with_both_measurements() -> None:
    localizer = HomeLocalizer(
        FixedAbsoluteAdapter(observation(Pose2D(-0.8, 0.0, math.pi))),
        config=HomeLocalizationConfig(
            maximum_position_disagreement_m=0.30,
            maximum_heading_disagreement_rad=math.radians(30.0),
        ),
        clock=lambda: 100.0,
    )

    estimate = localizer.estimate(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.10, 0.0, 0.0),
        odometry_age_s=0.02,
    )

    assert estimate.trusted is False
    assert estimate.pose_from_home is None
    assert "disagree" in str(estimate.unavailable_reason)
    assert estimate.evidence["disagreement"]["position_m"] == pytest.approx(0.9)
    assert estimate.evidence["disagreement"]["heading_rad"] == pytest.approx(
        math.pi
    )


class CorrectedHomeRecoveryHardware:
    """Raw odometry is outside the gate; the trusted fused estimate is inside."""

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
                "pose": {"x_m": 0.14, "y_m": 0.0, "yaw_rad": 0.0},
            },
            "motion": {"initialized": True, "armed": False, "fault": None},
        }

    def estimate_home(self, home: dict[str, object]) -> dict[str, object]:
        return {
            "state": "trusted",
            "trusted": True,
            "source": "odometry+absolute_fiducial",
            "unavailable_reason": None,
            "home_distance_m": 0.095,
            "heading_error_rad": -0.015,
            "pose_from_home": {"x_m": 0.095, "y_m": 0.0, "yaw_rad": 0.015},
            "evidence": {"absolute": {"reference_id": "home-tag-7"}},
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
        "odometry+absolute_fiducial"
    )
