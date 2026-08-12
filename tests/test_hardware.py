from __future__ import annotations

import asyncio
import json
import math
import time
from itertools import pairwise

import pytest

from border_collie_demo.config import HardwareConfig
from border_collie_demo.flight_recorder import FlightRecorder
from border_collie_demo.go2_lidar import LidarHandoffObservation
from border_collie_demo.go2_motion import MotionConfig
from border_collie_demo.go2_pose import Go2MotionEvidence, PoseStatus
from border_collie_demo.hardware import (
    FORWARD_PULSE_CONFIRMATION,
    CameraFailure,
    HardwareManager,
    HardwareUnavailable,
    RangeUnavailable,
    TargetLost,
)
from border_collie_demo.models import Pose, VelocityCommand
from border_collie_demo.qualified_tracking import SearchQualificationHandoff
from border_collie_demo.search_policy import SearchPolicy
from border_collie_demo.target_range import MetricArrivalGate, RangeCalibration


class FakeMotion:
    def __init__(self) -> None:
        self.armed = False
        self.initialized = False
        self.commands: list[VelocityCommand] = []
        self.stop_calls = 0
        self.postures: list[str] = []

    def status(self) -> dict[str, object]:
        return {
            "initialized": self.initialized,
            "armed": self.armed,
            "fault": None,
            "last_command": (
                self.commands[-1].to_dict()
                if self.commands
                else VelocityCommand(reason="idle").to_dict()
            ),
        }

    async def initialize(self) -> None:
        self.initialized = True

    async def arm(self, authority=None) -> str:
        self.armed = True
        self.authority = authority
        return "lease"

    async def command(self, lease: str, command: VelocityCommand) -> VelocityCommand:
        assert lease == "lease"
        assert self.armed
        self.commands.append(command)
        return command

    async def release(self, lease: str) -> None:
        assert lease == "lease"
        self.armed = False

    async def emergency_stop(self) -> list[str]:
        self.stop_calls += 1
        self.armed = False
        return []

    async def close(self) -> list[str]:
        self.armed = False
        return []

    async def stand_down(self) -> None:
        assert not self.armed
        self.postures.append("stand_down")

    async def stand_up(self, *, settle_s: float = 0.0) -> None:
        assert not self.armed
        self.postures.append("stand_up")


class FakePose:
    def __init__(self, *, healthy: bool = True) -> None:
        self.started = False
        self.healthy = healthy

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.started = False

    def status(self) -> PoseStatus:
        return PoseStatus(
            Pose(0.0, 0.0, 0.0, 1.0),
            0.0,
            self.started and self.healthy,
            None if self.started and self.healthy else "pose unavailable",
        )


class MetricRangePose(FakePose):
    def __init__(self, motion: FakeMotion, ranges: tuple[float, ...]) -> None:
        super().__init__()
        self._motion_adapter = motion
        self._ranges = iter(ranges)
        self._last_range = ranges[-1]

    def status(self) -> PoseStatus:
        self._last_range = next(self._ranges, self._last_range)
        command = (
            self._motion_adapter.commands[-1]
            if self._motion_adapter.commands
            else VelocityCommand(reason="idle")
        )
        moving = command.forward_mps > 0.0 or command.yaw_rps != 0.0
        return PoseStatus(
            Pose(0.0, 0.0, 0.0, 1.0),
            0.01,
            self.started,
            None,
            Go2MotionEvidence(
                source_timestamp_s=1.0,
                velocity_x_mps=command.forward_mps if moving else 0.0,
                velocity_y_mps=0.0,
                yaw_rate_rps=command.yaw_rps if moving else 0.0,
                imu_yaw_rate_rps=command.yaw_rps if moving else 0.0,
                mode=3 if moving else 1,
                gait_type=1 if moving else 0,
                obstacle_ranges_m=(0.0, self._last_range, 0.0, 0.0),
                foot_force=(20.0, 20.0, 20.0, 20.0),
                contact_feet=4,
                stationary_stance=not moving,
            ),
        )


class ScriptedLidarHandoff:
    def __init__(self, clearances_m: tuple[float, ...]) -> None:
        self._clearances = iter(clearances_m)
        self._last = clearances_m[-1]
        self.started = False
        self.calls: list[dict[str, object]] = []
        self.visual_calls: list[dict[str, object]] = []

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.started = False

    def status(self) -> dict[str, object]:
        return {"configured": True, "ready": self.started, "fresh": self.started}

    def note_visual_track(
        self,
        *,
        center_error_ratio: float | None,
        close_authorized: bool,
    ) -> None:
        self.visual_calls.append(
            {
                "center_error_ratio": center_error_ratio,
                "close_authorized": close_authorized,
            }
        )

    def observe(
        self,
        *,
        visual_close_authorized: bool,
        visual_center_error_ratio: float | None,
        allow_handoff: bool,
    ) -> LidarHandoffObservation:
        self.calls.append(
            {
                "visual_close_authorized": visual_close_authorized,
                "allow_handoff": allow_handoff,
            }
        )
        self._last = next(self._clearances, self._last)
        mode = "lidar_handoff" if allow_handoff else "lidar_visual_association"
        return LidarHandoffObservation(
            available=True,
            reason="scripted_lidar",
            front_clearance_m=self._last,
            age_s=0.01,
            confidence=0.80,
            association_valid=True,
            association_mode=mode,
            handoff_active=allow_handoff,
        )


class ContinuousFusionPose(FakePose):
    def __init__(self) -> None:
        super().__init__()
        self._source_time_s = 0.0

    def status(self) -> PoseStatus:
        self._source_time_s += 0.20
        return PoseStatus(
            Pose(0.0, 0.0, 0.0, self._source_time_s),
            0.0,
            self.started,
            None,
            Go2MotionEvidence(
                source_timestamp_s=self._source_time_s,
                velocity_x_mps=0.0,
                velocity_y_mps=0.0,
                yaw_rate_rps=0.0,
                imu_yaw_rate_rps=0.0,
                mode=5,
                gait_type=0,
                obstacle_ranges_m=None,
                foot_force=(0.0, 0.0, 0.0, 0.0),
                contact_feet=0,
                stationary_stance=True,
                stationary_evidence_source="verified_posture_kinematics",
            ),
        )


class TurningPose(FakePose):
    def __init__(self) -> None:
        super().__init__()
        self._yaws = iter((0.0, 0.0, 0.4, 0.9, 1.3, 1.52))
        self._last_yaw = 0.0

    def status(self) -> PoseStatus:
        if self.started:
            self._last_yaw = next(self._yaws, self._last_yaw)
        return PoseStatus(
            Pose(0.0, 0.0, self._last_yaw, 1.0),
            0.0,
            self.started,
            None if self.started else "pose unavailable",
        )


class ReturningPose(FakePose):
    def __init__(self) -> None:
        super().__init__()
        self._poses = iter(
            (
                Pose(1.0, 0.0, math.pi, 1.0),
                Pose(0.8, 0.0, math.pi, 1.1),
                Pose(0.5, 0.0, math.pi, 1.2),
                Pose(0.2, 0.0, math.pi, 1.3),
                Pose(0.09, 0.0, math.pi, 1.4),
            )
        )
        self._last = Pose(1.0, 0.0, math.pi, 1.0)

    def status(self) -> PoseStatus:
        if self.started:
            self._last = next(self._poses, self._last)
        return PoseStatus(
            self._last,
            0.0,
            self.started,
            None if self.started else "pose unavailable",
        )


class CommandDrivenReturningPose(FakePose):
    """Return trace fixture whose pose cadence is independent of instrumentation."""

    def __init__(self, motion: FakeMotion) -> None:
        super().__init__()
        self._motion = motion

    def status(self) -> PoseStatus:
        positions = (0.8, 0.5, 0.2, 0.09)
        index = min(len(self._motion.commands), len(positions) - 1)
        x_m = positions[index]
        return PoseStatus(
            Pose(x_m, 0.0, math.pi, 1.0 + index * 0.1),
            0.0,
            self.started,
            None if self.started else "pose unavailable",
        )


class BreadcrumbPose(FakePose):
    def __init__(self) -> None:
        super().__init__()
        self.returning = False
        self._outbound = iter((0.0, 0.25, 0.50, 0.75, 1.0))
        self._returning = iter((1.0, 1.0, 0.75, 0.50, 0.25, 0.05))
        self._x = 0.0

    def status(self) -> PoseStatus:
        if self.started:
            values = self._returning if self.returning else self._outbound
            self._x = next(values, self._x)
        yaw = math.pi if self.returning else 0.0
        return PoseStatus(Pose(self._x, 0.0, yaw, 1.0), 0.0, self.started, None)


class HomeTurnPose(FakePose):
    def __init__(self) -> None:
        super().__init__()
        self._poses = iter(
            Pose(1.0, 0.0, yaw, 1.0)
            for yaw in (0.0, 0.0, 0.0, 0.0, 0.8, 1.6, 2.4, 3.10)
        )
        self._last = Pose(1.0, 0.0, 0.0, 1.0)

    def status(self) -> PoseStatus:
        if self.started:
            self._last = next(self._poses, self._last)
        return PoseStatus(self._last, 0.0, self.started, None)


class RecoveringHomeTurnPose(FakePose):
    def __init__(self) -> None:
        super().__init__()
        self.recovered = False
        # Readiness and bearing checks take fresh samples before the measured
        # turn captures its own baseline.
        self._recovered_yaws = iter((0.0, 0.0, 0.0, 0.0, 0.8, 1.6, 2.4, 3.10))
        self._last_yaw = 0.0

    def status(self) -> PoseStatus:
        if self.started and self.recovered:
            self._last_yaw = next(self._recovered_yaws, self._last_yaw)
        return PoseStatus(
            Pose(1.0, 0.0, self._last_yaw, 1.0),
            0.0,
            self.started,
            None if self.started else "pose unavailable",
        )


class RecoveringMotion(FakeMotion):
    def __init__(self, pose: RecoveringHomeTurnPose) -> None:
        super().__init__()
        self._pose = pose

    async def stand_up(self, *, settle_s: float = 0.0) -> None:
        await super().stand_up(settle_s=settle_s)
        self._pose.recovered = True


class RestorePose(FakePose):
    def __init__(self) -> None:
        super().__init__()
        yaws = (math.pi / 2.0,) * 4 + (1.0, 0.5, 0.04, 0.04)
        self._poses = iter(Pose(0.08, 0.0, yaw, 1.0) for yaw in yaws)
        self._last = Pose(0.08, 0.0, math.pi / 2.0, 1.0)

    def status(self) -> PoseStatus:
        if self.started:
            self._last = next(self._poses, self._last)
        return PoseStatus(self._last, 0.0, self.started, None)


def live_config() -> HardwareConfig:
    return HardwareConfig(
        enabled=True,
        autonomy_enabled=True,
        lab_motion_enabled=True,
        network_interface="test0",
        forward_pulse_mps=0.55,
        forward_pulse_duration_s=0.04,
        command_heartbeat_s=0.01,
        command_watchdog_s=0.03,
        remote_api_settle_s=0.0,
        maximum_forward_mps=1.0,
    )


def test_hardware_start_connects_dds_motion_and_pose() -> None:
    async def scenario() -> None:
        dds_calls: list[str | None] = []
        motion = FakeMotion()
        pose = FakePose()
        manager = HardwareManager(
            live_config(),
            dds_initializer=dds_calls.append,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: pose,
        )

        await manager.start()

        assert dds_calls == ["test0"]
        assert motion.initialized is True
        assert pose.started is True
        assert manager.status()["connected"] is True
        assert manager.status()["can_pulse_forward"] is True
        await manager.close()

    asyncio.run(scenario())


def test_continuous_fusion_remains_trusted_through_seven_second_posture_sequence() -> (
    None
):
    async def scenario() -> None:
        motion = FakeMotion()
        pose = ContinuousFusionPose()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: pose,
        )
        await manager.start()
        home = manager.capture_home()

        samples = [manager.ingest_home_fusion_sample() for _ in range(35)]
        turn = await manager.turn_toward_home(
            home,
            yaw_rps=0.50,
            tolerance_rad=math.radians(5.0),
            timeout_s=1.0,
        )

        assert all(sample["trusted"] is True for sample in samples)
        fusion = manager.status()["continuous_home_fusion"]
        assert fusion["consecutive_trusted_samples"] >= 35
        assert fusion["latest"]["trusted"] is True
        assert turn["home_distance_m"] == pytest.approx(0.0)
        await manager.close()

    asyncio.run(scenario())


def test_continuous_fusion_adapter_failure_is_recorded_untrusted() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        pose = ContinuousFusionPose()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: pose,
        )
        await manager.start()
        manager.capture_home()

        def fail_status() -> PoseStatus:
            raise RuntimeError("pose reader failed")

        pose.status = fail_status  # type: ignore[method-assign]
        sample = manager.ingest_home_fusion_sample()

        assert sample == {
            "trusted": False,
            "unavailable_reason": (
                "continuous fusion ingestion failed: pose reader failed"
            ),
        }
        await manager.close()

    asyncio.run(scenario())


def test_continuous_home_fusion_is_written_to_the_run_black_box(tmp_path) -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        pose = ContinuousFusionPose()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: pose,
        )
        recorder = FlightRecorder(tmp_path)
        manager.set_flight_recorder(recorder)
        manager.set_motion_authority("run-1", "epoch-1", "capture_home")
        await manager.start()
        manager.capture_home()

        manager.ingest_home_fusion_sample()
        trace = recorder.snapshot_run("run-1")
        events = [json.loads(line) for line in trace.artifact.content.splitlines()]

        fusion_events = [
            event for event in events if event["kind"] == "home_fusion_sample"
        ]
        assert len(fusion_events) == 1
        payload = fusion_events[0]["payload"]
        assert payload["epoch"] == "epoch-1"
        assert payload["phase"] == "capture_home"
        assert payload["raw_pose"]["pose"] == {
            "x_m": 0.0,
            "y_m": 0.0,
            "yaw_rad": 0.0,
            "captured_monotonic_s": 0.4,
        }
        assert payload["estimate"]["trusted"] is True
        assert "covariance" in payload["estimate"]
        await manager.close()

    asyncio.run(scenario())


def test_measured_turn_stops_from_fresh_pose_progress() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: TurningPose(),
        )
        await manager.start()

        result = await manager.turn_relative(
            math.pi / 2.0,
            yaw_rps=0.8,
            tolerance_rad=0.06,
            timeout_s=0.25,
        )

        assert result["motion_path"] == "sport_client"
        assert result["requested_angle_rad"] == pytest.approx(math.pi / 2.0)
        assert result["measured_yaw_change_rad"] == pytest.approx(1.52)
        assert len(motion.commands) >= 3
        assert all(command.forward_mps == 0.0 for command in motion.commands)
        assert all(command.yaw_rps == 0.8 for command in motion.commands)
        assert motion.armed is False
        assert manager.status()["active_operation"] is None
        await manager.close()

    asyncio.run(scenario())


def test_posture_actions_run_disarmed_for_demo_stages() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        lowered = await manager.stand_down()
        raised = await manager.stand_up()

        assert motion.postures == ["stand_down", "stand_up"]
        assert lowered == {
            "posture": "stand_down",
            "motion_commands_sent": True,
        }
        assert raised == {
            "posture": "balance_stand",
            "motion_commands_sent": True,
            "settle_s": 1.0,
        }
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_find_target_turns_until_fresh_stable_perception_then_stops() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: TurningPose(),
        )
        await manager.start()
        manager.start_motion_trace("turn_to_fruit")
        statuses = iter(
            (
                {"camera_healthy": True, "target_ready": False, "detail": "no pear"},
                {"camera_healthy": True, "target_ready": False, "detail": "no pear"},
                {
                    "camera_healthy": True,
                    "target_ready": True,
                    "generation": "camera-1",
                    "source": {"pts": 123, "time_base": "1/90000"},
                    "detection": {
                        "label": "pear",
                        "generation": "camera-1",
                        "source_pts": 123,
                        "source_time_base": "1/90000",
                        "confidence": 0.81,
                        "consecutive_detections": 5,
                        "center_x_ratio": 0.50,
                        "center_y_ratio": 0.50,
                        "bottom_ratio": 0.65,
                        "bbox_area_ratio": 0.04,
                    },
                },
            )
        )

        result = await manager.find_target(
            lambda: next(
                statuses,
                {"camera_healthy": True, "target_ready": False},
            ),
            "pear",
            yaw_rps=0.20,
            sweep_rad=1.4,
            timeout_s=0.25,
        )

        assert result["label"] == "pear"
        assert result["motion_path"] == "sport_client"
        assert result["stable_detections"] == 5
        assert result["motion_commands_sent"] is True
        assert result["search_qualification"]["target_fruit"] == "pear"
        assert result["search_qualification"]["source_pts"] == 123
        assert all(command.forward_mps == 0.0 for command in motion.commands)
        assert all(
            command["phase"] == "turn_to_fruit" and command["forward_mps"] == 0.0
            for command in manager.motion_trace()
        )
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_find_target_fast_lock_replays_three_fresh_apple_observations() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: TurningPose(),
            search_policy=SearchPolicy.named("fast-lock"),
        )
        await manager.start()

        def apple_observation(consecutive: int, pts: int) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": False,
                "generation": "camera-apple",
                "source": {
                    "pts": pts,
                    "time_base": "1/90000",
                    "age_s": 0.02,
                },
                "detection": {
                    "label": "apple",
                    "generation": "camera-apple",
                    "source_pts": pts,
                    "source_time_base": "1/90000",
                    "confidence": 0.78,
                    "consecutive_detections": consecutive,
                    "inference_s": 0.08,
                    "age_s": 0.03,
                    "center_x_ratio": 0.48,
                    "center_y_ratio": 0.55,
                    "bottom_ratio": 0.67,
                    "bbox_area_ratio": 0.04,
                },
            }

        statuses = iter(
            (
                apple_observation(1, 100),
                apple_observation(2, 200),
                apple_observation(3, 300),
            )
        )

        result = await manager.find_target(
            lambda: next(statuses),
            "apple",
            yaw_rps=1.0,
            sweep_rad=2.0 * math.pi,
            timeout_s=1.0,
        )

        assert result["stable_detections"] == 3
        assert result["recognition"]["search_policy"] == "fast-lock"
        assert result["recognition"]["search_lock_minimum_detections"] == 3
        assert result["search_qualification"]["stable_detections"] == 3
        assert all(command.forward_mps == 0.0 for command in motion.commands)
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_find_target_slow_sweep_locks_three_fresh_apple_observations() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: TurningPose(),
            search_policy=SearchPolicy.named("slow-sweep"),
        )
        await manager.start()

        def apple_observation(consecutive: int, pts: int) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": False,
                "generation": "camera-apple",
                "source": {
                    "pts": pts,
                    "time_base": "1/90000",
                    "age_s": 0.02,
                },
                "detection": {
                    "label": "apple",
                    "generation": "camera-apple",
                    "source_pts": pts,
                    "source_time_base": "1/90000",
                    "confidence": 0.78,
                    "consecutive_detections": consecutive,
                    "inference_s": 0.08,
                    "age_s": 0.03,
                    "center_x_ratio": 0.48,
                    "center_y_ratio": 0.55,
                    "bottom_ratio": 0.67,
                    "bbox_area_ratio": 0.04,
                },
            }

        statuses = iter(
            (
                apple_observation(1, 100),
                apple_observation(2, 200),
                apple_observation(3, 300),
            )
        )

        result = await manager.find_target(
            lambda: next(statuses),
            "apple",
            yaw_rps=0.5,
            sweep_rad=2.0 * math.pi,
            timeout_s=1.0,
        )

        assert result["stable_detections"] == 3
        assert result["recognition"]["search_policy"] == "slow-sweep"
        assert result["recognition"]["search_lock_minimum_detections"] == 3
        assert result["search_qualification"]["stable_detections"] == 3
        assert all(command.forward_mps == 0.0 for command in motion.commands)
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_find_target_double_back_revisits_a_high_confidence_apple_bearing() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        clock = 0.0

        def monotonic() -> float:
            return clock

        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
            search_policy=SearchPolicy.named("double-back"),
            monotonic=monotonic,
        )
        await manager.start()

        def apple(
            source_pts: int,
            consecutive: int,
            *,
            ready: bool = False,
        ) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": ready,
                "generation": "camera-1",
                "source": {
                    "pts": source_pts,
                    "time_base": "1/90000",
                    "age_s": 0.02,
                },
                "detection": {
                    "label": "apple",
                    "generation": "camera-1",
                    "source_pts": source_pts,
                    "source_time_base": "1/90000",
                    "confidence": 0.806853175163269,
                    "consecutive_detections": consecutive,
                    "inference_s": 0.08,
                    "age_s": 0.01,
                    "center_x_ratio": 0.44,
                    "center_y_ratio": 0.50,
                    "bottom_ratio": 0.64,
                    "bbox_area_ratio": 0.04,
                },
            }

        statuses = iter(
            (
                {"camera_healthy": True, "target_ready": False},
                apple(100, 2),
                {"camera_healthy": True, "target_ready": False},
                {"camera_healthy": True, "target_ready": False},
                {"camera_healthy": True, "target_ready": False},
                {"camera_healthy": True, "target_ready": False},
                apple(200, 2),
                apple(300, 3, ready=True),
            )
        )

        def read_status() -> dict[str, object]:
            nonlocal clock
            clock += 0.2
            return next(statuses)

        result = await manager.find_target(
            read_status,
            "apple",
            yaw_rps=1.0,
            sweep_rad=2.0 * math.pi,
            timeout_s=5.0,
        )

        assert result["recognition"]["search_policy"] == "double-back"
        assert result["recognition"]["double_back_episodes"] == 1
        assert any(
            command.reason == "candidate_double_back" and command.yaw_rps < 0.0
            for command in motion.commands
        )
        assert all(command.forward_mps == 0.0 for command in motion.commands)
        assert result["stable_detections"] == 3
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "candidate",
    (
        {"label": "apple", "confidence": 0.49, "age_s": 0.01},
        {"label": "pear", "confidence": 0.95, "age_s": 0.01},
        {"label": "apple", "confidence": 0.95, "age_s": 0.251},
        {
            "label": "apple",
            "generation": "camera-old",
            "confidence": 0.95,
            "age_s": 0.01,
        },
    ),
)
def test_double_back_ignores_unsafe_candidate_evidence(
    candidate: dict[str, object],
) -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        configured_candidate = {
            "generation": "camera-1",
            "source_pts": 100,
            "source_time_base": "1/90000",
            "consecutive_detections": 1,
            **candidate,
        }
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
            search_policy=SearchPolicy.named("double-back"),
        )
        await manager.start()
        statuses = iter(
            (
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "generation": "camera-1",
                    "source": {"pts": 100, "time_base": "1/90000"},
                    "detection": configured_candidate,
                },
                {
                    "camera_healthy": True,
                    "target_ready": True,
                    "generation": "camera-1",
                    "source": {
                        "pts": 200,
                        "time_base": "1/90000",
                        "age_s": 0.02,
                    },
                    "detection": {
                        "label": "apple",
                        "generation": "camera-1",
                        "source_pts": 200,
                        "source_time_base": "1/90000",
                        "confidence": 0.85,
                        "consecutive_detections": 3,
                        "inference_s": 0.08,
                        "age_s": 0.01,
                        "center_x_ratio": 0.5,
                        "center_y_ratio": 0.5,
                        "bottom_ratio": 0.65,
                        "bbox_area_ratio": 0.04,
                    },
                },
            )
        )

        result = await manager.find_target(
            lambda: next(statuses),
            "apple",
            yaw_rps=1.0,
            sweep_rad=2.0 * math.pi,
            timeout_s=0.5,
        )

        assert [command.reason for command in motion.commands] == ["find_target"]
        assert result["recognition"]["double_back_episodes"] == 0
        assert all(command.forward_mps == 0.0 for command in motion.commands)
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_double_back_one_frame_noise_is_bounded_then_search_resumes() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        clock = 0.0
        sample = 0

        def monotonic() -> float:
            return clock

        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
            search_policy=SearchPolicy.named("double-back"),
            monotonic=monotonic,
        )
        await manager.start()

        def detection(pts: int, consecutive: int, ready: bool) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": ready,
                "generation": "camera-1",
                "source": {
                    "pts": pts,
                    "time_base": "1/90000",
                    "age_s": 0.02,
                },
                "detection": {
                    "label": "apple",
                    "generation": "camera-1",
                    "source_pts": pts,
                    "source_time_base": "1/90000",
                    "confidence": 0.81,
                    "consecutive_detections": consecutive,
                    "inference_s": 0.08,
                    "age_s": 0.01,
                    "center_x_ratio": 0.5,
                    "center_y_ratio": 0.5,
                    "bottom_ratio": 0.65,
                    "bbox_area_ratio": 0.04,
                },
            }

        def read_status() -> dict[str, object]:
            nonlocal clock, sample
            clock += 0.2
            sample += 1
            if sample == 1:
                return detection(100, 1, False)
            if sample == 14:
                return detection(200, 3, True)
            return {"camera_healthy": True, "target_ready": False}

        result = await manager.find_target(
            read_status,
            "apple",
            yaw_rps=1.0,
            sweep_rad=2.0 * math.pi,
            timeout_s=5.0,
        )

        reverse = [
            command
            for command in motion.commands
            if command.reason == "candidate_double_back"
        ]
        assert 1 <= len(reverse) <= 4
        assert any(command.reason == "find_target" for command in motion.commands[8:])
        assert result["recognition"]["double_back_episodes"] == 1
        assert result["recognition"]["double_back_maximum_episodes"] == 2
        assert all(command.forward_mps == 0.0 for command in motion.commands)
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_find_target_holds_during_crop_confirmation() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: TurningPose(),
        )
        await manager.start()

        tentative = {
            "camera_healthy": True,
            "target_ready": False,
            "detection": {
                "label": "pear",
                "confidence": 0.58,
                "consecutive_detections": 0,
                "age_s": 0.01,
                "center_x_ratio": 0.50,
                "crop_confirmation": {
                    "attempted": True,
                    "promoted": False,
                    "full_frame_confidence": 0.58,
                    "crop_confidence": 0.61,
                },
            },
        }
        statuses = iter(
            (
                {"camera_healthy": True, "target_ready": False},
                tentative,
                tentative,
                {"camera_healthy": True, "target_ready": False},
                {
                    "camera_healthy": True,
                    "target_ready": True,
                    "detection": {
                        "label": "pear",
                        "confidence": 0.81,
                        "consecutive_detections": 5,
                    },
                },
            )
        )

        result = await manager.find_target(
            lambda: next(
                statuses,
                {
                    "camera_healthy": True,
                    "target_ready": True,
                    "detection": {
                        "label": "pear",
                        "confidence": 0.81,
                        "consecutive_detections": 5,
                    },
                },
            ),
            "pear",
            yaw_rps=0.50,
            sweep_rad=2.0 * math.pi,
            timeout_s=0.25,
        )

        reasons = [command.reason for command in motion.commands]
        assert reasons == [
            "find_target",
            "candidate_alignment_hold",
            "candidate_alignment_hold",
            "candidate_alignment_hold_lost",
        ]
        assert [command.yaw_rps for command in motion.commands] == [
            0.50,
            0.0,
            0.0,
            0.0,
        ]
        assert result["recognition"]["crop_confirmation_samples"] == 2
        assert result["recognition"]["crop_slowdown_hold_samples"] == 2
        assert result["recognition"].get("crop_slowdown_turn_samples", 0) == 0
        assert result["recognition"]["crop_candidate_confidence_threshold"] == 0.50
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_find_target_centers_candidate_at_fine_rate_after_hold() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        clock = 0.0

        def monotonic() -> float:
            return clock

        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
            monotonic=monotonic,
        )
        await manager.start()

        def candidate(consecutive: int, *, ready: bool = False) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": ready,
                "detection": {
                    "label": "pear",
                    "confidence": 0.85,
                    "consecutive_detections": consecutive,
                    "age_s": 0.01,
                    "center_x_ratio": 0.20,
                },
            }

        statuses = iter(
            (
                candidate(1),
                candidate(2),
                candidate(3),
                candidate(4),
                candidate(4),
                candidate(4),
                candidate(5, ready=True),
            )
        )

        def read_status() -> dict[str, object]:
            nonlocal clock
            clock += 0.2
            return next(statuses)

        result = await manager.find_target(
            read_status,
            "pear",
            yaw_rps=1.0,
            sweep_rad=2.0 * math.pi,
            timeout_s=5.0,
        )

        assert result["stable_detections"] == 5
        assert [command.reason for command in motion.commands] == [
            "candidate_alignment_hold",
            "candidate_alignment_hold",
            "candidate_alignment_hold",
            "candidate_alignment_hold",
            "candidate_alignment_turn",
            "candidate_alignment_turn",
        ]
        assert [command.yaw_rps for command in motion.commands] == [
            0.0,
            0.0,
            0.0,
            0.0,
            0.2,
            0.2,
        ]
        assert result["recognition"]["candidate_lock_hold_s"] == 0.75
        assert result["recognition"]["candidate_lock_yaw_rps"] == 0.2
        assert result["recognition"]["crop_slowdown_hold_samples"] == 4
        assert result["recognition"]["crop_slowdown_turn_samples"] == 2
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_slow_sweep_second_scan_centers_candidate_at_fine_rate() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        clock = 0.0

        def monotonic() -> float:
            return clock

        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
            monotonic=monotonic,
        )
        await manager.start()

        def candidate(
            center_x_ratio: float,
            *,
            consecutive: int = 0,
            ready: bool = False,
        ) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": ready,
                "detection": {
                    "label": "pear",
                    "confidence": 0.66,
                    "consecutive_detections": consecutive,
                    "age_s": 0.01,
                    "center_x_ratio": center_x_ratio,
                },
            }

        statuses = iter(
            (
                candidate(0.10),
                candidate(0.10),
                candidate(0.10),
                candidate(0.10),
                candidate(0.10),
                candidate(0.40),
                candidate(0.60),
                candidate(0.70),
                candidate(0.70),
                candidate(0.50, consecutive=5, ready=True),
            )
        )

        def read_status() -> dict[str, object]:
            nonlocal clock
            clock += 0.2
            return next(statuses)

        result = await manager.find_target(
            read_status,
            "pear",
            yaw_rps=0.50,
            sweep_rad=2.0 * math.pi,
            timeout_s=5.0,
            search_policy=SearchPolicy.named("slow-sweep"),
        )

        assert result["stable_detections"] == 5
        assert [command.reason for command in motion.commands] == [
            "candidate_alignment_hold",
            "candidate_alignment_hold",
            "candidate_alignment_hold",
            "candidate_alignment_hold",
            "candidate_alignment_turn",
            "candidate_alignment_centered",
            "candidate_alignment_centered",
            "candidate_alignment_confirm_direction",
            "candidate_alignment_turn",
        ]
        assert [command.yaw_rps for command in motion.commands] == [
            0.0,
            0.0,
            0.0,
            0.0,
            0.20,
            0.0,
            0.0,
            0.0,
            -0.20,
        ]
        assert result["recognition"]["candidate_lock_confidence_threshold"] == 0.50
        assert result["recognition"]["candidate_alignment_yaw_rps"] == 0.20
        assert result["recognition"]["candidate_alignment_center_ratio"] == 0.12
        assert result["recognition"]["candidate_lock_count"] == 1
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_slow_sweep_second_scan_holds_through_brief_candidate_loss() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        clock = 0.0

        def monotonic() -> float:
            return clock

        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
            monotonic=monotonic,
        )
        await manager.start()

        candidate = {
            "camera_healthy": True,
            "target_ready": False,
            "detection": {
                "label": "pear",
                "confidence": 0.66,
                "consecutive_detections": 1,
                "age_s": 0.01,
                "center_x_ratio": 0.20,
            },
        }
        missing = {"camera_healthy": True, "target_ready": False}
        statuses = iter(
            (
                candidate,
                missing,
                missing,
                missing,
                missing,
                missing,
                missing,
                {
                    "camera_healthy": True,
                    "target_ready": True,
                    "detection": {
                        "label": "pear",
                        "confidence": 0.70,
                        "consecutive_detections": 5,
                    },
                },
            )
        )

        def read_status() -> dict[str, object]:
            nonlocal clock
            clock += 0.2
            return next(statuses)

        result = await manager.find_target(
            read_status,
            "pear",
            yaw_rps=0.50,
            sweep_rad=2.0 * math.pi,
            timeout_s=5.0,
            search_policy=SearchPolicy.named("slow-sweep"),
        )

        assert result["stable_detections"] == 5
        assert [command.reason for command in motion.commands] == [
            "candidate_alignment_hold",
            "candidate_alignment_hold_lost",
            "candidate_alignment_hold_lost",
            "candidate_alignment_hold_lost",
            "candidate_alignment_hold_lost",
            "candidate_alignment_hold_lost",
            "find_target",
        ]
        assert [command.yaw_rps for command in motion.commands] == [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.50,
        ]
        assert result["recognition"]["candidate_lock_count"] == 1
        assert result["recognition"]["candidate_lock_losses"] == 1
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_find_target_returns_to_broad_search_after_bounded_candidate_loss() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        clock = 0.0

        def monotonic() -> float:
            return clock

        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
            monotonic=monotonic,
        )
        await manager.start()

        statuses = iter(
            (
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "detection": {
                        "label": "pear",
                        "confidence": 0.85,
                        "consecutive_detections": 1,
                        "age_s": 0.01,
                        "center_x_ratio": 0.20,
                    },
                },
                {"camera_healthy": True, "target_ready": False},
                {"camera_healthy": True, "target_ready": False},
                {"camera_healthy": True, "target_ready": False},
                {"camera_healthy": True, "target_ready": False},
                {"camera_healthy": True, "target_ready": False},
                {"camera_healthy": True, "target_ready": False},
                {
                    "camera_healthy": True,
                    "target_ready": True,
                    "detection": {
                        "label": "pear",
                        "confidence": 0.85,
                        "consecutive_detections": 5,
                    },
                },
            )
        )

        def read_status() -> dict[str, object]:
            nonlocal clock
            clock += 0.2
            return next(statuses)

        result = await manager.find_target(
            read_status,
            "pear",
            yaw_rps=1.0,
            sweep_rad=2.0 * math.pi,
            timeout_s=5.0,
        )

        assert result["stable_detections"] == 5
        assert [command.reason for command in motion.commands] == [
            "candidate_alignment_hold",
            "candidate_alignment_hold_lost",
            "candidate_alignment_hold_lost",
            "candidate_alignment_hold_lost",
            "candidate_alignment_hold_lost",
            "candidate_alignment_hold_lost",
            "find_target",
        ]
        assert [command.yaw_rps for command in motion.commands] == [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        assert result["recognition"]["candidate_lock_losses"] == 1
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("confidence", "age_s"),
    ((0.49, 0.01), (0.85, 0.251)),
)
def test_find_target_does_not_lock_unqualified_candidate(
    confidence: float,
    age_s: float,
) -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()
        statuses = iter(
            (
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "detection": {
                        "label": "pear",
                        "confidence": confidence,
                        "consecutive_detections": 1,
                        "age_s": age_s,
                    },
                },
                {
                    "camera_healthy": True,
                    "target_ready": True,
                    "detection": {
                        "label": "pear",
                        "confidence": 0.85,
                        "consecutive_detections": 5,
                    },
                },
            )
        )

        result = await manager.find_target(
            lambda: next(statuses),
            "pear",
            yaw_rps=1.0,
            sweep_rad=2.0 * math.pi,
            timeout_s=0.5,
        )

        assert [command.reason for command in motion.commands] == ["find_target"]
        assert [command.yaw_rps for command in motion.commands] == [1.0]
        assert result["recognition"].get("candidate_lock_count", 0) == 0
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_find_target_holds_a_plausible_pear_long_enough_to_qualify() -> None:
    async def scenario() -> None:
        motion = FakeMotion()

        class MotionCoupledTurningPose(FakePose):
            def __init__(self) -> None:
                super().__init__()
                self.yaw = 0.0

            def status(self) -> PoseStatus:
                if self.started and motion.commands:
                    self.yaw += abs(motion.commands[-1].yaw_rps) * 0.2
                return PoseStatus(
                    Pose(0.0, 0.0, self.yaw, 1.0),
                    0.0,
                    self.started,
                    None if self.started else "pose unavailable",
                )

        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: MotionCoupledTurningPose(),
        )
        await manager.start()

        def candidate(consecutive: int, *, ready: bool = False) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": ready,
                "detection": {
                    "label": "pear",
                    "confidence": 0.85,
                    "consecutive_detections": consecutive,
                    "age_s": 0.01,
                    "center_x_ratio": 0.50,
                    "crop_confirmation": {
                        "attempted": True,
                        "promoted": ready,
                        "full_frame_confidence": 0.85,
                        "crop_confidence": 0.88,
                    },
                },
            }

        statuses = iter(
            (
                {"camera_healthy": True, "target_ready": False},
                candidate(1),
                candidate(2),
                candidate(3),
                candidate(4),
                candidate(4),
                candidate(5, ready=True),
            )
        )

        result = await manager.find_target(
            lambda: next(statuses),
            "pear",
            yaw_rps=1.0,
            sweep_rad=0.5,
            timeout_s=0.5,
        )

        assert result["stable_detections"] == 5
        assert [command.reason for command in motion.commands] == [
            "find_target",
            "candidate_alignment_hold",
            "candidate_alignment_hold",
            "candidate_alignment_hold",
            "candidate_alignment_hold",
            "candidate_alignment_hold",
        ]
        assert [command.yaw_rps for command in motion.commands] == [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_failed_search_reports_the_best_distant_pear_evidence() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: TurningPose(),
        )
        await manager.start()
        statuses = iter(
            (
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "detection": {
                        "label": "pear",
                        "confidence": 0.01,
                        "consecutive_detections": 0,
                        "source_pts": 100,
                        "bbox_xyxy": [100, 100, 228, 172],
                        "bbox_area_ratio": 0.01,
                    },
                },
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "detection": {
                        "label": "pear",
                        "confidence": 0.03,
                        "consecutive_detections": 0,
                        "source_pts": 200,
                        "bbox_xyxy": [100, 100, 356, 244],
                        "bbox_area_ratio": 0.04,
                    },
                },
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "detection": {
                        "label": "pear",
                        "confidence": 0.02,
                        "consecutive_detections": 0,
                        "source_pts": 300,
                        "bbox_xyxy": [120, 120, 248, 192],
                        "bbox_area_ratio": 0.01,
                    },
                },
            )
        )

        with pytest.raises(TargetLost) as failure:
            await manager.find_target(
                lambda: next(statuses),
                "pear",
                yaw_rps=0.20,
                sweep_rad=0.8,
                timeout_s=0.25,
            )

            assert failure.value.evidence == {
                "samples": 2,
                "pear_candidate_samples": 2,
                "maximum_confidence": 0.03,
                "maximum_consecutive_detections": 0,
                "maximum_bbox_area_ratio": 0.04,
                "search_policy": "slow-sweep",
                "search_lock_minimum_detections": 5,
                "closest_detection": {
                "source_pts": 200,
                "confidence": 0.03,
                "bbox_xyxy": [100, 100, 356, 244],
                "bbox_area_ratio": 0.04,
            },
            "search_progress_rad": pytest.approx(0.9),
        }
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_camera_final_approach_replay_survives_weak_and_stale_before_fresh_loss() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def seen(
            source_pts: int,
            *,
            center_x: float,
            center_y: float,
            bottom: float,
            confidence: float = 0.81,
            age_s: float = 0.01,
        ) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": confidence >= 0.65,
                "generation": "camera-1",
                "source": {"pts": source_pts, "age_s": 0.01},
                "detection": {
                    "label": "pear",
                    "generation": "camera-1",
                    "source_pts": source_pts,
                    "confidence": confidence,
                    "consecutive_detections": 5,
                    "center_x_ratio": center_x,
                    "center_y_ratio": center_y,
                    "bottom_ratio": bottom,
                    "bbox_area_ratio": 0.12,
                    "age_s": age_s,
                },
            }

        statuses = iter(
            (
                seen(1, center_x=0.53, center_y=0.50, bottom=0.65),
                seen(2, center_x=0.52, center_y=0.50, bottom=0.65),
                seen(3, center_x=0.51, center_y=0.50, bottom=0.65),
                seen(4, center_x=0.53, center_y=0.75, bottom=0.90),
                seen(5, center_x=0.52, center_y=0.76, bottom=0.91),
                seen(6, center_x=0.51, center_y=0.77, bottom=0.92),
                seen(
                    7,
                    center_x=0.51,
                    center_y=0.91,
                    bottom=0.997,
                    confidence=0.27,
                ),
                seen(
                    8,
                    center_x=0.51,
                    center_y=0.91,
                    bottom=0.997,
                    confidence=0.27,
                    age_s=0.40,
                ),
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "generation": "camera-1",
                    "source": {"pts": 9, "age_s": 0.01},
                    "detection": {},
                    "detail": "pear offscreen",
                },
            )
        )

        result = await manager.approach_target(
            lambda: next(
                statuses,
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "generation": "camera-1",
                    "source": {"pts": 10, "age_s": 0.01},
                    "detection": {},
                },
            ),
            "pear",
            forward_mps=0.55,
            maximum_yaw_rps=0.30,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            close_range_mps=0.55,
            final_push_mps=0.6,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        assert result["arrival_confirmed"] is True
        assert result["near_confirmations"] == 3
        assert result["final_push_mps"] == 0.6
        assert result["final_push_count"] == 1
        assert result["arrival_mode"] == "final_approach_loss_confirmed"
        assert result["final_approach_loss_samples"] == 2
        assert result["forward_pulse_count"] >= 4
        assert any(
            command.reason == "approach_target_slow" and command.forward_mps == 0.55
            for command in motion.commands
        )
        assert any(command.forward_mps == 0.55 for command in motion.commands)
        assert any(
            command.reason == "camera_final_push" and command.forward_mps == 0.6
            for command in motion.commands
        )
        assert motion.commands[-1].reason == "qualified_arrival_stop"
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_camera_bottom_clip_replay_stops_then_runs_one_configured_final_push() -> None:
    """Replay the decisive observations from run acfdb493 through hardware."""

    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def seen(
            source_pts: int,
            *,
            confidence: float,
            center_x: float,
            center_y: float,
            bottom: float,
            area: float,
        ) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": confidence >= 0.65,
                "generation": "camera-1",
                "source": {"pts": source_pts, "age_s": 0.01},
                "detection": {
                    "label": "pear",
                    "generation": "camera-1",
                    "source_pts": source_pts,
                    "confidence": confidence,
                    "consecutive_detections": 5,
                    "center_x_ratio": center_x,
                    "center_y_ratio": center_y,
                    "bottom_ratio": bottom,
                    "bbox_area_ratio": area,
                    "age_s": 0.01,
                },
            }

        statuses = iter(
            (
                *(
                    seen(
                        source_pts,
                        confidence=0.80,
                        center_x=0.54,
                        center_y=0.50,
                        bottom=0.65,
                        area=0.005,
                    )
                    for source_pts in (276000, 276060, 276180)
                ),
                seen(
                    276360,
                    confidence=0.821043848991394,
                    center_x=0.541015625,
                    center_y=0.9125,
                    bottom=0.9680555555555556,
                    area=0.005815972222222222,
                ),
                seen(
                    276540,
                    confidence=0.7732153534889221,
                    center_x=0.5453125,
                    center_y=0.9361111111111111,
                    bottom=0.9972222222222222,
                    area=0.006493055555555555,
                ),
                seen(
                    276660,
                    confidence=0.7592233419418335,
                    center_x=0.5546875,
                    center_y=0.9430555555555555,
                    bottom=1.0,
                    area=0.00640625,
                ),
                seen(
                    276780,
                    confidence=0.644224226474762,
                    center_x=0.56015625,
                    center_y=0.9597222222222223,
                    bottom=1.0,
                    area=0.0050347222222222225,
                ),
                seen(
                    276960,
                    confidence=0.8311417698860168,
                    center_x=0.57109375,
                    center_y=0.9729166666666667,
                    bottom=1.0,
                    area=0.0022851562500000003,
                ),
                seen(
                    277320,
                    confidence=0.1436695158481598,
                    center_x=0.57109375,
                    center_y=0.975,
                    bottom=0.9958333333333333,
                    area=0.0017578125,
                ),
            )
        )
        fallback = {
            "camera_healthy": True,
            "target_ready": False,
            "generation": "camera-1",
            "source": {"pts": 277440, "age_s": 0.01},
            "detection": {},
        }

        result = await manager.approach_target(
            lambda: next(statuses, fallback),
            "pear",
            forward_mps=0.55,
            maximum_yaw_rps=0.30,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            close_range_mps=0.55,
            final_push_mps=0.6,
            final_push_duration_s=1.0,
            timeout_s=2.0,
        )

        first_loss_stop = next(
            command
            for command in motion.commands
            if command.reason
            == "qualified_track_confirming_final_approach_loss"
        )
        push_commands = [
            command
            for command in motion.commands
            if command.reason == "camera_final_push"
        ]
        assert first_loss_stop.forward_mps == 0.0
        assert push_commands
        assert {command.forward_mps for command in push_commands} == {0.6}
        assert result["arrival_mode"] == "final_approach_loss_confirmed"
        assert result["final_approach_loss_samples"] == 2
        assert result["final_push_mps"] == 0.6
        assert result["final_push_duration_s"] == 1.0
        assert result["final_push_count"] == 1
        assert motion.commands[-1].reason == "qualified_arrival_stop"
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_metric_arrival_brakes_then_confirms_stopped_front_clearance() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        pose = MetricRangePose(motion, (0.55, 0.45, 0.35, 0.35))
        gate = MetricArrivalGate(
            RangeCalibration(
                forward_index=1,
                sensor_to_front_envelope_m=0.20,
                sensor_latency_s=0.10,
                braking_distance_m=0.04,
                noise_m=0.01,
            )
        )
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: pose,
            metric_arrival_gate=gate,
        )
        await manager.start()

        def seen(pts: int, *, close: bool = False) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": True,
                "generation": "camera-1",
                "detection": {
                    "label": "pear",
                    "generation": "camera-1",
                    "confidence": 0.81,
                    "center_x_ratio": 0.50,
                    "center_y_ratio": 0.74 if close else 0.50,
                    "bottom_ratio": 0.82 if close else 0.65,
                    "bbox_area_ratio": 0.10 if close else 0.04,
                    "age_s": 0.01,
                    "source_pts": pts,
                },
            }

        statuses = iter(
            (seen(1), seen(2), seen(3), seen(4, close=True), seen(5, close=True))
        )
        result = await manager.approach_target(
            lambda: next(statuses),
            "pear",
            forward_mps=1.0,
            maximum_yaw_rps=0.50,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            close_range_mps=0.55,
            final_push_mps=0.55,
            final_push_duration_s=0.001,
            timeout_s=0.5,
            metric_arrival_required=True,
        )

        assert result["arrival_confirmed"] is True
        assert result["arrival_mode"] == "metric_forward_range"
        assert result["front_clearance_m"] == pytest.approx(0.15)
        assert any(
            command.reason == "metric_arrival_predicted_stop"
            for command in motion.commands
        )
        assert motion.commands[-1].reason == "qualified_arrival_stop"
        await manager.close()

    asyncio.run(scenario())


def test_metric_arrival_cannot_fall_back_to_visual_geometry() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        pose = MetricRangePose(motion, (0.0,))
        gate = MetricArrivalGate(
            RangeCalibration(
                forward_index=1,
                sensor_to_front_envelope_m=0.20,
                sensor_latency_s=0.10,
                braking_distance_m=0.04,
                noise_m=0.01,
            )
        )
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: pose,
            metric_arrival_gate=gate,
        )
        await manager.start()
        pts = 0

        def seen() -> dict[str, object]:
            nonlocal pts
            pts += 1
            return {
                "camera_healthy": True,
                "target_ready": True,
                "generation": "camera-1",
                "detection": {
                    "label": "pear",
                    "generation": "camera-1",
                    "confidence": 0.81,
                    "center_x_ratio": 0.50,
                    "center_y_ratio": 0.76,
                    "bottom_ratio": 0.90,
                    "bbox_area_ratio": 0.12,
                    "age_s": 0.01,
                    "source_pts": pts,
                },
            }

        with pytest.raises(RangeUnavailable, match="forward range unavailable"):
            await manager.approach_target(
                seen,
                "pear",
                forward_mps=1.0,
                maximum_yaw_rps=0.50,
                near_bottom_ratio=0.86,
                near_center_ratio=0.72,
                near_confirmations=3,
                near_loss_grace_s=0.75,
                close_range_mps=0.55,
                final_push_mps=0.55,
                final_push_duration_s=0.001,
                timeout_s=0.5,
                metric_arrival_required=True,
            )
        assert motion.commands[-1].reason == "metric_range_unavailable"
        await manager.close()

    asyncio.run(scenario())


def test_close_visual_track_hands_off_to_lidar_until_stopped_18_inch_arrival() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        pose = MetricRangePose(motion, (0.0,))
        lidar = ScriptedLidarHandoff((0.90, 0.80, 0.70, 0.60, 0.48))
        gate = MetricArrivalGate(
            RangeCalibration(
                forward_index=0,
                sensor_to_front_envelope_m=0.0,
                sensor_latency_s=0.20,
                braking_distance_m=0.02,
                noise_m=0.02,
                target_clearance_m=0.4572,
                tolerance_m=0.0508,
                maximum_age_s=0.30,
            )
        )
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: pose,
            metric_arrival_gate=gate,
            metric_range_provider=lidar,
        )
        await manager.start()
        pts = 0

        def visible() -> dict[str, object]:
            nonlocal pts
            pts += 1
            return {
                "camera_healthy": True,
                "target_ready": True,
                "generation": "camera-1",
                "detection": {
                    "label": "pear",
                    "generation": "camera-1",
                    "confidence": 0.81,
                    "center_x_ratio": 0.50,
                    "center_y_ratio": 0.76,
                    "bottom_ratio": 0.90,
                    "bbox_area_ratio": 0.10,
                    "age_s": 0.01,
                    "source_pts": pts,
                },
            }

        statuses = iter(
            (
                visible(),
                visible(),
                visible(),
                visible(),
                {"camera_healthy": True, "target_ready": False},
                {"camera_healthy": True, "target_ready": False},
                {"camera_healthy": True, "target_ready": False},
                {"camera_healthy": True, "target_ready": False},
                {"camera_healthy": True, "target_ready": False},
            )
        )
        result = await manager.approach_target(
            lambda: next(statuses),
            "pear",
            forward_mps=1.0,
            maximum_yaw_rps=0.50,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=2,
            near_loss_grace_s=0.75,
            close_range_mps=0.55,
            final_push_mps=0.55,
            final_push_duration_s=0.001,
            timeout_s=0.5,
            metric_arrival_required=True,
        )

        assert result["arrival_confirmed"] is True
        assert result["metric_arrival_control_mode"] == "lidar_handoff"
        assert result["lidar_handoff_latched"] is True
        assert result["lidar_handoff_trigger_reason"] == "qualified_close_track_lost"
        assert result["range_association_mode"] == "lidar_handoff"
        assert result["front_clearance_m"] == pytest.approx(0.48)
        assert lidar.calls
        assert len(lidar.visual_calls) == 4
        assert all(call["close_authorized"] is True for call in lidar.visual_calls)
        assert all(call["allow_handoff"] is True for call in lidar.calls)
        assert all(call["visual_close_authorized"] is False for call in lidar.calls)
        assert any(
            command.reason == "visual_close_until_sight_loss"
            and command.forward_mps == 0.55
            for command in motion.commands
        )
        assert any(
            command.reason == "metric_lidar_handoff"
            and command.forward_mps == 0.55
            and command.yaw_rps == 0.0
            for command in motion.commands
        )
        assert any(
            command.reason == "metric_arrival_predicted_stop"
            for command in motion.commands
        )
        await manager.close()

    asyncio.run(scenario())


def test_lidar_handoff_is_monotonic_after_phantom_and_visual_reacquisition() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        pose = MetricRangePose(motion, (0.0,))
        lidar = ScriptedLidarHandoff((0.60, 0.70, 0.48, 0.48))
        gate = MetricArrivalGate(
            RangeCalibration(
                forward_index=0,
                sensor_to_front_envelope_m=0.0,
                sensor_latency_s=0.20,
                braking_distance_m=0.02,
                noise_m=0.02,
                target_clearance_m=0.4572,
                tolerance_m=0.0508,
                maximum_age_s=0.30,
            )
        )
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: pose,
            metric_arrival_gate=gate,
            metric_range_provider=lidar,
        )
        await manager.start()
        pts = 0

        def visible() -> dict[str, object]:
            nonlocal pts
            pts += 1
            return {
                "camera_healthy": True,
                "target_ready": True,
                "generation": "camera-1",
                "detection": {
                    "label": "pear",
                    "generation": "camera-1",
                    "confidence": 0.81,
                    "center_x_ratio": 0.50,
                    "center_y_ratio": 0.94,
                    "bottom_ratio": 1.0,
                    "bbox_area_ratio": 0.10,
                    "age_s": 0.01,
                    "source_pts": pts,
                },
            }

        weak_phantom = {
            "camera_healthy": True,
            "target_ready": False,
            "generation": "camera-1",
            "detection": {
                "label": "pear",
                "generation": "camera-1",
                "confidence": 0.01,
                "center_x_ratio": 0.56,
                "center_y_ratio": 0.13,
                "bottom_ratio": 0.16,
                "bbox_area_ratio": 0.002,
                "age_s": 0.01,
                "source_pts": 99,
            },
        }
        statuses = iter(
            (
                visible(),
                visible(),
                visible(),
                visible(),
                {"camera_healthy": True, "target_ready": False},
                weak_phantom,
                visible(),
                visible(),
            )
        )

        result = await manager.approach_target(
            lambda: next(statuses, weak_phantom),
            "pear",
            forward_mps=1.0,
            maximum_yaw_rps=0.50,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            close_range_mps=0.55,
            final_push_mps=0.55,
            final_push_duration_s=0.001,
            timeout_s=0.5,
            metric_arrival_required=True,
        )

        assert result["arrival_confirmed"] is True
        assert result["metric_arrival_control_mode"] == "lidar_handoff"
        assert result["lidar_handoff_latched"] is True
        assert result["lidar_handoff_trigger_reason"] == "qualified_close_track_lost"
        assert result["range_association_mode"] == "lidar_handoff"
        assert result["front_clearance_m"] == pytest.approx(0.48)
        assert len(lidar.calls) == 4
        assert all(call["allow_handoff"] is True for call in lidar.calls)
        assert len(lidar.visual_calls) == 4
        assert any(
            command.reason == "metric_arrival_predicted_stop"
            for command in motion.commands
        )
        assert any(
            command.reason == "metric_lidar_handoff" and command.forward_mps == 0.55
            for command in motion.commands
        )
        assert motion.commands[-1].reason == "qualified_arrival_stop"
        await manager.close()

    asyncio.run(scenario())


def test_approach_stays_slow_until_near_pear_leaves_camera() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def seen(*, center_y: float, bottom: float) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": True,
                "detection": {
                    "label": "pear",
                    "confidence": 0.81,
                    "consecutive_detections": 5,
                    "center_x_ratio": 0.50,
                    "center_y_ratio": center_y,
                    "bottom_ratio": bottom,
                    "age_s": 0.01,
                },
            }

        statuses = iter(
            (
                seen(center_y=0.50, bottom=0.65),
                seen(center_y=0.50, bottom=0.65),
                seen(center_y=0.50, bottom=0.65),
                seen(center_y=0.75, bottom=0.90),
                seen(center_y=0.76, bottom=0.91),
                seen(center_y=0.77, bottom=0.92),
                seen(center_y=0.78, bottom=0.93),
                seen(center_y=0.79, bottom=0.94),
                {"camera_healthy": True, "target_ready": False},
            )
        )

        result = await manager.approach_target(
            lambda: next(
                statuses,
                {"camera_healthy": True, "target_ready": False},
            ),
            "pear",
            forward_mps=1.0,
            maximum_yaw_rps=0.30,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            close_range_mps=0.55,
            final_push_mps=0.55,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        visible_forward = [
            command
            for command in motion.commands
            if command.reason in {"approach_target", "approach_target_slow"}
            and command.forward_mps == 1.0
        ]
        assert len(visible_forward) == 1
        assert (
            len(
                [
                    command
                    for command in motion.commands
                    if command.reason == "approach_target_slow"
                    and command.forward_mps == 0.55
                ]
            )
            == 2
        )
        assert not any(
            command.reason == "near_target_confirmed" for command in motion.commands
        )
        assert result["near_confirmations"] == 5
        assert result["forward_pulse_count"] >= 6
        assert result["final_push_count"] == 1
        assert any(
            command.reason == "visual_close_until_sight_loss"
            for command in motion.commands
        )
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_approach_holds_authorized_command_between_fresh_inference_frames() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def seen(
            source_pts: int,
            *,
            age_s: float = 0.01,
            near: bool = False,
        ) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": True,
                "generation": "camera-1",
                "source": {"pts": source_pts, "age_s": age_s},
                "detection": {
                    "label": "pear",
                    "generation": "camera-1",
                    "confidence": 0.81,
                    "center_x_ratio": 0.50,
                    "center_y_ratio": 0.76 if near else 0.50,
                    "bottom_ratio": 0.91 if near else 0.65,
                    "bbox_area_ratio": 0.12 if near else 0.04,
                    "age_s": age_s,
                    "source_pts": source_pts,
                },
            }

        # A 10 Hz control loop observes one or two duplicate samples between
        # fresh 4-8 Hz inference frames. Only the new source PTS values may
        # advance acquisition and arrival confirmation counts.
        statuses = iter(
            (
                seen(1),
                seen(1, age_s=0.10),
                seen(2),
                seen(2, age_s=0.10),
                seen(3),
                seen(3, age_s=0.10),
                seen(3, age_s=0.20),
                seen(3, age_s=0.251),
                seen(4, near=True),
                seen(4, age_s=0.10, near=True),
                seen(5, near=True),
                seen(6, near=True),
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "generation": "camera-1",
                    "source": {"pts": 8, "age_s": 0.01},
                    "detection": {},
                },
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "generation": "camera-1",
                    "source": {"pts": 9, "age_s": 0.01},
                    "detection": {},
                },
            )
        )

        result = await manager.approach_target(
            lambda: next(
                statuses,
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "generation": "camera-1",
                    "source": {"pts": 10, "age_s": 0.01},
                    "detection": {},
                },
            ),
            "pear",
            forward_mps=1.0,
            maximum_yaw_rps=1.0,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            close_range_mps=0.55,
            final_push_mps=0.6,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        assert motion.commands[0] == motion.commands[1]
        assert motion.commands[0].forward_mps == 0.0
        assert motion.commands[4] == motion.commands[5] == motion.commands[6]
        assert motion.commands[4].forward_mps == 1.0
        assert motion.commands[7] == VelocityCommand(
            reason="qualified_track_detection_stale"
        )
        assert motion.commands[8] == motion.commands[9]
        assert motion.commands[8].forward_mps == 0.55
        assert result["acquisition_samples"] == 3
        assert result["qualified_samples"] == 6
        assert result["near_samples"] == 3
        assert result["duplicate_samples"] == 5
        assert result["forward_pulse_count"] >= 7
        assert result["final_push_count"] == 1
        assert result["close_range_mps"] == 0.55
        await manager.close()

    asyncio.run(scenario())


def test_approach_replays_search_qualified_apple_confidence_drop() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def seen(
            source_pts: int,
            confidence: float,
            *,
            near: bool = False,
        ) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": confidence >= 0.50,
                "generation": "camera-1",
                "source": {
                    "pts": source_pts,
                    "time_base": "1/90000",
                    "age_s": 0.01,
                },
                "detection": {
                    "label": "apple",
                    "generation": "camera-1",
                    "source_pts": source_pts,
                    "source_time_base": "1/90000",
                    "confidence": confidence,
                    "consecutive_detections": 5,
                    "center_x_ratio": 0.50,
                    "center_y_ratio": 0.76 if near else 0.50,
                    "bottom_ratio": 0.91 if near else 0.65,
                    "bbox_area_ratio": 0.12 if near else 0.04,
                    "age_s": 0.01,
                },
            }

        qualified_at = time.monotonic()
        handoff = SearchQualificationHandoff(
            search_qualified=True,
            target_fruit="apple",
            generation="camera-1",
            source_pts=100,
            source_time_base="1/90000",
            qualified_monotonic_s=qualified_at,
            stable_detections=5,
            confidence=0.710628867149353,
            center_x_ratio=0.50,
            center_y_ratio=0.50,
            bottom_ratio=0.65,
            bbox_area_ratio=0.04,
        )
        statuses = iter(
            (
                seen(101, 0.6477978),
                seen(102, 0.60),
                seen(103, 0.59),
                seen(104, 0.58, near=True),
                seen(105, 0.57, near=True),
                seen(106, 0.56, near=True),
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "generation": "camera-1",
                    "source": {
                        "pts": 107,
                        "time_base": "1/90000",
                        "age_s": 0.01,
                    },
                    "detection": {},
                },
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "generation": "camera-1",
                    "source": {
                        "pts": 108,
                        "time_base": "1/90000",
                        "age_s": 0.01,
                    },
                    "detection": {},
                },
            )
        )

        result = await manager.approach_target(
            lambda: next(
                statuses,
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "generation": "camera-1",
                    "source": {
                        "pts": 109,
                        "time_base": "1/90000",
                        "age_s": 0.01,
                    },
                    "detection": {},
                },
            ),
            "apple",
            forward_mps=1.0,
            maximum_yaw_rps=0.50,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            close_range_mps=0.55,
            final_push_mps=0.6,
            final_push_duration_s=0.001,
            timeout_s=1.0,
            search_handoff=handoff,
        )

        assert [command.forward_mps for command in motion.commands[:3]] == [
            0.0,
            0.0,
            1.0,
        ]
        assert motion.commands[0].reason == "center_target_before_approach"
        assert result["search_handoff_accepted"] is True
        assert result["acquisition_samples"] == 0
        assert result["qualified_samples"] == 6
        assert result["forward_pulse_count"] >= 4
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_close_offset_uses_slew_limited_steering_without_in_place_pause() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def seen(
            source_pts: int,
            *,
            center_x: float,
            center_y: float = 0.50,
            bottom: float = 0.65,
            area: float = 0.04,
        ) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": True,
                "generation": "camera-1",
                "source": {"pts": source_pts, "age_s": 0.01},
                "detection": {
                    "label": "pear",
                    "generation": "camera-1",
                    "confidence": 0.81,
                    "center_x_ratio": center_x,
                    "center_y_ratio": center_y,
                    "bottom_ratio": bottom,
                    "bbox_area_ratio": area,
                    "age_s": 0.01,
                    "source_pts": source_pts,
                },
            }

        statuses = iter(
            (
                seen(1, center_x=0.50),
                seen(2, center_x=0.50),
                seen(3, center_x=0.50),
                seen(4, center_x=0.64, center_y=0.69, bottom=0.74, area=0.07),
                seen(5, center_x=0.66, center_y=0.75, bottom=0.83, area=0.12),
                seen(6, center_x=0.56, center_y=0.76, bottom=0.91, area=0.13),
                seen(7, center_x=0.53, center_y=0.77, bottom=0.92, area=0.14),
                seen(8, center_x=0.51, center_y=0.78, bottom=0.93, area=0.15),
                seen(9, center_x=0.50, center_y=0.79, bottom=0.94, area=0.16),
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "generation": "camera-1",
                    "source": {"pts": 10, "age_s": 0.01},
                    "detection": {},
                },
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "generation": "camera-1",
                    "source": {"pts": 11, "age_s": 0.01},
                    "detection": {},
                },
            )
        )

        def read_status() -> dict[str, object]:
            return next(
                statuses,
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "generation": "camera-1",
                    "source": {"pts": 12, "age_s": 0.01},
                    "detection": {},
                },
            )

        result = await manager.approach_target(
            read_status,
            "pear",
            forward_mps=1.0,
            maximum_yaw_rps=0.50,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            close_range_mps=0.55,
            final_push_mps=0.6,
            final_push_duration_s=0.001,
            timeout_s=1.0,
        )

        assert not any(
            command.reason == "recenter_close_target" for command in motion.commands
        )
        close_steering = [
            command
            for command in motion.commands
            if command.reason == "approach_target_slow"
            and command.yaw_rps < 0.0
        ]
        assert close_steering
        assert all(command.forward_mps == 0.55 for command in close_steering)
        assert abs(close_steering[0].yaw_rps) <= 0.20
        assert result["arrival_confirmed"] is True
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_approach_pauses_outside_middle_forty_then_slow_recenters() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def seen(
            source_pts: int,
            *,
            center_x: float,
            center_y: float = 0.50,
            bottom: float = 0.65,
            area: float = 0.04,
        ) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": True,
                "generation": "camera-1",
                "source": {"pts": source_pts, "age_s": 0.01},
                "detection": {
                    "label": "pear",
                    "generation": "camera-1",
                    "confidence": 0.81,
                    "center_x_ratio": center_x,
                    "center_y_ratio": center_y,
                    "bottom_ratio": bottom,
                    "bbox_area_ratio": area,
                    "age_s": 0.01,
                    "source_pts": source_pts,
                },
            }

        statuses = iter(
            (
                seen(1, center_x=0.50),
                seen(2, center_x=0.50),
                seen(3, center_x=0.50),
                seen(4, center_x=0.72),
                seen(5, center_x=0.74, center_y=0.51, bottom=0.66, area=0.05),
                seen(6, center_x=0.68, center_y=0.52, bottom=0.67, area=0.06),
                seen(7, center_x=0.50, center_y=0.76, bottom=0.91, area=0.12),
                seen(8, center_x=0.50, center_y=0.77, bottom=0.92, area=0.13),
                seen(9, center_x=0.50, center_y=0.78, bottom=0.93, area=0.14),
                seen(10, center_x=0.50, center_y=0.79, bottom=0.94, area=0.15),
                {"camera_healthy": True, "target_ready": False},
                {"camera_healthy": True, "target_ready": False},
            )
        )

        with pytest.raises(TargetLost, match="Arrival timed out"):
            await manager.approach_target(
                lambda: next(
                    statuses,
                    {"camera_healthy": True, "target_ready": False},
                ),
                "pear",
                forward_mps=1.0,
                maximum_yaw_rps=0.50,
                near_bottom_ratio=0.86,
                near_center_ratio=0.72,
                near_confirmations=3,
                near_loss_grace_s=0.75,
                close_range_mps=0.55,
                final_push_mps=0.6,
                final_push_duration_s=0.001,
                timeout_s=1.0,
            )

        pause_index = next(
            index
            for index, command in enumerate(motion.commands)
            if command.reason
            == "qualified_track_confirming_center_corridor_exit"
        )
        recenter_index = next(
            index
            for index, command in enumerate(motion.commands)
            if command.reason == "center_corridor_recenter"
        )
        assert motion.commands[pause_index].forward_mps == 0.0
        assert motion.commands[pause_index].yaw_rps == 0.0
        assert motion.commands[recenter_index].forward_mps == 0.0
        assert abs(motion.commands[recenter_index].yaw_rps) == pytest.approx(0.20)
        assert any(
            command.forward_mps > 0.0
            for command in motion.commands[recenter_index + 1 :]
        )
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_healthy_4hz_camera_does_not_toggle_motion_in_10hz_control_loop() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def seen(pts: int, age_s: float, *, near: bool = False) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": True,
                "generation": "camera-1",
                "source": {"pts": pts, "age_s": age_s},
                "detection": {
                    "label": "pear",
                    "generation": "camera-1",
                    "confidence": 0.81,
                    "center_x_ratio": 0.50,
                    "center_y_ratio": 0.76 if near else 0.50,
                    "bottom_ratio": 0.91 if near else 0.65,
                    "bbox_area_ratio": 0.12 if near else 0.04,
                    "age_s": age_s,
                    "source_pts": pts,
                },
            }

        statuses = iter(
            (
                seen(1, 0.01),
                seen(1, 0.11),
                seen(1, 0.21),
                seen(2, 0.01),
                seen(2, 0.11),
                seen(3, 0.01),
                seen(3, 0.11),
                seen(3, 0.21),
                seen(4, 0.01),
                seen(4, 0.11),
                seen(5, 0.01, near=True),
                seen(5, 0.11, near=True),
                seen(6, 0.01, near=True),
                seen(6, 0.11, near=True),
                seen(7, 0.01, near=True),
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "generation": "camera-1",
                    "source": {"pts": 8, "age_s": 0.01},
                    "detection": {},
                },
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "generation": "camera-1",
                    "source": {"pts": 9, "age_s": 0.01},
                    "detection": {},
                },
            )
        )
        result = await manager.approach_target(
            lambda: next(
                statuses,
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "generation": "camera-1",
                    "source": {"pts": 10, "age_s": 0.01},
                    "detection": {},
                },
            ),
            "pear",
            forward_mps=1.0,
            maximum_yaw_rps=0.50,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            close_range_mps=0.55,
            final_push_mps=0.55,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        first_forward = next(
            index
            for index, command in enumerate(motion.commands)
            if command.forward_mps > 0.0
        )
        first_loss_stop = next(
            index
            for index, command in enumerate(motion.commands)
            if command.reason == "qualified_track_confirming_final_approach_loss"
        )
        moving_window = motion.commands[first_forward:first_loss_stop]
        assert moving_window
        assert all(command.forward_mps > 0.0 for command in moving_window)
        assert motion.commands[first_loss_stop].forward_mps == 0.0
        assert result["duplicate_samples"] >= 7
        assert motion.commands[-1].reason == "qualified_arrival_stop"
        await manager.close()

    asyncio.run(scenario())


def test_approach_zeros_motion_before_reporting_camera_failure() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def seen(source_pts: int) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": True,
                "detection": {
                    "label": "pear",
                    "confidence": 0.81,
                    "center_x_ratio": 0.50,
                    "center_y_ratio": 0.50,
                    "bottom_ratio": 0.65,
                    "age_s": 0.01,
                    "source_pts": source_pts,
                },
            }

        statuses = iter(
            (
                seen(1),
                seen(2),
                seen(3),
                {"camera_healthy": False, "detail": "camera stalled"},
            )
        )

        with pytest.raises(CameraFailure, match="camera stalled"):
            await manager.approach_target(
                lambda: next(statuses),
                "pear",
                forward_mps=1.0,
                maximum_yaw_rps=1.0,
                near_bottom_ratio=0.86,
                near_center_ratio=0.72,
                near_confirmations=3,
                near_loss_grace_s=0.75,
                close_range_mps=0.55,
                final_push_mps=1.0,
                final_push_duration_s=1.0,
                timeout_s=0.5,
            )

        assert any(command.forward_mps == 1.0 for command in motion.commands)
        assert motion.commands[-1] == VelocityCommand(
            reason="qualified_track_camera_unhealthy"
        )
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_approach_rejects_sub_breakaway_slow_speed_before_arming() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        with pytest.raises(ValueError, match="minimum 0.55"):
            await manager.approach_target(
                lambda: {"camera_healthy": True, "target_ready": False},
                "pear",
                forward_mps=1.0,
                maximum_yaw_rps=0.30,
                near_bottom_ratio=0.86,
                near_center_ratio=0.72,
                near_confirmations=3,
                near_loss_grace_s=0.75,
                close_range_mps=0.54,
                final_push_mps=0.54,
                final_push_duration_s=0.001,
                timeout_s=0.001,
            )

        assert motion.commands == []
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_approach_centers_pear_before_first_forward_command() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def seen(center_x: float, *, near: bool = False) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": True,
                "detection": {
                    "label": "pear",
                    "center_x_ratio": center_x,
                    "center_y_ratio": 0.76 if near else 0.50,
                    "bottom_ratio": 0.91 if near else 0.65,
                    "age_s": 0.01,
                },
            }

        statuses = iter(
            (
                seen(0.75),
                seen(0.66),
                seen(0.56),
                seen(0.54),
                seen(0.52),
                seen(0.51, near=True),
                seen(0.50, near=True),
                seen(0.50, near=True),
                {"camera_healthy": True, "target_ready": False},
            )
        )

        result = await manager.approach_target(
            lambda: next(
                statuses,
                {"camera_healthy": True, "target_ready": False},
            ),
            "pear",
            forward_mps=1.0,
            maximum_yaw_rps=0.30,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            close_range_mps=0.55,
            final_push_mps=1.0,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        first_forward = next(
            index
            for index, command in enumerate(motion.commands)
            if command.forward_mps > 0.0
        )
        assert first_forward == 6
        assert all(
            command.forward_mps == 0.0 for command in motion.commands[:first_forward]
        )
        assert all(
            command.reason == "center_target_before_approach"
            for command in motion.commands[:first_forward]
        )
        assert [command.yaw_rps for command in motion.commands[:first_forward]] == [
            -0.30,
            -0.30,
            -0.30,
            -0.30,
            0.0,
            0.0,
        ]
        assert result["initial_center_confirmations"] == 3
        assert result["initial_center_tolerance_ratio"] == 0.05
        assert result["initial_center_yaw_rps"] == 0.30
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_approach_tracks_an_acquired_pear_at_sixty_percent_confidence() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def seen(
            confidence: float,
            *,
            near: bool = False,
            target_ready: bool,
        ) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": target_ready,
                "detection": {
                    "label": "pear",
                    "confidence": confidence,
                    "center_x_ratio": 0.50,
                    "center_y_ratio": 0.76 if near else 0.50,
                    "bottom_ratio": 0.91 if near else 0.65,
                    "age_s": 0.01,
                },
            }

        statuses = iter(
            (
                seen(0.80, target_ready=True),
                seen(0.80, target_ready=True),
                seen(0.80, target_ready=True),
                seen(0.60, target_ready=False),
                seen(0.60, target_ready=False),
                seen(0.60, near=True, target_ready=False),
                seen(0.60, near=True, target_ready=False),
                seen(0.60, near=True, target_ready=False),
                {"camera_healthy": True, "target_ready": False},
            )
        )

        result = await manager.approach_target(
            lambda: next(
                statuses,
                {"camera_healthy": True, "target_ready": False},
            ),
            "pear",
            forward_mps=0.55,
            maximum_yaw_rps=1.00,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            close_range_mps=0.55,
            final_push_mps=0.55,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        assert result["arrival_confirmed"] is True
        assert result["tracking_minimum_confidence"] == 0.55
        assert result["minimum_observed_tracking_confidence"] == 0.60
        assert any(
            command.reason == "approach_target" and command.forward_mps == 0.55
            for command in motion.commands
        )
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_approach_steering_enters_and_exits_with_smooth_hysteresis() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def seen(center_x: float, *, near: bool = False) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": True,
                "detection": {
                    "label": "pear",
                    "confidence": 0.81,
                    "center_x_ratio": center_x,
                    "center_y_ratio": 0.77 if near else 0.50,
                    "bottom_ratio": 0.92 if near else 0.65,
                    "age_s": 0.01,
                },
            }

        statuses = iter(
            (
                seen(0.53),
                seen(0.52),
                seen(0.51),
                seen(0.68),
                seen(0.74),
                seen(0.78),
                seen(0.80),
                seen(0.68),
                seen(0.60),
                seen(0.51, near=True),
                seen(0.50, near=True),
                seen(0.50, near=True),
                {"camera_healthy": True, "target_ready": False},
            )
        )

        result = await manager.approach_target(
            lambda: next(
                statuses,
                {"camera_healthy": True, "target_ready": False},
            ),
            "pear",
            forward_mps=1.0,
            maximum_yaw_rps=0.50,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            close_range_mps=0.55,
            final_push_mps=0.55,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        assert result["arrival_confirmed"] is True
        moving_yaws = [
            command.yaw_rps
            for command in motion.commands
            if command.reason == "approach_target" and command.forward_mps == 1.0
        ]
        first_correction = next(yaw for yaw in moving_yaws if yaw != 0.0)
        assert first_correction == pytest.approx(-0.02)
        assert all(
            abs(current - previous) <= 0.020001
            for previous, current in pairwise(moving_yaws)
        )
        assert any(
            command.reason == "approach_target"
            and command.forward_mps == 1.0
            and command.yaw_rps == 0.0
            for command in motion.commands
        )
        assert result["moving_steering_enter_ratio"] == 0.12
        assert result["moving_steering_exit_ratio"] == 0.08
        assert result["moving_yaw_gain"] == 1.50
        assert result["moving_yaw_maximum_rps"] == 0.50
        assert result["moving_yaw_slew_rps_per_s"] == 2.0
        assert result["stationary_recenter_error_ratio"] == 0.20
        first_forward = next(
            index
            for index, command in enumerate(motion.commands)
            if command.forward_mps > 0.0
        )
        assert all(
            command.forward_mps == 0.0 for command in motion.commands[:first_forward]
        )
        await manager.close()

    asyncio.run(scenario())


def test_approach_uses_close_range_continuity_at_new_apple_tracking_floor() -> (
    None
):
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def apple(
            confidence: float,
            *,
            center_x: float,
            center_y: float,
            bottom: float,
            target_ready: bool,
        ) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": target_ready,
                "detection": {
                    "label": "apple",
                    "confidence": confidence,
                    "center_x_ratio": center_x,
                    "center_y_ratio": center_y,
                    "bottom_ratio": bottom,
                    "age_s": 0.01,
                },
            }

        statuses = iter(
            (
                apple(
                    0.84, center_x=0.53, center_y=0.66, bottom=0.68, target_ready=True
                ),
                apple(
                    0.83, center_x=0.52, center_y=0.66, bottom=0.68, target_ready=True
                ),
                apple(
                    0.81, center_x=0.51, center_y=0.66, bottom=0.68, target_ready=True
                ),
                apple(
                    0.76, center_x=0.56, center_y=0.73, bottom=0.76, target_ready=True
                ),
                apple(
                    0.60, center_x=0.61, center_y=0.79, bottom=0.83, target_ready=False
                ),
                apple(
                    0.57, center_x=0.65, center_y=0.83, bottom=0.87, target_ready=False
                ),
                apple(
                    0.50, center_x=0.65, center_y=0.89, bottom=0.93, target_ready=False
                ),
                apple(
                    0.60, center_x=0.67, center_y=0.87, bottom=0.91, target_ready=False
                ),
                {"camera_healthy": True, "target_ready": False},
            )
        )

        result = await manager.approach_target(
            lambda: next(
                statuses,
                {"camera_healthy": True, "target_ready": False},
            ),
            "apple",
            forward_mps=1.0,
            maximum_yaw_rps=0.30,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            close_range_mps=0.55,
            final_push_mps=0.55,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        assert result["arrival_confirmed"] is True
        assert result["tracking_minimum_confidence"] == pytest.approx(0.50)
        assert result["close_range_continuation_samples"] >= 4
        assert result["minimum_observed_tracking_confidence"] == pytest.approx(0.50)
        await manager.close()

    asyncio.run(scenario())


def test_close_range_continuity_rejects_a_discontinuous_low_confidence_apple() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        acquired = {
            "camera_healthy": True,
            "target_ready": True,
            "detection": {
                "label": "apple",
                "confidence": 0.82,
                "center_x_ratio": 0.50,
                "center_y_ratio": 0.70,
                "bottom_ratio": 0.74,
                "age_s": 0.01,
            },
        }
        discontinuous = {
            "camera_healthy": True,
            "target_ready": False,
            "detection": {
                "label": "apple",
                "confidence": 0.20,
                "center_x_ratio": 0.90,
                "center_y_ratio": 0.85,
                "bottom_ratio": 0.90,
                "age_s": 0.01,
            },
        }
        statuses = iter((acquired, acquired, acquired, discontinuous))

        with pytest.raises(TargetLost, match="Arrival timed out") as caught:
            await manager.approach_target(
                lambda: next(statuses, discontinuous),
                "apple",
                forward_mps=1.0,
                maximum_yaw_rps=0.30,
                near_bottom_ratio=0.86,
                near_center_ratio=0.72,
                near_confirmations=3,
                near_loss_grace_s=0.75,
                close_range_mps=0.55,
                final_push_mps=0.55,
                final_push_duration_s=0.001,
                timeout_s=0.08,
            )

        assert caught.value.evidence["close_range_mps"] == 0.55
        assert all(
            command.reason != "fruit_offscreen_final_push"
            for command in motion.commands
        )
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_camera_failure_during_final_push_stops_and_fails() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()
        near = {
            "camera_healthy": True,
            "target_ready": True,
            "detection": {
                "label": "pear",
                "center_x_ratio": 0.5,
                "center_y_ratio": 0.8,
                "bottom_ratio": 0.95,
                "age_s": 0.01,
            },
        }
        statuses = iter(
            (
                near,
                near,
                near,
                {"camera_healthy": True, "target_ready": False},
                {"camera_healthy": False, "detail": "source progress is stale"},
            )
        )

        with pytest.raises(CameraFailure, match="source progress is stale"):
            await manager.approach_target(
                lambda: next(
                    statuses,
                    {
                        "camera_healthy": False,
                        "detail": "source progress is stale",
                    },
                ),
                "pear",
                forward_mps=0.55,
                maximum_yaw_rps=0.30,
                near_bottom_ratio=0.86,
                near_center_ratio=0.72,
                near_confirmations=3,
                near_loss_grace_s=0.75,
                close_range_mps=0.55,
                final_push_mps=0.6,
                final_push_duration_s=0.05,
                timeout_s=0.5,
            )

        assert motion.commands[-1].forward_mps == 0.0
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_return_home_replays_outbound_pulses_and_logs_measured_home_distance(
    tmp_path,
) -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: CommandDrivenReturningPose(motion),
        )
        recorder = FlightRecorder(tmp_path)
        manager.set_flight_recorder(recorder)
        manager.set_motion_authority("run-1", "epoch-1", "return_home")
        await manager.start()

        result = await manager.return_home(
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            forward_mps=1.0,
            forward_pulse_count=3,
            arrival_tolerance_m=0.10,
            heading_gate_rad=math.radians(20.0),
            maximum_yaw_rps=0.30,
            minimum_progress_m=0.03,
            stall_timeout_s=0.10,
            timeout_s=0.50,
        )

        assert result["home_distance_m"] == pytest.approx(0.09)
        assert result["arrival_tolerance_m"] == 0.10
        assert result["requested_forward_pulses"] == 3
        assert result["replayed_forward_pulses"] == 3
        assert result["motion_path"] == "direct_fused_closed_loop"
        assert result["planned_breadcrumbs"] == 0
        assert len(motion.commands) == 3
        assert all(command.forward_mps == 1.0 for command in motion.commands)
        assert motion.armed is False
        events = [
            json.loads(line)
            for line in recorder.snapshot_run("run-1").artifact.content.splitlines()
        ]
        decisions = [
            event["payload"]
            for event in events
            if event["kind"] == "home_navigation_sample"
        ]
        assert len(decisions) == 4
        assert decisions[0]["home_localization"]["home_distance_m"] == 0.8
        assert decisions[0]["plan"] == {
            "mode": "drive_to_home",
            "distance_m": 0.8,
            "heading_error_rad": 0.0,
            "forward_mps": 1.0,
            "yaw_rps": 0.0,
        }
        assert decisions[0]["thresholds"]["arrival_tolerance_m"] == 0.10
        assert decisions[0]["thresholds"]["heading_gate_rad"] == pytest.approx(
            math.radians(20.0)
        )
        await manager.close()

    asyncio.run(scenario())


def test_return_home_follows_recorded_outbound_breadcrumbs_in_reverse() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        pose = BreadcrumbPose()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: pose,
        )
        await manager.start()
        home = manager.capture_home()
        await manager.run_forward_pulse(FORWARD_PULSE_CONFIRMATION)
        pose.returning = True
        motion.commands.clear()

        result = await manager.return_home(
            home,
            forward_mps=1.0,
            forward_pulse_count=6,
            arrival_tolerance_m=0.10,
            heading_gate_rad=math.radians(20.0),
            maximum_yaw_rps=0.30,
            minimum_progress_m=0.03,
            stall_timeout_s=0.10,
            timeout_s=0.50,
        )

        assert result["motion_path"] == "breadcrumb_closed_loop"
        assert result["planned_breadcrumbs"] >= 2
        assert result["reached_breadcrumbs"] == result["planned_breadcrumbs"]
        assert result["home_distance_m"] == pytest.approx(0.05)
        assert result["replayed_forward_pulses"] < 6
        assert all(command.forward_mps == 1.0 for command in motion.commands)
        await manager.close()

    asyncio.run(scenario())


def test_turn_toward_home_uses_current_bearing_and_measured_yaw() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: HomeTurnPose(),
        )
        await manager.start()

        result = await manager.turn_toward_home(
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            yaw_rps=0.30,
            tolerance_rad=math.radians(5.0),
            timeout_s=0.50,
        )

        assert result["home_distance_m"] == pytest.approx(1.0)
        assert result["home_bearing_error_rad"] == pytest.approx(math.pi)
        assert result["measured_yaw_change_rad"] == pytest.approx(3.10)
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_turn_toward_home_recovers_once_when_first_yaw_lease_does_not_move() -> None:
    async def scenario() -> None:
        pose = RecoveringHomeTurnPose()
        motion = RecoveringMotion(pose)
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: pose,
        )
        await manager.start()

        result = await manager.turn_toward_home(
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            yaw_rps=0.50,
            tolerance_rad=math.radians(5.0),
            response_timeout_s=0.03,
            response_min_progress_rad=math.radians(2.0),
            recovery_settle_s=0.0,
            timeout_s=0.50,
        )

        assert motion.postures == ["stand_up"]
        assert result["turn_recovery_count"] == 1
        assert result["measured_yaw_change_rad"] == pytest.approx(3.10)
        assert any(command.yaw_rps == 0.50 for command in motion.commands)
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_restore_heading_rechecks_both_home_position_and_heading() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: RestorePose(),
        )
        await manager.start()

        result = await manager.restore_home_heading(
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            yaw_rps=0.30,
            heading_tolerance_rad=math.radians(5.0),
            position_tolerance_m=0.10,
            timeout_s=0.50,
        )

        assert result["home_distance_m"] == pytest.approx(0.08)
        assert result["heading_error_rad"] == pytest.approx(-0.04)
        assert result["position_tolerance_m"] == 0.10
        assert result["heading_tolerance_rad"] == pytest.approx(math.radians(5.0))
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_guarded_pulse_renews_fixed_command_then_releases() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        result = await manager.run_forward_pulse(FORWARD_PULSE_CONFIRMATION)

        assert len(motion.commands) >= 2
        assert all(command.forward_mps == 0.55 for command in motion.commands)
        assert all(command.yaw_rps == 0.0 for command in motion.commands)
        assert result["motion_path"] == "factory_avoidance"
        assert result["stopped"] is True
        assert motion.armed is False
        assert manager.status()["active_operation"] is None
        await manager.close()

    asyncio.run(scenario())


def test_guarded_pulse_requires_exact_phrase_and_both_feature_gates() -> None:
    async def scenario() -> None:
        manager = HardwareManager(live_config())
        with pytest.raises(HardwareUnavailable, match="type exactly"):
            await manager.run_forward_pulse("almost")

        disabled = HardwareManager(HardwareConfig())
        with pytest.raises(HardwareUnavailable, match="hardware is disabled"):
            await disabled.run_forward_pulse(FORWARD_PULSE_CONFIRMATION)

        read_only = HardwareManager(
            HardwareConfig(enabled=True, lab_motion_enabled=False)
        )
        with pytest.raises(HardwareUnavailable, match="lab motion"):
            await read_only.run_forward_pulse(FORWARD_PULSE_CONFIRMATION)

    asyncio.run(scenario())


def test_guarded_pulse_requires_fresh_pose() -> None:
    async def scenario() -> None:
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: FakeMotion(),
            pose_factory=lambda _age: FakePose(healthy=False),
        )
        await manager.start()

        assert manager.status()["clients_initialized"] is True
        assert manager.status()["connected"] is False
        assert manager.status()["can_pulse_forward"] is False
        with pytest.raises(HardwareUnavailable, match="fresh Go2 pose"):
            await manager.run_forward_pulse(FORWARD_PULSE_CONFIRMATION)
        await manager.close()

    asyncio.run(scenario())


def test_home_capture_returns_one_fresh_disarmed_pose() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        home = manager.capture_home()

        assert home == {
            "x_m": 0.0,
            "y_m": 0.0,
            "yaw_rad": 0.0,
            "captured_monotonic_s": 1.0,
            "age_s": 0.0,
            "source": "rt/sportmodestate",
            "visual_odometry": {
                "state": "unavailable",
                "generation": None,
                "frame_sequence": None,
                "error": None,
                "sensor_fusion": "legacy",
            },
        }
        assert motion.armed is False
        assert motion.commands == []
        await manager.close()

    asyncio.run(scenario())


def test_motion_factory_receives_the_validated_limits() -> None:
    async def scenario() -> None:
        received: list[MotionConfig] = []
        motion = FakeMotion()

        def factory(config: MotionConfig) -> FakeMotion:
            received.append(config)
            return motion

        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=factory,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        assert received[0].maximum_forward_mps == 1.0
        assert received[0].minimum_forward_mps == 0.55
        assert received[0].command_watchdog_s == 0.03
        assert received[0].remote_api_settle_s == 0.0
        await manager.close()

    asyncio.run(scenario())
