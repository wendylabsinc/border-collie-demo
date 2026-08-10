from __future__ import annotations

import asyncio
import math

import pytest

from border_collie_demo.config import HardwareConfig
from border_collie_demo.go2_motion import MotionConfig
from border_collie_demo.go2_pose import PoseStatus
from border_collie_demo.hardware import (
    FORWARD_PULSE_CONFIRMATION,
    CameraFailure,
    HardwareManager,
    HardwareUnavailable,
    TargetLost,
)
from border_collie_demo.models import Pose, VelocityCommand


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
        assert all(command.forward_mps == 0.0 for command in motion.commands)
        assert all(
            command["phase"] == "turn_to_fruit" and command["forward_mps"] == 0.0
            for command in manager.motion_trace()
        )
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
            "crop_confirm_hold",
            "crop_confirm_hold",
            "crop_confirm_hold",
        ]
        assert [command.yaw_rps for command in motion.commands] == [
            0.50,
            0.0,
            0.0,
            0.0,
        ]
        assert result["recognition"]["crop_confirmation_samples"] == 2
        assert result["recognition"]["crop_slowdown_hold_samples"] == 3
        assert result["recognition"].get("crop_slowdown_turn_samples", 0) == 0
        assert result["recognition"]["crop_candidate_confidence_threshold"] == 0.50
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_find_target_resumes_candidate_tracking_at_half_rate_after_hold() -> None:
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
            "crop_confirm_hold",
            "crop_confirm_hold",
            "crop_confirm_hold",
            "crop_confirm_hold",
            "crop_confirm_slow_turn",
            "crop_confirm_slow_turn",
        ]
        assert [command.yaw_rps for command in motion.commands] == [
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.5,
        ]
        assert result["recognition"]["candidate_lock_hold_s"] == 0.75
        assert result["recognition"]["candidate_lock_yaw_rps"] == 0.5
        assert result["recognition"]["crop_slowdown_hold_samples"] == 4
        assert result["recognition"]["crop_slowdown_turn_samples"] == 2
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_find_target_returns_to_broad_search_after_candidate_loss() -> None:
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
                    },
                },
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
            "crop_confirm_hold",
            "crop_confirm_hold",
            "crop_confirm_hold",
            "find_target",
        ]
        assert [command.yaw_rps for command in motion.commands] == [
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
            "crop_confirm_hold",
            "crop_confirm_hold",
            "crop_confirm_hold",
            "crop_confirm_hold",
            "crop_confirm_hold",
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


def test_approach_stops_on_confirmed_visible_geometry_without_a_final_push() -> None:
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
            *, center_x: float, center_y: float, bottom: float
        ) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": True,
                "detection": {
                    "label": "pear",
                    "confidence": 0.81,
                    "consecutive_detections": 5,
                    "center_x_ratio": center_x,
                    "center_y_ratio": center_y,
                    "bottom_ratio": bottom,
                    "age_s": 0.01,
                },
            }

        statuses = iter(
            (
                seen(center_x=0.53, center_y=0.50, bottom=0.65),
                seen(center_x=0.52, center_y=0.50, bottom=0.65),
                seen(center_x=0.51, center_y=0.50, bottom=0.65),
                seen(center_x=0.53, center_y=0.75, bottom=0.90),
                seen(center_x=0.52, center_y=0.76, bottom=0.91),
                seen(center_x=0.51, center_y=0.77, bottom=0.92),
                {
                    "camera_healthy": True,
                    "target_ready": False,
                    "detail": "pear offscreen",
                },
            )
        )

        result = await manager.approach_target(
            lambda: next(
                statuses,
                {"camera_healthy": True, "target_ready": False},
            ),
            "pear",
            forward_mps=0.55,
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

        assert result["arrival_confirmed"] is True
        assert result["near_confirmations"] == 3
        assert result["final_push_mps"] == 1.0
        assert result["final_push_count"] == 0
        assert result["forward_pulse_count"] == 3
        assert any(
            command.reason == "approach_target_slow" and command.forward_mps == 0.55
            for command in motion.commands
        )
        assert any(command.forward_mps == 0.55 for command in motion.commands)
        assert not any(command.forward_mps == 1.0 for command in motion.commands)
        assert motion.commands[-1].reason == "qualified_arrival_stop"
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_approach_slows_then_stops_while_near_pear_remains_visible() -> None:
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
        assert result["near_confirmations"] == 3
        assert result["forward_pulse_count"] == 3
        assert result["final_push_count"] == 0
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
            )
        )

        result = await manager.approach_target(
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
        assert result["forward_pulse_count"] == 6
        assert result["close_range_mps"] == 0.55
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
        assert first_forward == 5
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


def test_approach_may_combine_forward_and_yaw_after_initial_centering() -> None:
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
                seen(0.72),
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
        assert any(
            command.reason == "approach_target"
            and command.forward_mps == 1.0
            and command.yaw_rps == pytest.approx(-0.5)
            for command in motion.commands
        )
        assert any(
            command.reason == "approach_target"
            and command.forward_mps == 1.0
            and command.yaw_rps == 0.0
            for command in motion.commands
        )
        assert result["moving_yaw_deadband_ratio"] == 0.20
        assert result["stationary_recenter_error_ratio"] == 0.40
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


def test_approach_uses_close_range_continuity_after_red_apple_confidence_drops() -> (
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
                    0.30, center_x=0.61, center_y=0.79, bottom=0.83, target_ready=False
                ),
                apple(
                    0.17, center_x=0.65, center_y=0.83, bottom=0.87, target_ready=False
                ),
                apple(
                    0.15, center_x=0.65, center_y=0.89, bottom=0.93, target_ready=False
                ),
                apple(
                    0.30, center_x=0.67, center_y=0.87, bottom=0.91, target_ready=False
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
        assert result["close_range_continuation_samples"] >= 4
        assert result["minimum_observed_tracking_confidence"] == pytest.approx(0.15)
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


def test_visible_arrival_finishes_before_a_later_camera_failure() -> None:
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

        result = await manager.approach_target(
            lambda: next(
                statuses,
                {"camera_healthy": False, "detail": "source progress is stale"},
            ),
            "pear",
            forward_mps=0.55,
            maximum_yaw_rps=0.30,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            close_range_mps=0.55,
            final_push_mps=1.0,
            final_push_duration_s=0.05,
            timeout_s=0.5,
        )

        assert result["arrival_confirmed"] is True
        assert result["final_push_count"] == 0
        assert not any(
            command.reason == "fruit_offscreen_final_push"
            for command in motion.commands
        )
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_return_home_replays_outbound_pulses_and_logs_measured_home_distance() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: ReturningPose(),
        )
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
