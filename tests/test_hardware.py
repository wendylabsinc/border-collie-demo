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
    StepBackNoResponse,
    TargetLost,
)
from border_collie_demo.models import Pose, VelocityCommand


class FakeMotion:
    def __init__(self) -> None:
        self.armed = False
        self.initialized = False
        self.commands: list[VelocityCommand] = []
        self.step_back_commands: list[VelocityCommand] = []
        self.stop_calls = 0
        self.postures: list[str] = []
        self.avoidance_suspended = False
        self.suspend_calls = 0
        self.resume_calls = 0
        self.hold_calls: list[tuple[float, float]] = []

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

    async def arm(self) -> str:
        self.armed = True
        return "lease"

    async def command(self, lease: str, command: VelocityCommand) -> VelocityCommand:
        assert lease == "lease"
        assert self.armed
        self.commands.append(command)
        return command

    async def step_back_hold(
        self, lease: str, reverse_mps: float, duration_s: float
    ) -> dict[str, object]:
        assert lease == "lease"
        assert self.armed
        assert self.avoidance_suspended, (
            "reverse must only be commanded inside the avoidance-off window"
        )
        command = VelocityCommand(-reverse_mps, 0.0, "step_back_hold")
        self.commands.append(command)
        self.step_back_commands.append(command)
        self.hold_calls.append((reverse_mps, duration_s))
        return {
            "hold_pattern": "single_setpoint_hold",
            "hold_move_commands": 1,
            "command_timestamps_monotonic_s": [1.0],
            "hold_duration_s": duration_s,
            "watchdog_renewals": 3,
            "watchdog_renewed_without_resend": True,
            "stop_issued_by": "window_end",
            "stop_confirmed": True,
        }

    async def suspend_avoidance_for_step_back(self, lease: str) -> dict[str, object]:
        assert lease == "lease"
        assert self.armed
        assert not self.avoidance_suspended
        self.avoidance_suspended = True
        self.suspend_calls += 1
        return {
            "avoidance_prior_enabled": True,
            "avoidance_switch_settle_s": 0.0,
        }

    async def resume_avoidance_after_step_back(self, lease: str) -> dict[str, object]:
        assert lease == "lease"
        assert self.armed
        assert self.avoidance_suspended
        self.avoidance_suspended = False
        self.resume_calls += 1
        return {
            "avoidance_restored": True,
            "avoidance_switched_off_s": 0.123,
        }

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


class SteppingBackPose(FakePose):
    """Fresh poses for one step back: readiness, before, and after samples."""

    def __init__(self) -> None:
        super().__init__()
        self._poses = iter(
            (
                Pose(1.0, 0.0, 0.0, 1.0),
                Pose(1.0, 0.0, 0.0, 1.1),
                Pose(0.72, 0.05, 0.0, 1.2),
            )
        )
        self._last = Pose(1.0, 0.0, 0.0, 1.0)

    def status(self) -> PoseStatus:
        if self.started:
            self._last = next(self._poses, self._last)
        return PoseStatus(self._last, 0.0, self.started, None)


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
        forward_pulse_mps=0.50,
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

        assert result["motion_path"] == "factory_avoidance"
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


def test_step_back_reverses_measures_displacement_then_disarms() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: SteppingBackPose(),
        )
        await manager.start()
        manager.start_motion_trace("step_back")

        result = await manager.step_back(
            reverse_mps=1.0,
            duration_s=0.03,
            minimum_backward_m=0.02,
        )

        assert result["motion_path"] == "direct_sport_reverse"
        assert result["commanded_reverse_mps"] == 1.0
        assert result["commanded_duration_s"] == 0.03
        assert result["hold_pattern"] == "single_setpoint_hold"
        assert result["hold_move_commands"] == 1
        assert result["hold_duration_s"] == 0.03
        assert result["stop_issued_by"] == "window_end"
        assert result["watchdog_renewed_without_resend"] is True
        assert result["measured_backward_m"] == pytest.approx(0.28)
        assert result["measured_lateral_m"] == pytest.approx(0.05)
        assert result["minimum_backward_m"] == 0.02
        assert result["motion_commands_sent"] is True
        assert result["avoidance_prior_enabled"] is True
        assert result["avoidance_restored"] is True
        assert result["avoidance_switched_off_s"] == 0.123
        assert result["pose_before"] == {"x_m": 1.0, "y_m": 0.0, "yaw_rad": 0.0}
        assert result["pose_after"] == {"x_m": 0.72, "y_m": 0.05, "yaw_rad": 0.0}
        assert motion.suspend_calls == 1
        assert motion.resume_calls == 1
        assert motion.hold_calls == [(1.0, 0.03)]
        assert len(motion.step_back_commands) == 1
        assert all(
            command.forward_mps == -1.0 and command.yaw_rps == 0.0
            for command in motion.step_back_commands
        )
        assert all(
            command["phase"] == "step_back"
            and command["forward_mps"] == -1.0
            and command["reason"] == "step_back_hold"
            for command in manager.motion_trace()
        )
        assert motion.armed is False
        assert manager.status()["active_operation"] is None
        await manager.close()

    asyncio.run(scenario())


def test_step_back_fails_closed_when_odometry_measures_no_movement() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        with pytest.raises(StepBackNoResponse, match="measured only") as failure:
            await manager.step_back(
                reverse_mps=1.0,
                duration_s=0.03,
                minimum_backward_m=0.02,
            )

        assert len(motion.step_back_commands) >= 1
        # Avoidance was restored before the fail-closed gate fired, and the
        # sealed evidence carries the full window record.
        assert motion.resume_calls == 1
        evidence = failure.value.evidence
        assert evidence["motion_path"] == "direct_sport_reverse"
        assert evidence["avoidance_restored"] is True
        assert evidence["measured_backward_m"] == pytest.approx(0.0)
        assert evidence["hold_move_commands"] == 1
        assert motion.armed is False
        assert manager.status()["active_operation"] is None
        await manager.close()

    asyncio.run(scenario())


def test_step_back_hard_faults_when_avoidance_restore_fails() -> None:
    """The demo must never continue with avoidance silently off."""

    class RestoreFailingMotion(FakeMotion):
        async def resume_avoidance_after_step_back(
            self, lease: str
        ) -> dict[str, object]:
            raise RuntimeError("SwitchSet(True) returned 3203")

    async def scenario() -> None:
        motion = RestoreFailingMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: SteppingBackPose(),
        )
        await manager.start()

        with pytest.raises(
            HardwareUnavailable, match="avoidance restore failed"
        ) as failure:
            await manager.step_back(
                reverse_mps=1.0,
                duration_s=0.03,
                minimum_backward_m=0.02,
            )

        # The reverse window itself succeeded; only the restore failed, and
        # that alone must fail the stage with the evidence sealed.
        assert len(motion.step_back_commands) >= 1
        assert failure.value.evidence["avoidance_restored"] is False
        assert failure.value.evidence["motion_commands_sent"] is True
        assert motion.armed is False
        assert manager.status()["active_operation"] is None
        await manager.close()

    asyncio.run(scenario())


def test_step_back_window_starts_only_after_the_motion_arm_settle() -> None:
    """Regression for the 2026-08-07 run-2 trap: a settle-consumed window."""

    class SlowArmMotion(FakeMotion):
        def __init__(self, settle_s: float) -> None:
            super().__init__()
            self._settle_s = settle_s

        async def arm(self) -> str:
            await asyncio.sleep(self._settle_s)
            return await super().arm()

    async def scenario() -> None:
        motion = SlowArmMotion(settle_s=0.08)
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: SteppingBackPose(),
        )
        await manager.start()

        # The arm settle (0.08 s) is longer than the whole reverse window
        # (0.02 s). If the window budget were consumed before the settle
        # finished, no setpoint would be sent; the hold must receive its
        # full requested window regardless of how long arming takes.
        result = await manager.step_back(
            reverse_mps=1.0,
            duration_s=0.02,
            minimum_backward_m=0.01,
        )

        assert motion.hold_calls == [(1.0, 0.02)]
        assert len(motion.step_back_commands) == 1
        assert result["hold_move_commands"] == 1
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_step_back_rejects_inconsistent_command_values() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: SteppingBackPose(),
        )
        await manager.start()

        with pytest.raises(ValueError, match="configured limit"):
            await manager.step_back(
                reverse_mps=1.5,
                duration_s=0.4,
                minimum_backward_m=0.05,
            )
        with pytest.raises(ValueError, match="displacement gate"):
            await manager.step_back(
                reverse_mps=1.0,
                duration_s=0.03,
                minimum_backward_m=0.05,
            )

        assert motion.commands == []
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
        assert result["stable_detections"] == 5
        assert result["motion_commands_sent"] is True
        assert all(command.forward_mps == 0.0 for command in motion.commands)
        assert all(
            command["phase"] == "turn_to_fruit"
            and command["forward_mps"] == 0.0
            for command in manager.motion_trace()
        )
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_find_target_slows_but_keeps_rotating_during_crop_confirmation() -> None:
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
            "crop_confirm_slow_turn",
            "find_target",
        ]
        assert [command.yaw_rps for command in motion.commands] == [
            0.50,
            0.0,
            0.50,
            0.50,
        ]
        assert result["recognition"]["crop_confirmation_samples"] == 2
        assert result["recognition"]["crop_slowdown_hold_samples"] == 1
        assert result["recognition"]["crop_slowdown_turn_samples"] == 1
        assert result["recognition"]["crop_candidate_confidence_threshold"] == 0.50
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


def test_approach_requires_near_geometry_then_one_offscreen_final_push() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def seen(*, center_x: float, center_y: float, bottom: float) -> dict[str, object]:
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
                {"camera_healthy": True, "target_ready": False, "detail": "pear offscreen"},
            )
        )

        result = await manager.approach_target(
            lambda: next(
                statuses,
                {"camera_healthy": True, "target_ready": False},
            ),
            "pear",
            forward_mps=0.50,
            maximum_yaw_rps=0.30,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            final_push_mps=1.0,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        assert result["arrival_confirmed"] is True
        assert result["near_confirmations"] == 3
        assert result["final_push_mps"] == 1.0
        assert result["final_push_count"] == 1
        assert result["forward_pulse_count"] == 5
        assert any(
            command.reason == "approach_target_near_visible"
            and command.forward_mps == 0.50
            for command in motion.commands
        )
        assert any(command.forward_mps == 0.50 for command in motion.commands)
        assert any(command.forward_mps == 1.0 for command in motion.commands)
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_approach_keeps_moving_while_near_pear_remains_visible() -> None:
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
            final_push_mps=0.30,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        visible_forward = [
            command
            for command in motion.commands
            if command.reason
            in {"approach_target", "approach_target_near_visible"}
            and command.forward_mps == 1.0
        ]
        assert len(visible_forward) == 6
        assert len(
            [
                command
                for command in motion.commands
                if command.reason == "approach_target_near_visible"
                and command.forward_mps == 1.0
            ]
        ) == 3
        assert not any(
            command.reason == "near_target_confirmed"
            for command in motion.commands
        )
        assert result["near_confirmations"] == 5
        assert result["forward_pulse_count"] == 7
        assert result["final_push_count"] == 1
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
            final_push_mps=1.0,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        first_forward = next(
            index
            for index, command in enumerate(motion.commands)
            if command.forward_mps > 0.0
        )
        assert first_forward == 4
        assert all(
            command.forward_mps == 0.0
            for command in motion.commands[:first_forward]
        )
        assert all(
            command.reason == "center_target_before_approach"
            for command in motion.commands[:first_forward]
        )
        assert [
            command.yaw_rps for command in motion.commands[:first_forward]
        ] == [-0.50, -0.50, 0.0, 0.0]
        assert result["initial_center_confirmations"] == 3
        assert result["initial_center_tolerance_ratio"] == 0.08
        assert result["initial_center_yaw_rps"] == 0.50
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_late_approach_tightens_centering_before_the_blind_push() -> None:
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
                    # A constant 0.06 offset: inside the wide 0.08 band,
                    # outside the tighter 0.04 late-approach band.
                    "center_x_ratio": 0.56,
                    "center_y_ratio": center_y,
                    "bottom_ratio": bottom,
                },
            }

        statuses = iter(
            (
                seen(center_y=0.50, bottom=0.65),
                seen(center_y=0.50, bottom=0.65),
                seen(center_y=0.50, bottom=0.65),
                seen(center_y=0.50, bottom=0.65),
                seen(center_y=0.76, bottom=0.91),
                seen(center_y=0.76, bottom=0.91),
                seen(center_y=0.77, bottom=0.92),
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
            final_push_mps=0.30,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        forward_commands = [
            command
            for command in motion.commands
            if command.reason in {"approach_target", "approach_target_near_visible"}
        ]
        early = [command for command in forward_commands if command.yaw_rps == 0.0]
        late = [command for command in forward_commands if command.yaw_rps == -0.30]
        # Far approach: the 0.06 offset stays inside the 0.08 band, no yaw.
        # The first late sample is the hysteresis confirmation and also sends
        # zero yaw.
        assert len(early) == 3
        # Late approach: after two consecutive off-band samples the same
        # offset draws the fixed correction while forward translation
        # continues, so every iteration still records exactly one forward
        # pulse.
        assert len(late) == 2
        assert all(command.forward_mps == 1.0 for command in forward_commands)
        assert result["approach_center_tolerance_ratio"] == 0.08
        assert result["late_center_tolerance_ratio"] == 0.04
        assert result["late_center_confirmations"] == 2
        assert result["late_center_corrections"] == 2
        assert result["forward_pulse_count"] == len(forward_commands) + 1
        assert result["final_push_count"] == 1
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_late_centering_ignores_single_frame_jitter_across_the_tight_band() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def seen(center_x: float) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": True,
                "detection": {
                    "label": "pear",
                    "confidence": 0.81,
                    "consecutive_detections": 5,
                    "center_x_ratio": center_x,
                    "center_y_ratio": 0.76,
                    "bottom_ratio": 0.91,
                },
            }

        # Alternating 0.05/0.03 offsets straddle the 0.04 tight band but
        # never hold two consecutive off-band samples.
        statuses = iter(
            (
                seen(0.55),
                seen(0.53),
                seen(0.55),
                seen(0.53),
                seen(0.55),
                seen(0.53),
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
            final_push_mps=0.30,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        assert result["arrival_confirmed"] is True
        assert result["late_center_corrections"] == 0
        assert all(command.yaw_rps == 0.0 for command in motion.commands)
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_final_push_fires_despite_a_static_low_confidence_phantom() -> None:
    """Regression for supervised run 45a1e796 (2026-08-08).

    After a confirmed near track, a static 0.010-0.016 confidence detection
    with a matching label held the arrival gate open for ~12 seconds and
    suppressed the final push until the approach deadline. A detection below
    the fruit's close-range tracking confidence must count as NOT visible
    for arrival purposes.
    """

    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def near() -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": True,
                "detection": {
                    "label": "pear",
                    "confidence": 0.81,
                    "consecutive_detections": 5,
                    "center_x_ratio": 0.50,
                    "center_y_ratio": 0.76,
                    "bottom_ratio": 0.91,
                },
            }

        # The measured phantom: matching label, ~0.015 confidence, frozen
        # geometry far above the arrival region.
        phantom = {
            "camera_healthy": True,
            "target_ready": False,
            "detection": {
                "label": "pear",
                "confidence": 0.015,
                "center_x_ratio": 0.42,
                "center_y_ratio": 0.35,
                "bottom_ratio": 0.3861,
            },
        }
        statuses = iter((near(), near(), near(), near(), near(), phantom))

        result = await manager.approach_target(
            lambda: next(statuses, phantom),
            "pear",
            forward_mps=1.0,
            maximum_yaw_rps=0.30,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            final_push_mps=0.30,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        assert result["arrival_confirmed"] is True
        assert result["final_push_count"] == 1
        assert any(
            command.reason == "fruit_offscreen_final_push"
            for command in motion.commands
        )
        assert result["arrival_visibility_confidence_floor"] == 0.20
        assert result["subthreshold_visibility_samples"] >= 1
        assert result["near_gate_confirmed"] is True
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_push_fires_despite_a_mid_confidence_spatially_inconsistent_phantom() -> None:
    """The lowered pear close-range floor must not re-open the phantom hole.

    A phantom above the 0.20 floor but spatially discontinuous with the last
    accepted track (static box far above the arrival region) cannot hold the
    arrival gate open: visibility also requires continuity with the fruit we
    were actually approaching.
    """

    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def near() -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": True,
                "detection": {
                    "label": "pear",
                    "confidence": 0.81,
                    "consecutive_detections": 5,
                    "center_x_ratio": 0.50,
                    "center_y_ratio": 0.76,
                    "bottom_ratio": 0.91,
                },
            }

        mid_phantom = {
            "camera_healthy": True,
            "target_ready": False,
            "detection": {
                "label": "pear",
                "confidence": 0.30,
                "center_x_ratio": 0.42,
                "center_y_ratio": 0.35,
                "bottom_ratio": 0.3861,
            },
        }
        statuses = iter((near(), near(), near(), near(), near(), mid_phantom))

        result = await manager.approach_target(
            lambda: next(statuses, mid_phantom),
            "pear",
            forward_mps=1.0,
            maximum_yaw_rps=0.30,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            final_push_mps=0.30,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        assert result["arrival_confirmed"] is True
        assert result["final_push_count"] == 1
        assert any(
            command.reason == "fruit_offscreen_final_push"
            for command in motion.commands
        )
        assert result["subthreshold_visibility_samples"] >= 1
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_continuation_bridges_the_pear_confidence_collapse_at_arrival() -> None:
    """Regression for supervised run d740a5f2 (2026-08-08).

    A real pear filling the frame at arrival distance collapsed from 0.89 to
    0.2658 confidence at bbox bottom 0.9972. The old 0.55 pear close-range
    value (identical to the normal tracking floor) rejected every collapsed
    frame, so the near gate starved at zero confirmations and Arrival timed
    out. Geometrically continuous sub-floor frames at or above 0.20 must now
    keep the track alive and count toward near confirmation.
    """

    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def qualified(*, center_y: float, bottom: float) -> dict[str, object]:
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
                },
            }

        def collapsed(
            confidence: float, *, center_y: float, bottom: float
        ) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": False,
                "detection": {
                    "label": "pear",
                    "confidence": confidence,
                    "center_x_ratio": 0.50,
                    "center_y_ratio": center_y,
                    "bottom_ratio": bottom,
                },
            }

        statuses = iter(
            (
                qualified(center_y=0.50, bottom=0.65),
                qualified(center_y=0.50, bottom=0.65),
                qualified(center_y=0.50, bottom=0.65),
                qualified(center_y=0.60, bottom=0.78),
                qualified(center_y=0.68, bottom=0.825),
                collapsed(0.27, center_y=0.78, bottom=0.86),
                collapsed(0.24, center_y=0.82, bottom=0.92),
                collapsed(0.22, center_y=0.85, bottom=0.99),
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
            final_push_mps=0.30,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        assert result["arrival_confirmed"] is True
        assert result["near_gate_confirmed"] is True
        assert result["near_confirmations"] == 3
        assert result["close_range_continuation_samples"] == 3
        assert result["close_range_tracking_confidence"] == 0.20
        assert result["minimum_observed_tracking_confidence"] == pytest.approx(0.22)
        assert result["final_push_count"] == 1
        assert any(
            command.reason == "fruit_offscreen_final_push"
            for command in motion.commands
        )
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_arrival_timeout_carries_the_full_approach_evidence() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        def far() -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": True,
                "detection": {
                    "label": "pear",
                    "confidence": 0.81,
                    "consecutive_detections": 5,
                    "center_x_ratio": 0.50,
                    "center_y_ratio": 0.50,
                    "bottom_ratio": 0.65,
                },
            }

        phantom = {
            "camera_healthy": True,
            "target_ready": False,
            "detection": {
                "label": "pear",
                "confidence": 0.012,
                "center_x_ratio": 0.42,
                "center_y_ratio": 0.35,
                "bottom_ratio": 0.3861,
            },
        }
        statuses = iter((far(), far(), far(), far()))

        with pytest.raises(TargetLost, match="Arrival timed out") as failure:
            await manager.approach_target(
                lambda: next(statuses, phantom),
                "pear",
                forward_mps=1.0,
                maximum_yaw_rps=0.30,
                near_bottom_ratio=0.86,
                near_center_ratio=0.72,
                near_confirmations=3,
                near_loss_grace_s=0.75,
                final_push_mps=0.30,
                final_push_duration_s=0.001,
                timeout_s=0.08,
            )

        evidence = failure.value.evidence
        assert evidence["arrival_confirmed"] is False
        assert evidence["near_gate_confirmed"] is False
        assert evidence["final_push_count"] == 0
        assert evidence["forward_pulse_count"] >= 1
        assert evidence["subthreshold_visibility_samples"] >= 1
        assert evidence["arrival_visibility_confidence_floor"] == 0.20
        assert evidence["motion_commands_sent"] is True
        assert evidence["last_track_geometry"] == {
            "center_x_ratio": 0.50,
            "center_y_ratio": 0.50,
            "bottom_ratio": 0.65,
        }
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

        def low_confidence(*, near: bool = False) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "target_ready": False,
                "detection": {
                    "label": "pear",
                    "confidence": 0.60,
                    "center_x_ratio": 0.50,
                    "center_y_ratio": 0.76 if near else 0.50,
                    "bottom_ratio": 0.91 if near else 0.65,
                },
            }

        statuses = iter(
            (
                low_confidence(),
                low_confidence(),
                low_confidence(),
                low_confidence(),
                low_confidence(),
                low_confidence(near=True),
                low_confidence(near=True),
                low_confidence(near=True),
                {"camera_healthy": True, "target_ready": False},
            )
        )

        result = await manager.approach_target(
            lambda: next(
                statuses,
                {"camera_healthy": True, "target_ready": False},
            ),
            "pear",
            forward_mps=0.50,
            maximum_yaw_rps=0.30,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            final_push_mps=0.30,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        assert result["arrival_confirmed"] is True
        assert result["tracking_minimum_confidence"] == 0.55
        assert result["minimum_observed_tracking_confidence"] == 0.60
        assert any(
            command.reason == "approach_target" and command.forward_mps == 0.50
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
                },
            }

        statuses = iter(
            (
                seen(0.53),
                seen(0.52),
                seen(0.51),
                seen(0.70),
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
            final_push_mps=0.30,
            final_push_duration_s=0.001,
            timeout_s=0.5,
        )

        assert result["arrival_confirmed"] is True
        assert any(
            command.reason == "approach_target"
            and command.forward_mps == 1.0
            and command.yaw_rps == -0.30
            for command in motion.commands
        )
        first_forward = next(
            index
            for index, command in enumerate(motion.commands)
            if command.forward_mps > 0.0
        )
        assert all(
            command.forward_mps == 0.0
            for command in motion.commands[:first_forward]
        )
        await manager.close()

    asyncio.run(scenario())


def test_approach_uses_close_range_continuity_after_red_apple_confidence_drops() -> None:
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
                },
            }

        statuses = iter(
            (
                apple(0.84, center_x=0.53, center_y=0.66, bottom=0.68, target_ready=True),
                apple(0.83, center_x=0.52, center_y=0.66, bottom=0.68, target_ready=True),
                apple(0.81, center_x=0.51, center_y=0.66, bottom=0.68, target_ready=True),
                apple(0.76, center_x=0.56, center_y=0.73, bottom=0.76, target_ready=True),
                apple(0.30, center_x=0.61, center_y=0.79, bottom=0.83, target_ready=False),
                apple(0.17, center_x=0.65, center_y=0.83, bottom=0.87, target_ready=False),
                apple(0.15, center_x=0.65, center_y=0.89, bottom=0.93, target_ready=False),
                apple(0.30, center_x=0.67, center_y=0.87, bottom=0.91, target_ready=False),
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
            final_push_mps=0.30,
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
            },
        }
        statuses = iter((acquired, acquired, acquired, discontinuous))

        with pytest.raises(TargetLost, match="Arrival timed out"):
            await manager.approach_target(
                lambda: next(statuses, discontinuous),
                "apple",
                forward_mps=1.0,
                maximum_yaw_rps=0.30,
                near_bottom_ratio=0.86,
                near_center_ratio=0.72,
                near_confirmations=3,
                near_loss_grace_s=0.75,
                final_push_mps=0.30,
                final_push_duration_s=0.001,
                timeout_s=0.08,
            )

        assert all(command.reason != "fruit_offscreen_final_push" for command in motion.commands)
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_final_push_stops_if_camera_source_fails_during_the_bounded_push() -> None:
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
                    {"camera_healthy": False, "detail": "source progress is stale"},
                ),
                "pear",
                forward_mps=0.50,
                maximum_yaw_rps=0.30,
                near_bottom_ratio=0.86,
                near_center_ratio=0.72,
                near_confirmations=3,
                near_loss_grace_s=0.75,
                final_push_mps=1.0,
                final_push_duration_s=0.05,
                timeout_s=0.5,
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
        assert result["motion_path"] == "factory_avoidance"
        assert len(motion.commands) == 3
        assert all(command.forward_mps == 1.0 for command in motion.commands)
        assert motion.armed is False
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
        assert all(command.forward_mps == 0.50 for command in motion.commands)
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
        assert received[0].command_watchdog_s == 0.03
        assert received[0].remote_api_settle_s == 0.0
        await manager.close()

    asyncio.run(scenario())
