from __future__ import annotations

import asyncio
import math

import pytest

from border_collie_demo.black_box import RunBlackBox
from border_collie_demo.config import HardwareConfig
from border_collie_demo.fruit_bearing_map import FruitBearingMap
from border_collie_demo.go2_motion import MotionConfig
from border_collie_demo.go2_pose import PoseStatus
from border_collie_demo.guidance import FruitGuidance, GuidanceConfig, GuidancePhase
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
        self.mode: str | None = None
        self.command_modes: list[str] = []

    def status(self) -> dict[str, object]:
        return {
            "initialized": self.initialized,
            "armed": self.armed,
            "mode": self.mode,
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
        self.mode = "factory_avoidance"
        return "lease"

    async def arm_sport_yaw(self) -> str:
        self.armed = True
        self.mode = "sport_yaw"
        return "lease"

    async def command(self, lease: str, command: VelocityCommand) -> VelocityCommand:
        assert lease == "lease"
        assert self.armed
        self.commands.append(command)
        assert self.mode is not None
        self.command_modes.append(self.mode)
        return command

    async def release(self, lease: str) -> None:
        assert lease == "lease"
        self.armed = False
        self.mode = None

    async def emergency_stop(self) -> list[str]:
        self.stop_calls += 1
        self.armed = False
        self.mode = None
        return []

    async def close(self) -> list[str]:
        self.armed = False
        self.mode = None
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


class HeadingEscapePose(FakePose):
    def __init__(self) -> None:
        super().__init__()
        self._poses = iter(
            (
                Pose(1.0, 0.0, math.pi, 1.0),
                Pose(0.9, 0.0, math.pi, 1.05),
                Pose(0.8, 0.0, math.pi, 1.1),
                Pose(0.6, 0.0, 0.0, 1.2),
            )
        )
        self._last = Pose(0.6, 0.0, 0.0, 1.2)

    def status(self) -> PoseStatus:
        if self.started:
            self._last = next(self._poses, self._last)
        return PoseStatus(self._last, 0.0, self.started, None)


class NearHomeHeadingEscapePose(FakePose):
    """Minimized end of Banana run 1d5129e4 including post-stop settling."""

    def __init__(self, motion: FakeMotion) -> None:
        super().__init__()
        self._motion = motion
        self._poses = iter(
            (
                Pose(0.30, 0.0, math.pi, 1.0),
                Pose(0.25, 0.0, math.pi, 1.05),
                Pose(0.20, 0.0, math.pi, 1.1),
                Pose(0.12, 0.0, 0.0, 1.2),
                Pose(0.065, 0.0, 0.0, 1.3),
            )
        )
        self._last = Pose(0.30, 0.0, math.pi, 1.0)
        self.samples = 0

    def status(self) -> PoseStatus:
        if self.started:
            self._last = next(self._poses, self._last)
            self.samples += 1
            if self.samples >= 5:
                assert self._motion.armed is False
        return PoseStatus(self._last, 0.0, self.started, None)


class PhysicalReturnHeadingDriftPose(FakePose):
    """Minimized pose replay from run 864c856b's failed Home return."""

    def __init__(self) -> None:
        super().__init__()
        # Home is the origin, so the bearing is pi while x remains positive.
        # The physical run began near -0.10 rad, drifted through -0.30 rad,
        # then crossed the existing 20-degree fail-closed gate.
        heading_errors = (-0.099, -0.072, -0.122, -0.180, -0.223, -0.299, -0.351)
        self._poses = iter(
            Pose(distance, 0.0, math.pi - error, 1.0 + index * 0.1)
            for index, (distance, error) in enumerate(
                zip((1.49, 1.42, 1.31, 1.18, 1.02, 0.83, 0.70), heading_errors),
                start=1,
            )
        )
        self._last = Pose(1.49, 0.0, math.pi + 0.099, 1.0)

    def status(self) -> PoseStatus:
        if self.started:
            self._last = next(self._poses, self._last)
        return PoseStatus(self._last, 0.0, self.started, None)


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


class HomeMotionSequencePose(HomeTurnPose):
    def __init__(self) -> None:
        super().__init__()
        self._returning: ReturningPose | None = None

    def begin_return(self) -> None:
        self._returning = ReturningPose()
        self._returning.started = self.started

    def status(self) -> PoseStatus:
        if self._returning is not None:
            return self._returning.status()
        return super().status()


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
            command["phase"] == "turn_to_fruit" and command["forward_mps"] == 0.0
            for command in manager.motion_trace()
        )
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_motion_commands_stream_to_the_run_black_box_before_stage_completion(
    tmp_path,
) -> None:
    async def scenario() -> None:
        from uuid import uuid4

        recorder = RunBlackBox(tmp_path)
        run_id = str(uuid4())
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: TurningPose(),
            black_box=recorder,
        )
        await manager.start()
        manager.start_motion_trace("turn_to_fruit", run_id=run_id)
        statuses = iter(
            (
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

        await manager.find_target(
            lambda: next(
                statuses,
                {"camera_healthy": True, "target_ready": True},
            ),
            "pear",
            yaw_rps=0.20,
            sweep_rad=1.4,
            timeout_s=0.25,
        )

        events = recorder.read(run_id)
        commands = [event for event in events if event["kind"] == "motion_command"]
        assert commands
        assert commands[0]["phase"] == "turn_to_fruit"
        assert commands[0]["payload"]["yaw_rps"] == 0.20
        await manager.close()

    asyncio.run(scenario())


def test_slow_motion_rpc_diagnostic_streams_to_the_active_run_black_box(
    tmp_path,
) -> None:
    async def scenario() -> None:
        from uuid import uuid4

        class DiagnosticMotion(FakeMotion):
            diagnostic_sink = None

            def set_diagnostic_sink(self, sink) -> None:
                self.diagnostic_sink = sink

        recorder = RunBlackBox(tmp_path)
        run_id = str(uuid4())
        motion = DiagnosticMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
            black_box=recorder,
        )
        await manager.start()
        manager.start_motion_trace("approach_fruit", run_id=run_id)
        assert motion.diagnostic_sink is not None

        motion.diagnostic_sink(
            {
                "kind": "motion_rpc_slow",
                "method": "SwitchGet",
                "slow_threshold_s": 1.0,
                "timeout_s": 5.0,
            }
        )

        events = recorder.read(run_id)
        assert events[-1]["kind"] == "motion_rpc_slow"
        assert events[-1]["phase"] == "approach_fruit"
        assert events[-1]["payload"] == {
            "method": "SwitchGet",
            "slow_threshold_s": 1.0,
            "timeout_s": 5.0,
        }
        await manager.close()

    asyncio.run(scenario())


def test_mission_lifetime_pear_guidance_arrives_stopped_after_disappearance() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()
        guidance = FruitGuidance(
            "pear",
            config=GuidanceConfig(
                duplicate_hold_s=0.025,
                final_push_duration_s=0.10,
            ),
        )
        pts = 0

        def status(*, near: bool = False, visible: bool = True) -> dict[str, object]:
            nonlocal pts
            pts += 1
            return {
                "camera_healthy": True,
                "generation": "camera-1",
                "source": {"pts": pts, "time_base": "1/90000", "age_s": 0.01},
                "detection": (
                    {
                        "label": "pear",
                        "confidence": 0.80,
                        "generation": "camera-1",
                        "source_pts": pts,
                        "source_time_base": "1/90000",
                        "age_s": 0.01,
                        "center_x_ratio": 0.50,
                        "center_y_ratio": 0.80 if near else 0.55,
                        "bottom_ratio": 0.92 if near else 0.65,
                    }
                    if visible
                    else None
                ),
            }

        locked = await manager.guide_target(
            lambda: status(),
            guidance,
            allow_forward=False,
            timeout_s=0.2,
        )
        approach_samples = 0

        def approach_status() -> dict[str, object]:
            nonlocal approach_samples
            approach_samples += 1
            return status(near=True, visible=approach_samples <= 3)

        arrived = await manager.guide_target(
            approach_status,
            guidance,
            allow_forward=True,
            timeout_s=0.2,
        )

        assert locked["acquisition_epoch"] == 1
        assert locked["confidence_summary"] == {
            "detected_frames": 3,
            "minimum": 0.8,
            "maximum": 0.8,
            "average": pytest.approx(0.8),
            "lock_confidence": 0.8,
        }
        assert [sample["source_pts"] for sample in locked["search_trace"]] == [
            1,
            2,
            3,
        ]
        assert all(
            sample["target_fruit"] == "pear"
            and sample["confidence"] == 0.8
            and sample["measured_yaw_rad"] == pytest.approx(0.0)
            and "commanded_yaw_rps" in sample
            and "search_progress_rad" in sample
            and "guidance_action" in sample
            for sample in locked["search_trace"]
        )
        assert locked["search_trace"][-1]["locked"] is True
        assert arrived["acquisition_epoch"] == 1
        assert arrived["arrival_confirmed"] is True
        assert arrived["guidance_reason"] == "first_qualified_lower_edge_arrival"
        assert arrived["final_push_count"] == 0
        assert arrived["forward_pulse_count"] == 0
        assert not any(
            command.reason == "fruit_lower_edge_final_push"
            for command in motion.commands
        )
        assert motion.commands[-1].forward_mps == 0.0
        assert motion.commands[-1].yaw_rps == 0.0
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_search_records_all_fruits_in_bearing_map_but_selected_target_drives_lock() -> None:
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
        guidance = FruitGuidance("pear")
        bearing_map = FruitBearingMap()
        pts = 0

        def status() -> dict[str, object]:
            nonlocal pts
            pts += 1
            # The decoder can already have accepted the next camera frame while
            # inference still publishes observations for the processed frame.
            # Bearing evidence must follow the observations' identity, not the
            # newer scheduler/source identity.
            latest_source_pts = pts + 1
            shared = {
                "generation": "camera-map",
                "source_pts": pts,
                "source_time_base": "1/90000",
            }
            return {
                "camera_healthy": True,
                "generation": "camera-map",
                "source": {
                    "pts": latest_source_pts,
                    "time_base": "1/90000",
                    "age_s": 0.01,
                },
                "detection": {
                    **shared,
                    "label": "pear",
                    "confidence": 0.80,
                    "age_s": 0.01,
                    "center_x_ratio": 0.50,
                    "center_y_ratio": 0.55,
                    "bottom_ratio": 0.65,
                },
                "observations": {
                    fruit: {
                        **shared,
                        "label": fruit,
                        "confidence": confidence,
                        "center_x_ratio": 0.50,
                    }
                    for fruit, confidence in {
                        "apple": 0.90,
                        "banana": 0.85,
                        "pear": 0.80,
                    }.items()
                },
            }

        result = await manager.guide_target(
            status,
            guidance,
            allow_forward=False,
            timeout_s=0.2,
            bearing_map=bearing_map,
            home_pose=home,
        )

        assert result["label"] == "pear"
        assert set(result["fruit_bearing_map"]["fruits"]) == {
            "apple",
            "banana",
            "pear",
        }
        assert result["fruit_bearing_map"]["fruits"]["apple"]["sample_count"] == 3
        await manager.close()

    asyncio.run(scenario())


def test_approach_trace_records_each_guidance_decision_and_arrival_counter() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()
        guidance = FruitGuidance(
            "banana",
            config=GuidanceConfig(final_push_duration_s=0.001),
        )

        def status(
            pts: int,
            *,
            confidence: float = 0.80,
            center_x: float = 0.50,
            center_y: float = 0.55,
            bottom: float = 0.65,
            visible: bool = True,
        ) -> dict[str, object]:
            return {
                "camera_healthy": True,
                "generation": "camera-approach",
                "source": {
                    "pts": pts,
                    "time_base": "1/90000",
                    "age_s": 0.01,
                },
                "detection": (
                    {
                        "label": "banana",
                        "confidence": confidence,
                        "generation": "camera-approach",
                        "source_pts": pts,
                        "source_time_base": "1/90000",
                        "age_s": 0.02,
                        "center_x_ratio": center_x,
                        "center_y_ratio": center_y,
                        "bottom_ratio": bottom,
                    }
                    if visible
                    else None
                ),
                "inference": {
                    "latest": {
                        "source_pts": pts,
                        "detection_pts": pts if visible else None,
                        "model_route": {"full_frame": {"selected": "general"}},
                        "inference_start_monotonic_s": 10.0 + pts,
                        "inference_end_monotonic_s": 10.08 + pts,
                        "inference_duration_s": 0.08,
                        "inference_total_ms": 80.0,
                        "inference_overrun": False,
                        "error": None,
                    },
                    "summary": {
                        "processed_frames": pts,
                        "timed_frames": pts,
                        "overrun_frames": 0,
                        "minimum_ms": 75.0,
                        "maximum_ms": 90.0,
                        "average_ms": 80.0,
                        "overrun_threshold_ms": 200.0,
                    },
                },
            }

        for pts in (1, 2, 3):
            guidance.observe(status(pts), now_s=float(pts) / 10.0)
        statuses = iter(
            (
                status(4, confidence=0.19),
                status(5, visible=False),
                status(6, center_x=0.80),
                status(6, center_x=0.80),
                status(7, center_y=0.80, bottom=0.92),
                status(8, center_y=0.80, bottom=0.92),
                status(9, center_y=0.80, bottom=0.92),
                status(10, visible=False),
                status(11, visible=False),
                status(12, visible=False),
            )
        )

        result = await manager.guide_target(
            lambda: next(statuses, status(13, visible=False)),
            guidance,
            allow_forward=True,
            timeout_s=0.5,
        )

        trace = result["approach_trace"]
        assert [sample["guidance_action"] for sample in trace[:4]] == [
            "stop",
            "stop",
            "align",
            "hold",
        ]
        assert [sample["guidance_reason"] for sample in trace[:4]] == [
            "tracking_confidence_below_floor",
            "target_missing_after_lock",
            "target_outside_outer_corridor",
            "duplicate_frame_bounded_hold",
        ]
        assert trace[0] == {
            **trace[0],
            "sample": 1,
            "generation": "camera-approach",
            "source_pts": 4,
            "source_time_base": "1/90000",
            "source_age_s": 0.01,
            "detection_age_s": 0.02,
            "raw_label": "banana",
            "confidence": 0.19,
            "acquisition_confidence": 0.20,
            "tracking_confidence": 0.20,
            "center_x_ratio": 0.50,
            "center_y_ratio": 0.55,
            "bottom_ratio": 0.65,
            "guidance_phase": "locked",
            "guidance_action": "stop",
            "guidance_reason": "tracking_confidence_below_floor",
            "terminal": False,
            "arrival_eligible": False,
            "near_fresh_samples": 0,
            "near_loss_samples": 0,
            "centered_fresh_samples": 3,
            "frame_advanced": True,
            "resulting_command": {
                "forward_mps": 0.0,
                "yaw_rps": 0.0,
                "reason": "tracking_confidence_below_floor",
            },
            "command_sent": True,
        }
        assert trace[0]["model_route"] == {
            "full_frame": {"selected": "general"}
        }
        assert trace[0]["inference_start_monotonic_s"] == 14.0
        assert trace[0]["inference_end_monotonic_s"] == 14.08
        assert trace[0]["inference_total_ms"] == 80.0
        assert trace[0]["inference_overrun"] is False
        assert trace[0]["detection_pts"] == 4
        assert trace[0]["focus_active"] is False
        assert result["approach_summary"]["inference"] == {
            "processed_frames": 7,
            "timed_frames": 7,
            "overrun_frames": 0,
            "minimum_ms": 75.0,
            "maximum_ms": 90.0,
            "average_ms": 80.0,
            "overrun_threshold_ms": 200.0,
        }
        assert trace[3]["frame_advanced"] is False
        assert trace[3]["resulting_command"]["yaw_rps"] == -0.50
        assert trace[4]["near_fresh_samples"] == 0
        assert trace[4]["arrival_eligible"] is True
        assert trace[4]["guidance_action"] == "arrived"
        assert trace[4]["guidance_reason"] == "first_qualified_lower_edge_arrival"
        assert result["approach_trace_limit"] == 256
        assert result["approach_trace_dropped"] == 0
        summary = result["approach_summary"]
        assert summary["samples"] == 5
        assert summary["action_counts"] == {
            "align": 1,
            "arrived": 1,
            "hold": 1,
            "stop": 2,
        }
        assert summary["reason_counts"]["first_qualified_lower_edge_arrival"] == 1
        assert summary["final_guidance_reason"] == (
            "first_qualified_lower_edge_arrival"
        )
        assert summary["forward_decisions"] == 0
        assert summary["stop_decisions"] == 2
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_approach_trace_records_terminal_camera_failure_before_disarm() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()
        guidance = FruitGuidance("pear")
        guidance.acquisition_epoch = 1
        guidance.phase = GuidancePhase.LOCKED
        unhealthy = {
            "camera_healthy": False,
            "generation": "camera-1",
            "source": {"pts": 90, "time_base": "1/90000", "age_s": 0.40},
            "detection": None,
            "detail": "source progress is stale",
        }

        with pytest.raises(CameraFailure) as raised:
            await manager.guide_target(
                lambda: unhealthy,
                guidance,
                allow_forward=True,
                timeout_s=0.2,
            )

        evidence = raised.value.evidence
        assert evidence["guidance_reason"] == "camera_unhealthy"
        assert evidence["approach_trace"][-1]["guidance_reason"] == "camera_unhealthy"
        assert evidence["approach_trace"][-1]["resulting_command"] == {
            "forward_mps": 0.0,
            "yaw_rps": 0.0,
            "reason": "camera_unhealthy",
        }
        assert evidence["approach_trace"][-1]["terminal"] is True
        assert evidence["approach_trace"][-1]["command_sent"] is False
        assert evidence["approach_summary"]["reason_counts"] == {"camera_unhealthy": 1}
        assert evidence["approach_summary"]["final_guidance_reason"] == (
            "camera_unhealthy"
        )
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_approach_timeout_aggregates_actual_low_confidence_stops() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()
        manager.start_motion_trace("approach_fruit")
        guidance = FruitGuidance("pear")
        guidance.acquisition_epoch = 1
        guidance.phase = GuidancePhase.LOCKED
        calls = 0

        def weak_status() -> dict[str, object]:
            nonlocal calls
            calls += 1
            return {
                "camera_healthy": True,
                "generation": "camera-timeout",
                "source": {
                    "pts": calls,
                    "time_base": "1/90000",
                    "age_s": 0.01,
                },
                "detection": {
                    "label": "pear",
                    "confidence": 0.54,
                    "generation": "camera-timeout",
                    "source_pts": calls,
                    "source_time_base": "1/90000",
                    "age_s": 0.02,
                    "center_x_ratio": 0.50,
                    "center_y_ratio": 0.55,
                    "bottom_ratio": 0.65,
                },
            }

        with pytest.raises(TargetLost, match="camera guidance timed out") as raised:
            await manager.guide_target(
                weak_status,
                guidance,
                allow_forward=True,
                timeout_s=0.045,
            )

        summary = raised.value.evidence["approach_summary"]
        assert calls >= 3
        assert summary == {
            "samples": calls,
            "recorded_samples": calls,
            "dropped_samples": 0,
            "action_counts": {"stop": calls},
            "reason_counts": {"tracking_confidence_below_floor": calls},
            "confidence": {
                "detected_frames": calls,
                "minimum": 0.54,
                "maximum": 0.54,
                "average": pytest.approx(0.54),
                "lock_confidence": None,
            },
            "forward_decisions": 0,
            "stop_decisions": calls,
            "forward_commands_sent": 0,
            "stop_commands_sent": calls,
            "final_guidance_reason": "tracking_confidence_below_floor",
        }
        assert len(raised.value.evidence["approach_trace"]) == calls
        assert all(
            command["reason"] == "tracking_confidence_below_floor"
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


@pytest.mark.parametrize(
    ("final_push_duration_s", "final_push_count", "forward_pulse_count"),
    [(0.001, 1, 5), (0.0, 0, 4)],
)
def test_approach_requires_near_geometry_then_bounded_optional_final_push(
    final_push_duration_s: float,
    final_push_count: int,
    forward_pulse_count: int,
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
            forward_mps=0.50,
            maximum_yaw_rps=0.30,
            near_bottom_ratio=0.86,
            near_center_ratio=0.72,
            near_confirmations=3,
            near_loss_grace_s=0.75,
            final_push_mps=1.0,
            final_push_duration_s=final_push_duration_s,
            timeout_s=0.5,
        )

        assert result["arrival_confirmed"] is True
        assert result["near_confirmations"] == 3
        assert result["final_push_mps"] == 1.0
        assert result["final_push_count"] == final_push_count
        assert result["forward_pulse_count"] == forward_pulse_count
        assert any(
            command.reason == "approach_target_near_visible"
            and command.forward_mps == 0.50
            for command in motion.commands
        )
        assert any(command.forward_mps == 0.50 for command in motion.commands)
        assert any(command.forward_mps == 1.0 for command in motion.commands) is (
            final_push_count == 1
        )
        if final_push_count == 0:
            assert any(
                command.reason == "bounded_final_push_disabled"
                and command.forward_mps == 0.0
                and command.yaw_rps == 0.0
                for command in motion.commands
            )
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
            if command.reason in {"approach_target", "approach_target_near_visible"}
            and command.forward_mps == 1.0
        ]
        assert len(visible_forward) == 6
        assert (
            len(
                [
                    command
                    for command in motion.commands
                    if command.reason == "approach_target_near_visible"
                    and command.forward_mps == 1.0
                ]
            )
            == 3
        )
        assert not any(
            command.reason == "near_target_confirmed" for command in motion.commands
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
            command.forward_mps == 0.0 for command in motion.commands[:first_forward]
        )
        assert all(
            command.reason == "center_target_before_approach"
            for command in motion.commands[:first_forward]
        )
        assert [command.yaw_rps for command in motion.commands[:first_forward]] == [
            -0.50,
            -0.50,
            0.0,
            0.0,
        ]
        assert result["initial_center_confirmations"] == 3
        assert result["initial_center_tolerance_ratio"] == 0.08
        assert result["initial_center_yaw_rps"] == 0.50
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

        assert all(
            command.reason != "fruit_offscreen_final_push"
            for command in motion.commands
        )
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
            maximum_yaw_rps=0.50,
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


def test_position_only_return_uses_fresh_pose_until_home_then_disarms() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: ReturningPose(),
        )
        await manager.start()

        result = await manager.return_home_position(
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            forward_mps=1.0,
            arrival_tolerance_m=0.10,
            heading_gate_rad=math.radians(20.0),
            maximum_yaw_rps=0.50,
            minimum_progress_m=0.03,
            stall_timeout_s=0.10,
            timeout_s=0.50,
        )

        assert result["home_distance_m"] == pytest.approx(0.09)
        assert result["measured_after_disarm"] is True
        assert result["heading_restoration_skipped"] is True
        assert result["pose_source"] == "rt/sportmodestate"
        assert len(motion.commands) == 2
        assert all(command.forward_mps == 1.0 for command in motion.commands)
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_position_only_return_does_not_arm_when_already_home() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: FakePose(),
        )
        await manager.start()

        result = await manager.return_home_position(
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 2.5},
            forward_mps=1.0,
            arrival_tolerance_m=0.10,
            heading_gate_rad=math.radians(20.0),
            maximum_yaw_rps=0.50,
            minimum_progress_m=0.03,
            stall_timeout_s=0.10,
            timeout_s=0.50,
        )

        assert result["home_distance_m"] == 0.0
        assert result["motion_commands_sent"] is False
        assert result["heading_restoration_skipped"] is True
        assert motion.commands == []
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_position_only_return_stops_when_heading_escapes_instead_of_yaw_only() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: HeadingEscapePose(),
        )
        await manager.start()

        with pytest.raises(HardwareUnavailable, match="heading escaped"):
            await manager.return_home_position(
                {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
                forward_mps=1.0,
                arrival_tolerance_m=0.10,
                heading_gate_rad=math.radians(20.0),
                maximum_yaw_rps=0.50,
                minimum_progress_m=0.03,
                stall_timeout_s=0.10,
                timeout_s=0.50,
            )

        assert len(motion.commands) == 1
        assert motion.commands[0].forward_mps == 1.0
        assert all(command.forward_mps > 0.0 for command in motion.commands)
        assert motion.command_modes == ["factory_avoidance"]
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_near_home_heading_escape_reconciles_from_fresh_post_disarm_position() -> None:
    """Replay 1d5129e4: measured position wins after the heading-gate stop."""

    async def scenario() -> None:
        motion = FakeMotion()
        pose = NearHomeHeadingEscapePose(motion)
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: pose,
        )
        await manager.start()

        result = await manager.return_home_position(
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            forward_mps=1.0,
            arrival_tolerance_m=0.10,
            heading_gate_rad=math.radians(20.0),
            maximum_yaw_rps=0.50,
            minimum_progress_m=0.03,
            stall_timeout_s=0.10,
            timeout_s=0.50,
        )

        assert result["home_distance_m"] == pytest.approx(0.065)
        assert result["heading_gate_escape_reconciled"] is True
        assert result["measured_after_disarm"] is True
        assert result["pose_source"] == "rt/sportmodestate"
        assert motion.armed is False
        assert len(motion.commands) == 1
        assert motion.commands[0].forward_mps == 1.0
        await manager.close()

    asyncio.run(scenario())


def test_position_return_replay_never_emits_subthreshold_moving_yaw() -> None:
    async def scenario() -> None:
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: PhysicalReturnHeadingDriftPose(),
        )
        await manager.start()

        with pytest.raises(HardwareUnavailable, match="heading escaped"):
            await manager.return_home_position(
                {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
                forward_mps=1.0,
                arrival_tolerance_m=0.10,
                heading_gate_rad=math.radians(20.0),
                maximum_yaw_rps=0.50,
                minimum_progress_m=0.03,
                stall_timeout_s=0.50,
                timeout_s=1.0,
            )

        moving = [command for command in motion.commands if command.forward_mps > 0.0]
        assert moving
        assert all(abs(command.yaw_rps) >= 0.50 for command in moving)
        assert all(command.forward_mps > 0.0 for command in motion.commands)
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
        assert abs(result["home_bearing_error_rad"]) <= math.radians(5.0)
        assert result["measured_yaw_change_rad"] == pytest.approx(3.10)
        assert result["motion_path"] == "sport_yaw"
        assert motion.command_modes
        assert set(motion.command_modes) == {"sport_yaw"}
        assert all(command.forward_mps == 0.0 for command in motion.commands)
        assert motion.armed is False
        await manager.close()

    asyncio.run(scenario())


def test_home_motion_orders_all_sports_yaw_before_factory_forward_translation() -> None:
    async def scenario() -> None:
        pose = HomeMotionSequencePose()
        motion = FakeMotion()
        manager = HardwareManager(
            live_config(),
            dds_initializer=lambda _interface: None,
            motion_factory=lambda _config: motion,
            pose_factory=lambda _age: pose,
        )
        await manager.start()

        turned = await manager.turn_toward_home(
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            yaw_rps=0.50,
            tolerance_rad=math.radians(5.0),
            timeout_s=0.50,
        )
        pose.begin_return()
        returned = await manager.return_home_position(
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            forward_mps=1.0,
            arrival_tolerance_m=0.10,
            heading_gate_rad=math.radians(20.0),
            maximum_yaw_rps=0.50,
            minimum_progress_m=0.03,
            stall_timeout_s=0.10,
            timeout_s=0.50,
        )

        first_forward = next(
            index
            for index, command in enumerate(motion.commands)
            if command.forward_mps > 0.0
        )
        assert all(
            command.forward_mps == 0.0 and command.yaw_rps != 0.0
            for command in motion.commands[:first_forward]
        )
        assert all(
            command.forward_mps > 0.0 for command in motion.commands[first_forward:]
        )
        assert motion.command_modes[:first_forward] == ["sport_yaw"] * first_forward
        assert motion.command_modes[first_forward:] == ["factory_avoidance"] * (
            len(motion.commands) - first_forward
        )
        assert turned["motion_path"] == "sport_yaw"
        assert returned["motion_path"] == "factory_avoidance"
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
            "odometry_epoch": home["odometry_epoch"],
        }
        assert isinstance(home["odometry_epoch"], str) and home["odometry_epoch"]
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
