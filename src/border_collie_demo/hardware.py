from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from typing import Protocol
from uuid import uuid4

from .config import HardwareConfig
from .flight_recorder import FlightRecorder
from .go2_motion import (
    MotionConfig,
    MotionError,
    MotionNotReady,
    create_go2_motion,
    initialize_dds,
)
from .go2_pose import Go2PoseProvider, PoseStatus
from .home_localization import (
    HomeEstimate,
    HomeLocalizationConfig,
    HomeLocalizer,
    VisualOdometryAdapter,
    pose_to_world,
)
from .models import VelocityCommand
from .motion_guardian import MotionAuthority
from .qualified_tracking import (
    MotionRecommendation,
    QualifiedFruitTracker,
    QualifiedTrackingConfig,
    SearchQualificationHandoff,
    search_handoff_from_status,
)
from .return_home import (
    Pose2D,
    ReturnMode,
    ReturnPlannerConfig,
    ReturnStep,
    normalize_angle,
    plan_return_step,
)
from .search_policy import (
    CandidateAlignmentController,
    DoubleBackSearchController,
    SearchGenerationChanged,
    SearchPolicy,
)
from .target_range import (
    MetricArrivalAction,
    MetricArrivalControlMode,
    MetricArrivalGate,
    RangeCalibration,
    RangeObservation,
)

FORWARD_PULSE_CONFIRMATION = "PATH CLEAR - MOVE WOOF FORWARD"
INITIAL_CENTER_TOLERANCE_RATIO = 0.05
INITIAL_CENTER_CONFIRMATIONS = 3
SEARCH_CROP_CANDIDATE_CONFIDENCE = 0.50
# QualifiedFruitTracker owns the filtered hysteresis state. Hardware translates
# its authorized steering error to a bounded, slew-limited moving command.
APPROACH_YAW_GAIN = 1.50
APPROACH_MOVING_MAX_YAW_RPS = 0.50
APPROACH_YAW_SLEW_RPS_PER_S = 2.0
APPROACH_RECENTER_YAW_RPS = 0.20
FORWARD_CAPABLE_OPERATIONS = frozenset(
    {"forward_pulse", "approach_target", "return_home"}
)
LIDAR_HANDOFF_TRACKING_REASONS = frozenset(
    {
        "qualified_close_track_lost",
        "qualified_close_track_confidence_collapsed",
    }
)


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _home_pose(home: dict[str, object]) -> Pose2D:
    values = tuple(_finite_float(home.get(name)) for name in ("x_m", "y_m", "yaw_rad"))
    if any(value is None for value in values):
        raise ValueError("captured Home pose is missing or invalid")
    x_m, y_m, yaw_rad = (float(value) for value in values if value is not None)
    return Pose2D(x_m, y_m, yaw_rad)


def _pose_payload(pose: Pose2D) -> dict[str, float]:
    return {"x_m": pose.x_m, "y_m": pose.y_m, "yaw_rad": pose.yaw_rad}


def _return_step_payload(step: ReturnStep) -> dict[str, object]:
    return {
        "mode": step.mode.value,
        "distance_m": step.distance_m,
        "heading_error_rad": step.heading_error_rad,
        "forward_mps": step.forward_mps,
        "yaw_rps": step.yaw_rps,
    }


class HardwareUnavailable(RuntimeError):
    pass


class CameraFailure(HardwareUnavailable):
    pass


class RangeUnavailable(HardwareUnavailable):
    def __init__(
        self,
        message: str,
        *,
        evidence: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.evidence = dict(evidence or {})


class TargetLost(HardwareUnavailable):
    def __init__(
        self,
        message: str,
        *,
        evidence: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.evidence = dict(evidence or {})


class TargetLostOffAxis(TargetLost):
    """A close qualified track left the camera corridor before handoff."""


class TurnNoResponse(HardwareUnavailable):
    """The SDK accepted yaw commands but odometry measured no physical turn."""


class MotionAdapterProtocol(Protocol):
    @property
    def armed(self) -> bool: ...

    def status(self) -> dict[str, object]: ...

    async def initialize(self) -> None: ...

    async def arm(self, authority: MotionAuthority) -> str: ...

    async def command(
        self, lease: str, command: VelocityCommand
    ) -> VelocityCommand: ...

    async def release(self, lease: str) -> None: ...

    async def emergency_stop(self) -> list[str]: ...

    async def stand_down(self) -> None: ...

    async def stand_up(self, *, settle_s: float = 1.0) -> None: ...

    async def close(self) -> list[str]: ...


class PoseProviderProtocol(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...

    def status(self) -> PoseStatus: ...


class MetricRangeProviderProtocol(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...

    def status(self) -> dict[str, object]: ...

    def observe(
        self,
        *,
        visual_close_authorized: bool,
        visual_center_error_ratio: float | None,
        allow_handoff: bool,
    ) -> object: ...


MotionFactory = Callable[[MotionConfig], MotionAdapterProtocol]
PoseFactory = Callable[[float], PoseProviderProtocol]
DdsInitializer = Callable[[str | None], None]


class HardwareManager:
    def __init__(
        self,
        config: HardwareConfig | None = None,
        *,
        dds_initializer: DdsInitializer = initialize_dds,
        motion_factory: MotionFactory | None = None,
        pose_factory: PoseFactory | None = None,
        home_localizer: HomeLocalizer | None = None,
        visual_odometry: VisualOdometryAdapter | None = None,
        metric_arrival_gate: MetricArrivalGate | None = None,
        metric_range_provider: MetricRangeProviderProtocol | None = None,
        search_policy: SearchPolicy | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or HardwareConfig()
        self._dds_initializer = dds_initializer
        self._motion_factory = motion_factory or create_go2_motion
        self._pose_factory = pose_factory or (
            lambda maximum_age_s: Go2PoseProvider(
                maximum_age_s=maximum_age_s,
                foot_contact_minimum_force=(self.config.foot_contact_minimum_force),
                stationary_maximum_speed_mps=(self.config.stationary_maximum_speed_mps),
                stationary_maximum_yaw_rate_rps=(
                    self.config.stationary_maximum_yaw_rate_rps
                ),
            )
        )
        self._home_localizer = home_localizer or HomeLocalizer(
            visual_odometry,
            config=HomeLocalizationConfig(
                maximum_odometry_age_s=self.config.pose_maximum_age_s
            ),
        )
        self._metric_arrival_gate = metric_arrival_gate or self._configured_range_gate()
        self._metric_range_provider = metric_range_provider
        self._search_policy = search_policy or SearchPolicy.named("slow-sweep")
        self._monotonic = monotonic
        self._motion: MotionAdapterProtocol | None = None
        self._pose: PoseProviderProtocol | None = None
        self._connected = False
        self._fault: str | None = None
        self._operation_lock = asyncio.Lock()
        self._active_operation: str | None = None
        self._last_pulse: dict[str, object] | None = None
        self._motion_trace_phase: str | None = None
        self._motion_trace: list[dict[str, object]] = []
        self._motion_run_id = "system"
        self._motion_run_epoch = str(uuid4())
        self._motion_authority_phase = "idle"
        self._flight_recorder: FlightRecorder | None = None
        self._captured_home_pose: Pose2D | None = None
        self._breadcrumbs: list[Pose2D] = []
        self._fusion_ingestion_task: asyncio.Task[None] | None = None
        self._continuous_fusion_latest: dict[str, object] | None = None
        self._continuous_fusion_consecutive_trusted = 0
        self._continuous_fusion_interval_s = 0.20

    async def start(self) -> None:
        if not self.config.enabled or self._connected:
            return
        try:
            self._dds_initializer(self.config.network_interface)
            self._pose = self._pose_factory(self.config.pose_maximum_age_s)
            self._pose.start()
            if self._metric_range_provider is not None:
                self._metric_range_provider.start()
            self._motion = self._motion_factory(
                MotionConfig(
                    minimum_forward_mps=self.config.minimum_forward_mps,
                    maximum_forward_mps=self.config.maximum_forward_mps,
                    maximum_yaw_rps=self.config.maximum_yaw_rps,
                    command_watchdog_s=self.config.command_watchdog_s,
                    authority_ttl_s=self.config.motion_authority_ttl_s,
                    rpc_timeout_s=self.config.rpc_timeout_s,
                    client_timeout_s=self.config.client_timeout_s,
                    remote_api_settle_s=self.config.remote_api_settle_s,
                )
            )
            await self._motion.initialize()
            self._connected = True
            self._fault = None
        except Exception as exc:  # noqa: BLE001 - hardware factories are untyped
            self._fault = f"hardware connection failed: {exc}"
            await self._best_effort_stop()
            if self._pose is not None:
                self._pose.close()
            self._pose = None
            self._motion = None
            self._connected = False

    async def run_forward_pulse(self, confirmation: str) -> dict[str, object]:
        if confirmation.strip().upper() != FORWARD_PULSE_CONFIRMATION:
            raise HardwareUnavailable(f'type exactly "{FORWARD_PULSE_CONFIRMATION}"')
        if not self.config.enabled:
            raise HardwareUnavailable("Go2 hardware is disabled")
        if not self.config.lab_motion_enabled:
            raise HardwareUnavailable("lab motion endpoint is disabled")
        if not self._connected or self._motion is None:
            raise HardwareUnavailable(self._fault or "Go2 hardware is not connected")
        if self._pose is None or not self._pose.status().healthy:
            raise HardwareUnavailable("fresh Go2 pose is required before motion")

        async with self._operation_lock:
            if self._active_operation is not None:
                raise HardwareUnavailable(
                    f"hardware operation already active: {self._active_operation}"
                )
            self._active_operation = "forward_pulse"
            lease: str | None = None
            started = time.monotonic()
            release_error: str | None = None
            operation_error: Exception | None = None
            try:
                lease = await self._motion.arm(self._authority_for_active_operation())
                deadline = started + self.config.forward_pulse_duration_s
                while True:
                    now = time.monotonic()
                    if now >= deadline:
                        break
                    await self._send_motion_command(
                        lease,
                        VelocityCommand(
                            self.config.forward_pulse_mps,
                            0.0,
                            "guarded_forward_pulse",
                        ),
                    )
                    await asyncio.sleep(
                        min(self.config.command_heartbeat_s, deadline - now)
                    )
            except Exception as exc:  # noqa: BLE001 - stop in finally, then report
                operation_error = exc
            finally:
                if lease is not None and self._motion.armed:
                    try:
                        await self._motion.release(lease)
                    except MotionError as exc:
                        release_error = str(exc)
                        stop_errors = await self._motion.emergency_stop()
                        if stop_errors:
                            release_error = "; ".join([release_error, *stop_errors])
                else:
                    stop_errors = await self._motion.emergency_stop()
                    if stop_errors:
                        release_error = "; ".join(stop_errors)
                self._active_operation = None

            elapsed_s = time.monotonic() - started
            self._last_pulse = {
                "motion_path": "factory_avoidance",
                "forward_mps": self.config.forward_pulse_mps,
                "requested_duration_s": self.config.forward_pulse_duration_s,
                "elapsed_s": round(elapsed_s, 3),
                "stopped": not self._motion.armed,
                "release_error": release_error,
            }
            if operation_error is not None:
                raise HardwareUnavailable(
                    f"forward pulse failed: {operation_error}"
                ) from operation_error
            if release_error is not None:
                raise HardwareUnavailable(f"forward pulse stop failed: {release_error}")
            return dict(self._last_pulse)

    async def emergency_stop(self) -> list[str]:
        if self._motion is None:
            return []
        return await self._motion.emergency_stop()

    async def stand_down(self) -> dict[str, object]:
        return await self._run_posture_action(
            operation="stand_down",
            evidence_posture="stand_down",
        )

    async def stand_up(self, *, settle_s: float = 1.0) -> dict[str, object]:
        return await self._run_posture_action(
            operation="stand_up",
            evidence_posture="balance_stand",
            settle_s=settle_s,
        )

    async def turn_relative(
        self,
        angle_rad: float,
        *,
        yaw_rps: float,
        tolerance_rad: float,
        timeout_s: float,
        response_timeout_s: float = 0.75,
        response_min_progress_rad: float = math.radians(2.0),
    ) -> dict[str, object]:
        """Turn through a measured robot-local yaw change, then disarm."""
        requested = float(angle_rad)
        rate = float(yaw_rps)
        tolerance = float(tolerance_rad)
        timeout = float(timeout_s)
        response_timeout = float(response_timeout_s)
        response_min_progress = float(response_min_progress_rad)
        if not all(
            math.isfinite(value)
            for value in (
                requested,
                rate,
                tolerance,
                timeout,
                response_timeout,
                response_min_progress,
            )
        ):
            raise ValueError("turn values must be finite")
        if requested == 0.0 or abs(requested) > 2.0 * math.pi:
            raise ValueError("turn angle must be non-zero and at most one revolution")
        if rate <= 0.0 or rate > self.config.maximum_yaw_rps:
            raise ValueError("turn rate is outside the configured limit")
        if tolerance <= 0.0 or tolerance >= abs(requested):
            raise ValueError("turn tolerance is inconsistent with the angle")
        if timeout <= 0.0:
            raise ValueError("turn timeout must be positive")
        if response_timeout <= 0.0 or response_min_progress <= 0.0:
            raise ValueError("turn response gate must be positive")
        self._require_autonomy_ready()

        async with self._operation_lock:
            if self._active_operation is not None:
                raise HardwareUnavailable(
                    f"hardware operation already active: {self._active_operation}"
                )
            self._active_operation = "measured_turn"
            lease: str | None = None
            release_error: str | None = None
            operation_error: Exception | None = None
            direction = 1.0 if requested > 0.0 else -1.0
            progress = 0.0
            started = time.monotonic()
            try:
                assert self._pose is not None and self._motion is not None
                initial = self._pose.status()
                if not initial.healthy or initial.pose is None:
                    raise HardwareUnavailable(
                        initial.error or "fresh Go2 pose is required before motion"
                    )
                previous_yaw = initial.pose.yaw_rad
                lease = await self._motion.arm(self._authority_for_active_operation())
                command_started = time.monotonic()
                deadline = started + timeout
                while time.monotonic() < deadline:
                    sample = self._pose.status()
                    if not sample.healthy or sample.pose is None:
                        raise HardwareUnavailable(
                            sample.error or "Go2 pose became stale during turn"
                        )
                    yaw_step = math.atan2(
                        math.sin(sample.pose.yaw_rad - previous_yaw),
                        math.cos(sample.pose.yaw_rad - previous_yaw),
                    )
                    previous_yaw = sample.pose.yaw_rad
                    progress += max(0.0, direction * yaw_step)
                    if progress >= abs(requested) - tolerance:
                        break
                    response_elapsed = time.monotonic() - command_started
                    if (
                        response_elapsed >= response_timeout
                        and progress < response_min_progress
                    ):
                        raise TurnNoResponse(
                            "yaw command was accepted but measured only "
                            f"{math.degrees(progress):.1f} degrees in "
                            f"{response_elapsed:.2f} seconds"
                        )
                    await self._send_motion_command(
                        lease,
                        VelocityCommand(0.0, direction * rate, "measured_turn"),
                    )
                    await asyncio.sleep(self.config.command_heartbeat_s)
                else:
                    raise HardwareUnavailable("measured turn timed out")
            except Exception as exc:  # noqa: BLE001 - always disarm below
                operation_error = exc
            finally:
                if (
                    lease is not None
                    and self._motion is not None
                    and self._motion.armed
                ):
                    try:
                        await self._motion.release(lease)
                    except MotionError as exc:
                        release_error = str(exc)
                        stop_errors = await self._motion.emergency_stop()
                        if stop_errors:
                            release_error = "; ".join([release_error, *stop_errors])
                elif self._motion is not None:
                    stop_errors = await self._motion.emergency_stop()
                    if stop_errors:
                        release_error = "; ".join(stop_errors)
                self._active_operation = None

            if isinstance(operation_error, TurnNoResponse):
                raise operation_error
            if operation_error is not None:
                raise HardwareUnavailable(
                    f"measured turn failed: {operation_error}"
                ) from operation_error
            if release_error is not None:
                raise HardwareUnavailable(f"measured turn stop failed: {release_error}")
            return {
                "motion_path": "sport_client",
                "requested_angle_rad": requested,
                "measured_yaw_change_rad": progress,
                "yaw_rps": direction * rate,
                "elapsed_s": round(time.monotonic() - started, 3),
                "stopped": True,
                "motion_commands_sent": True,
            }

    async def find_target(
        self,
        status_reader: Callable[[], dict[str, object]],
        target_fruit: str,
        *,
        yaw_rps: float,
        sweep_rad: float,
        timeout_s: float,
        search_policy: SearchPolicy | None = None,
    ) -> dict[str, object]:
        """Run one measured bounded search until fresh target evidence locks."""
        rate = float(yaw_rps)
        sweep = float(sweep_rad)
        timeout = float(timeout_s)
        if not all(math.isfinite(value) for value in (rate, sweep, timeout)):
            raise ValueError("search values must be finite")
        if rate <= 0.0 or rate > self.config.maximum_yaw_rps:
            raise ValueError("search rate is outside the configured limit")
        if sweep <= 0.0 or sweep > 2.0 * math.pi or timeout <= 0.0:
            raise ValueError("search bounds are invalid")
        self._require_autonomy_ready()
        policy = search_policy or self._search_policy

        async with self._operation_lock:
            if self._active_operation is not None:
                raise HardwareUnavailable(
                    f"hardware operation already active: {self._active_operation}"
                )
            self._active_operation = "find_target"
            lease: str | None = None
            release_error: str | None = None
            operation_error: Exception | None = None
            commands_sent = False
            progress = 0.0
            started = self._monotonic()
            evidence: dict[str, object] | None = None
            double_back_controller = (
                DoubleBackSearchController(
                    target_fruit,
                    rate,
                    policy,
                    started_at=started,
                )
                if policy.candidate_mode == "double-back"
                else None
            )
            candidate_alignment_controller = (
                CandidateAlignmentController(target_fruit, rate, policy)
                if double_back_controller is None
                else None
            )
            required_search_detections = policy.required_consecutive_detections(
                target_fruit
            )
            recognition: dict[str, object] = {
                "samples": 0,
                "pear_candidate_samples": 0,
                "maximum_confidence": None,
                "maximum_consecutive_detections": 0,
                "maximum_bbox_area_ratio": None,
                "closest_detection": None,
                "search_policy": policy.name,
                "search_lock_minimum_detections": (
                    required_search_detections
                ),
            }
            if double_back_controller is not None:
                recognition.update(double_back_controller.evidence())
            try:
                assert self._pose is not None and self._motion is not None
                initial = self._pose.status()
                if not initial.healthy or initial.pose is None:
                    raise HardwareUnavailable(
                        initial.error or "fresh Go2 pose is required before search"
                    )
                previous_yaw = initial.pose.yaw_rad
                lease = await self._motion.arm(self._authority_for_active_operation())
                deadline = started + timeout
                while self._monotonic() < deadline:
                    status = status_reader()
                    sampled_at = self._monotonic()
                    self._record_perception_sample(status, target_fruit)
                    search_lock = policy.evaluate(status, target_fruit)
                    recognition["samples"] = int(recognition["samples"]) + 1
                    if not status.get("camera_healthy"):
                        raise CameraFailure(
                            str(
                                status.get("detail")
                                or "camera evidence became unhealthy"
                            )
                        )
                    detection = status.get("detection")
                    if (
                        double_back_controller is not None
                        and status.get("target_ready") is not True
                    ):
                        try:
                            double_back_controller.sample(status, now_s=sampled_at)
                        except SearchGenerationChanged as exc:
                            raise CameraFailure(str(exc)) from exc
                        recognition.update(double_back_controller.evidence())
                    if isinstance(detection, dict):
                        label = str(detection.get("label") or "").casefold()
                        if label == target_fruit.casefold():
                            recognition["pear_candidate_samples"] = (
                                int(recognition["pear_candidate_samples"]) + 1
                            )
                        confidence = _finite_float(detection.get("confidence"))
                        if confidence is not None:
                            current_maximum = _finite_float(
                                recognition["maximum_confidence"]
                            )
                            recognition["maximum_confidence"] = (
                                confidence
                                if current_maximum is None
                                else max(current_maximum, confidence)
                            )
                        consecutive = detection.get("consecutive_detections")
                        if isinstance(consecutive, int) and not isinstance(
                            consecutive, bool
                        ):
                            recognition["maximum_consecutive_detections"] = max(
                                int(recognition["maximum_consecutive_detections"]),
                                consecutive,
                            )
                        area_ratio = _finite_float(detection.get("bbox_area_ratio"))
                        maximum_area = _finite_float(
                            recognition["maximum_bbox_area_ratio"]
                        )
                        if area_ratio is not None and (
                            maximum_area is None or area_ratio > maximum_area
                        ):
                            bbox = detection.get("bbox_xyxy")
                            recognition["maximum_bbox_area_ratio"] = area_ratio
                            recognition["closest_detection"] = {
                                "source_pts": detection.get("source_pts"),
                                "confidence": confidence,
                                "bbox_xyxy": (
                                    list(bbox)
                                    if isinstance(bbox, (list, tuple))
                                    else None
                                ),
                                "bbox_area_ratio": area_ratio,
                            }
                        crop_confirmation = detection.get("crop_confirmation")
                        if isinstance(crop_confirmation, dict):
                            full_frame_confidence = _finite_float(
                                crop_confirmation.get("full_frame_confidence")
                            )
                            crop_candidate = bool(
                                label == target_fruit.casefold()
                                and crop_confirmation.get("attempted") is True
                                and full_frame_confidence is not None
                                and full_frame_confidence
                                >= SEARCH_CROP_CANDIDATE_CONFIDENCE
                            )
                            if crop_candidate:
                                recognition["crop_confirmation_samples"] = (
                                    int(
                                        recognition.get(
                                            "crop_confirmation_samples",
                                            0,
                                        )
                                    )
                                    + 1
                                )
                                recognition["crop_candidate_confidence_threshold"] = (
                                    SEARCH_CROP_CANDIDATE_CONFIDENCE
                                )
                    search_target_ready = (
                        search_lock.qualified
                        if target_fruit.casefold() == "apple"
                        or policy.name == "fast-lock"
                        else status.get("target_ready") is True
                    )
                    if search_target_ready and isinstance(detection, dict):
                        label = str(detection.get("label") or "").casefold()
                        if label == target_fruit.casefold():
                            handoff = search_handoff_from_status(
                                (
                                    status
                                    if status.get("target_ready") is True
                                    else {**status, "target_ready": True}
                                ),
                                target_fruit,
                                qualified_monotonic_s=sampled_at,
                                minimum_stable_detections=(
                                    required_search_detections
                                ),
                            )
                            evidence = {
                                "motion_path": "sport_client",
                                "label": target_fruit,
                                "confidence": detection.get("confidence"),
                                "stable_detections": detection.get(
                                    "consecutive_detections"
                                ),
                                "search_progress_rad": progress,
                                "recognition": {
                                    **recognition,
                                    "search_progress_rad": progress,
                                    "search_lock_reason": search_lock.reason,
                                },
                                "motion_commands_sent": commands_sent,
                            }
                            if handoff is not None:
                                evidence.update(
                                    generation=handoff.generation,
                                    source_pts=handoff.source_pts,
                                    source_time_base=handoff.source_time_base,
                                    qualified_monotonic_s=(
                                        handoff.qualified_monotonic_s
                                    ),
                                    center_x_ratio=handoff.center_x_ratio,
                                    center_y_ratio=handoff.center_y_ratio,
                                    bottom_ratio=handoff.bottom_ratio,
                                    bbox_area_ratio=handoff.bbox_area_ratio,
                                    search_qualification=handoff.to_evidence(),
                                )
                            break
                    sample = self._pose.status()
                    if not sample.healthy or sample.pose is None:
                        raise HardwareUnavailable(
                            sample.error or "Go2 pose became stale during search"
                        )
                    yaw_step = math.atan2(
                        math.sin(sample.pose.yaw_rad - previous_yaw),
                        math.cos(sample.pose.yaw_rad - previous_yaw),
                    )
                    progress += max(0.0, yaw_step)
                    previous_yaw = sample.pose.yaw_rad
                    if progress >= sweep:
                        raise TargetLost(
                            f"{target_fruit} was not found in the bounded search sweep",
                            evidence={
                                **recognition,
                                "search_progress_rad": progress,
                            },
                        )
                    now = self._monotonic()
                    if double_back_controller is not None:
                        directive = double_back_controller.command(
                            now_s=now,
                            measured_yaw_step_rad=yaw_step,
                        )
                        recognition.update(double_back_controller.evidence())
                        await self._send_motion_command(
                            lease,
                            VelocityCommand(0.0, directive.yaw_rps, directive.reason),
                        )
                        commands_sent = commands_sent or directive.yaw_rps != 0.0
                        await asyncio.sleep(self.config.command_heartbeat_s)
                        continue
                    assert candidate_alignment_controller is not None
                    directive = candidate_alignment_controller.command(
                        status,
                        now_s=now,
                    )
                    recognition.update(candidate_alignment_controller.evidence())
                    await self._send_motion_command(
                        lease,
                        VelocityCommand(0.0, directive.yaw_rps, directive.reason),
                    )
                    commands_sent = commands_sent or directive.yaw_rps != 0.0
                    await asyncio.sleep(self.config.command_heartbeat_s)
                else:
                    raise TargetLost(
                        f"{target_fruit} search timed out",
                        evidence={
                            **recognition,
                            "search_progress_rad": progress,
                        },
                    )
            except Exception as exc:  # noqa: BLE001 - always disarm below
                operation_error = exc
            finally:
                if (
                    lease is not None
                    and self._motion is not None
                    and self._motion.armed
                ):
                    try:
                        await self._motion.release(lease)
                    except MotionError as exc:
                        release_error = str(exc)
                        await self._motion.emergency_stop()
                self._active_operation = None

            if operation_error is not None:
                raise operation_error
            if release_error is not None:
                raise HardwareUnavailable(f"search stop failed: {release_error}")
            assert evidence is not None
            return evidence

    async def approach_target(
        self,
        status_reader: Callable[[], dict[str, object]],
        target_fruit: str,
        *,
        forward_mps: float,
        maximum_yaw_rps: float,
        near_bottom_ratio: float,
        near_center_ratio: float,
        near_confirmations: int,
        near_loss_grace_s: float,
        close_range_mps: float,
        final_push_mps: float,
        final_push_duration_s: float,
        timeout_s: float,
        metric_arrival_required: bool = False,
        search_handoff: SearchQualificationHandoff | None = None,
    ) -> dict[str, object]:
        """Follow temporal fruit recommendations and stop on qualified Arrival."""
        numeric = (
            forward_mps,
            maximum_yaw_rps,
            near_bottom_ratio,
            near_center_ratio,
            near_loss_grace_s,
            close_range_mps,
            final_push_mps,
            final_push_duration_s,
            timeout_s,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("approach values must be finite")
        if not 0.0 < forward_mps <= self.config.maximum_forward_mps:
            raise ValueError("approach speed is outside the configured limit")
        if not 0.0 < final_push_mps <= self.config.maximum_forward_mps:
            raise ValueError("final push is outside the configured limit")
        if forward_mps < self.config.minimum_forward_mps:
            raise ValueError(
                "approach speed is below minimum "
                f"{self.config.minimum_forward_mps:.2f} m/s"
            )
        if not 0.0 < close_range_mps <= forward_mps:
            raise ValueError("close-range speed must not exceed approach speed")
        if close_range_mps < self.config.minimum_forward_mps:
            raise ValueError(
                "close-range speed is below minimum "
                f"{self.config.minimum_forward_mps:.2f} m/s"
            )
        if not 0.0 < maximum_yaw_rps <= self.config.maximum_yaw_rps:
            raise ValueError("approach yaw is outside the configured limit")
        if not 0.0 < near_bottom_ratio <= 1.0 or not 0.0 < near_center_ratio <= 1.0:
            raise ValueError("near-fruit geometry thresholds are invalid")
        if (
            near_confirmations < 1
            or min(near_loss_grace_s, final_push_duration_s, timeout_s) <= 0.0
        ):
            raise ValueError("approach timing or confirmation count is invalid")
        self._require_autonomy_ready()
        if metric_arrival_required and self._metric_arrival_gate is None:
            raise RangeUnavailable(
                "metric Arrival is unavailable: forward range is not calibrated",
                evidence={"range_calibration": {"configured": False}},
            )

        async with self._operation_lock:
            if self._active_operation is not None:
                raise HardwareUnavailable(
                    f"hardware operation already active: {self._active_operation}"
                )
            self._active_operation = "approach_target"
            lease: str | None = None
            release_error: str | None = None
            operation_error: Exception | None = None
            evidence: dict[str, object] | None = None
            commands_sent = False
            forward_pulse_count = 0
            metric_final_approach_pulses = 0
            slow_speed_scale = close_range_mps / forward_mps
            tracker = QualifiedFruitTracker(
                QualifiedTrackingConfig.for_fruit(
                    target_fruit,
                    acquisition_confirmations=(self.config.pear_tracking_confirmations),
                    center_tolerance_ratio=INITIAL_CENTER_TOLERANCE_RATIO,
                    center_confirmations=INITIAL_CENTER_CONFIRMATIONS,
                    near_bottom_ratio=near_bottom_ratio,
                    near_center_ratio=near_center_ratio,
                    near_confirmations=near_confirmations,
                    sight_loss_grace_s=near_loss_grace_s,
                    slow_speed_scale=slow_speed_scale,
                    minimum_tracking_confidence=(
                        self.config.pear_tracking_minimum_confidence
                        if target_fruit.casefold().strip() == "pear"
                        else None
                    ),
                    final_approach_latch_enabled=not metric_arrival_required,
                ),
                search_handoff=search_handoff,
            )
            last_decision_evidence: dict[str, object] = {}
            last_authorized_command: VelocityCommand | None = None
            last_moving_yaw_rps = 0.0
            metric_control_mode = MetricArrivalControlMode.CAMERA_TRACKING
            lidar_handoff_trigger_reason: str | None = None
            lidar_handoff_latched_at: float | None = None
            visual_handoff_mode: object | None = None
            started = time.monotonic()
            try:
                assert self._motion is not None
                lease = await self._motion.arm(self._authority_for_active_operation())
                deadline = started + timeout_s
                while time.monotonic() < deadline:
                    now = time.monotonic()
                    status = status_reader()
                    self._record_perception_sample(status, target_fruit)
                    if not status.get("camera_healthy"):
                        last_authorized_command = await self._send_motion_command(
                            lease,
                            VelocityCommand(reason="qualified_track_camera_unhealthy"),
                        )
                        raise CameraFailure(
                            str(
                                status.get("detail")
                                or "camera evidence became unhealthy"
                            )
                        )
                    decision = tracker.observe(status, now_s=now)
                    last_decision_evidence = dict(decision.evidence)
                    note_visual_track = getattr(
                        self._metric_range_provider,
                        "note_visual_track",
                        None,
                    )
                    visible_detection = status.get("detection")
                    if (
                        metric_control_mode is MetricArrivalControlMode.CAMERA_TRACKING
                        and callable(note_visual_track)
                        and target_fruit.casefold().strip() == "pear"
                        and isinstance(visible_detection, dict)
                        and str(visible_detection.get("label") or "").casefold().strip()
                        == target_fruit.casefold().strip()
                        and decision.reason
                        in {
                            "confirming_target_identity",
                            "confirming_target_reacquisition",
                            "centering_acquired_target",
                            "qualified_track",
                            "close_range_track",
                            "qualified_visible_arrival",
                            "large_tracking_error",
                            "close_range_steering",
                        }
                        and int(last_decision_evidence.get("close_range_samples", 0))
                        > 0
                    ):
                        filtered_center = _finite_float(
                            last_decision_evidence.get("filtered_center_x_ratio")
                        )
                        note_visual_track(
                            center_error_ratio=(
                                None
                                if filtered_center is None
                                else filtered_center - 0.5
                            ),
                            close_authorized=True,
                        )
                    metric_decision = None
                    close_samples = int(
                        last_decision_evidence.get("close_range_samples", 0)
                    )
                    handoff_authorized = bool(
                        metric_arrival_required
                        and self._metric_range_provider is not None
                        and close_samples >= 2
                        and decision.reason in LIDAR_HANDOFF_TRACKING_REASONS
                    )
                    if (
                        metric_control_mode is MetricArrivalControlMode.CAMERA_TRACKING
                        and handoff_authorized
                    ):
                        metric_control_mode = MetricArrivalControlMode.LIDAR_HANDOFF
                        lidar_handoff_trigger_reason = decision.reason
                        lidar_handoff_latched_at = now
                        visual_handoff_mode = decision.evidence.get("arrival_mode")
                    allow_lidar_handoff = (
                        metric_control_mode is MetricArrivalControlMode.LIDAR_HANDOFF
                    )
                    last_decision_evidence.update(
                        {
                            "metric_arrival_control_mode": metric_control_mode.value,
                            "lidar_handoff_latched": allow_lidar_handoff,
                            "lidar_handoff_trigger_reason": (
                                lidar_handoff_trigger_reason
                            ),
                            "lidar_handoff_elapsed_s": (
                                None
                                if lidar_handoff_latched_at is None
                                else max(0.0, now - lidar_handoff_latched_at)
                            ),
                        }
                    )
                    metric_path_active = bool(
                        allow_lidar_handoff
                        or (
                            self._metric_range_provider is None
                            and decision.recommendation
                            in {
                                MotionRecommendation.SLOW,
                                MotionRecommendation.ARRIVAL,
                            }
                        )
                    )
                    if metric_arrival_required and metric_path_active:
                        assert self._metric_arrival_gate is not None
                        range_observation = self._range_observation(
                            status,
                            target_fruit=target_fruit,
                            filtered_center_x=last_decision_evidence.get(
                                "filtered_center_x_ratio"
                            ),
                            last_command=last_authorized_command,
                            close_speed_mps=close_range_mps,
                            visual_close_authorized=False,
                            allow_lidar_handoff=allow_lidar_handoff,
                        )
                        metric_decision = self._metric_arrival_gate.observe(
                            range_observation
                        )
                        last_decision_evidence.update(metric_decision.evidence)
                        last_decision_evidence.update(
                            {
                                "range_source": range_observation.range_source,
                                "range_association_mode": (
                                    range_observation.association_mode
                                ),
                                "range_association_valid": (
                                    range_observation.association_valid
                                ),
                            }
                        )
                        if metric_decision.action is MetricArrivalAction.UNAVAILABLE:
                            if metric_decision.reason in {
                                "pear_lidar_visual_association_pending",
                                "pear_lidar_loss_association_pending",
                            }:
                                last_authorized_command = (
                                    await self._send_motion_command(
                                        lease,
                                        VelocityCommand(
                                            reason="metric_lidar_association_pending"
                                        ),
                                    )
                                )
                                last_moving_yaw_rps = 0.0
                                await asyncio.sleep(self.config.command_heartbeat_s)
                                continue
                            last_authorized_command = await self._send_motion_command(
                                lease,
                                VelocityCommand(reason="metric_range_unavailable"),
                            )
                            raise RangeUnavailable(
                                "metric Arrival unavailable: "
                                + metric_decision.reason.replace("_", " "),
                                evidence=dict(metric_decision.evidence),
                            )
                        if metric_decision.action is MetricArrivalAction.BRAKE:
                            last_authorized_command = await self._send_motion_command(
                                lease,
                                VelocityCommand(reason="metric_arrival_predicted_stop"),
                            )
                            last_moving_yaw_rps = 0.0
                            await asyncio.sleep(self.config.command_heartbeat_s)
                            continue

                    lidar_handoff_advance = bool(
                        metric_decision is not None
                        and metric_decision.action is MetricArrivalAction.ADVANCE
                        and last_decision_evidence.get("range_association_mode")
                        == "lidar_handoff"
                    )

                    camera_arrival_confirmed = bool(
                        not metric_arrival_required
                        and decision.recommendation is MotionRecommendation.ARRIVAL
                        and decision.reason != "qualified_visible_arrival"
                    )
                    arrival_confirmed = camera_arrival_confirmed or (
                        metric_decision is not None
                        and metric_decision.action is MetricArrivalAction.ARRIVAL
                    )
                    if arrival_confirmed:
                        camera_final_push_count = 0
                        if camera_arrival_confirmed:
                            push_deadline = time.monotonic() + final_push_duration_s
                            while time.monotonic() < push_deadline:
                                push_status = status_reader()
                                self._record_perception_sample(
                                    push_status,
                                    target_fruit,
                                )
                                if not push_status.get("camera_healthy"):
                                    last_authorized_command = (
                                        await self._send_motion_command(
                                            lease,
                                            VelocityCommand(
                                                reason=(
                                                    "camera_final_push_unhealthy"
                                                )
                                            ),
                                        )
                                    )
                                    raise CameraFailure(
                                        str(
                                            push_status.get("detail")
                                            or "camera became unhealthy during final push"
                                        )
                                    )
                                last_authorized_command = (
                                    await self._send_motion_command(
                                        lease,
                                        VelocityCommand(
                                            final_push_mps,
                                            0.0,
                                            "camera_final_push",
                                        ),
                                    )
                                )
                                forward_pulse_count += 1
                                commands_sent = True
                                remaining = push_deadline - time.monotonic()
                                if remaining > 0.0:
                                    await asyncio.sleep(
                                        min(
                                            self.config.command_heartbeat_s,
                                            remaining,
                                        )
                                    )
                            camera_final_push_count = 1
                        last_authorized_command = await self._send_motion_command(
                            lease,
                            VelocityCommand(reason="qualified_arrival_stop"),
                        )
                        evidence = {
                            **last_decision_evidence,
                            "arrival_confirmed": True,
                            "arrival_mode": (
                                "metric_forward_range"
                                if metric_arrival_required
                                else last_decision_evidence.get("arrival_mode")
                            ),
                            "visual_final_approach_mode": (
                                visual_handoff_mode
                                if allow_lidar_handoff
                                else decision.evidence.get("arrival_mode")
                            ),
                            "near_confirmations": last_decision_evidence.get(
                                "near_samples", 0
                            ),
                            "close_range_mps": close_range_mps,
                            "final_push_mps": final_push_mps,
                            "final_push_duration_s": final_push_duration_s,
                            "final_push_count": (
                                metric_final_approach_pulses
                                if metric_arrival_required
                                else camera_final_push_count
                            ),
                            "initial_center_confirmations": (
                                INITIAL_CENTER_CONFIRMATIONS
                            ),
                            "initial_center_tolerance_ratio": (
                                INITIAL_CENTER_TOLERANCE_RATIO
                            ),
                            "initial_center_yaw_rps": maximum_yaw_rps,
                            "moving_yaw_deadband_ratio": (
                                tracker.config.moving_steering_enter_ratio
                            ),
                            "moving_steering_enter_ratio": (
                                tracker.config.moving_steering_enter_ratio
                            ),
                            "moving_steering_exit_ratio": (
                                tracker.config.moving_steering_exit_ratio
                            ),
                            "moving_yaw_gain": APPROACH_YAW_GAIN,
                            "moving_yaw_maximum_rps": min(
                                maximum_yaw_rps,
                                APPROACH_MOVING_MAX_YAW_RPS,
                            ),
                            "moving_yaw_slew_rps_per_s": (APPROACH_YAW_SLEW_RPS_PER_S),
                            "stationary_recenter_error_ratio": (
                                tracker.config.stationary_recenter_error_ratio
                            ),
                            "stationary_recenter_yaw_rps": min(
                                maximum_yaw_rps,
                                APPROACH_RECENTER_YAW_RPS,
                            ),
                            "forward_pulse_count": forward_pulse_count,
                            "forward_pulse_period_s": self.config.command_heartbeat_s,
                            "motion_commands_sent": commands_sent,
                            "tracking_minimum_confidence": (
                                tracker.config.tracking_confidence
                            ),
                            "close_range_continuation_samples": (
                                last_decision_evidence.get("close_range_samples", 0)
                            ),
                        }
                        break
                    if decision.recommendation is MotionRecommendation.HOLD:
                        held_command = last_authorized_command or VelocityCommand(
                            reason="qualified_track_hold_without_authority"
                        )
                        last_authorized_command = await self._send_motion_command(
                            lease,
                            held_command,
                        )
                        if last_authorized_command.forward_mps > 0.0:
                            forward_pulse_count += 1
                        commands_sent = commands_sent or (
                            last_authorized_command.forward_mps != 0.0
                            or last_authorized_command.yaw_rps != 0.0
                        )
                        await asyncio.sleep(self.config.command_heartbeat_s)
                        continue
                    if lidar_handoff_advance:
                        last_authorized_command = await self._send_motion_command(
                            lease,
                            VelocityCommand(
                                close_range_mps,
                                0.0,
                                "metric_lidar_handoff",
                            ),
                        )
                        last_moving_yaw_rps = 0.0
                        forward_pulse_count += 1
                        metric_final_approach_pulses += 1
                        commands_sent = True
                        await asyncio.sleep(self.config.command_heartbeat_s)
                        continue
                    if decision.recommendation in {
                        MotionRecommendation.SEARCH,
                        MotionRecommendation.STOP,
                    }:
                        last_authorized_command = await self._send_motion_command(
                            lease,
                            VelocityCommand(
                                reason=f"qualified_track_{decision.reason}"
                            ),
                        )
                        last_moving_yaw_rps = 0.0
                        if decision.reason == "target_lost_off_axis":
                            raise TargetLostOffAxis(
                                f"qualified {target_fruit} left the centered "
                                "close-range handoff corridor",
                                evidence=dict(last_decision_evidence),
                            )
                        await asyncio.sleep(self.config.command_heartbeat_s)
                        continue

                    horizontal_error = decision.horizontal_error
                    if decision.recommendation is MotionRecommendation.ALIGN:
                        recenter_yaw_rps = (
                            min(maximum_yaw_rps, APPROACH_RECENTER_YAW_RPS)
                            if decision.reason == "center_corridor_recenter"
                            else maximum_yaw_rps
                        )
                        yaw = (
                            0.0
                            if abs(horizontal_error) <= INITIAL_CENTER_TOLERANCE_RATIO
                            else -math.copysign(
                                recenter_yaw_rps,
                                horizontal_error,
                            )
                        )
                        last_authorized_command = await self._send_motion_command(
                            lease,
                            VelocityCommand(
                                0.0,
                                yaw,
                                (
                                    "center_corridor_recenter"
                                    if decision.reason == "center_corridor_recenter"
                                    else (
                                        "recenter_close_target"
                                        if decision.reason == "close_tracking_recenter"
                                        else "center_target_before_approach"
                                    )
                                ),
                            ),
                        )
                        last_moving_yaw_rps = 0.0
                        commands_sent = commands_sent or yaw != 0.0
                        await asyncio.sleep(self.config.command_heartbeat_s)
                        continue

                    moving_yaw_limit = min(
                        maximum_yaw_rps,
                        APPROACH_MOVING_MAX_YAW_RPS,
                    )
                    desired_yaw = max(
                        -moving_yaw_limit,
                        min(
                            moving_yaw_limit,
                            -APPROACH_YAW_GAIN * horizontal_error,
                        ),
                    )
                    maximum_yaw_delta = (
                        APPROACH_YAW_SLEW_RPS_PER_S * self.config.command_heartbeat_s
                    )
                    yaw = max(
                        last_moving_yaw_rps - maximum_yaw_delta,
                        min(
                            last_moving_yaw_rps + maximum_yaw_delta,
                            desired_yaw,
                        ),
                    )
                    if abs(yaw) < 1e-9:
                        yaw = 0.0
                    last_moving_yaw_rps = yaw
                    commanded_scale = decision.forward_scale
                    command_reason = (
                        "approach_target_slow"
                        if decision.recommendation is MotionRecommendation.SLOW
                        else "approach_target"
                    )
                    if decision.recommendation is MotionRecommendation.ARRIVAL and (
                        not metric_arrival_required
                        or (
                            self._metric_range_provider is not None
                            or (
                                metric_decision is not None
                                and metric_decision.action
                                is MetricArrivalAction.ADVANCE
                            )
                        )
                    ):
                        commanded_scale = slow_speed_scale
                        command_reason = (
                            "visual_close_until_sight_loss"
                            if metric_decision is None
                            else "metric_final_approach"
                        )
                        if metric_arrival_required:
                            metric_final_approach_pulses += 1
                    last_authorized_command = await self._send_motion_command(
                        lease,
                        VelocityCommand(
                            forward_mps * commanded_scale,
                            yaw,
                            command_reason,
                        ),
                    )
                    forward_pulse_count += 1
                    commands_sent = True
                    await asyncio.sleep(self.config.command_heartbeat_s)
                else:
                    raise TargetLost(
                        f"qualified {target_fruit} Arrival timed out",
                        evidence={
                            **last_decision_evidence,
                            "close_range_mps": close_range_mps,
                            "moving_yaw_deadband_ratio": (
                                tracker.config.moving_steering_enter_ratio
                            ),
                            "moving_steering_enter_ratio": (
                                tracker.config.moving_steering_enter_ratio
                            ),
                            "moving_steering_exit_ratio": (
                                tracker.config.moving_steering_exit_ratio
                            ),
                            "stationary_recenter_error_ratio": (
                                tracker.config.stationary_recenter_error_ratio
                            ),
                        },
                    )
            except Exception as exc:  # noqa: BLE001 - always disarm below
                operation_error = exc
            finally:
                if (
                    lease is not None
                    and self._motion is not None
                    and self._motion.armed
                ):
                    try:
                        await self._motion.release(lease)
                    except MotionError as exc:
                        release_error = str(exc)
                        await self._motion.emergency_stop()
                self._active_operation = None

            if operation_error is not None:
                raise operation_error
            if release_error is not None:
                raise HardwareUnavailable(f"approach stop failed: {release_error}")
            assert evidence is not None
            return evidence

    def _configured_range_gate(self) -> MetricArrivalGate | None:
        values = (
            self.config.forward_range_index,
            self.config.range_sensor_to_front_envelope_m,
            self.config.range_sensor_latency_s,
            self.config.range_braking_distance_m,
            self.config.range_noise_m,
        )
        if any(value is None for value in values):
            return None
        index, offset, latency, braking, noise = values
        assert isinstance(index, int)
        return MetricArrivalGate(
            RangeCalibration(
                forward_index=index,
                sensor_to_front_envelope_m=float(offset),
                sensor_latency_s=float(latency),
                braking_distance_m=float(braking),
                noise_m=float(noise),
                target_clearance_m=self.config.arrival_clearance_m,
                tolerance_m=self.config.arrival_clearance_tolerance_m,
                maximum_age_s=self.config.range_maximum_age_s,
            )
        )

    def _range_observation(
        self,
        status: dict[str, object],
        *,
        target_fruit: str,
        filtered_center_x: object,
        last_command: VelocityCommand | None,
        close_speed_mps: float,
        visual_close_authorized: bool = False,
        allow_lidar_handoff: bool = False,
    ) -> RangeObservation:
        pose_status = None if self._pose is None else self._pose.status()
        motion = None if pose_status is None else pose_status.motion
        detection = status.get("detection")
        detection_age = (
            _finite_float(detection.get("age_s"))
            if isinstance(detection, dict)
            else None
        )
        label = (
            str(detection.get("label") or "").casefold()
            if isinstance(detection, dict)
            else ""
        )
        visual_fresh = bool(
            status.get("camera_healthy") is True
            and label == target_fruit.casefold().strip()
            and detection_age is not None
            and 0.0 <= detection_age <= 0.25
        )
        center_x = _finite_float(filtered_center_x)
        commanded_stopped = bool(
            last_command is not None
            and last_command.forward_mps == 0.0
            and last_command.yaw_rps == 0.0
        )
        robot_stopped: bool | None = False
        if commanded_stopped:
            velocity_x = None if motion is None else motion.velocity_x_mps
            velocity_y = None if motion is None else motion.velocity_y_mps
            yaw_rate = (
                None
                if motion is None
                else (
                    motion.imu_yaw_rate_rps
                    if motion.imu_yaw_rate_rps is not None
                    else motion.yaw_rate_rps
                )
            )
            if velocity_x is None or velocity_y is None or yaw_rate is None:
                robot_stopped = None
            else:
                robot_stopped = bool(
                    math.hypot(velocity_x, velocity_y)
                    <= self.config.stationary_maximum_speed_mps
                    and abs(yaw_rate) <= self.config.stationary_maximum_yaw_rate_rps
                )
        associated_range_m = None
        association_confidence = None
        association_valid = False
        association_mode = "directional"
        range_source = "directional"
        unavailable_reason = None
        range_age_s = None if pose_status is None else pose_status.age_s
        ranges_m = None if motion is None else motion.obstacle_ranges_m
        if self._metric_range_provider is not None:
            range_source = "lidar_temporal_pear_handoff"
            ranges_m = None
            projected = self._metric_range_provider.observe(
                visual_close_authorized=visual_close_authorized,
                visual_center_error_ratio=(
                    None if center_x is None else center_x - 0.5
                ),
                allow_handoff=allow_lidar_handoff,
            )
            range_age_s = _finite_float(getattr(projected, "age_s", None))
            if getattr(projected, "available", False) is True:
                associated_range_m = _finite_float(
                    getattr(projected, "front_clearance_m", None)
                )
                association_confidence = _finite_float(
                    getattr(projected, "confidence", None)
                )
                association_valid = bool(getattr(projected, "association_valid", False))
                association_mode = str(
                    getattr(projected, "association_mode", "unavailable")
                )
            else:
                unavailable_reason = str(
                    getattr(projected, "reason", "pear_lidar_cloud_unavailable")
                )
        return RangeObservation(
            ranges_m=ranges_m,
            age_s=range_age_s,
            pear_center_error_ratio=(None if center_x is None else center_x - 0.5),
            visual_evidence_fresh=visual_fresh,
            robot_stopped=robot_stopped,
            close_speed_mps=close_speed_mps,
            associated_range_m=associated_range_m,
            association_confidence=association_confidence,
            association_valid=association_valid,
            association_mode=association_mode,
            range_source=range_source,
            unavailable_reason=unavailable_reason,
        )

    async def return_home(
        self,
        home: dict[str, object],
        *,
        forward_mps: float,
        forward_pulse_count: int,
        arrival_tolerance_m: float,
        heading_gate_rad: float,
        maximum_yaw_rps: float,
        minimum_progress_m: float,
        stall_timeout_s: float,
        timeout_s: float,
    ) -> dict[str, object]:
        """Replay outbound forward pulses toward Home with fresh pose guards."""
        home_pose = _home_pose(home)
        config = ReturnPlannerConfig(
            arrival_tolerance_m=arrival_tolerance_m,
            heading_tolerance_rad=math.radians(5.0),
            heading_gate_rad=heading_gate_rad,
            forward_mps=forward_mps,
            maximum_yaw_rps=maximum_yaw_rps,
        )
        if forward_mps > self.config.maximum_forward_mps:
            raise ValueError("return speed is outside the configured limit")
        if forward_mps < self.config.minimum_forward_mps:
            raise ValueError(
                "return speed is below minimum "
                f"{self.config.minimum_forward_mps:.2f} m/s"
            )
        if (
            isinstance(forward_pulse_count, bool)
            or not isinstance(forward_pulse_count, int)
            or forward_pulse_count < 1
        ):
            raise ValueError("return requires a positive outbound forward pulse count")
        if maximum_yaw_rps > self.config.maximum_yaw_rps:
            raise ValueError("return yaw is outside the configured limit")
        if minimum_progress_m <= 0.0 or stall_timeout_s <= 0.0 or timeout_s <= 0.0:
            raise ValueError("return progress and timing values must be positive")
        self._require_autonomy_ready()

        async with self._operation_lock:
            if self._active_operation is not None:
                raise HardwareUnavailable(
                    f"hardware operation already active: {self._active_operation}"
                )
            self._active_operation = "return_home"
            lease: str | None = None
            release_error: str | None = None
            operation_error: Exception | None = None
            evidence: dict[str, object] | None = None
            started = time.monotonic()
            best_distance = math.inf
            progress_at = started
            samples = 0
            replayed_forward_pulses = 0
            commands_sent = False
            route_waypoints = list(reversed(self._breadcrumbs[1:]))
            planned_breadcrumbs = len(route_waypoints)
            reached_breadcrumbs = 0
            active_target: Pose2D | None = None
            try:
                assert self._motion is not None and self._pose is not None
                lease = await self._motion.arm(self._authority_for_active_operation())
                deadline = started + timeout_s
                while time.monotonic() < deadline:
                    sample = self._pose.status()
                    estimate = self._trusted_home_estimate(
                        home_pose,
                        sample,
                        operation="return Home",
                    )
                    assert estimate.pose_from_home is not None
                    current_world = pose_to_world(home_pose, estimate.pose_from_home)
                    home_distance = estimate.home_distance_m
                    assert home_distance is not None
                    samples += 1
                    if home_distance <= arrival_tolerance_m:
                        terminal_step = plan_return_step(home_pose, current_world, config)
                        self._record_flight(
                            "home_navigation_sample",
                            {
                                "epoch": self._motion_run_epoch,
                                "phase": self._motion_authority_phase,
                                "raw_pose": sample.to_dict(),
                                "home_localization": estimate.to_dict(),
                                "current_world": _pose_payload(current_world),
                                "target": _pose_payload(home_pose),
                                "target_kind": "home",
                                "plan": _return_step_payload(terminal_step),
                                "progress": {
                                    "samples": samples,
                                    "best_target_distance_m": (
                                        None if math.isinf(best_distance) else best_distance
                                    ),
                                    "seconds_since_progress": (
                                        time.monotonic() - progress_at
                                    ),
                                    "requested_forward_pulses": forward_pulse_count,
                                    "replayed_forward_pulses": replayed_forward_pulses,
                                    "planned_breadcrumbs": planned_breadcrumbs,
                                    "reached_breadcrumbs": reached_breadcrumbs,
                                },
                                "thresholds": {
                                    "arrival_tolerance_m": arrival_tolerance_m,
                                    "heading_gate_rad": heading_gate_rad,
                                    "heading_tolerance_rad": config.heading_tolerance_rad,
                                    "minimum_progress_m": minimum_progress_m,
                                    "stall_timeout_s": stall_timeout_s,
                                    "timeout_s": timeout_s,
                                },
                            },
                        )
                        evidence = {
                            "home_distance_m": home_distance,
                            "arrival_tolerance_m": arrival_tolerance_m,
                            "requested_forward_pulses": forward_pulse_count,
                            "replayed_forward_pulses": replayed_forward_pulses,
                            "playback_stopped_at_home": (
                                replayed_forward_pulses < forward_pulse_count
                            ),
                            "pose_samples": samples,
                            "motion_path": (
                                "breadcrumb_closed_loop"
                                if planned_breadcrumbs
                                else "direct_fused_closed_loop"
                            ),
                            "planned_breadcrumbs": planned_breadcrumbs,
                            "reached_breadcrumbs": reached_breadcrumbs,
                            "motion_commands_sent": commands_sent,
                            "home_localization": estimate.to_dict(),
                        }
                        break
                    while route_waypoints:
                        waypoint = route_waypoints[0]
                        waypoint_distance = math.hypot(
                            waypoint.x_m - current_world.x_m,
                            waypoint.y_m - current_world.y_m,
                        )
                        if waypoint_distance > self.config.breadcrumb_reach_m:
                            break
                        route_waypoints.pop(0)
                        reached_breadcrumbs += 1
                        active_target = None
                    target = route_waypoints[0] if route_waypoints else home_pose
                    if active_target != target:
                        active_target = target
                        best_distance = math.inf
                        progress_at = time.monotonic()
                    step = plan_return_step(target, current_world, config)
                    if replayed_forward_pulses >= forward_pulse_count:
                        raise HardwareUnavailable(
                            "return pulse playback completed "
                            f"{home_distance:.3f} m from Home"
                        )
                    now = time.monotonic()
                    progressed = step.distance_m <= best_distance - minimum_progress_m
                    if progressed:
                        best_distance = step.distance_m
                        progress_at = now
                    self._record_flight(
                        "home_navigation_sample",
                        {
                            "epoch": self._motion_run_epoch,
                            "phase": self._motion_authority_phase,
                            "raw_pose": sample.to_dict(),
                            "home_localization": estimate.to_dict(),
                            "current_world": _pose_payload(current_world),
                            "target": _pose_payload(target),
                            "target_kind": (
                                "breadcrumb" if route_waypoints else "home"
                            ),
                            "plan": _return_step_payload(step),
                            "progress": {
                                "samples": samples,
                                "best_target_distance_m": best_distance,
                                "seconds_since_progress": now - progress_at,
                                "requested_forward_pulses": forward_pulse_count,
                                "replayed_forward_pulses": replayed_forward_pulses,
                                "planned_breadcrumbs": planned_breadcrumbs,
                                "reached_breadcrumbs": reached_breadcrumbs,
                            },
                            "thresholds": {
                                "arrival_tolerance_m": arrival_tolerance_m,
                                "heading_gate_rad": heading_gate_rad,
                                "heading_tolerance_rad": config.heading_tolerance_rad,
                                "minimum_progress_m": minimum_progress_m,
                                "stall_timeout_s": stall_timeout_s,
                                "timeout_s": timeout_s,
                            },
                        },
                    )
                    if not progressed and now - progress_at > stall_timeout_s:
                        raise HardwareUnavailable(
                            "return Home stalled "
                            f"{step.distance_m:.3f} m from its active waypoint"
                        )
                    command = (
                        VelocityCommand(0.0, step.yaw_rps, "return_course_correction")
                        if step.mode is ReturnMode.TURN_TO_HOME
                        else VelocityCommand(
                            step.forward_mps,
                            step.yaw_rps,
                            "return_home",
                        )
                    )
                    await self._send_motion_command(lease, command)
                    commands_sent = True
                    if command.forward_mps > 0.0:
                        replayed_forward_pulses += 1
                    await asyncio.sleep(self.config.command_heartbeat_s)
                else:
                    raise HardwareUnavailable("return Home timed out")
            except Exception as exc:  # noqa: BLE001 - always disarm below
                operation_error = exc
            finally:
                if (
                    lease is not None
                    and self._motion is not None
                    and self._motion.armed
                ):
                    try:
                        await self._motion.release(lease)
                    except MotionError as exc:
                        release_error = str(exc)
                        await self._motion.emergency_stop()
                self._active_operation = None

            if operation_error is not None:
                raise operation_error
            if release_error is not None:
                raise HardwareUnavailable(f"return Home stop failed: {release_error}")
            assert evidence is not None
            return evidence

    async def turn_toward_home(
        self,
        home: dict[str, object],
        *,
        yaw_rps: float,
        tolerance_rad: float,
        response_timeout_s: float = 0.75,
        response_min_progress_rad: float = math.radians(2.0),
        recovery_settle_s: float = 1.0,
        timeout_s: float,
    ) -> dict[str, object]:
        home_pose = _home_pose(home)
        self._require_autonomy_ready()
        self._require_continuous_fusion_ready(operation="begin the Home turn")
        deadline = time.monotonic() + timeout_s
        recovery_count = 0
        while True:
            assert self._pose is not None
            sample = self._pose.status()
            estimate = self._trusted_home_estimate(
                home_pose,
                sample,
                operation="turn toward Home",
            )
            assert estimate.pose_from_home is not None
            current = estimate.pose_from_home
            dx = -current.x_m
            dy = -current.y_m
            if abs(dy) < 1e-12:
                dy = 0.0
            distance = math.hypot(dx, dy)
            if distance <= 0.10:
                return {
                    "home_distance_m": distance,
                    "home_bearing_error_rad": 0.0,
                    "measured_yaw_change_rad": 0.0,
                    "turn_recovery_count": recovery_count,
                    "motion_commands_sent": False,
                    "home_localization": estimate.to_dict(),
                }
            bearing_error = normalize_angle(math.atan2(dy, dx) - current.yaw_rad)
            if abs(bearing_error) <= tolerance_rad:
                return {
                    "home_distance_m": distance,
                    "home_bearing_error_rad": bearing_error,
                    "measured_yaw_change_rad": 0.0,
                    "turn_recovery_count": recovery_count,
                    "motion_commands_sent": False,
                    "home_localization": estimate.to_dict(),
                }
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                raise HardwareUnavailable("turn toward Home timed out")
            try:
                evidence = await self.turn_relative(
                    bearing_error,
                    yaw_rps=yaw_rps,
                    tolerance_rad=tolerance_rad,
                    timeout_s=remaining_s,
                    response_timeout_s=response_timeout_s,
                    response_min_progress_rad=response_min_progress_rad,
                )
            except TurnNoResponse as exc:
                if recovery_count >= 1:
                    raise HardwareUnavailable(
                        "Home turn did not respond after StandUp/BalanceStand "
                        f"recovery: {exc}"
                    ) from exc
                await self.emergency_stop()
                await self.stand_up(settle_s=recovery_settle_s)
                recovery_count += 1
                continue
            return {
                **evidence,
                "home_distance_m": distance,
                "home_bearing_error_rad": bearing_error,
                "turn_recovery_count": recovery_count,
                "home_localization": estimate.to_dict(),
            }

    async def restore_home_heading(
        self,
        home: dict[str, object],
        *,
        yaw_rps: float,
        heading_tolerance_rad: float,
        position_tolerance_m: float,
        timeout_s: float,
    ) -> dict[str, object]:
        home_pose = _home_pose(home)
        self._require_autonomy_ready()
        assert self._pose is not None
        before = self._pose.status()
        before_estimate = self._trusted_home_estimate(
            home_pose,
            before,
            operation="restore Home heading",
        )
        assert before_estimate.pose_from_home is not None
        initial_distance = before_estimate.home_distance_m
        assert initial_distance is not None
        if initial_distance > position_tolerance_m:
            raise HardwareUnavailable(
                f"Home position was lost before heading restore: {initial_distance:.3f} m"
            )
        initial_error = before_estimate.heading_error_rad
        assert initial_error is not None
        motion_sent = False
        if abs(initial_error) > heading_tolerance_rad:
            await self.turn_relative(
                initial_error,
                yaw_rps=yaw_rps,
                tolerance_rad=heading_tolerance_rad,
                timeout_s=timeout_s,
            )
            motion_sent = True

        after = self._pose.status()
        after_estimate = self._trusted_home_estimate(
            home_pose,
            after,
            operation="restore Home heading",
        )
        distance = after_estimate.home_distance_m
        heading_error = after_estimate.heading_error_rad
        assert distance is not None and heading_error is not None
        if distance > position_tolerance_m:
            raise HardwareUnavailable(
                f"Home position was lost during heading restore: {distance:.3f} m"
            )
        if abs(heading_error) > heading_tolerance_rad:
            raise HardwareUnavailable(
                f"Home heading restore missed tolerance: {heading_error:.3f} rad"
            )
        return {
            "home_distance_m": distance,
            "heading_error_rad": heading_error,
            "position_tolerance_m": position_tolerance_m,
            "heading_tolerance_rad": heading_tolerance_rad,
            "motion_commands_sent": motion_sent,
            "home_localization": after_estimate.to_dict(),
        }

    def capture_home(self) -> dict[str, object]:
        """Capture one fresh, disarmed robot-local pose as Home."""
        if not self.config.enabled:
            raise HardwareUnavailable("Go2 hardware is disabled")
        if not self._connected or self._pose is None or self._motion is None:
            raise HardwareUnavailable(self._fault or "Go2 hardware is not connected")
        if self._active_operation is not None or self._motion.armed:
            raise HardwareUnavailable("motion must be disarmed before Home capture")

        status = self._pose.status()
        if not status.healthy or status.pose is None or status.age_s is None:
            raise HardwareUnavailable(status.error or "fresh Go2 pose is unavailable")
        home_pose = Pose2D(status.pose.x_m, status.pose.y_m, status.pose.yaw_rad)
        self._captured_home_pose = home_pose
        self._breadcrumbs = [home_pose]
        visual_capture = self._home_localizer.capture_home(
            home_pose,
            captured_monotonic_s=status.pose.captured_monotonic_s,
            motion=status.motion,
        )
        self._continuous_fusion_latest = None
        self._continuous_fusion_consecutive_trusted = 0
        self._start_continuous_fusion_ingestion()
        return {
            "x_m": status.pose.x_m,
            "y_m": status.pose.y_m,
            "yaw_rad": status.pose.yaw_rad,
            "captured_monotonic_s": status.pose.captured_monotonic_s,
            "age_s": status.age_s,
            "source": "rt/sportmodestate",
            "visual_odometry": visual_capture,
        }

    def estimate_home(self, home: dict[str, object]) -> dict[str, object]:
        """Return explicit trusted/unavailable Home evidence without motion."""
        try:
            home_pose = _home_pose(home)
        except ValueError as exc:
            return self._unavailable_home_estimate(str(exc))
        if self._pose is None:
            return self._unavailable_home_estimate("Go2 pose provider is unavailable")
        sample = self._pose.status()
        if not sample.healthy or sample.pose is None or sample.age_s is None:
            return self._unavailable_home_estimate(
                sample.error or "fresh Go2 pose is unavailable"
            )
        return self._home_localizer.estimate(
            home_pose,
            Pose2D(sample.pose.x_m, sample.pose.y_m, sample.pose.yaw_rad),
            odometry_age_s=sample.age_s,
            captured_monotonic_s=sample.pose.captured_monotonic_s,
            motion=sample.motion,
        ).to_dict()

    def ingest_home_fusion_sample(self) -> dict[str, object]:
        """Advance captured-Home fusion independently of motion commands."""
        raw_pose: dict[str, object] | None = None
        try:
            if self._captured_home_pose is None or self._pose is None:
                result: dict[str, object] = {
                    "trusted": False,
                    "unavailable_reason": "Home fusion is not initialized",
                }
            else:
                sample = self._pose.status()
                raw_pose = sample.to_dict()
                if (
                    not sample.healthy
                    or sample.pose is None
                    or sample.age_s is None
                    or sample.motion is None
                ):
                    result = {
                        "trusted": False,
                        "unavailable_reason": (
                            sample.error
                            or "fresh pose and IMU evidence are unavailable"
                        ),
                    }
                else:
                    result = self._home_localizer.track_kinematics(
                        self._captured_home_pose,
                        Pose2D(sample.pose.x_m, sample.pose.y_m, sample.pose.yaw_rad),
                        odometry_age_s=sample.age_s,
                        captured_monotonic_s=sample.pose.captured_monotonic_s,
                        motion=sample.motion,
                    )
        except Exception as exc:  # noqa: BLE001 - fusion must fail closed
            result = {
                "trusted": False,
                "unavailable_reason": f"continuous fusion ingestion failed: {exc}",
            }
        if result.get("trusted") is True:
            self._continuous_fusion_consecutive_trusted += 1
        else:
            self._continuous_fusion_consecutive_trusted = 0
        self._continuous_fusion_latest = {
            **result,
            "recorded_monotonic_s": self._monotonic(),
        }
        self._record_flight(
            "home_fusion_sample",
            {
                "epoch": self._motion_run_epoch,
                "phase": self._motion_authority_phase,
                "captured_home": (
                    None
                    if self._captured_home_pose is None
                    else {
                        "x_m": self._captured_home_pose.x_m,
                        "y_m": self._captured_home_pose.y_m,
                        "yaw_rad": self._captured_home_pose.yaw_rad,
                    }
                ),
                "raw_pose": raw_pose,
                "estimate": dict(self._continuous_fusion_latest),
            },
        )
        return dict(result)

    def _start_continuous_fusion_ingestion(self) -> None:
        task = self._fusion_ingestion_task
        if task is not None:
            task.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._fusion_ingestion_task = None
            return
        self._fusion_ingestion_task = loop.create_task(
            self._continuous_fusion_ingestion_loop()
        )

    async def _continuous_fusion_ingestion_loop(self) -> None:
        while self._connected and self._captured_home_pose is not None:
            await asyncio.sleep(self._continuous_fusion_interval_s)
            self.ingest_home_fusion_sample()

    def _require_continuous_fusion_ready(self, *, operation: str) -> None:
        if not self._home_localizer.fusion_initialized:
            return
        self.ingest_home_fusion_sample()
        latest = self._continuous_fusion_latest or {}
        recorded = _finite_float(latest.get("recorded_monotonic_s"))
        age_s = None if recorded is None else self._monotonic() - recorded
        if (
            latest.get("trusted") is not True
            or self._continuous_fusion_consecutive_trusted < 3
            or age_s is None
            or age_s > self._continuous_fusion_interval_s * 2.0
        ):
            raise HardwareUnavailable(
                f"continuous Home fusion is not ready to {operation}: "
                + str(
                    latest.get("unavailable_reason") or "three trusted samples required"
                )
            )

    async def close(self) -> list[str]:
        errors: list[str] = []
        task, self._fusion_ingestion_task = self._fusion_ingestion_task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._motion is not None:
            errors.extend(await self._motion.close())
        if self._pose is not None:
            self._pose.close()
        if self._metric_range_provider is not None:
            self._metric_range_provider.close()
        self._connected = False
        return errors

    def status(self) -> dict[str, object]:
        motion = None if self._motion is None else self._motion.status()
        pose_status = None if self._pose is None else self._pose.status()
        pose = None if pose_status is None else pose_status.to_dict()
        connected = bool(
            self._connected and pose_status is not None and pose_status.healthy
        )
        can_pulse = bool(
            self.config.enabled
            and self.config.lab_motion_enabled
            and self._connected
            and connected
            and self._fault is None
            and self._active_operation is None
            and motion is not None
            and motion.get("initialized")
            and not motion.get("armed")
        )
        metric_arrival = (
            {"configured": False, "ready": False}
            if self._metric_arrival_gate is None
            else self._metric_arrival_gate.describe()
        )
        if self._metric_arrival_gate is not None:
            range_provider = (
                None
                if self._metric_range_provider is None
                else self._metric_range_provider.status()
            )
            metric_arrival = {
                **metric_arrival,
                "configured": bool(
                    range_provider is None or range_provider.get("configured")
                ),
                "ready": bool(range_provider is None or range_provider.get("ready")),
                "range_provider": range_provider,
            }
        return {
            "configured": self.config.enabled,
            "autonomy_enabled": self.config.autonomy_enabled,
            "lab_motion_enabled": self.config.lab_motion_enabled,
            "clients_initialized": self._connected,
            "connected": connected,
            "fault": self._fault,
            "network_interface": self.config.network_interface,
            "active_operation": self._active_operation,
            "can_pulse_forward": can_pulse,
            "forward_pulse": {
                "mps": self.config.forward_pulse_mps,
                "duration_s": self.config.forward_pulse_duration_s,
                "confirmation": FORWARD_PULSE_CONFIRMATION,
            },
            "motion": motion,
            "pose": pose,
            "home_localization": self._home_localizer.describe(),
            "continuous_home_fusion": {
                "running": bool(
                    self._fusion_ingestion_task is not None
                    and not self._fusion_ingestion_task.done()
                ),
                "interval_s": self._continuous_fusion_interval_s,
                "consecutive_trusted_samples": (
                    self._continuous_fusion_consecutive_trusted
                ),
                "latest_age_s": (
                    None
                    if self._continuous_fusion_latest is None
                    else max(
                        0.0,
                        self._monotonic()
                        - float(
                            self._continuous_fusion_latest[
                                "recorded_monotonic_s"
                            ]
                        ),
                    )
                ),
                "latest": self._continuous_fusion_latest,
            },
            "metric_arrival_required": self.config.metric_arrival_required,
            "metric_arrival": metric_arrival,
            "last_pulse": self._last_pulse,
        }

    def _trusted_home_estimate(
        self,
        home: Pose2D,
        sample: PoseStatus,
        *,
        operation: str,
    ) -> HomeEstimate:
        if not sample.healthy or sample.pose is None or sample.age_s is None:
            raise HardwareUnavailable(
                sample.error or f"fresh Go2 pose is required to {operation}"
            )
        estimate = self._home_localizer.estimate(
            home,
            Pose2D(sample.pose.x_m, sample.pose.y_m, sample.pose.yaw_rad),
            odometry_age_s=sample.age_s,
            captured_monotonic_s=sample.pose.captured_monotonic_s,
            motion=sample.motion,
        )
        if not estimate.trusted:
            raise HardwareUnavailable(
                f"Home localization unavailable during {operation}: "
                f"{estimate.unavailable_reason}"
            )
        return estimate

    def _unavailable_home_estimate(self, reason: str) -> dict[str, object]:
        return {
            "state": "unavailable",
            "trusted": False,
            "source": None,
            "unavailable_reason": reason,
            "home_distance_m": None,
            "heading_error_rad": None,
            "pose_from_home": None,
            "evidence": {
                "capabilities": self._home_localizer.describe(),
            },
        }

    def start_motion_trace(self, phase: str) -> None:
        self._motion_trace_phase = str(phase)
        self._motion_trace = []

    def set_motion_authority(self, run_id: str, epoch: str, phase: str) -> None:
        if self._active_operation is not None:
            raise HardwareUnavailable(
                "motion authority cannot change while hardware operation is active"
            )
        if not run_id.strip() or not epoch.strip() or not phase.strip():
            raise ValueError("motion authority identifiers must be non-empty")
        self._motion_run_id = run_id
        self._motion_run_epoch = epoch
        self._motion_authority_phase = phase
        self._record_flight(
            "motion_authority_selected",
            {"epoch": epoch, "phase": phase},
        )

    def set_flight_recorder(self, recorder: FlightRecorder) -> None:
        if self._active_operation is not None:
            raise HardwareUnavailable(
                "flight recorder cannot change while hardware operation is active"
            )
        self._flight_recorder = recorder

    def motion_trace(self) -> list[dict[str, object]]:
        return [dict(command) for command in self._motion_trace]

    async def _send_motion_command(
        self,
        lease: str,
        command: VelocityCommand,
    ) -> VelocityCommand:
        if self._motion is None:
            raise HardwareUnavailable("Go2 motion adapter is not connected")
        if 0.0 < command.forward_mps < self.config.minimum_forward_mps:
            raise HardwareUnavailable(
                f"forward command {command.forward_mps:.2f} m/s is below minimum "
                f"{self.config.minimum_forward_mps:.2f} m/s"
            )
        if (
            command.forward_mps > 0.0
            and self._active_operation not in FORWARD_CAPABLE_OPERATIONS
        ):
            raise HardwareUnavailable(
                "forward command blocked before approach or return motion"
            )
        if (
            self._home_localizer.fusion_initialized
            and self._captured_home_pose is not None
            and self._pose is not None
        ):
            sample = self._pose.status()
            if (
                not sample.healthy
                or sample.pose is None
                or sample.age_s is None
                or sample.motion is None
            ):
                raise HardwareUnavailable(
                    sample.error
                    or "fresh fused pose evidence is required before motion"
                )
            tracked = self._home_localizer.track_kinematics(
                self._captured_home_pose,
                Pose2D(sample.pose.x_m, sample.pose.y_m, sample.pose.yaw_rad),
                odometry_age_s=sample.age_s,
                captured_monotonic_s=sample.pose.captured_monotonic_s,
                motion=sample.motion,
            )
            if tracked.get("trusted") is not True:
                raise HardwareUnavailable(
                    "pose fusion unavailable before motion: "
                    + str(tracked.get("unavailable_reason") or "unknown")
                )
        if command.forward_mps > 0.0 and self._active_operation != "return_home":
            self._record_breadcrumb()
        sent = await self._motion.command(lease, command)
        self._motion_trace.append(
            {
                "sequence": len(self._motion_trace) + 1,
                "phase": self._motion_trace_phase,
                "recorded_monotonic_s": time.monotonic(),
                **sent.to_dict(),
            }
        )
        if self._flight_recorder is not None:
            pose = None
            if self._pose is not None:
                pose = self._pose.status().to_dict()
            self._record_flight(
                "motion_command",
                {
                    "epoch": self._motion_run_epoch,
                    "phase": self._motion_trace_phase,
                    "operation": self._active_operation,
                    "command": sent.to_dict(),
                    "pose": pose,
                },
            )
        return sent

    def _record_breadcrumb(self) -> None:
        """Capture sparse outbound poses without making motion wait on vision."""
        if self._captured_home_pose is None or self._pose is None:
            return
        sample = self._pose.status()
        if not sample.healthy or sample.pose is None:
            return
        pose = Pose2D(sample.pose.x_m, sample.pose.y_m, sample.pose.yaw_rad)
        if self._breadcrumbs:
            previous = self._breadcrumbs[-1]
            if math.hypot(pose.x_m - previous.x_m, pose.y_m - previous.y_m) < (
                self.config.breadcrumb_spacing_m
            ):
                return
        self._breadcrumbs.append(pose)
        if len(self._breadcrumbs) > self.config.maximum_breadcrumbs:
            # Preserve Home and the most recent path samples.
            recent_count = self.config.maximum_breadcrumbs - 1
            self._breadcrumbs = [
                self._breadcrumbs[0],
                *self._breadcrumbs[-recent_count:],
            ]

    def _record_perception_sample(
        self,
        status: dict[str, object],
        target_fruit: str,
    ) -> None:
        detection = status.get("detection")
        self._record_flight(
            "perception_sample",
            {
                "epoch": self._motion_run_epoch,
                "phase": self._motion_authority_phase,
                "target_fruit": target_fruit,
                "camera_healthy": status.get("camera_healthy"),
                "target_ready": status.get("target_ready"),
                "generation": status.get("generation"),
                "detection": detection if isinstance(detection, dict) else None,
            },
        )

    def _record_flight(self, kind: str, payload: dict[str, object]) -> None:
        if self._flight_recorder is not None:
            self._flight_recorder.record(
                kind,
                payload,
                run_id=self._motion_run_id,
            )

    def _authority_for_active_operation(self) -> MotionAuthority:
        operation = self._active_operation
        if operation is None:
            raise HardwareUnavailable("motion authority requires an active operation")
        return MotionAuthority(
            run_id=self._motion_run_id,
            epoch=self._motion_run_epoch,
            operation=f"{self._motion_authority_phase}:{operation}",
            allow_forward=operation in FORWARD_CAPABLE_OPERATIONS,
            maximum_forward_mps=self.config.maximum_forward_mps,
            maximum_yaw_rps=self.config.maximum_yaw_rps,
            ttl_s=self.config.motion_authority_ttl_s,
        )

    async def _best_effort_stop(self) -> None:
        if self._motion is None:
            return
        try:
            await self._motion.emergency_stop()
        except (MotionError, MotionNotReady):
            pass

    async def _run_posture_action(
        self,
        *,
        operation: str,
        evidence_posture: str,
        settle_s: float = 0.0,
    ) -> dict[str, object]:
        self._require_autonomy_ready()
        async with self._operation_lock:
            if self._active_operation is not None:
                raise HardwareUnavailable(
                    f"hardware operation already active: {self._active_operation}"
                )
            assert self._motion is not None
            if self._motion.armed:
                raise HardwareUnavailable(
                    "motion must be disarmed before posture change"
                )
            self._active_operation = operation
            try:
                if operation == "stand_down":
                    await self._motion.stand_down()
                else:
                    await self._motion.stand_up(settle_s=settle_s)
            except Exception as exc:
                raise HardwareUnavailable(f"{operation} failed: {exc}") from exc
            finally:
                self._active_operation = None
            evidence: dict[str, object] = {
                "posture": evidence_posture,
                "motion_commands_sent": True,
            }
            if operation == "stand_up":
                evidence["settle_s"] = settle_s
            return evidence

    def _require_autonomy_ready(self) -> None:
        if not self.config.enabled:
            raise HardwareUnavailable("Go2 hardware is disabled")
        if not self.config.autonomy_enabled:
            raise HardwareUnavailable("autonomous demo motion is disabled")
        if not self._connected or self._motion is None or self._pose is None:
            raise HardwareUnavailable(self._fault or "Go2 hardware is not connected")
        pose = self._pose.status()
        if not pose.healthy:
            raise HardwareUnavailable(
                pose.error or "fresh Go2 pose is required before motion"
            )
