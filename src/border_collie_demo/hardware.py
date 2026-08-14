from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable
from typing import Protocol

from .black_box import RunBlackBox
from .config import HardwareConfig
from .fruit_bearing_map import FruitBearingMap
from .fruits import fruit_policy
from .go2_motion import (
    MotionConfig,
    MotionError,
    MotionNotReady,
    create_go2_motion,
    initialize_dds,
)
from .go2_pose import Go2PoseProvider, PoseStatus
from .guidance import FruitGuidance, GuidanceAction, GuidanceDecision, GuidancePhase
from .models import VelocityCommand
from .return_home import (
    Pose2D,
    ReturnMode,
    ReturnPlannerConfig,
    normalize_angle,
    plan_position_return_step,
    plan_return_step,
)
from .search_experiment import confidence_summary

FORWARD_PULSE_CONFIRMATION = "PATH CLEAR - MOVE WOOF FORWARD"
INITIAL_CENTER_TOLERANCE_RATIO = 0.08
INITIAL_CENTER_CONFIRMATIONS = 3
INITIAL_CENTER_YAW_RPS = 0.50
SEARCH_CROP_CANDIDATE_CONFIDENCE = 0.50
APPROACH_CENTER_TOLERANCE_RATIO = 0.08
CLOSE_RANGE_MINIMUM_BOTTOM_RATIO = 0.70
CLOSE_RANGE_MAXIMUM_CENTER_DELTA_RATIO = 0.20
CLOSE_RANGE_MAXIMUM_VERTICAL_RETREAT_RATIO = 0.08
FORWARD_CAPABLE_OPERATIONS = frozenset(
    {"forward_pulse", "approach_target", "guide_target", "return_home"}
)
APPROACH_TRACE_LIMIT = 256


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


class HardwareUnavailable(RuntimeError):
    pass


class CameraFailure(HardwareUnavailable):
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


class TurnNoResponse(HardwareUnavailable):
    """The SDK accepted yaw commands but odometry measured no physical turn."""


class MotionAdapterProtocol(Protocol):
    @property
    def armed(self) -> bool: ...

    def status(self) -> dict[str, object]: ...

    async def initialize(self) -> None: ...

    async def arm(self) -> str: ...

    async def arm_sport_yaw(self) -> str: ...

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


MotionFactory = Callable[[MotionConfig], MotionAdapterProtocol]
PoseFactory = Callable[[float], PoseProviderProtocol]
DdsInitializer = Callable[[str | None], None]


class _ApproachRecorder:
    """Bounded, image-free record of every approach guidance decision."""

    def __init__(
        self,
        guidance: FruitGuidance,
        *,
        started_s: float,
        event_sink: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self._guidance = guidance
        self._started_s = started_s
        self._trace: deque[dict[str, object]] = deque(maxlen=APPROACH_TRACE_LIMIT)
        self._action_counts: Counter[str] = Counter()
        self._reason_counts: Counter[str] = Counter()
        self._confidence_count = 0
        self._confidence_minimum: float | None = None
        self._confidence_maximum: float | None = None
        self._confidence_total = 0.0
        self._forward_decisions = 0
        self._stop_decisions = 0
        self._forward_commands_sent = 0
        self._stop_commands_sent = 0
        self._samples = 0
        self._event_sink = event_sink
        self._latest_inference_summary: dict[str, object] | None = None

    def record(
        self,
        status: dict[str, object],
        decision: GuidanceDecision,
        *,
        now_s: float,
    ) -> None:
        self._samples += 1
        source = status.get("source")
        detection = status.get("detection")
        source_record = source if isinstance(source, dict) else {}
        detection_record = detection if isinstance(detection, dict) else {}
        inference = status.get("inference")
        inference_record = inference if isinstance(inference, dict) else {}
        inference_latest = inference_record.get("latest")
        inference_latest_record = (
            inference_latest if isinstance(inference_latest, dict) else {}
        )
        inference_summary = inference_record.get("summary")
        if isinstance(inference_summary, dict):
            self._latest_inference_summary = dict(inference_summary)
        confidence = _finite_float(detection_record.get("confidence"))
        if confidence is not None:
            self._confidence_count += 1
            self._confidence_total += confidence
            self._confidence_minimum = (
                confidence
                if self._confidence_minimum is None
                else min(self._confidence_minimum, confidence)
            )
            self._confidence_maximum = (
                confidence
                if self._confidence_maximum is None
                else max(self._confidence_maximum, confidence)
            )
        action = decision.action.value
        reason = decision.reason
        self._action_counts[action] += 1
        self._reason_counts[reason] += 1
        if decision.command.forward_mps > 0.0:
            self._forward_decisions += 1
        if decision.action is GuidanceAction.STOP:
            self._stop_decisions += 1
        sample = {
            "sample": self._samples,
            "recorded_monotonic_s": now_s,
            "elapsed_s": round(now_s - self._started_s, 4),
            "target_fruit": self._guidance.target_fruit,
            "camera_healthy": status.get("camera_healthy") is True,
            "generation": status.get("generation"),
            "source_pts": source_record.get("pts"),
            "source_time_base": source_record.get("time_base"),
            "source_age_s": _finite_float(source_record.get("age_s")),
            "detection_age_s": _finite_float(detection_record.get("age_s")),
            "detection_pts": inference_latest_record.get("detection_pts"),
            "model_route": inference_latest_record.get("model_route"),
            "inference_start_monotonic_s": _finite_float(
                inference_latest_record.get("inference_start_monotonic_s")
            ),
            "inference_end_monotonic_s": _finite_float(
                inference_latest_record.get("inference_end_monotonic_s")
            ),
            "inference_duration_s": _finite_float(
                inference_latest_record.get("inference_duration_s")
            ),
            "inference_total_ms": _finite_float(
                inference_latest_record.get("inference_total_ms")
            ),
            "inference_overrun": inference_latest_record.get("inference_overrun"),
            "inference_error": inference_latest_record.get("error"),
            "raw_label": detection_record.get("label"),
            "confidence": confidence,
            "focus_confidence": self._guidance.policy.focus_confidence,
            "acquisition_confidence": (self._guidance.policy.acquisition_confidence),
            "tracking_confidence": (
                self._guidance.policy.close_range_tracking_confidence
            ),
            "center_x_ratio": _finite_float(detection_record.get("center_x_ratio")),
            "center_y_ratio": _finite_float(detection_record.get("center_y_ratio")),
            "bottom_ratio": _finite_float(detection_record.get("bottom_ratio")),
            "guidance_phase": decision.phase.value,
            "guidance_action": action,
            "guidance_reason": reason,
            "terminal": decision.terminal,
            "arrival_confirmed": decision.arrival_confirmed,
            "arrival_eligible": decision.arrival_eligible,
            "centered_fresh_samples": decision.centered_fresh_samples,
            "near_fresh_samples": decision.near_fresh_samples,
            "near_loss_samples": decision.near_loss_samples,
            "frame_advanced": decision.frame_advanced,
            "focus_active": decision.focus_active,
            "focus_direction": decision.focus_direction,
            "focus_grace_remaining_s": decision.focus_grace_remaining_s,
            "resulting_command": decision.command.to_dict(),
            "command_sent": False,
        }
        self._trace.append(sample)
        if self._event_sink is not None:
            self._event_sink(dict(sample))

    def mark_command_sent(self) -> None:
        if not self._trace:
            return
        self._trace[-1]["command_sent"] = True
        command = self._trace[-1]["resulting_command"]
        if not isinstance(command, dict):
            return
        forward_mps = _finite_float(command.get("forward_mps"))
        if forward_mps is not None and forward_mps > 0.0:
            self._forward_commands_sent += 1
        if self._trace[-1]["guidance_action"] == GuidanceAction.STOP.value:
            self._stop_commands_sent += 1

    def evidence(self) -> dict[str, object]:
        recorded = len(self._trace)
        dropped = self._samples - recorded
        confidence_average = (
            self._confidence_total / self._confidence_count
            if self._confidence_count
            else None
        )
        return {
            "approach_trace": [dict(sample) for sample in self._trace],
            "approach_trace_limit": APPROACH_TRACE_LIMIT,
            "approach_trace_dropped": dropped,
            "approach_summary": {
                "samples": self._samples,
                "recorded_samples": recorded,
                "dropped_samples": dropped,
                "action_counts": dict(sorted(self._action_counts.items())),
                "reason_counts": dict(sorted(self._reason_counts.items())),
                "confidence": {
                    "detected_frames": self._confidence_count,
                    "minimum": self._confidence_minimum,
                    "maximum": self._confidence_maximum,
                    "average": confidence_average,
                    "lock_confidence": None,
                },
                "forward_decisions": self._forward_decisions,
                "stop_decisions": self._stop_decisions,
                "forward_commands_sent": self._forward_commands_sent,
                "stop_commands_sent": self._stop_commands_sent,
                "final_guidance_reason": (
                    self._trace[-1]["guidance_reason"] if self._trace else None
                ),
                **(
                    {"inference": dict(self._latest_inference_summary)}
                    if self._latest_inference_summary is not None
                    else {}
                ),
            },
        }


class HardwareManager:
    def __init__(
        self,
        config: HardwareConfig | None = None,
        *,
        dds_initializer: DdsInitializer = initialize_dds,
        motion_factory: MotionFactory | None = None,
        pose_factory: PoseFactory | None = None,
        black_box: RunBlackBox | None = None,
    ) -> None:
        self.config = config or HardwareConfig()
        self._dds_initializer = dds_initializer
        self._motion_factory = motion_factory or create_go2_motion
        self._pose_factory = pose_factory or (
            lambda maximum_age_s: Go2PoseProvider(maximum_age_s=maximum_age_s)
        )
        self._motion: MotionAdapterProtocol | None = None
        self._pose: PoseProviderProtocol | None = None
        self._connected = False
        self._fault: str | None = None
        self._operation_lock = asyncio.Lock()
        self._active_operation: str | None = None
        self._last_pulse: dict[str, object] | None = None
        self._motion_trace_phase: str | None = None
        self._motion_trace: list[dict[str, object]] = []
        self._posture = "unknown"
        self._black_box = black_box
        self._motion_trace_run_id: str | None = None
        # Process-local epoch: app restart or provider recreation must invalidate
        # advisory fruit bearings rather than reusing a shifted odometry origin.
        self._odometry_epoch = uuid.uuid4().hex

    async def start(self) -> None:
        if not self.config.enabled or self._connected:
            return
        try:
            self._dds_initializer(self.config.network_interface)
            self._pose = self._pose_factory(self.config.pose_maximum_age_s)
            self._pose.start()
            self._motion = self._motion_factory(
                MotionConfig(
                    maximum_forward_mps=self.config.maximum_forward_mps,
                    maximum_yaw_rps=self.config.maximum_yaw_rps,
                    command_watchdog_s=self.config.command_watchdog_s,
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
                lease = await self._motion.arm()
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
        motion_path: str = "factory_avoidance",
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
        if motion_path not in {"factory_avoidance", "sport_yaw"}:
            raise ValueError("turn motion_path must be factory_avoidance or sport_yaw")
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
                lease = (
                    await self._motion.arm_sport_yaw()
                    if motion_path == "sport_yaw"
                    else await self._motion.arm()
                )
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
                "motion_path": motion_path,
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
            started = time.monotonic()
            evidence: dict[str, object] | None = None
            crop_slow_turn_next = False
            recognition: dict[str, object] = {
                "samples": 0,
                "pear_candidate_samples": 0,
                "maximum_confidence": None,
                "maximum_consecutive_detections": 0,
                "maximum_bbox_area_ratio": None,
                "closest_detection": None,
            }
            try:
                assert self._pose is not None and self._motion is not None
                initial = self._pose.status()
                if not initial.healthy or initial.pose is None:
                    raise HardwareUnavailable(
                        initial.error or "fresh Go2 pose is required before search"
                    )
                previous_yaw = initial.pose.yaw_rad
                lease = await self._motion.arm()
                deadline = started + timeout
                while time.monotonic() < deadline:
                    status = status_reader()
                    slow_for_crop_confirmation = False
                    recognition["samples"] = int(recognition["samples"]) + 1
                    if not status.get("camera_healthy"):
                        raise CameraFailure(
                            str(
                                status.get("detail")
                                or "camera evidence became unhealthy"
                            )
                        )
                    detection = status.get("detection")
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
                            slow_for_crop_confirmation = bool(
                                label == target_fruit.casefold()
                                and crop_confirmation.get("attempted") is True
                                and full_frame_confidence is not None
                                and full_frame_confidence
                                >= SEARCH_CROP_CANDIDATE_CONFIDENCE
                            )
                            if slow_for_crop_confirmation:
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
                    if status.get("target_ready") and isinstance(detection, dict):
                        label = str(detection.get("label") or "").casefold()
                        if label == target_fruit.casefold():
                            evidence = {
                                "label": target_fruit,
                                "confidence": detection.get("confidence"),
                                "stable_detections": detection.get(
                                    "consecutive_detections"
                                ),
                                "search_progress_rad": progress,
                                "recognition": {
                                    **recognition,
                                    "search_progress_rad": progress,
                                },
                                "motion_commands_sent": commands_sent,
                            }
                            break
                    sample = self._pose.status()
                    if not sample.healthy or sample.pose is None:
                        raise HardwareUnavailable(
                            sample.error or "Go2 pose became stale during search"
                        )
                    progress += max(
                        0.0,
                        math.atan2(
                            math.sin(sample.pose.yaw_rad - previous_yaw),
                            math.cos(sample.pose.yaw_rad - previous_yaw),
                        ),
                    )
                    previous_yaw = sample.pose.yaw_rad
                    if progress >= sweep:
                        raise TargetLost(
                            f"{target_fruit} was not found in the bounded search sweep",
                            evidence={
                                **recognition,
                                "search_progress_rad": progress,
                            },
                        )
                    command_rate = rate
                    command_reason = "find_target"
                    if slow_for_crop_confirmation:
                        crop_slow_turn_next = not crop_slow_turn_next
                        if crop_slow_turn_next:
                            command_rate = 0.0
                            command_reason = "crop_confirm_hold"
                            recognition["crop_slowdown_hold_samples"] = (
                                int(
                                    recognition.get(
                                        "crop_slowdown_hold_samples",
                                        0,
                                    )
                                )
                                + 1
                            )
                        else:
                            command_reason = "crop_confirm_slow_turn"
                            recognition["crop_slowdown_turn_samples"] = (
                                int(
                                    recognition.get(
                                        "crop_slowdown_turn_samples",
                                        0,
                                    )
                                )
                                + 1
                            )
                    else:
                        crop_slow_turn_next = False
                    await self._send_motion_command(
                        lease,
                        VelocityCommand(0.0, command_rate, command_reason),
                    )
                    commands_sent = commands_sent or command_rate != 0.0
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
        final_push_mps: float,
        final_push_duration_s: float,
        timeout_s: float,
    ) -> dict[str, object]:
        """Approach a fresh Target Fruit track and confirm lower-edge Arrival."""
        numeric = (
            forward_mps,
            maximum_yaw_rps,
            near_bottom_ratio,
            near_center_ratio,
            near_loss_grace_s,
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
        if not 0.0 < maximum_yaw_rps <= self.config.maximum_yaw_rps:
            raise ValueError("approach yaw is outside the configured limit")
        if not 0.0 < near_bottom_ratio <= 1.0 or not 0.0 < near_center_ratio <= 1.0:
            raise ValueError("near-fruit geometry thresholds are invalid")
        if (
            near_confirmations < 1
            or min(near_loss_grace_s, timeout_s) <= 0.0
            or final_push_duration_s < 0.0
        ):
            raise ValueError("approach timing or confirmation count is invalid")
        self._require_autonomy_ready()

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
            confirmations = 0
            near_at: float | None = None
            commands_sent = False
            forward_pulse_count = 0
            initial_centered = False
            initial_center_confirmations = 0
            tracking_confirmations = 0
            minimum_observed_tracking_confidence: float | None = None
            close_range_continuation_samples = 0
            track_acquired = False
            last_track_geometry: tuple[float, float, float] | None = None
            policy = fruit_policy(target_fruit)
            started = time.monotonic()
            try:
                assert self._motion is not None
                lease = await self._motion.arm()
                deadline = started + timeout_s
                while time.monotonic() < deadline:
                    now = time.monotonic()
                    status = status_reader()
                    if not status.get("camera_healthy"):
                        raise CameraFailure(
                            str(
                                status.get("detail")
                                or "camera evidence became unhealthy"
                            )
                        )
                    detection = status.get("detection")
                    target_ready = bool(status.get("target_ready"))
                    detection_label = (
                        str(detection.get("label") or "").casefold()
                        if isinstance(detection, dict)
                        else ""
                    )
                    detection_confidence = (
                        _finite_float(detection.get("confidence"))
                        if isinstance(detection, dict)
                        else None
                    )
                    center_x = (
                        _finite_float(detection.get("center_x_ratio"))
                        if isinstance(detection, dict)
                        else None
                    )
                    center_y = (
                        _finite_float(detection.get("center_y_ratio"))
                        if isinstance(detection, dict)
                        else None
                    )
                    bottom = (
                        _finite_float(detection.get("bottom_ratio"))
                        if isinstance(detection, dict)
                        else None
                    )
                    if (
                        target_ready
                        and isinstance(detection, dict)
                        and detection_label != target_fruit.casefold()
                    ):
                        raise TargetLost(
                            f"qualified {target_fruit} track changed identity"
                        )
                    tracking_candidate = (
                        isinstance(detection, dict)
                        and detection_label == target_fruit.casefold()
                        and detection_confidence is not None
                        and detection_confidence
                        >= self.config.pear_tracking_minimum_confidence
                    )
                    close_range_continuation = (
                        track_acquired
                        and isinstance(detection, dict)
                        and detection_label == target_fruit.casefold()
                        and detection_confidence is not None
                        and detection_confidence
                        >= policy.close_range_tracking_confidence
                        and center_x is not None
                        and center_y is not None
                        and bottom is not None
                        and last_track_geometry is not None
                        and max(bottom, last_track_geometry[2])
                        >= CLOSE_RANGE_MINIMUM_BOTTOM_RATIO
                        and abs(center_x - last_track_geometry[0])
                        <= CLOSE_RANGE_MAXIMUM_CENTER_DELTA_RATIO
                        and center_y
                        >= last_track_geometry[1]
                        - CLOSE_RANGE_MAXIMUM_VERTICAL_RETREAT_RATIO
                        and bottom
                        >= last_track_geometry[2]
                        - CLOSE_RANGE_MAXIMUM_VERTICAL_RETREAT_RATIO
                    )
                    requested_target_ready = (
                        target_ready
                        and isinstance(detection, dict)
                        and detection_label == target_fruit.casefold()
                    )
                    if requested_target_ready:
                        tracking_confirmations = self.config.pear_tracking_confirmations
                    elif tracking_candidate:
                        tracking_confirmations += 1
                    elif close_range_continuation:
                        tracking_confirmations = max(
                            tracking_confirmations,
                            self.config.pear_tracking_confirmations,
                        )
                    else:
                        tracking_confirmations = 0
                    track_ready = (
                        (requested_target_ready)
                        or (
                            tracking_candidate
                            and tracking_confirmations
                            >= self.config.pear_tracking_confirmations
                        )
                        or close_range_continuation
                    )
                    if not track_ready:
                        if not initial_centered:
                            initial_center_confirmations = 0
                        target_still_visible = (
                            isinstance(detection, dict)
                            and detection_label == target_fruit.casefold()
                        )
                        if (
                            target_still_visible
                            or near_at is None
                            or now - near_at > near_loss_grace_s
                        ):
                            await self._send_motion_command(
                                lease,
                                VelocityCommand(reason="target_not_visible"),
                            )
                            await asyncio.sleep(self.config.command_heartbeat_s)
                            continue
                        if final_push_duration_s == 0.0:
                            await self._send_motion_command(
                                lease,
                                VelocityCommand(reason="bounded_final_push_disabled"),
                            )
                        push_deadline = now + final_push_duration_s
                        while time.monotonic() < push_deadline:
                            push_status = status_reader()
                            if not push_status.get("camera_healthy"):
                                raise CameraFailure(
                                    str(
                                        push_status.get("detail")
                                        or "camera evidence failed during final push"
                                    )
                                )
                            await self._send_motion_command(
                                lease,
                                VelocityCommand(
                                    final_push_mps,
                                    0.0,
                                    "fruit_offscreen_final_push",
                                ),
                            )
                            commands_sent = True
                            forward_pulse_count += 1
                            await asyncio.sleep(
                                min(
                                    self.config.command_heartbeat_s,
                                    max(0.0, push_deadline - time.monotonic()),
                                )
                            )
                        evidence = {
                            "arrival_confirmed": True,
                            "near_confirmations": confirmations,
                            "final_push_mps": final_push_mps,
                            "final_push_duration_s": final_push_duration_s,
                            "final_push_count": (
                                0 if final_push_duration_s == 0.0 else 1
                            ),
                            "initial_center_confirmations": (
                                initial_center_confirmations
                            ),
                            "initial_center_tolerance_ratio": (
                                INITIAL_CENTER_TOLERANCE_RATIO
                            ),
                            "initial_center_yaw_rps": INITIAL_CENTER_YAW_RPS,
                            "forward_pulse_count": forward_pulse_count,
                            "forward_pulse_period_s": self.config.command_heartbeat_s,
                            "motion_commands_sent": commands_sent,
                            "tracking_minimum_confidence": (
                                self.config.pear_tracking_minimum_confidence
                            ),
                            "minimum_observed_tracking_confidence": (
                                minimum_observed_tracking_confidence
                            ),
                            "close_range_tracking_confidence": (
                                policy.close_range_tracking_confidence
                            ),
                            "close_range_continuation_samples": (
                                close_range_continuation_samples
                            ),
                        }
                        break

                    assert isinstance(detection, dict)
                    label = detection_label
                    if label != target_fruit.casefold():
                        raise TargetLost(
                            f"qualified {target_fruit} track changed identity"
                        )
                    if detection_confidence is not None:
                        minimum_observed_tracking_confidence = (
                            detection_confidence
                            if minimum_observed_tracking_confidence is None
                            else min(
                                minimum_observed_tracking_confidence,
                                detection_confidence,
                            )
                        )
                    if center_x is None or center_y is None or bottom is None:
                        raise CameraFailure(
                            f"{target_fruit} geometry is missing from fresh evidence"
                        )
                    track_acquired = True
                    if close_range_continuation and not (
                        requested_target_ready or tracking_candidate
                    ):
                        close_range_continuation_samples += 1
                    last_track_geometry = (center_x, center_y, bottom)
                    if not initial_centered:
                        centered_sample = (
                            abs(center_x - 0.5) <= INITIAL_CENTER_TOLERANCE_RATIO
                        )
                        if centered_sample:
                            initial_center_confirmations += 1
                        else:
                            initial_center_confirmations = 0
                        if initial_center_confirmations < INITIAL_CENTER_CONFIRMATIONS:
                            yaw = (
                                0.0
                                if centered_sample
                                else -math.copysign(
                                    INITIAL_CENTER_YAW_RPS,
                                    center_x - 0.5,
                                )
                            )
                            await self._send_motion_command(
                                lease,
                                VelocityCommand(
                                    0.0,
                                    yaw,
                                    "center_target_before_approach",
                                ),
                            )
                            commands_sent = commands_sent or yaw != 0.0
                            await asyncio.sleep(self.config.command_heartbeat_s)
                            continue
                        initial_centered = True
                    near = bottom >= near_bottom_ratio and center_y >= near_center_ratio
                    confirmations = confirmations + 1 if near else 0
                    near_confirmed = confirmations >= near_confirmations
                    if near_confirmed:
                        near_at = now
                    horizontal_error = center_x - 0.5
                    yaw = (
                        0.0
                        if abs(horizontal_error) <= APPROACH_CENTER_TOLERANCE_RATIO
                        else -math.copysign(
                            maximum_yaw_rps,
                            horizontal_error,
                        )
                    )
                    await self._send_motion_command(
                        lease,
                        VelocityCommand(
                            forward_mps,
                            yaw,
                            (
                                "approach_target_near_visible"
                                if near_confirmed
                                else "approach_target"
                            ),
                        ),
                    )
                    forward_pulse_count += 1
                    commands_sent = True
                    await asyncio.sleep(self.config.command_heartbeat_s)
                else:
                    raise TargetLost(f"qualified {target_fruit} Arrival timed out")
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

    async def guide_target(
        self,
        status_reader: Callable[[], dict[str, object]],
        guidance: FruitGuidance,
        *,
        allow_forward: bool,
        timeout_s: float,
        bearing_map: FruitBearingMap | None = None,
        home_pose: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Execute one mission-lifetime guidance module until lock or Arrival."""
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("guidance timeout must be finite and positive")
        self._require_autonomy_ready()

        async with self._operation_lock:
            if self._active_operation is not None:
                raise HardwareUnavailable(
                    f"hardware operation already active: {self._active_operation}"
                )
            self._active_operation = "guide_target"
            lease: str | None = None
            release_error: str | None = None
            operation_error: Exception | None = None
            evidence: dict[str, object] | None = None
            commands_sent = False
            forward_pulse_count = 0
            samples = 0
            search_progress_rad = 0.0
            previous_search_yaw: float | None = None
            search_trace: list[dict[str, object]] = []
            inference_evidence: dict[str, object] = {}
            started = time.monotonic()
            approach_recorder = (
                _ApproachRecorder(
                    guidance,
                    started_s=started,
                    event_sink=lambda event: self._record_black_box(
                        "guidance_decision", event
                    ),
                )
                if allow_forward
                else None
            )
            try:
                assert self._motion is not None and self._pose is not None
                if not allow_forward:
                    initial_pose = self._pose.status()
                    if not initial_pose.healthy or initial_pose.pose is None:
                        raise HardwareUnavailable(
                            initial_pose.error
                            or "fresh Go2 pose is required before Target Fruit search"
                        )
                    previous_search_yaw = initial_pose.pose.yaw_rad
                lease = await self._motion.arm()
                deadline = started + timeout_s
                while time.monotonic() < deadline:
                    now = time.monotonic()
                    measured_yaw_rad: float | None = None
                    if not allow_forward:
                        pose = self._pose.status()
                        if not pose.healthy or pose.pose is None:
                            raise HardwareUnavailable(
                                pose.error
                                or "Go2 pose became stale during Target Fruit search"
                            )
                        assert previous_search_yaw is not None
                        measured_yaw_rad = pose.pose.yaw_rad
                        yaw_delta = math.atan2(
                            math.sin(pose.pose.yaw_rad - previous_search_yaw),
                            math.cos(pose.pose.yaw_rad - previous_search_yaw),
                        )
                        search_progress_rad += max(0.0, yaw_delta)
                        previous_search_yaw = pose.pose.yaw_rad
                        if search_progress_rad >= guidance.config.search_sweep_rad:
                            raise TargetLost(
                                f"{guidance.target_fruit} was not found in the bounded search sweep",
                                evidence={
                                    "guidance_phase": guidance.phase.value,
                                    "search_progress_rad": search_progress_rad,
                                    "search_sweep_rad": guidance.config.search_sweep_rad,
                                    "samples": samples,
                                    "search_trace": search_trace,
                                    "confidence_summary": confidence_summary(
                                        search_trace
                                    ),
                                    **inference_evidence,
                                },
                            )
                    status = status_reader()
                    bearing_map_status: dict[str, object] | None = None
                    if (
                        not allow_forward
                        and bearing_map is not None
                        and home_pose is not None
                        and measured_yaw_rad is not None
                    ):
                        source = status.get("source")
                        observations = status.get("observations")
                        if isinstance(source, dict) and isinstance(observations, dict):
                            bearing_map_status = bearing_map.observe(
                                home_pose,
                                {
                                    "x_m": pose.pose.x_m,
                                    "y_m": pose.pose.y_m,
                                    "yaw_rad": pose.pose.yaw_rad,
                                    "age_s": pose.age_s,
                                },
                                {
                                    "generation": status.get("generation"),
                                    "source_pts": source.get("pts"),
                                    "source_time_base": source.get("time_base"),
                                    "odometry_epoch": self._odometry_epoch,
                                },
                                observations,
                            )
                            self._record_black_box(
                                "fruit_bearing_map", bearing_map_status
                            )
                    raw_inference = status.get("inference")
                    if isinstance(raw_inference, dict) and isinstance(
                        raw_inference.get("summary"), dict
                    ):
                        inference_evidence = {
                            "inference_summary": dict(raw_inference["summary"])
                        }
                    decision = guidance.observe(
                        status,
                        now_s=now,
                        allow_forward=allow_forward,
                    )
                    samples += 1
                    if approach_recorder is not None:
                        approach_recorder.record(status, decision, now_s=now)
                    if not allow_forward:
                        source = status.get("source")
                        detection = status.get("detection")
                        inference = status.get("inference")
                        inference_record = (
                            inference if isinstance(inference, dict) else {}
                        )
                        inference_latest = inference_record.get("latest")
                        inference_latest_record = (
                            inference_latest
                            if isinstance(inference_latest, dict)
                            else {}
                        )
                        search_event = {
                            "sample": samples,
                            "elapsed_s": round(now - started, 4),
                            "target_fruit": guidance.target_fruit,
                            "source_pts": (
                                source.get("pts") if isinstance(source, dict) else None
                            ),
                            "confidence": (
                                detection.get("confidence")
                                if isinstance(detection, dict)
                                else None
                            ),
                            "detection_age_s": (
                                _finite_float(detection.get("age_s"))
                                if isinstance(detection, dict)
                                else None
                            ),
                            "detection_pts": inference_latest_record.get(
                                "detection_pts"
                            ),
                            "model_route": inference_latest_record.get("model_route"),
                            "inference_start_monotonic_s": _finite_float(
                                inference_latest_record.get(
                                    "inference_start_monotonic_s"
                                )
                            ),
                            "inference_end_monotonic_s": _finite_float(
                                inference_latest_record.get(
                                    "inference_end_monotonic_s"
                                )
                            ),
                            "inference_duration_s": _finite_float(
                                inference_latest_record.get("inference_duration_s")
                            ),
                            "inference_total_ms": _finite_float(
                                inference_latest_record.get("inference_total_ms")
                            ),
                            "inference_overrun": inference_latest_record.get(
                                "inference_overrun"
                            ),
                            "inference_error": inference_latest_record.get("error"),
                            "center_x_ratio": (
                                detection.get("center_x_ratio")
                                if isinstance(detection, dict)
                                else None
                            ),
                            "measured_yaw_rad": measured_yaw_rad,
                            "search_progress_rad": search_progress_rad,
                            "commanded_yaw_rps": decision.command.yaw_rps,
                            "guidance_action": decision.action.value,
                            "guidance_reason": decision.reason,
                            "resulting_command": decision.command.to_dict(),
                            "frame_advanced": decision.frame_advanced,
                            "focus_active": decision.focus_active,
                            "focus_direction": decision.focus_direction,
                            "focus_grace_remaining_s": (
                                decision.focus_grace_remaining_s
                            ),
                            "locked": decision.phase is GuidancePhase.LOCKED,
                            "fruit_bearing_map": bearing_map_status,
                        }
                        search_trace.append(search_event)
                        self._record_black_box("guidance_decision", search_event)
                    if decision.action is GuidanceAction.STOP and decision.terminal:
                        diagnostics = (
                            approach_recorder.evidence()
                            if approach_recorder is not None
                            else {}
                        )
                        if decision.camera_failure:
                            raise CameraFailure(
                                f"camera guidance stopped: {decision.reason}",
                                evidence={
                                    "guidance_reason": decision.reason,
                                    "search_trace": search_trace,
                                    "confidence_summary": confidence_summary(
                                        search_trace
                                    ),
                                    **inference_evidence,
                                    **diagnostics,
                                },
                            )
                        raise TargetLost(
                            f"{guidance.target_fruit} guidance stopped: "
                            f"{decision.reason}",
                            evidence={
                                "guidance_reason": decision.reason,
                                "search_trace": search_trace,
                                "confidence_summary": confidence_summary(search_trace),
                                **inference_evidence,
                                **diagnostics,
                            },
                        )

                    await self._send_motion_command(lease, decision.command)
                    if approach_recorder is not None:
                        approach_recorder.mark_command_sent()
                    commands_sent = commands_sent or (
                        decision.command.forward_mps != 0.0
                        or decision.command.yaw_rps != 0.0
                    )
                    if decision.command.forward_mps > 0.0:
                        forward_pulse_count += 1

                    terminal_for_call = (
                        not allow_forward and decision.phase is GuidancePhase.LOCKED
                    ) or decision.arrival_confirmed
                    if terminal_for_call:
                        evidence = {
                            "label": guidance.target_fruit,
                            "guidance_phase": decision.phase.value,
                            "guidance_reason": decision.reason,
                            "acquisition_epoch": guidance.acquisition_epoch,
                            "centered_fresh_samples": (decision.centered_fresh_samples),
                            "near_confirmations": decision.near_fresh_samples,
                            "arrival_confirmed": decision.arrival_confirmed,
                            "final_push_mps": guidance.config.final_push_mps,
                            "final_push_duration_s": (
                                guidance.config.final_push_duration_s
                            ),
                            "final_push_count": guidance.final_push_count,
                            "forward_pulse_count": forward_pulse_count,
                            "forward_pulse_period_s": (self.config.command_heartbeat_s),
                            "samples": samples,
                            "search_progress_rad": search_progress_rad,
                            "search_trace": search_trace,
                            "confidence_summary": confidence_summary(search_trace),
                            "fruit_bearing_map": (
                                bearing_map.status()
                                if bearing_map is not None
                                else None
                            ),
                            **inference_evidence,
                            "motion_commands_sent": commands_sent,
                            **(
                                approach_recorder.evidence()
                                if approach_recorder is not None
                                else {}
                            ),
                        }
                        break
                    await asyncio.sleep(self.config.command_heartbeat_s)
                else:
                    raise TargetLost(
                        f"{guidance.target_fruit} camera guidance timed out",
                        evidence={
                            "guidance_phase": guidance.phase.value,
                            "samples": samples,
                            "search_trace": search_trace,
                            "confidence_summary": confidence_summary(search_trace),
                            **inference_evidence,
                            **(
                                approach_recorder.evidence()
                                if approach_recorder is not None
                                else {}
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
                raise HardwareUnavailable(
                    f"camera guidance stop failed: {release_error}"
                )
            assert evidence is not None
            return evidence

    async def return_home(
        self,
        home: dict[str, object],
        *,
        forward_mps: float,
        forward_pulse_count: int,
        arrival_tolerance_m: float,
        heading_gate_rad: float,
        maximum_yaw_rps: float,
        minimum_yaw_rps: float = 0.50,
        heading_tolerance_rad: float = math.radians(5.0),
        minimum_progress_m: float,
        stall_timeout_s: float,
        timeout_s: float,
    ) -> dict[str, object]:
        """Replay outbound forward pulses toward Home with fresh pose guards."""
        home_pose = _home_pose(home)
        config = ReturnPlannerConfig(
            arrival_tolerance_m=arrival_tolerance_m,
            heading_tolerance_rad=heading_tolerance_rad,
            heading_gate_rad=heading_gate_rad,
            forward_mps=forward_mps,
            maximum_yaw_rps=maximum_yaw_rps,
            minimum_yaw_rps=minimum_yaw_rps,
        )
        if forward_mps > self.config.maximum_forward_mps:
            raise ValueError("return speed is outside the configured limit")
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
            try:
                assert self._motion is not None and self._pose is not None
                lease = await self._motion.arm()
                deadline = started + timeout_s
                while time.monotonic() < deadline:
                    sample = self._pose.status()
                    if not sample.healthy or sample.pose is None:
                        raise HardwareUnavailable(
                            sample.error or "Go2 pose became stale during return Home"
                        )
                    current = Pose2D(
                        sample.pose.x_m,
                        sample.pose.y_m,
                        sample.pose.yaw_rad,
                    )
                    step = plan_return_step(home_pose, current, config)
                    samples += 1
                    if step.distance_m <= arrival_tolerance_m:
                        evidence = {
                            "home_distance_m": step.distance_m,
                            "arrival_tolerance_m": arrival_tolerance_m,
                            "requested_forward_pulses": forward_pulse_count,
                            "replayed_forward_pulses": replayed_forward_pulses,
                            "playback_stopped_at_home": (
                                replayed_forward_pulses < forward_pulse_count
                            ),
                            "pose_samples": samples,
                            "motion_path": "factory_avoidance",
                            "motion_commands_sent": commands_sent,
                        }
                        break
                    if replayed_forward_pulses >= forward_pulse_count:
                        raise HardwareUnavailable(
                            "return pulse playback completed "
                            f"{step.distance_m:.3f} m from Home"
                        )
                    now = time.monotonic()
                    if step.distance_m <= best_distance - minimum_progress_m:
                        best_distance = step.distance_m
                        progress_at = now
                    elif now - progress_at > stall_timeout_s:
                        raise HardwareUnavailable(
                            f"return Home stalled at {step.distance_m:.3f} m"
                        )
                    if step.mode is ReturnMode.TURN_TO_HOME:
                        raise HardwareUnavailable(
                            "return Home heading escaped the forward steering gate"
                        )
                    command = VelocityCommand(
                        step.forward_mps,
                        step.yaw_rps,
                        "return_home",
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

    def measure_home_position(
        self,
        home: dict[str, object],
    ) -> dict[str, object]:
        """Read one fresh authoritative Go2 pose relative to captured Home."""
        home_pose = _home_pose(home)
        if not self.config.enabled:
            raise HardwareUnavailable("Go2 hardware is disabled")
        if not self._connected or self._pose is None or self._motion is None:
            raise HardwareUnavailable(self._fault or "Go2 hardware is not connected")
        sample = self._pose.status()
        if not sample.healthy or sample.pose is None or sample.age_s is None:
            raise HardwareUnavailable(
                sample.error or "fresh Go2 pose is unavailable for Home measurement"
            )
        return {
            "home_distance_m": math.hypot(
                home_pose.x_m - sample.pose.x_m,
                home_pose.y_m - sample.pose.y_m,
            ),
            "pose_age_s": sample.age_s,
            "pose_captured_monotonic_s": sample.pose.captured_monotonic_s,
            "pose_source": "rt/sportmodestate",
        }

    async def return_home_position(
        self,
        home: dict[str, object],
        *,
        forward_mps: float,
        arrival_tolerance_m: float,
        heading_gate_rad: float,
        maximum_yaw_rps: float,
        minimum_yaw_rps: float = 0.50,
        heading_tolerance_rad: float = math.radians(5.0),
        minimum_progress_m: float,
        stall_timeout_s: float,
        timeout_s: float,
    ) -> dict[str, object]:
        """Return to measured Home position with no pulse or heading credit."""
        home_pose = _home_pose(home)
        config = ReturnPlannerConfig(
            arrival_tolerance_m=arrival_tolerance_m,
            heading_tolerance_rad=heading_tolerance_rad,
            heading_gate_rad=heading_gate_rad,
            forward_mps=forward_mps,
            maximum_yaw_rps=maximum_yaw_rps,
            minimum_yaw_rps=minimum_yaw_rps,
        )
        if forward_mps > self.config.maximum_forward_mps:
            raise ValueError("return speed is outside the configured limit")
        if maximum_yaw_rps > self.config.maximum_yaw_rps:
            raise ValueError("return yaw is outside the configured limit")
        if minimum_progress_m <= 0.0 or stall_timeout_s <= 0.0 or timeout_s <= 0.0:
            raise ValueError("return progress and timing values must be positive")
        self._require_autonomy_ready()

        initial = self.measure_home_position(home)
        initial_distance = float(initial["home_distance_m"])
        if initial_distance <= arrival_tolerance_m:
            return {
                **initial,
                "arrival_tolerance_m": arrival_tolerance_m,
                "pose_samples": 1,
                "motion_path": "factory_avoidance",
                "motion_commands_sent": False,
                "heading_restoration_skipped": True,
                "measured_after_disarm": True,
            }

        async with self._operation_lock:
            if self._active_operation is not None:
                raise HardwareUnavailable(
                    f"hardware operation already active: {self._active_operation}"
                )
            self._active_operation = "return_home"
            lease: str | None = None
            release_error: str | None = None
            operation_error: Exception | None = None
            started = time.monotonic()
            best_distance = initial_distance
            progress_at = started
            samples = 1
            commands_sent = False
            heading_gate_escape = False
            try:
                assert self._motion is not None and self._pose is not None
                lease = await self._motion.arm()
                deadline = started + timeout_s
                while time.monotonic() < deadline:
                    sample = self._pose.status()
                    if not sample.healthy or sample.pose is None:
                        raise HardwareUnavailable(
                            sample.error or "Go2 pose became stale during return Home"
                        )
                    current = Pose2D(
                        sample.pose.x_m,
                        sample.pose.y_m,
                        sample.pose.yaw_rad,
                    )
                    step = plan_position_return_step(home_pose, current, config)
                    samples += 1
                    if step.mode is ReturnMode.COMPLETE:
                        break
                    now = time.monotonic()
                    if step.distance_m <= best_distance - minimum_progress_m:
                        best_distance = step.distance_m
                        progress_at = now
                    elif now - progress_at > stall_timeout_s:
                        raise HardwareUnavailable(
                            f"return Home stalled at {step.distance_m:.3f} m"
                        )
                    if step.mode is ReturnMode.TURN_TO_HOME:
                        heading_gate_escape = True
                        break
                    command = VelocityCommand(
                        step.forward_mps,
                        step.yaw_rps,
                        "return_home_position",
                    )
                    await self._send_motion_command(lease, command)
                    commands_sent = True
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
                elif self._motion is not None:
                    stop_errors = await self._motion.emergency_stop()
                    if stop_errors:
                        release_error = "; ".join(stop_errors)
                self._active_operation = None

            if operation_error is not None:
                raise operation_error
            if release_error is not None:
                raise HardwareUnavailable(f"return Home stop failed: {release_error}")
            try:
                terminal = self.measure_home_position(home)
            except HardwareUnavailable as exc:
                if heading_gate_escape:
                    self._record_black_box(
                        "return_home_reconciliation",
                        {
                            "home_distance_m": None,
                            "arrival_tolerance_m": arrival_tolerance_m,
                            "heading_gate_escape_reconciled": False,
                            "measured_after_disarm": True,
                            "measurement_error": str(exc),
                        },
                    )
                    raise HardwareUnavailable(
                        "return Home heading escaped the forward steering gate; "
                        f"fresh post-disarm position was unavailable: {exc}"
                    ) from exc
                raise
            terminal_distance = float(terminal["home_distance_m"])
            if heading_gate_escape:
                self._record_black_box(
                    "return_home_reconciliation",
                    {
                        **terminal,
                        "arrival_tolerance_m": arrival_tolerance_m,
                        "heading_gate_escape_reconciled": (
                            terminal_distance <= arrival_tolerance_m
                        ),
                        "measured_after_disarm": True,
                    },
                )
            if terminal_distance > arrival_tolerance_m:
                if heading_gate_escape:
                    raise HardwareUnavailable(
                        "return Home heading escaped the forward steering gate; "
                        "fresh post-disarm position remained outside Home at "
                        f"{terminal_distance:.3f} m"
                    )
                raise HardwareUnavailable(
                    f"return ended outside Home after disarm: {terminal_distance:.3f} m"
                )
            return {
                **terminal,
                "arrival_tolerance_m": arrival_tolerance_m,
                "pose_samples": samples + 1,
                "motion_path": "factory_avoidance",
                "motion_commands_sent": commands_sent,
                "heading_restoration_skipped": True,
                "measured_after_disarm": True,
                "heading_gate_escape_reconciled": heading_gate_escape,
            }

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
        deadline = time.monotonic() + timeout_s
        recovery_count = 0
        measured_yaw_change = 0.0
        commands_sent = False
        while True:
            assert self._pose is not None
            sample = self._pose.status()
            if not sample.healthy or sample.pose is None:
                raise HardwareUnavailable(
                    sample.error or "fresh Go2 pose is required to turn toward Home"
                )
            dx = home_pose.x_m - sample.pose.x_m
            dy = home_pose.y_m - sample.pose.y_m
            distance = math.hypot(dx, dy)
            if distance <= 0.10:
                return {
                    "home_distance_m": distance,
                    "home_bearing_error_rad": 0.0,
                    "measured_yaw_change_rad": measured_yaw_change,
                    "turn_recovery_count": recovery_count,
                    "motion_path": "sport_yaw",
                    "pose_age_s": sample.age_s,
                    "bearing_tolerance_rad": tolerance_rad,
                    "motion_commands_sent": commands_sent,
                }
            bearing_error = normalize_angle(math.atan2(dy, dx) - sample.pose.yaw_rad)
            if abs(bearing_error) <= tolerance_rad:
                return {
                    "home_distance_m": distance,
                    "home_bearing_error_rad": bearing_error,
                    "measured_yaw_change_rad": measured_yaw_change,
                    "turn_recovery_count": recovery_count,
                    "motion_path": "sport_yaw",
                    "pose_age_s": sample.age_s,
                    "bearing_tolerance_rad": tolerance_rad,
                    "motion_commands_sent": commands_sent,
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
                    motion_path="sport_yaw",
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
            measured_yaw_change += float(evidence["measured_yaw_change_rad"])
            commands_sent = True

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
        if not before.healthy or before.pose is None:
            raise HardwareUnavailable(
                before.error or "fresh Go2 pose is required to restore Home heading"
            )
        initial_distance = math.hypot(
            home_pose.x_m - before.pose.x_m,
            home_pose.y_m - before.pose.y_m,
        )
        if initial_distance > position_tolerance_m:
            raise HardwareUnavailable(
                f"Home position was lost before heading restore: {initial_distance:.3f} m"
            )
        initial_error = normalize_angle(home_pose.yaw_rad - before.pose.yaw_rad)
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
        if not after.healthy or after.pose is None:
            raise HardwareUnavailable(
                after.error or "Go2 pose became stale after heading restore"
            )
        distance = math.hypot(
            home_pose.x_m - after.pose.x_m,
            home_pose.y_m - after.pose.y_m,
        )
        heading_error = normalize_angle(home_pose.yaw_rad - after.pose.yaw_rad)
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
        return {
            "x_m": status.pose.x_m,
            "y_m": status.pose.y_m,
            "yaw_rad": status.pose.yaw_rad,
            "captured_monotonic_s": status.pose.captured_monotonic_s,
            "age_s": status.age_s,
            "source": "rt/sportmodestate",
            "odometry_epoch": self._odometry_epoch,
        }

    def record_bearing_route(self, evidence: dict[str, object]) -> None:
        """Persist one non-image advisory routing decision in the run black box."""
        self._record_black_box("fruit_bearing_route", dict(evidence))

    async def close(self) -> list[str]:
        errors: list[str] = []
        if self._motion is not None:
            errors.extend(await self._motion.close())
        if self._pose is not None:
            self._pose.close()
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
        return {
            "configured": self.config.enabled,
            "autonomy_enabled": self.config.autonomy_enabled,
            "lab_motion_enabled": self.config.lab_motion_enabled,
            "clients_initialized": self._connected,
            "connected": connected,
            "fault": self._fault,
            "network_interface": self.config.network_interface,
            "active_operation": self._active_operation,
            "posture": self._posture,
            "can_pulse_forward": can_pulse,
            "forward_pulse": {
                "mps": self.config.forward_pulse_mps,
                "duration_s": self.config.forward_pulse_duration_s,
                "confirmation": FORWARD_PULSE_CONFIRMATION,
            },
            "motion": motion,
            "pose": pose,
            "last_pulse": self._last_pulse,
        }

    def start_motion_trace(self, phase: str, *, run_id: str | None = None) -> None:
        self._motion_trace_phase = str(phase)
        self._motion_trace_run_id = run_id
        self._motion_trace = []

    def motion_trace(self) -> list[dict[str, object]]:
        return [dict(command) for command in self._motion_trace]

    async def _send_motion_command(
        self,
        lease: str,
        command: VelocityCommand,
    ) -> VelocityCommand:
        if self._motion is None:
            raise HardwareUnavailable("Go2 motion adapter is not connected")
        if (
            command.forward_mps > 0.0
            and self._active_operation not in FORWARD_CAPABLE_OPERATIONS
        ):
            raise HardwareUnavailable(
                "forward command blocked before approach or return motion"
            )
        sent = await self._motion.command(lease, command)
        motion_status = self._motion.status()
        event = {
            "sequence": len(self._motion_trace) + 1,
            "phase": self._motion_trace_phase,
            "recorded_monotonic_s": time.monotonic(),
            "motion_path": motion_status.get("mode"),
            **sent.to_dict(),
        }
        self._motion_trace.append(event)
        self._record_black_box("motion_command", event)
        return sent

    def _record_black_box(self, kind: str, payload: dict[str, object]) -> None:
        if self._black_box is None or self._motion_trace_run_id is None:
            return
        self._black_box.record(
            self._motion_trace_run_id,
            kind,
            phase=self._motion_trace_phase,
            payload=payload,
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
                    self._posture = "down"
                else:
                    await self._motion.stand_up(settle_s=settle_s)
                    self._posture = "standing"
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
