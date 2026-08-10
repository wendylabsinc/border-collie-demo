"""Production stage adapter for a qualified-fruit Demo Run."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from .hardware import CameraFailure, HardwareUnavailable, TargetLost
from .media import BarkFailure
from .models import MissionPhase
from .orchestrator import (
    DEFAULT_STAGE_FAILURE_REASONS,
    StageContext,
    StageFailure,
)


class BarkPort(Protocol):
    async def bark(self) -> dict[str, object]: ...


DOWN_HOLD_S = 5.0
ARRIVAL_STOP_SETTLE_S = 1.0
STAND_UP_SETTLE_S = 1.0


class ProductionStageExecutor:
    """Map the eight public Demo Run stages onto guarded production adapters."""

    def __init__(
        self,
        hardware: Any,
        perception_status: Callable[[], dict[str, object]],
        bark: BarkPort,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._hardware = hardware
        self._perception_status = perception_status
        self._bark = bark
        self._sleep = sleep

    async def execute(
        self,
        phase: MissionPhase,
        context: StageContext,
    ) -> dict[str, Any]:
        set_authority = getattr(self._hardware, "set_motion_authority", None)
        if callable(set_authority):
            set_authority(context.run_id, context.run_epoch, phase.value)
        start_trace = getattr(self._hardware, "start_motion_trace", None)
        if callable(start_trace):
            start_trace(phase.value)
        try:
            evidence = await self._execute(phase, context)
        except CameraFailure as exc:
            raise StageFailure(
                "CAMERA_FAILURE",
                str(exc),
                details=self._failure_details(),
            ) from exc
        except TargetLost as exc:
            search_phase = phase in (
                MissionPhase.TURN_TO_FRUIT,
                MissionPhase.FIND_FRUIT,
            )
            raise StageFailure(
                (
                    "TARGET_RECOGNITION_FAILURE"
                    if search_phase
                    else DEFAULT_STAGE_FAILURE_REASONS[phase]
                ),
                str(exc),
                details=self._failure_details(
                    {"recognition": exc.evidence} if exc.evidence else None
                ),
            ) from exc
        except HardwareUnavailable as exc:
            raise StageFailure(
                DEFAULT_STAGE_FAILURE_REASONS[phase],
                str(exc),
                details=self._failure_details(),
            ) from exc
        except BarkFailure as exc:
            raise StageFailure(
                "ACTION_FAILURE",
                str(exc),
                details=self._failure_details(),
            ) from exc
        trace = self._read_motion_trace()
        if trace is not None:
            evidence = {**evidence, "motion_commands": trace}
        return evidence

    def _read_motion_trace(self) -> list[dict[str, object]] | None:
        read_trace = getattr(self._hardware, "motion_trace", None)
        if not callable(read_trace):
            return None
        return read_trace()

    def _failure_details(
        self,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        combined = dict(details or {})
        trace = self._read_motion_trace()
        if trace is not None:
            combined["motion_commands"] = trace
        return combined or None

    async def _execute(
        self,
        phase: MissionPhase,
        context: StageContext,
    ) -> dict[str, Any]:
        if phase is MissionPhase.ORIENT_FOR_RUN:
            requested_degrees = float(context.orientation_degrees)
            requested_rad = math.radians(requested_degrees)
            if requested_degrees == 0.0:
                return {
                    "requested_angle_degrees": 0.0,
                    "requested_angle_rad": 0.0,
                    "measured_yaw_change_rad": 0.0,
                    "orientation_skipped": True,
                    "motion_commands_sent": False,
                }
            evidence = await self._hardware.turn_relative(
                requested_rad,
                yaw_rps=1.00,
                tolerance_rad=min(math.radians(3.0), requested_rad / 2.0),
                timeout_s=30.0,
            )
            return {
                **evidence,
                "requested_angle_degrees": requested_degrees,
            }
        if phase is MissionPhase.TURN_TO_FRUIT:
            if visible := self._visible_target_evidence(context.target_fruit):
                return visible
            return await self._hardware.find_target(
                self._perception_status,
                context.target_fruit,
                yaw_rps=1.00,
                sweep_rad=2.0 * math.pi,
                timeout_s=30.0,
            )
        if phase is MissionPhase.FIND_FRUIT:
            if visible := self._visible_target_evidence(context.target_fruit):
                return visible
            return await self._hardware.find_target(
                self._perception_status,
                context.target_fruit,
                yaw_rps=0.20,
                sweep_rad=math.radians(75.0),
                timeout_s=9.0,
            )
        if phase is MissionPhase.APPROACH_FRUIT:
            return await self._hardware.approach_target(
                self._perception_status,
                context.target_fruit,
                # Target acquisition must not switch into a lower motion
                # profile. Geometry qualification owns the Arrival stop.
                forward_mps=1.0,
                maximum_yaw_rps=1.0,
                near_bottom_ratio=0.86,
                near_center_ratio=0.72,
                near_confirmations=3,
                near_loss_grace_s=0.75,
                final_push_mps=1.0,
                final_push_duration_s=1.0,
                timeout_s=20.0,
            )
        if phase is MissionPhase.SIT_AND_BARK:
            stop_errors = await self._hardware.emergency_stop()
            if stop_errors:
                raise HardwareUnavailable(
                    "arrival stop failed: " + "; ".join(stop_errors)
                )
            await self._sleep(ARRIVAL_STOP_SETTLE_S)
            evidence = await self._hardware.stand_down()
            bark = await self._bark.bark()
            await self._sleep(DOWN_HOLD_S)
            return {
                **evidence,
                **bark,
                "arrival_stop_confirmed": True,
                "arrival_stop_settle_s": ARRIVAL_STOP_SETTLE_S,
                "down_hold_s": DOWN_HOLD_S,
            }
        if phase is MissionPhase.STAND:
            return await self._hardware.stand_up(settle_s=STAND_UP_SETTLE_S)
        if phase is MissionPhase.TURN_TOWARD_HOME:
            return await self._hardware.turn_toward_home(
                context.home,
                yaw_rps=0.50,
                tolerance_rad=math.radians(5.0),
                response_timeout_s=0.75,
                response_min_progress_rad=math.radians(2.0),
                recovery_settle_s=1.0,
                timeout_s=30.0,
            )
        if phase is MissionPhase.RETURN_HOME:
            return await self._hardware.return_home(
                context.home,
                forward_mps=1.0,
                forward_pulse_count=context.outbound_forward_pulses,
                arrival_tolerance_m=0.10,
                heading_gate_rad=math.radians(20.0),
                maximum_yaw_rps=0.30,
                minimum_progress_m=0.03,
                stall_timeout_s=2.0,
                timeout_s=30.0,
            )
        if phase is MissionPhase.RESTORE_HEADING:
            return await self._hardware.restore_home_heading(
                context.home,
                yaw_rps=0.30,
                heading_tolerance_rad=math.radians(5.0),
                position_tolerance_m=0.10,
                timeout_s=15.0,
            )
        raise StageFailure("INTERNAL_ERROR", f"production stage is not implemented: {phase.value}")

    def _visible_target_evidence(self, target_fruit: str) -> dict[str, Any] | None:
        """Skip broad search only for current, fully qualified target evidence."""
        status = self._perception_status()
        detection = status.get("detection")
        selected_target = str(status.get("target_fruit") or "").casefold()
        detection_label = (
            str(detection.get("label") or "").casefold()
            if isinstance(detection, dict)
            else ""
        )
        target = target_fruit.casefold()
        if not (
            status.get("camera_healthy") is True
            and status.get("target_ready") is True
            and selected_target == target
            and detection_label == target
            and isinstance(detection, dict)
        ):
            return None
        return {
            "label": target_fruit,
            "confidence": detection.get("confidence"),
            "stable_detections": detection.get("consecutive_detections"),
            "detection_age_s": detection.get("age_s"),
            "center_x_ratio": detection.get("center_x_ratio"),
            "center_y_ratio": detection.get("center_y_ratio"),
            "bottom_ratio": detection.get("bottom_ratio"),
            "search_progress_rad": 0.0,
            "search_skipped": True,
            "skip_reason": "target_already_visible",
            "motion_commands_sent": False,
        }

    async def stop(self) -> list[str]:
        return await self._hardware.emergency_stop()
