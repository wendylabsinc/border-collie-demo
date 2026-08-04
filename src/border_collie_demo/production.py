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
        if phase is MissionPhase.TURN_TO_FRUIT:
            return await self._hardware.find_target(
                self._perception_status,
                context.target_fruit,
                yaw_rps=0.50,
                sweep_rad=2.0 * math.pi,
                timeout_s=30.0,
            )
        if phase is MissionPhase.FIND_FRUIT:
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
                # The legacy factory-avoidance calibration found 0.50 m/s to
                # be the deadband edge and 1.0 m/s to produce a reliable
                # physical step during camera-guided approach. Once qualified
                # lower-edge disappearance proves arrival, soften the one
                # bounded final movement before the stop-and-lie-down stage.
                forward_mps=1.0,
                maximum_yaw_rps=0.30,
                near_bottom_ratio=0.86,
                near_center_ratio=0.72,
                near_confirmations=3,
                near_loss_grace_s=0.75,
                final_push_mps=0.3,
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

    async def stop(self) -> list[str]:
        return await self._hardware.emergency_stop()
