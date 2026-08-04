"""Production stage adapter for the qualified pear Demo Run."""

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
        try:
            return await self._execute(phase, context)
        except CameraFailure as exc:
            raise StageFailure("CAMERA_FAILURE", str(exc)) from exc
        except TargetLost as exc:
            raise StageFailure(DEFAULT_STAGE_FAILURE_REASONS[phase], str(exc)) from exc
        except HardwareUnavailable as exc:
            raise StageFailure(DEFAULT_STAGE_FAILURE_REASONS[phase], str(exc)) from exc
        except BarkFailure as exc:
            raise StageFailure("ACTION_FAILURE", str(exc)) from exc

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
                # physical step. Keep short travel bounded by camera arrival
                # and timeout instead of commanding at the unreliable edge.
                forward_mps=1.0,
                maximum_yaw_rps=0.30,
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

    async def stop(self) -> list[str]:
        return await self._hardware.emergency_stop()
