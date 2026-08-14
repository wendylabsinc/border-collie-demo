"""Production stage adapter for a qualified-fruit Demo Run."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from .fruit_bearing_map import FruitBearingMap
from .fruits import fruit_policy
from .guidance import FruitGuidance, GuidanceConfig, GuidancePhase
from .hardware import CameraFailure, HardwareUnavailable, TargetLost
from .media import BarkFailure
from .models import MissionPhase
from .orchestrator import (
    DEFAULT_STAGE_FAILURE_REASONS,
    StageContext,
    StageFailure,
)
from .run_tuning import RunTuning


class BarkPort(Protocol):
    async def bark(self) -> dict[str, object]: ...


DOWN_HOLD_S = 5.0
ARRIVAL_STOP_SETTLE_S = 1.0
STAND_UP_SETTLE_S = 1.0
BEARING_ROUTE_YAW_RPS = 0.50


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
        self._guidance_run_id: str | None = None
        self._guidance: FruitGuidance | None = None
        self._run_tuning: RunTuning | None = None
        self._bearing_map = FruitBearingMap()
        self._last_bearing_route: dict[str, object] | None = None
        self._settled_home_run_id: str | None = None
        self._settled_home_evidence: dict[str, object] | None = None

    def bearing_map_status(self) -> dict[str, object]:
        """Expose the advisory process-local map without motion authority."""
        return {
            **self._bearing_map.status(),
            "routing": (
                None
                if self._last_bearing_route is None
                else dict(self._last_bearing_route)
            ),
        }

    async def execute(
        self,
        phase: MissionPhase,
        context: StageContext,
    ) -> dict[str, Any]:
        start_trace = getattr(self._hardware, "start_motion_trace", None)
        if callable(start_trace):
            start_trace(phase.value, run_id=context.run_id)
        try:
            evidence = await self._execute(phase, context)
        except CameraFailure as exc:
            raise StageFailure(
                "CAMERA_FAILURE",
                str(exc),
                details=self._failure_details(
                    {"recognition": exc.evidence} if exc.evidence else None
                ),
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
                details=self._failure_details({"safety_class": "motion"}),
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
        combined["fruit_bearing_map"] = self._bearing_map.status()
        trace = self._read_motion_trace()
        if trace is not None:
            combined["motion_commands"] = trace
        return combined or None

    async def _execute(
        self,
        phase: MissionPhase,
        context: StageContext,
    ) -> dict[str, Any]:
        guide_target = getattr(self._hardware, "guide_target", None)
        if callable(guide_target) and phase in (
            MissionPhase.TURN_TO_FRUIT,
            MissionPhase.FIND_FRUIT,
            MissionPhase.APPROACH_FRUIT,
        ):
            if self._guidance_run_id != context.run_id or self._guidance is None:
                payload = context.run_tuning
                if payload is None and context.search_experiment is not None:
                    legacy = context.search_experiment
                    payload = {
                        "search": {"yaw_rps": legacy.get("search_yaw_rps", 0.40)},
                        "recognition": {
                            "focus_confidence": legacy.get("focus_confidence"),
                            "lock_confidence": legacy.get("lock_confidence"),
                            "required_frames": legacy.get("center_confirmations", 3),
                        },
                        "centering": {
                            "lock_tolerance_ratio": legacy.get(
                                "center_tolerance_ratio", 0.08
                            )
                        },
                    }
                tuning = RunTuning.from_payload(context.target_fruit, payload)
                config = GuidanceConfig(
                    search_yaw_rps=tuning.search.yaw_rps,
                    search_sweep_rad=tuning.search.sweep_rad,
                    focus_yaw_rps=tuning.search.focus_yaw_rps,
                    focus_missing_grace_s=tuning.search.focus_missing_grace_s,
                    center_tolerance_ratio=tuning.centering.lock_tolerance_ratio,
                    center_confirmations=tuning.recognition.required_frames,
                    approach_forward_mps=tuning.approach.forward_mps,
                    approach_yaw_rps=tuning.centering.approach_yaw_rps,
                    outer_corridor_ratio=tuning.centering.outer_corridor_ratio,
                    recenter_yaw_rps=tuning.centering.recenter_yaw_rps,
                    duplicate_hold_s=tuning.approach.duplicate_hold_s,
                    source_maximum_age_s=tuning.approach.source_maximum_age_s,
                    detection_maximum_age_s=tuning.approach.detection_maximum_age_s,
                    slow_inference_grace_s=(
                        tuning.approach.slow_inference_grace_s
                    ),
                    near_bottom_ratio=tuning.arrival.near_bottom_ratio,
                    disappearance_bottom_ratio=(
                        tuning.arrival.disappearance_bottom_ratio
                    ),
                    near_center_ratio=tuning.arrival.near_center_ratio,
                    near_confirmations=tuning.arrival.near_confirmations,
                    near_loss_confirmations=tuning.arrival.loss_confirmations,
                    near_loss_grace_s=tuning.arrival.loss_grace_s,
                    final_push_mps=tuning.arrival.final_push_mps,
                    final_push_duration_s=tuning.arrival.final_push_duration_s,
                )
                policy = fruit_policy(context.target_fruit)
                policy = type(policy)(
                    acquisition_confidence=tuning.recognition.lock_confidence,
                    close_range_tracking_confidence=tuning.recognition.tracking_confidence,
                    motion_qualified=policy.motion_qualified,
                    focus_confidence=tuning.recognition.focus_confidence,
                )
                self._guidance_run_id = context.run_id
                self._run_tuning = tuning
                self._guidance = FruitGuidance(
                    context.target_fruit,
                    config=config,
                    policy=policy,
                )
            if phase is MissionPhase.FIND_FRUIT and (
                self._guidance.phase is GuidancePhase.LOCKED
            ):
                return {
                    "label": context.target_fruit,
                    "guidance_phase": self._guidance.phase.value,
                    "acquisition_epoch": self._guidance.acquisition_epoch,
                    "search_skipped": True,
                    "skip_reason": "mission_lifetime_identity_already_locked",
                    "motion_commands_sent": False,
                }
            map_enabled = bool(
                isinstance(context.home.get("odometry_epoch"), str)
                and isinstance(context.home.get("age_s"), (int, float))
            )
            map_home = context.home
            route_evidence: dict[str, object] | None = None
            if map_enabled and phase is MissionPhase.TURN_TO_FRUIT:
                route_started = time.monotonic()
                if self._run_tuning.search.bearing_routing_enabled:
                    map_home, route_evidence = await self._prepare_bearing_route(context)
                else:
                    route_evidence = {
                        "available": False,
                        "reason": "routing_disabled",
                        "authority": "yaw_route_only",
                    }
                route_used = "turn" in route_evidence
                fallback_reason = (
                    None
                    if route_used
                    else (
                        "already_aligned"
                        if route_evidence.get("available") is True
                        else route_evidence.get("reason")
                    )
                )
                self._last_bearing_route = {
                    "run_id": context.run_id,
                    "target_fruit": context.target_fruit,
                    "bearing_routing_enabled": (
                        self._run_tuning.search.bearing_routing_enabled
                    ),
                    "bearing_route_used": route_used,
                    "bearing_route_fallback_reason": fallback_reason,
                    "bearing_route_planning_ms": (
                        time.monotonic() - route_started
                    )
                    * 1000.0,
                    "route": dict(route_evidence),
                }
                record_route = getattr(self._hardware, "record_bearing_route", None)
                if callable(record_route):
                    record_route(self._last_bearing_route)
            guide_options: dict[str, object] = {}
            if map_enabled:
                guide_options = {
                    "bearing_map": self._bearing_map,
                    "home_pose": map_home,
                }
            result = await guide_target(
                self._perception_status,
                self._guidance,
                allow_forward=phase is MissionPhase.APPROACH_FRUIT,
                timeout_s=(
                    self._run_tuning.approach.timeout_s
                    if phase is MissionPhase.APPROACH_FRUIT
                    else self._run_tuning.search.timeout_s
                ),
                **guide_options,
            )
            if route_evidence is not None:
                assert self._last_bearing_route is not None
                result = {
                    **result,
                    "fruit_bearing_route": route_evidence,
                    "bearing_routing_enabled": self._last_bearing_route[
                        "bearing_routing_enabled"
                    ],
                    "bearing_route_used": self._last_bearing_route[
                        "bearing_route_used"
                    ],
                    "bearing_route_fallback_reason": self._last_bearing_route[
                        "bearing_route_fallback_reason"
                    ],
                    "bearing_route_planning_ms": self._last_bearing_route[
                        "bearing_route_planning_ms"
                    ],
                }
            return result
        if phase is MissionPhase.TURN_TO_FRUIT:
            if visible := self._visible_target_evidence(context.target_fruit):
                return visible
            return await self._hardware.find_target(
                self._perception_status,
                context.target_fruit,
                yaw_rps=0.50,
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
            tuning = RunTuning.from_payload(context.target_fruit, context.run_tuning)
            return await self._hardware.approach_target(
                self._perception_status,
                context.target_fruit,
                # The legacy factory-avoidance calibration found 0.50 m/s to
                # be the deadband edge and 1.0 m/s to produce a reliable
                # physical step during camera-guided approach. Once qualified
                # lower-edge disappearance proves arrival, soften the one
                # bounded final movement before the stop-and-lie-down stage.
                forward_mps=tuning.approach.forward_mps,
                maximum_yaw_rps=tuning.centering.approach_yaw_rps,
                near_bottom_ratio=tuning.arrival.near_bottom_ratio,
                near_center_ratio=tuning.arrival.near_center_ratio,
                near_confirmations=tuning.arrival.near_confirmations,
                near_loss_grace_s=tuning.arrival.loss_grace_s,
                final_push_mps=tuning.arrival.final_push_mps,
                final_push_duration_s=tuning.arrival.final_push_duration_s,
                timeout_s=tuning.approach.timeout_s,
            )
        if phase is MissionPhase.SIT_AND_BARK:
            stop_errors = await self._hardware.emergency_stop()
            if stop_errors:
                raise HardwareUnavailable(
                    "arrival stop failed: " + "; ".join(stop_errors)
                )
            await self._sleep(ARRIVAL_STOP_SETTLE_S)
            evidence = await self._hardware.stand_down()
            try:
                bark = await self._bark.bark()
            except Exception as exc:  # noqa: BLE001 - bark is audience-only
                bark = {
                    "bark_played": False,
                    "bark_error": str(exc)[:240] or type(exc).__name__,
                    "bark_error_type": type(exc).__name__,
                }
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
            tuning = RunTuning.from_payload(context.target_fruit, context.run_tuning)
            tolerance_rad = math.radians(tuning.home.align_tolerance_deg)
            evidence = await self._hardware.turn_toward_home(
                context.home,
                yaw_rps=tuning.home.align_yaw_rps,
                tolerance_rad=tolerance_rad,
                response_timeout_s=0.75,
                response_min_progress_rad=math.radians(2.0),
                recovery_settle_s=1.0,
                timeout_s=tuning.home.align_timeout_s,
            )
            error = evidence.get("home_bearing_error_rad")
            if (
                not isinstance(error, (int, float))
                or isinstance(error, bool)
                or not math.isfinite(float(error))
                or abs(float(error)) > tolerance_rad
            ):
                raise HardwareUnavailable(
                    "turn toward Home did not finish inside the fresh bearing gate"
                )
            if evidence.get("motion_path") != "sport_yaw":
                raise HardwareUnavailable(
                    "turn toward Home did not use regular SportClient yaw"
                )
            return evidence
        if phase is MissionPhase.RETURN_HOME:
            tuning = RunTuning.from_payload(context.target_fruit, context.run_tuning)
            evidence = await self._hardware.return_home_position(
                context.home,
                forward_mps=tuning.home.return_forward_mps,
                arrival_tolerance_m=tuning.home.arrival_tolerance_m,
                heading_gate_rad=math.radians(tuning.home.heading_gate_deg),
                heading_tolerance_rad=math.radians(tuning.home.return_yaw_deadband_deg),
                minimum_yaw_rps=tuning.home.return_minimum_yaw_rps,
                maximum_yaw_rps=tuning.home.return_yaw_rps,
                minimum_progress_m=tuning.home.minimum_progress_m,
                stall_timeout_s=tuning.home.stall_timeout_s,
                timeout_s=tuning.home.return_timeout_s,
                settle_interval_s=tuning.home.settle_interval_s,
                settled_sample_count=tuning.home.settled_sample_count,
                settled_maximum_spread_m=(
                    tuning.home.settled_maximum_spread_m
                ),
                settled_sample_timeout_s=tuning.home.settled_sample_timeout_s,
                settled_retry_count=tuning.home.settled_retry_count,
            )
            if evidence.get("settled_home_verified") is True:
                self._settled_home_run_id = context.run_id
                self._settled_home_evidence = dict(evidence)
            return evidence
        if phase is MissionPhase.RESTORE_HEADING:
            if (
                self._settled_home_run_id != context.run_id
                or self._settled_home_evidence is None
                or self._settled_home_evidence.get("settled_home_verified") is not True
            ):
                raise HardwareUnavailable(
                    "legacy restore_heading phase has no settled Home verification"
                )
            return {
                **self._settled_home_evidence,
                "heading_restoration_skipped": True,
                "legacy_restore_heading_semantics": (
                    "settled_position_verification_already_complete"
                ),
                "motion_commands_sent": False,
            }
        raise StageFailure(
            "INTERNAL_ERROR", f"production stage is not implemented: {phase.value}"
        )

    async def _prepare_bearing_route(
        self, context: StageContext
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Qualify canonical Home, then optionally perform one advisory map turn."""
        current_home = dict(context.home)
        map_status = self._bearing_map.status()
        if not map_status["valid"]:
            return current_home, {
                "available": False,
                "reason": "map_unanchored",
                "authority": "yaw_route_only",
            }
        camera_status = self._perception_status()
        generation = camera_status.get("generation")
        current_home["generation"] = generation
        route = self._bearing_map.route_to(context.target_fruit, current_home)
        route_evidence = {**route.to_dict(), "authority": "yaw_route_only"}
        if (
            route.available
            and route.angular_delta_rad is not None
            and abs(route.angular_delta_rad)
            > math.radians(self._run_tuning.home.align_tolerance_deg)
        ):
            route_evidence["turn"] = await self._hardware.turn_relative(
                route.angular_delta_rad,
                yaw_rps=BEARING_ROUTE_YAW_RPS,
                tolerance_rad=math.radians(
                    self._run_tuning.home.align_tolerance_deg
                ),
                timeout_s=self._run_tuning.home.align_timeout_s,
                motion_path="sport_yaw",
            )
        return current_home, route_evidence

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
