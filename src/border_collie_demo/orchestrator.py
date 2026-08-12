"""End-to-end Demo Run orchestration over one stage-executor interface."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Protocol

from .evidence import EvidenceArtifact
from .models import MissionPhase
from .qualified_tracking import SearchQualificationHandoff
from .run_coordinator import RunCoordinator
from .run_results import RunResultStore


@dataclass(frozen=True)
class StageContext:
    run_id: str
    target_fruit: str
    home: dict[str, Any]
    run_epoch: str = "simulation"
    orientation_degrees: float = 0.0
    search_policy: str = "slow-sweep"
    outbound_forward_pulses: int = 0
    search_qualification_handoff: SearchQualificationHandoff | None = None


class StageExecutor(Protocol):
    async def execute(
        self,
        phase: MissionPhase,
        context: StageContext,
    ) -> dict[str, Any]: ...

    async def stop(self) -> list[str]: ...


class AutomaticFailureRecovery(Protocol):
    async def recover_automatically(self, run_id: str) -> dict[str, Any] | None: ...


class StageFailure(RuntimeError):
    def __init__(
        self,
        reason: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.details = dict(details or {})


EXECUTED_STAGES = (
    MissionPhase.ORIENT_FOR_RUN,
    MissionPhase.TURN_TO_FRUIT,
    MissionPhase.FIND_FRUIT,
    MissionPhase.APPROACH_FRUIT,
    MissionPhase.SIT_AND_BARK,
    MissionPhase.STAND,
    MissionPhase.TURN_TOWARD_HOME,
    MissionPhase.RETURN_HOME,
    MissionPhase.RESTORE_HEADING,
)

DEFAULT_STAGE_FAILURE_REASONS = {
    MissionPhase.ORIENT_FOR_RUN: "ORIENTATION_FAILURE",
    MissionPhase.TURN_TO_FRUIT: "TARGET_RECOGNITION_FAILURE",
    MissionPhase.FIND_FRUIT: "TARGET_RECOGNITION_FAILURE",
    MissionPhase.APPROACH_FRUIT: "ARRIVAL_FAILURE",
    MissionPhase.SIT_AND_BARK: "ACTION_FAILURE",
    MissionPhase.STAND: "ACTION_FAILURE",
    MissionPhase.TURN_TOWARD_HOME: "RETURN_HOME_FAILURE",
    MissionPhase.RETURN_HOME: "RETURN_HOME_FAILURE",
    MissionPhase.RESTORE_HEADING: "RETURN_HOME_FAILURE",
}


class DemoOrchestrator:
    def __init__(
        self,
        coordinator: RunCoordinator,
        results: RunResultStore,
        stages: StageExecutor,
        terminal_evidence: Callable[[], list[EvidenceArtifact]] | None = None,
        automatic_failure_recovery: AutomaticFailureRecovery | None = None,
        lie_down_evidence: (
            Callable[[str, str], Awaitable[dict[str, Any]]] | None
        ) = None,
    ) -> None:
        self._coordinator = coordinator
        self._results = results
        self._stages = stages
        self._terminal_evidence = terminal_evidence
        self._automatic_failure_recovery = automatic_failure_recovery
        self._lie_down_evidence = lie_down_evidence

    async def _capture_terminal_evidence(self, run_id: str) -> None:
        if self._terminal_evidence is None:
            self._results.record_evidence_unavailable(
                run_id,
                "terminal evidence adapter is not configured",
            )
            return
        try:
            artifacts = await asyncio.to_thread(self._terminal_evidence)
            self._results.record_artifacts(run_id, artifacts)
        except Exception as exc:  # noqa: BLE001 - evidence must not mask safety
            self._results.record_evidence_unavailable(
                run_id,
                f"terminal evidence capture failed: {exc}",
            )

    async def run(self, run_id: str) -> dict[str, Any]:
        run = self._results.get(run_id)
        context = StageContext(
            run_id=run_id,
            run_epoch=str(run["run_epoch"]),
            target_fruit=run["target_fruit"],
            home=run["home"],
            orientation_degrees=float(run.get("orientation_degrees", 0.0)),
            search_policy=str(run.get("search_policy", "slow-sweep")),
        )
        try:
            for phase in EXECUTED_STAGES:
                self._coordinator.advance(
                    run_id,
                    reason=f"starting {phase.value}",
                    event_reason=f"{phase.value.upper()}_STARTED",
                    message=f"{phase.value} started",
                )
                evidence = await self._stages.execute(phase, context)
                if (
                    phase is MissionPhase.SIT_AND_BARK
                    and self._lie_down_evidence is not None
                ):
                    evidence = {
                        **evidence,
                        "lie_down_snapshot": await self._lie_down_evidence(
                            run_id,
                            "audience_action",
                        ),
                    }
                self._results.record_stage(run_id, phase.value, evidence)
                if phase in (
                    MissionPhase.TURN_TO_FRUIT,
                    MissionPhase.FIND_FRUIT,
                ):
                    raw_handoff = evidence.get("search_qualification")
                    handoff = None
                    if isinstance(raw_handoff, Mapping):
                        try:
                            handoff = SearchQualificationHandoff.from_evidence(
                                raw_handoff
                            )
                        except (TypeError, ValueError):
                            handoff = None
                    context = replace(
                        context,
                        search_qualification_handoff=handoff,
                    )
                if phase is MissionPhase.APPROACH_FRUIT:
                    pulse_count = evidence.get("forward_pulse_count")
                    if not isinstance(pulse_count, int) or pulse_count < 1:
                        raise StageFailure(
                            "ARRIVAL_FAILURE",
                            "approach did not record a positive forward pulse count",
                        )
                    context = replace(
                        context,
                        outbound_forward_pulses=pulse_count,
                    )
                    self._coordinator.advance(
                        run_id,
                        reason="Arrival confirmed",
                        event_reason="ARRIVAL_CONFIRMED",
                        message="qualified near-fruit Arrival confirmed",
                    )

            stop_errors = await self._stages.stop()
            if stop_errors:
                raise RuntimeError("; ".join(stop_errors))
            return self._coordinator.finish(
                run_id,
                terminal_phase=MissionPhase.COMPLETE,
                outcome="COMPLETED",
                reason="SUCCESS",
                message="Demo Run completed at Home",
                final_safety_state="DISARMED_CONFIRMED",
            )
        except StageFailure as exc:
            failed_phase = self._results.get(run_id)["current_phase"]
            stop_errors = await self._stages.stop()
            await self._capture_terminal_evidence(run_id)
            failed = self._coordinator.finish(
                run_id,
                terminal_phase=MissionPhase.FAILED,
                outcome="FAILED",
                reason=exc.reason,
                message=exc.message,
                final_safety_state=(
                    "DISARMED_CONFIRMED"
                    if not stop_errors
                    else "STOP_REQUESTED_UNCONFIRMED"
                ),
                failed_phase=failed_phase,
                failure_details=exc.details,
            )
            if self._automatic_failure_recovery is not None:
                await self._automatic_failure_recovery.recover_automatically(run_id)
                self._coordinator.refresh_black_box(run_id)
                return self._results.get(run_id)
            return failed
        except Exception as exc:  # noqa: BLE001 - terminal safety boundary
            failed_phase = self._results.get(run_id)["current_phase"]
            stop_errors = await self._stages.stop()
            await self._capture_terminal_evidence(run_id)
            return self._coordinator.finish(
                run_id,
                terminal_phase=MissionPhase.FAILED,
                outcome="FAILED",
                reason="INTERNAL_ERROR",
                message=f"Demo Run failed: {exc}",
                final_safety_state=(
                    "DISARMED_CONFIRMED"
                    if not stop_errors
                    else "STOP_REQUESTED_UNCONFIRMED"
                ),
                failed_phase=failed_phase,
            )


class SimulatedStageExecutor:
    """Deterministic non-hardware adapter for the complete base Demo Run."""

    _EVIDENCE: ClassVar[dict[MissionPhase, dict[str, Any]]] = {
        MissionPhase.ORIENT_FOR_RUN: {
            "requested_angle_degrees": 0.0,
            "requested_angle_rad": 0.0,
            "measured_yaw_change_rad": 0.0,
            "motion_commands_sent": False,
        },
        MissionPhase.TURN_TO_FRUIT: {
            "measured_yaw_change_rad": 3.14,
            "motion_commands_sent": False,
        },
        MissionPhase.FIND_FRUIT: {
            "label": "pear",
            "confidence": 0.81,
            "stable_detections": 5,
            "motion_commands_sent": False,
        },
        MissionPhase.APPROACH_FRUIT: {
            "arrival_confirmed": True,
            "close_range_mps": 1.0,
            "final_push_mps": 0.55,
            "final_push_duration_s": 1.0,
            "forward_pulse_count": 7,
            "motion_commands_sent": False,
        },
        MissionPhase.SIT_AND_BARK: {
            "posture": "stand_down",
            "bark_played": True,
            "down_hold_s": 5.0,
            "motion_commands_sent": False,
        },
        MissionPhase.STAND: {
            "posture": "balance_stand",
            "motion_commands_sent": False,
        },
        MissionPhase.TURN_TOWARD_HOME: {
            "home_bearing_error_rad": 0.03,
            "motion_commands_sent": False,
        },
        MissionPhase.RETURN_HOME: {
            "home_distance_m": 0.08,
            "motion_commands_sent": False,
        },
        MissionPhase.RESTORE_HEADING: {
            "heading_error_rad": 0.04,
            "motion_commands_sent": False,
        },
    }

    def __init__(
        self,
        *,
        fail_at: str | None = None,
        failure_reason: str | None = None,
        failure_message: str = "simulated stage failure",
        delay_at: str | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self._fail_at = fail_at
        self._failure_reason = failure_reason
        self._failure_message = failure_message
        self._delay_at = delay_at
        self._delay_s = delay_s

    async def execute(
        self,
        phase: MissionPhase,
        context: StageContext,
    ) -> dict[str, Any]:
        if phase.value == self._delay_at and self._delay_s > 0.0:
            await asyncio.sleep(self._delay_s)
        if phase.value == self._fail_at:
            raise StageFailure(
                self._failure_reason or DEFAULT_STAGE_FAILURE_REASONS[phase],
                self._failure_message,
            )
        evidence = dict(self._EVIDENCE[phase])
        if phase is MissionPhase.ORIENT_FOR_RUN:
            requested_rad = math.radians(context.orientation_degrees)
            evidence.update(
                requested_angle_degrees=context.orientation_degrees,
                requested_angle_rad=requested_rad,
                measured_yaw_change_rad=requested_rad,
            )
        if phase is MissionPhase.FIND_FRUIT:
            evidence["label"] = context.target_fruit
        if phase is MissionPhase.RETURN_HOME:
            evidence.update(
                requested_forward_pulses=context.outbound_forward_pulses,
                replayed_forward_pulses=context.outbound_forward_pulses,
            )
        return evidence

    async def stop(self) -> list[str]:
        return []
