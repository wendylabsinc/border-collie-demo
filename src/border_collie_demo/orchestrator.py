"""End-to-end Demo Run orchestration over one stage-executor interface."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Protocol

from .evidence import EvidenceArtifact
from .mission import MissionMachine
from .models import MissionPhase
from .run_results import RunResultStore


@dataclass(frozen=True)
class StageContext:
    run_id: str
    target_fruit: str
    home: dict[str, Any]
    outbound_forward_pulses: int = 0


class StageExecutor(Protocol):
    async def execute(
        self,
        phase: MissionPhase,
        context: StageContext,
    ) -> dict[str, Any]: ...

    async def stop(self) -> list[str]: ...


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
    MissionPhase.TURN_TO_FRUIT,
    MissionPhase.FIND_FRUIT,
    MissionPhase.APPROACH_FRUIT,
    MissionPhase.SIT_AND_BARK,
    MissionPhase.STAND,
    MissionPhase.STEP_BACK,
    MissionPhase.TURN_TOWARD_HOME,
    MissionPhase.RETURN_HOME,
    MissionPhase.RESTORE_HEADING,
)

DEFAULT_STAGE_FAILURE_REASONS = {
    MissionPhase.TURN_TO_FRUIT: "TARGET_RECOGNITION_FAILURE",
    MissionPhase.FIND_FRUIT: "TARGET_RECOGNITION_FAILURE",
    MissionPhase.APPROACH_FRUIT: "ARRIVAL_FAILURE",
    MissionPhase.SIT_AND_BARK: "ACTION_FAILURE",
    MissionPhase.STAND: "ACTION_FAILURE",
    MissionPhase.STEP_BACK: "ACTION_FAILURE",
    MissionPhase.TURN_TOWARD_HOME: "RETURN_HOME_FAILURE",
    MissionPhase.RETURN_HOME: "RETURN_HOME_FAILURE",
    MissionPhase.RESTORE_HEADING: "RETURN_HOME_FAILURE",
}


class DemoOrchestrator:
    def __init__(
        self,
        mission: MissionMachine,
        results: RunResultStore,
        stages: StageExecutor,
        terminal_evidence: Callable[[], list[EvidenceArtifact]] | None = None,
    ) -> None:
        self._mission = mission
        self._results = results
        self._stages = stages
        self._terminal_evidence = terminal_evidence

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
            target_fruit=run["target_fruit"],
            home=run["home"],
        )
        try:
            for phase in EXECUTED_STAGES:
                self._mission.advance(f"starting {phase.value}")
                self._results.enter_phase(
                    run_id,
                    phase=phase.value,
                    reason=f"{phase.value.upper()}_STARTED",
                    message=f"{phase.value} started",
                )
                evidence = await self._stages.execute(phase, context)
                self._results.record_stage(run_id, phase.value, evidence)
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
                    self._mission.advance("Arrival confirmed")
                    self._results.enter_phase(
                        run_id,
                        phase=self._mission.phase.value,
                        reason="ARRIVAL_CONFIRMED",
                        message="qualified near-fruit Arrival confirmed",
                    )

            stop_errors = await self._stages.stop()
            if stop_errors:
                raise RuntimeError("; ".join(stop_errors))
            await self._capture_terminal_evidence(run_id)
            self._mission.advance("Demo Run completed and disarmed")
            return self._results.seal(
                run_id,
                phase=self._mission.phase.value,
                outcome="COMPLETED",
                reason="SUCCESS",
                message="Demo Run completed at Home",
                final_safety_state="DISARMED_CONFIRMED",
            )
        except StageFailure as exc:
            failed_phase = self._mission.phase.value
            stop_errors = await self._stages.stop()
            await self._capture_terminal_evidence(run_id)
            self._mission.fail(exc.message)
            return self._results.seal(
                run_id,
                phase=self._mission.phase.value,
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
        except Exception as exc:  # noqa: BLE001 - terminal safety boundary
            failed_phase = self._mission.phase.value
            stop_errors = await self._stages.stop()
            await self._capture_terminal_evidence(run_id)
            self._mission.fail(f"Demo Run failed: {exc}")
            return self._results.seal(
                run_id,
                phase=self._mission.phase.value,
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
            "final_push_mps": 0.3,
            "final_push_duration_s": 0.6,
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
        MissionPhase.STEP_BACK: {
            "commanded_reverse_mps": 0.5,
            "commanded_duration_s": 0.5,
            "measured_backward_m": 0.24,
            "avoidance_prior_enabled": True,
            "avoidance_restored": True,
            "avoidance_switched_off_s": 0.9,
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
