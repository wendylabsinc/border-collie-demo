"""Durable authority for Demo Run lifecycle transitions.

The coordinator keeps the in-memory MissionMachine as a projection of the
durable Run Result.  Consequential transitions are written first so a process
failure can never leave motion-oriented state that exists only in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .flight_recorder import FlightRecorder
from .mission import MissionMachine
from .models import MissionPhase
from .run_results import RunResultStore


@dataclass(frozen=True)
class RunActivation:
    target_fruit: str
    activation_source: str
    orientation_degrees: float = 0.0
    idempotency_key: str | None = None


@dataclass(frozen=True)
class ActivationDecision:
    run: dict[str, Any]
    created: bool


class RunCoordinator:
    """Deep module that serializes and durably orders Demo Run state changes."""

    def __init__(
        self,
        mission: MissionMachine,
        results: RunResultStore,
        recorder: FlightRecorder | None = None,
    ) -> None:
        self._mission = mission
        self._results = results
        self._recorder = recorder

    @property
    def active_run_id(self) -> str | None:
        return self._results.active_run_id

    def activate(self, activation: RunActivation) -> ActivationDecision:
        run, created = self._results.start_or_reuse_run(
            target_fruit=activation.target_fruit,
            activation_source=activation.activation_source,
            orientation_degrees=activation.orientation_degrees,
            idempotency_key=activation.idempotency_key,
        )
        if not created:
            self._record("activation_reused", run, run_id=run["run_id"])
            return ActivationDecision(run=run, created=False)

        # The activation record is durable before the in-memory projection
        # changes. A crash here is recovered as an interrupted, disarmed run.
        self._mission.begin_run("Demo Run activation persisted")
        run = self._results.enter_phase(
            run["run_id"],
            phase=self._mission.phase.value,
            reason="PREFLIGHT_STARTED",
            message="preflight entered; verifying production motion and media gates",
            expected_phase="idle",
        )
        self._record("run_activated", run, run_id=run["run_id"])
        return ActivationDecision(run=run, created=True)

    def advance(
        self,
        run_id: str,
        *,
        reason: str,
        event_reason: str,
        message: str,
    ) -> dict[str, Any]:
        current = self._mission.phase
        next_phase = self._mission.next_phase()
        run = self._results.enter_phase(
            run_id,
            phase=next_phase.value,
            reason=event_reason,
            message=message,
            expected_phase=current.value,
        )
        self._record("run_transition", run["events"][-1], run_id=run_id)
        self._mission.advance(reason)
        return run

    def finish(
        self,
        run_id: str,
        *,
        terminal_phase: MissionPhase,
        outcome: str,
        reason: str,
        message: str,
        final_safety_state: str,
        failed_phase: str | None = None,
        failure_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run = self._results.seal(
            run_id,
            phase=terminal_phase.value,
            outcome=outcome,
            reason=reason,
            message=message,
            final_safety_state=final_safety_state,
            failed_phase=failed_phase,
            failure_details=failure_details,
        )
        self._record(
            "run_terminal",
            {
                "outcome": outcome,
                "reason": reason,
                "phase": terminal_phase.value,
                "final_safety_state": final_safety_state,
            },
            run_id=run_id,
        )
        if terminal_phase is MissionPhase.COMPLETE:
            self._mission.advance(message)
        elif terminal_phase is MissionPhase.STOPPED:
            self._mission.stop(message)
        elif terminal_phase is MissionPhase.FAILED:
            self._mission.fail(message)
        else:
            raise ValueError(f"unsupported terminal phase: {terminal_phase.value}")
        return run

    def recover_interrupted(self, *, final_safety_state: str) -> list[dict[str, Any]]:
        sealed = self._results.seal_interrupted_runs(
            final_safety_state=final_safety_state
        )
        if sealed:
            self._mission.fail("interrupted Demo Run recovered at process startup")
            for run in sealed:
                self._record("interrupted_run_sealed", run, run_id=run["run_id"])
        return sealed

    def _record(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        run_id: str,
    ) -> None:
        if self._recorder is not None:
            self._recorder.record(kind, payload, run_id=run_id)
