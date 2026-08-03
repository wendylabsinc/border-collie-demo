from __future__ import annotations

from dataclasses import dataclass, field

from .models import MissionEvent, MissionPhase, RemoteInput


class MissionError(RuntimeError):
    """Raised when a requested mission transition is unsafe or invalid."""


class RestartRequired(MissionError):
    """Raised after a physical remote has permanently taken over this process."""


NEXT_PHASE: dict[MissionPhase, MissionPhase] = {
    MissionPhase.IDLE: MissionPhase.PREFLIGHT,
    MissionPhase.PREFLIGHT: MissionPhase.CAPTURE_HOME,
    MissionPhase.CAPTURE_HOME: MissionPhase.WAIT_FOR_COMMAND,
    MissionPhase.WAIT_FOR_COMMAND: MissionPhase.TURN_TO_FRUIT,
    MissionPhase.TURN_TO_FRUIT: MissionPhase.FIND_FRUIT,
    MissionPhase.FIND_FRUIT: MissionPhase.APPROACH_FRUIT,
    MissionPhase.APPROACH_FRUIT: MissionPhase.ARRIVED,
    MissionPhase.ARRIVED: MissionPhase.SIT_AND_BARK,
    MissionPhase.SIT_AND_BARK: MissionPhase.STAND,
    MissionPhase.STAND: MissionPhase.TURN_TOWARD_HOME,
    MissionPhase.TURN_TOWARD_HOME: MissionPhase.RETURN_HOME,
    MissionPhase.RETURN_HOME: MissionPhase.RESTORE_HEADING,
    MissionPhase.RESTORE_HEADING: MissionPhase.COMPLETE,
}


@dataclass
class MissionMachine:
    """Pure mission state; hardware orchestration will be added in later slices."""

    phase: MissionPhase = MissionPhase.IDLE
    reason: str = "waiting to start"
    takeover_latched: bool = False
    history: list[MissionEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.history:
            self.history.append(MissionEvent.record(self.phase, self.reason))

    def advance(self, reason: str) -> MissionPhase:
        self._require_process_control()
        next_phase = NEXT_PHASE.get(self.phase)
        if next_phase is None:
            raise MissionError(f"cannot advance from {self.phase.value}")
        self._record(next_phase, reason)
        return self.phase

    def stop(self, reason: str = "operator stop") -> MissionPhase:
        self._require_process_control()
        self._record(MissionPhase.STOPPED, reason)
        return self.phase

    def fail(self, reason: str) -> MissionPhase:
        self._require_process_control()
        self._record(MissionPhase.FAILED, reason)
        return self.phase

    def remote_takeover(self, remote_input: RemoteInput) -> MissionPhase:
        if self.takeover_latched:
            return self.phase
        self.takeover_latched = True
        self._record(
            MissionPhase.REMOTE_TAKEOVER,
            f"{remote_input.source}: {remote_input.control}",
        )
        return self.phase

    def status(self) -> dict[str, object]:
        return {
            "phase": self.phase.value,
            "reason": self.reason,
            "remote_takeover_latched": self.takeover_latched,
            "restart_required": self.takeover_latched,
            "hardware_enabled": False,
            "can_move": False,
            "history": [
                {
                    "phase": event.phase.value,
                    "reason": event.reason,
                    "recorded_monotonic_s": event.recorded_monotonic_s,
                }
                for event in self.history
            ],
        }

    def _require_process_control(self) -> None:
        if self.takeover_latched:
            raise RestartRequired(
                "physical remote takeover is latched; restart the application"
            )

    def _record(self, phase: MissionPhase, reason: str) -> None:
        self.phase = phase
        self.reason = reason
        self.history.append(MissionEvent.record(phase, reason))
