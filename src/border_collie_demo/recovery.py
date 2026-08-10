"""Bounded, durable recovery of a failed Demo Run to its captured Home."""

from __future__ import annotations

import asyncio
import math
from typing import Any, Protocol

from .hardware import HardwareUnavailable
from .home_localization import HomeLocalizer
from .return_home import Pose2D
from .run_results import RunResultStore

RECOVERY_CONFIRMATION = "RECOVER FAILED RUN TO CAPTURED HOME"
HOME_POSITION_TOLERANCE_M = 0.10
RECOVERY_PULSE_RESERVE_FACTOR = 1.5


def recovery_forward_pulse_budget(outbound_pulses: int) -> int:
    """Return a bounded reserve while pose feedback remains authoritative."""
    return math.ceil(outbound_pulses * RECOVERY_PULSE_RESERVE_FACTOR)


class RecoveryHardware(Protocol):
    def status(self) -> dict[str, object]: ...

    def start_motion_trace(self, phase: str) -> None: ...

    def set_motion_authority(self, run_id: str, epoch: str, phase: str) -> None: ...

    def motion_trace(self) -> list[dict[str, object]]: ...

    async def turn_toward_home(
        self, home: dict[str, object], **options: float
    ) -> dict[str, object]: ...

    async def return_home(
        self, home: dict[str, object], **options: object
    ) -> dict[str, object]: ...

    async def emergency_stop(self) -> list[str]: ...


def recoverable_forward_pulses(run: dict[str, Any]) -> tuple[int, str]:
    """Recover the bounded return budget from completed or failed approach evidence."""
    approach = (run.get("stage_results") or {}).get("approach_fruit") or {}
    completed_count = approach.get("forward_pulse_count")
    if (
        isinstance(completed_count, int)
        and not isinstance(completed_count, bool)
        and completed_count > 0
    ):
        return completed_count, "stage_results.approach_fruit.forward_pulse_count"

    commands = (run.get("failure_details") or {}).get("motion_commands") or []
    failed_count = sum(
        1
        for command in commands
        if isinstance(command, dict)
        and command.get("phase") == "approach_fruit"
        and isinstance(command.get("forward_mps"), (int, float))
        and not isinstance(command.get("forward_mps"), bool)
        and float(command["forward_mps"]) > 0.0
    )
    if failed_count:
        return failed_count, "failure_details.motion_commands"
    return 0, "no recorded forward approach commands"


def _recovery_blockers(status: dict[str, object]) -> list[str]:
    blockers: list[str] = []
    if status.get("configured") is not True:
        blockers.append("Go2 hardware is not configured")
    if status.get("autonomy_enabled") is not True:
        blockers.append("autonomous motion is disabled")
    if status.get("connected") is not True:
        blockers.append("Go2 hardware or pose is not connected")
    if status.get("fault") is not None:
        blockers.append(f"hardware fault: {status['fault']}")
    if status.get("active_operation") is not None:
        blockers.append(f"hardware operation active: {status['active_operation']}")

    pose = status.get("pose")
    if not isinstance(pose, dict) or pose.get("healthy") is not True:
        blockers.append("fresh Go2 pose is unavailable")
    motion = status.get("motion")
    if not isinstance(motion, dict):
        blockers.append("motion controller status is unavailable")
    else:
        if motion.get("initialized") is not True:
            blockers.append("motion controller is not initialized")
        if motion.get("armed") is not False:
            blockers.append("motion controller is not disarmed")
        if motion.get("fault") is not None:
            blockers.append(f"motion fault: {motion['fault']}")
    return blockers


def _with_trace(
    hardware: RecoveryHardware, evidence: dict[str, object]
) -> dict[str, object]:
    return {**evidence, "motion_commands": hardware.motion_trace()}


def _home_estimate_from_status(
    home: dict[str, object],
    status: dict[str, object],
    hardware: RecoveryHardware,
) -> dict[str, object]:
    estimate_home = getattr(hardware, "estimate_home", None)
    if callable(estimate_home):
        estimate = estimate_home(home)
        if isinstance(estimate, dict):
            return estimate

    pose_status = status.get("pose")
    pose = pose_status.get("pose") if isinstance(pose_status, dict) else None
    if not isinstance(pose, dict):
        return _unavailable_home_estimate("fresh Go2 pose is unavailable")
    values = (
        home.get("x_m"),
        home.get("y_m"),
        home.get("yaw_rad"),
        pose.get("x_m"),
        pose.get("y_m"),
        pose.get("yaw_rad"),
        pose_status.get("age_s") if isinstance(pose_status, dict) else None,
    )
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in values
    ):
        return _unavailable_home_estimate("Home or Go2 pose is invalid")
    home_x, home_y, home_yaw, pose_x, pose_y, pose_yaw, pose_age = (
        float(value) for value in values
    )
    return HomeLocalizer().estimate(
        Pose2D(home_x, home_y, home_yaw),
        Pose2D(pose_x, pose_y, pose_yaw),
        odometry_age_s=pose_age,
    ).to_dict()


def _unavailable_home_estimate(reason: str) -> dict[str, object]:
    return {
        "state": "unavailable",
        "trusted": False,
        "source": None,
        "unavailable_reason": reason,
        "home_distance_m": None,
        "heading_error_rad": None,
        "pose_from_home": None,
        "evidence": {},
    }


def _trusted_home_distance(estimate: dict[str, object]) -> float | None:
    distance = estimate.get("home_distance_m")
    if (
        estimate.get("trusted") is True
        and isinstance(distance, (int, float))
        and not isinstance(distance, bool)
        and math.isfinite(float(distance))
    ):
        return float(distance)
    return None


class FailedRunHomeRecovery:
    """Run one forward-only, position-only recovery attempt for a failed run.

    Failed-run recovery deliberately does not restore the captured heading. The
    production heading turn can move the feet outside the position gate; a
    recovery must prefer a stable Home position over an oscillating turn and
    translate loop. The original failed run remains failed, while the recovery
    attempt gets its own durable outcome.
    """

    def __init__(self, hardware: RecoveryHardware, results: RunResultStore) -> None:
        self._hardware = hardware
        self._results = results

    def validate(self, run: dict[str, Any]) -> tuple[int, str]:
        if run.get("outcome") != "FAILED":
            raise HardwareUnavailable("only failed Demo Runs can be recovered")
        if run.get("final_safety_state") != "DISARMED_CONFIRMED":
            raise HardwareUnavailable(
                "failed run does not have a confirmed disarmed terminal state"
            )
        home = run.get("home")
        if not isinstance(home, dict):
            raise HardwareUnavailable("failed run has no captured Home pose")
        pulse_count, pulse_source = recoverable_forward_pulses(run)
        status = self._hardware.status()
        blockers = _recovery_blockers(status)
        if blockers:
            raise HardwareUnavailable("; ".join(blockers))
        if pulse_count < 1:
            home_estimate = _home_estimate_from_status(home, status, self._hardware)
            home_distance = _trusted_home_distance(home_estimate)
            if home_distance is None or home_distance > HOME_POSITION_TOLERANCE_M:
                raise HardwareUnavailable(
                    "failed run is outside Home and recorded no forward approach "
                    "commands to bound recovery"
                )
        return pulse_count, pulse_source

    async def run(self, run_id: str, recovery_id: str) -> dict[str, Any]:
        run = self._results.get(run_id)
        home = run["home"]
        pulse_count, pulse_source = recoverable_forward_pulses(run)
        recovery_pulse_budget = recovery_forward_pulse_budget(pulse_count)
        self._results.record_recovery_step(
            run_id,
            recovery_id,
            "preflight",
            {
                "hardware": self._hardware.status(),
                "captured_home": home,
                "outbound_forward_pulses": pulse_count,
                "maximum_forward_pulses": recovery_pulse_budget,
                "recovery_forward_pulse_budget": recovery_pulse_budget,
                "recovery_pulse_reserve_factor": RECOVERY_PULSE_RESERVE_FACTOR,
                "forward_pulse_source": pulse_source,
                "position_tolerance_m": HOME_POSITION_TOLERANCE_M,
                "heading_restore_planned": False,
                "heading_restore_reason": (
                    "failed-run recovery is position-only to avoid turn-induced "
                    "position drift"
                ),
            },
        )
        try:
            blockers = _recovery_blockers(self._hardware.status())
            if blockers:
                raise HardwareUnavailable("; ".join(blockers))

            if pulse_count < 1:
                stop_errors = await self._hardware.emergency_stop()
                final_status = self._hardware.status()
                home_estimate = _home_estimate_from_status(
                    home, final_status, self._hardware
                )
                home_distance = _trusted_home_distance(home_estimate)
                recovered = (
                    home_distance is not None
                    and home_distance <= HOME_POSITION_TOLERANCE_M
                    and not stop_errors
                    and not _recovery_blockers(final_status)
                )
                return self._results.seal_recovery(
                    run_id,
                    recovery_id,
                    outcome="COMPLETED" if recovered else "FAILED",
                    reason=(
                        "HOME_POSITION_ALREADY_RECOVERED"
                        if recovered
                        else "RECOVERY_POSITION_GATE_FAILED"
                    ),
                    message=(
                        "failed run remained inside the captured Home position gate"
                        if recovered
                        else "failed run recovery did not satisfy the Home position gate"
                    ),
                    final_safety_state=(
                        "DISARMED_CONFIRMED"
                        if not stop_errors
                        else "STOP_REQUESTED_UNCONFIRMED"
                    ),
                    final_evidence={
                        "hardware": final_status,
                        "stop_errors": stop_errors,
                        "home_distance_m": home_distance,
                        "home_localization": home_estimate,
                        "heading_restored": False,
                        "motion_commands": [],
                    },
                )

            set_authority = getattr(self._hardware, "set_motion_authority", None)
            if callable(set_authority):
                set_authority(run_id, recovery_id, "recovery_turn_toward_home")
            self._hardware.start_motion_trace("recovery_turn_toward_home")
            turn = await self._hardware.turn_toward_home(
                home,
                yaw_rps=0.50,
                tolerance_rad=math.radians(5.0),
                response_timeout_s=0.75,
                response_min_progress_rad=math.radians(2.0),
                recovery_settle_s=1.0,
                timeout_s=30.0,
            )
            self._results.record_recovery_step(
                run_id,
                recovery_id,
                "turn_toward_home",
                _with_trace(self._hardware, turn),
            )

            if callable(set_authority):
                set_authority(run_id, recovery_id, "recovery_return_home")
            self._hardware.start_motion_trace("recovery_return_home")
            returned = await self._hardware.return_home(
                home,
                forward_mps=1.0,
                forward_pulse_count=recovery_pulse_budget,
                arrival_tolerance_m=HOME_POSITION_TOLERANCE_M,
                heading_gate_rad=math.radians(20.0),
                maximum_yaw_rps=0.30,
                minimum_progress_m=0.03,
                stall_timeout_s=2.0,
                timeout_s=30.0,
            )
            self._results.record_recovery_step(
                run_id,
                recovery_id,
                "return_home",
                _with_trace(self._hardware, returned),
            )
            stop_errors = await self._hardware.emergency_stop()
            final_status = self._hardware.status()
            controller_home_distance = returned.get("home_distance_m")
            home_estimate = _home_estimate_from_status(
                home, final_status, self._hardware
            )
            home_distance = _trusted_home_distance(home_estimate)
            recovered = (
                home_distance is not None
                and home_distance <= HOME_POSITION_TOLERANCE_M
                and not stop_errors
                and not _recovery_blockers(final_status)
            )
            return self._results.seal_recovery(
                run_id,
                recovery_id,
                outcome="COMPLETED" if recovered else "FAILED",
                reason=(
                    "HOME_POSITION_RECOVERED"
                    if recovered
                    else "RECOVERY_POSITION_GATE_FAILED"
                ),
                message=(
                    "failed run returned inside the captured Home position gate"
                    if recovered
                    else "failed run recovery did not satisfy the Home position gate"
                ),
                final_safety_state=(
                    "DISARMED_CONFIRMED" if not stop_errors else "STOP_REQUESTED_UNCONFIRMED"
                ),
                final_evidence={
                    "hardware": final_status,
                    "stop_errors": stop_errors,
                    "home_distance_m": home_distance,
                    "home_localization": home_estimate,
                    "controller_home_distance_m": controller_home_distance,
                    "heading_restored": False,
                },
            )
        except asyncio.CancelledError:
            stop_errors = await self._hardware.emergency_stop()
            self._results.seal_recovery(
                run_id,
                recovery_id,
                outcome="STOPPED",
                reason="OPERATOR_STOP",
                message="operator stopped failed-run Home recovery",
                final_safety_state=(
                    "DISARMED_CONFIRMED" if not stop_errors else "STOP_REQUESTED_UNCONFIRMED"
                ),
                final_evidence={
                    "hardware": self._hardware.status(),
                    "stop_errors": stop_errors,
                },
            )
            raise
        except Exception as exc:  # noqa: BLE001 - recovery must always fail closed
            stop_errors = await self._hardware.emergency_stop()
            return self._results.seal_recovery(
                run_id,
                recovery_id,
                outcome="FAILED",
                reason="RECOVERY_FAILURE",
                message=str(exc),
                final_safety_state=(
                    "DISARMED_CONFIRMED" if not stop_errors else "STOP_REQUESTED_UNCONFIRMED"
                ),
                final_evidence={
                    "hardware": self._hardware.status(),
                    "stop_errors": stop_errors,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "motion_commands": self._hardware.motion_trace(),
                },
            )
