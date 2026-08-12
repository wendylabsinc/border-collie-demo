"""Bounded, durable recovery of a failed Demo Run to its captured Home."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from .hardware import HardwareUnavailable
from .home_localization import HomeLocalizer
from .return_home import Pose2D
from .run_results import RunResultStore

RECOVERY_CONFIRMATION = "RECOVER FAILED RUN TO CAPTURED HOME"
HOME_POSITION_TOLERANCE_M = 0.10
RECOVERY_PULSE_RESERVE_FACTOR = 1.5
FAILURE_DOWN_HOLD_S = 5.0
FAILURE_STAND_SETTLE_S = 1.0
POSE_FRESHNESS_LIMIT_S = 0.50
FUSION_FRESHNESS_LIMIT_S = 0.50
# The production approach is bounded to 20 s at a 100 ms heartbeat. Recovery
# refuses evidence that could not have come from that bounded control window.
MAXIMUM_RECOVERABLE_OUTBOUND_PULSES = 200
AUTOMATIC_FAILURE_RECOVERY_REASONS = frozenset(
    {"ARRIVAL_FAILURE", "TARGET_LOST_OFF_AXIS"}
)
AUTOMATIC_RECOVERY_CONFIRMATION = "AUTOMATIC RECOVERABLE FAILURE TO CAPTURED HOME"


class AutomaticRecoveryIneligible(HardwareUnavailable):
    """Raised when evidence does not authorize autonomous recovery motion."""


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

    async def stand_down(self) -> dict[str, object]: ...

    async def stand_up(self, *, settle_s: float = 1.0) -> dict[str, object]: ...


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
    elif (
        not isinstance(pose.get("age_s"), (int, float))
        or isinstance(pose.get("age_s"), bool)
        or not math.isfinite(float(pose["age_s"]))
        or float(pose["age_s"]) < 0.0
        or float(pose["age_s"]) > POSE_FRESHNESS_LIMIT_S
    ):
        blockers.append("Go2 pose is stale or has invalid age evidence")
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


def _continuous_fusion_blockers(status: dict[str, object]) -> list[str]:
    fusion = status.get("continuous_home_fusion")
    if not isinstance(fusion, dict):
        return ["continuous Home fusion status is unavailable"]
    blockers: list[str] = []
    if fusion.get("running") is not True:
        blockers.append("continuous Home fusion is not running")
    samples = fusion.get("consecutive_trusted_samples")
    if (
        not isinstance(samples, int)
        or isinstance(samples, bool)
        or samples < 3
    ):
        blockers.append("continuous Home fusion has fewer than three trusted samples")
    latest = fusion.get("latest")
    if not isinstance(latest, dict) or latest.get("trusted") is not True:
        blockers.append("latest continuous Home fusion sample is untrusted")
    latest_age_s = fusion.get("latest_age_s")
    if (
        not isinstance(latest_age_s, (int, float))
        or isinstance(latest_age_s, bool)
        or not math.isfinite(float(latest_age_s))
        or float(latest_age_s) < 0.0
        or float(latest_age_s) > FUSION_FRESHNESS_LIMIT_S
    ):
        blockers.append("latest continuous Home fusion sample is stale")
    return blockers


def _validate_captured_home(home: object) -> dict[str, object]:
    if not isinstance(home, dict):
        raise AutomaticRecoveryIneligible("failed run has no captured Home pose")
    for field in ("x_m", "y_m", "yaw_rad", "captured_monotonic_s", "age_s"):
        value = home.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise AutomaticRecoveryIneligible(
                f"captured Home has invalid {field} evidence"
            )
    age_s = float(home["age_s"])
    if age_s < 0.0 or age_s > POSE_FRESHNESS_LIMIT_S:
        raise AutomaticRecoveryIneligible(
            "captured Home was not fresh when the Demo Run began"
        )
    return home


def _confirmed_disarm_state(
    status: dict[str, object], stop_errors: list[str]
) -> str:
    motion = status.get("motion")
    return (
        "DISARMED_CONFIRMED"
        if (
            not stop_errors
            and isinstance(motion, dict)
            and motion.get("armed") is False
        )
        else "STOP_REQUESTED_UNCONFIRMED"
    )


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

    def __init__(
        self,
        hardware: RecoveryHardware,
        results: RunResultStore,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        automatic_recovery_guard: Callable[[], str | None] | None = None,
        lie_down_evidence: (
            Callable[[str, str], Awaitable[dict[str, object]]] | None
        ) = None,
    ) -> None:
        self._hardware = hardware
        self._results = results
        self._sleep = sleep
        self._automatic_recovery_guard = automatic_recovery_guard
        self._lie_down_evidence = lie_down_evidence

    def _automatic_guard_blocker(self) -> str | None:
        if self._automatic_recovery_guard is None:
            return None
        try:
            return self._automatic_recovery_guard()
        except Exception as exc:  # noqa: BLE001 - a broken guard must fail closed
            return f"automatic recovery safety guard failed: {exc}"

    async def _safe_stop(self) -> list[str]:
        try:
            return await self._hardware.emergency_stop()
        except Exception as exc:  # noqa: BLE001 - stop evidence must be durable
            return [f"emergency stop raised {type(exc).__name__}: {exc}"]

    async def recover_automatically(self, run_id: str) -> dict[str, Any] | None:
        """Run the failure posture and Home correction for eligible failures."""
        run = self._results.get(run_id)
        if run.get("reason") not in AUTOMATIC_FAILURE_RECOVERY_REASONS:
            return None
        attempts = run.get("recovery_attempts") or []
        if attempts:
            return attempts[-1] if isinstance(attempts[-1], dict) else None
        attempt = self._results.start_recovery(
            run_id,
            confirmation=AUTOMATIC_RECOVERY_CONFIRMATION,
        )
        recovery_id = str(attempt["recovery_id"])
        try:
            self.validate(run, automatic=True)
            stop_errors = await self._safe_stop()
            if stop_errors:
                raise HardwareUnavailable(
                    "failure posture stop failed: " + "; ".join(stop_errors)
                )
            down = await self._hardware.stand_down()
            if down.get("posture") != "stand_down":
                raise HardwareUnavailable(
                    "failure posture did not confirm stand_down"
                )
            self._results.record_recovery_step(
                run_id,
                recovery_id,
                "failure_posture_down",
                {**down, "bark_played": False},
            )
            await self._sleep(FAILURE_DOWN_HOLD_S)
            lie_down_snapshot = (
                await self._lie_down_evidence(run_id, "failure_recovery")
                if self._lie_down_evidence is not None
                else None
            )
            self._results.record_recovery_step(
                run_id,
                recovery_id,
                "failure_posture_hold",
                {
                    "down_hold_s": FAILURE_DOWN_HOLD_S,
                    "bark_played": False,
                    **(
                        {"lie_down_snapshot": lie_down_snapshot}
                        if lie_down_snapshot is not None
                        else {}
                    ),
                },
            )
            stood = await self._hardware.stand_up(settle_s=FAILURE_STAND_SETTLE_S)
            if stood.get("posture") != "balance_stand":
                raise HardwareUnavailable(
                    "failure posture did not confirm balance_stand"
                )
            post_stand_status = self._hardware.status()
            post_stand_blockers = [
                *_recovery_blockers(post_stand_status),
                *_continuous_fusion_blockers(post_stand_status),
            ]
            guard_blocker = self._automatic_guard_blocker()
            if guard_blocker is not None:
                post_stand_blockers.append(guard_blocker)
            if post_stand_blockers:
                raise AutomaticRecoveryIneligible("; ".join(post_stand_blockers))
            self._results.record_recovery_step(
                run_id,
                recovery_id,
                "failure_posture_stand",
                {
                    **stood,
                    "stand_settle_s": FAILURE_STAND_SETTLE_S,
                    "home_fusion": post_stand_status.get("continuous_home_fusion"),
                },
            )
        except asyncio.CancelledError:
            stop_errors = await self._safe_stop()
            final_status = self._hardware.status()
            self._results.seal_recovery(
                run_id,
                recovery_id,
                outcome="STOPPED",
                reason="OPERATOR_STOP",
                message="operator stopped automatic failed-run recovery",
                final_safety_state=_confirmed_disarm_state(
                    final_status, stop_errors
                ),
                final_evidence={
                    "hardware": final_status,
                    "stop_errors": stop_errors,
                },
            )
            raise
        except AutomaticRecoveryIneligible as exc:
            stop_errors = await self._safe_stop()
            final_status = self._hardware.status()
            return self._results.seal_recovery(
                run_id,
                recovery_id,
                outcome="FAILED",
                reason="AUTOMATIC_RECOVERY_SKIPPED",
                message=str(exc),
                final_safety_state=_confirmed_disarm_state(
                    final_status, stop_errors
                ),
                final_evidence={
                    "hardware": final_status,
                    "stop_errors": stop_errors,
                    "recovery_motion_authorized": False,
                },
            )
        except Exception as exc:  # noqa: BLE001 - posture must fail closed
            stop_errors = await self._safe_stop()
            final_status = self._hardware.status()
            return self._results.seal_recovery(
                run_id,
                recovery_id,
                outcome="FAILED",
                reason="RECOVERY_FAILURE",
                message=str(exc),
                final_safety_state=_confirmed_disarm_state(
                    final_status, stop_errors
                ),
                final_evidence={
                    "hardware": final_status,
                    "stop_errors": stop_errors,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
        return await self.run(run_id, recovery_id, automatic=True)

    def validate(
        self, run: dict[str, Any], *, automatic: bool = False
    ) -> tuple[int, str]:
        if run.get("outcome") != "FAILED":
            raise HardwareUnavailable("only failed Demo Runs can be recovered")
        if run.get("final_safety_state") != "DISARMED_CONFIRMED":
            raise HardwareUnavailable(
                "failed run does not have a confirmed disarmed terminal state"
            )
        if automatic and run.get("failed_phase") != "approach_fruit":
            raise AutomaticRecoveryIneligible(
                "automatic recovery is limited to approach_fruit failures"
            )
        home = run.get("home")
        if automatic:
            home = _validate_captured_home(home)
            guard_blocker = self._automatic_guard_blocker()
            if guard_blocker is not None:
                raise AutomaticRecoveryIneligible(guard_blocker)
        elif not isinstance(home, dict):
            raise HardwareUnavailable("failed run has no captured Home pose")
        assert isinstance(home, dict)
        pulse_count, pulse_source = recoverable_forward_pulses(run)
        status = self._hardware.status()
        blockers = _recovery_blockers(status)
        if blockers:
            error_type = (
                AutomaticRecoveryIneligible if automatic else HardwareUnavailable
            )
            raise error_type("; ".join(blockers))
        if automatic:
            estimate = _home_estimate_from_status(home, status, self._hardware)
            if _trusted_home_distance(estimate) is None:
                raise AutomaticRecoveryIneligible(
                    "captured Home cannot be associated with a fresh trusted pose"
                )
            if pulse_count < 1:
                raise AutomaticRecoveryIneligible(
                    "failed approach recorded no forward pulse evidence"
                )
            if pulse_count > MAXIMUM_RECOVERABLE_OUTBOUND_PULSES:
                raise AutomaticRecoveryIneligible(
                    "failed approach forward pulse evidence exceeds its recovery bound"
                )
        if pulse_count < 1:
            home_estimate = _home_estimate_from_status(home, status, self._hardware)
            home_distance = _trusted_home_distance(home_estimate)
            if home_distance is None or home_distance > HOME_POSITION_TOLERANCE_M:
                raise HardwareUnavailable(
                    "failed run is outside Home and recorded no forward approach "
                    "commands to bound recovery"
                )
        return pulse_count, pulse_source

    async def run(
        self, run_id: str, recovery_id: str, *, automatic: bool = False
    ) -> dict[str, Any]:
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
            current_status = self._hardware.status()
            blockers = _recovery_blockers(current_status)
            if automatic:
                blockers.extend(_continuous_fusion_blockers(current_status))
                guard_blocker = self._automatic_guard_blocker()
                if guard_blocker is not None:
                    blockers.append(guard_blocker)
            if blockers:
                error_type = (
                    AutomaticRecoveryIneligible
                    if automatic
                    else HardwareUnavailable
                )
                raise error_type("; ".join(blockers))

            if pulse_count < 1:
                stop_errors = await self._safe_stop()
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
                    final_safety_state=_confirmed_disarm_state(
                        final_status, stop_errors
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
            stop_errors = await self._safe_stop()
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
                final_safety_state=_confirmed_disarm_state(
                    final_status, stop_errors
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
            stop_errors = await self._safe_stop()
            final_status = self._hardware.status()
            self._results.seal_recovery(
                run_id,
                recovery_id,
                outcome="STOPPED",
                reason="OPERATOR_STOP",
                message="operator stopped failed-run Home recovery",
                final_safety_state=_confirmed_disarm_state(
                    final_status, stop_errors
                ),
                final_evidence={
                    "hardware": final_status,
                    "stop_errors": stop_errors,
                },
            )
            raise
        except Exception as exc:  # noqa: BLE001 - recovery must always fail closed
            stop_errors = await self._safe_stop()
            final_status = self._hardware.status()
            return self._results.seal_recovery(
                run_id,
                recovery_id,
                outcome="FAILED",
                reason=(
                    "AUTOMATIC_RECOVERY_SKIPPED"
                    if isinstance(exc, AutomaticRecoveryIneligible)
                    else "RECOVERY_FAILURE"
                ),
                message=str(exc),
                final_safety_state=_confirmed_disarm_state(
                    final_status, stop_errors
                ),
                final_evidence={
                    "hardware": final_status,
                    "stop_errors": stop_errors,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "motion_commands": self._hardware.motion_trace(),
                },
            )
