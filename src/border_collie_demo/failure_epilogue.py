"""One bounded, capability-gated Home recovery after a failed Demo Run."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol


class PositionHomeRobot(Protocol):
    async def emergency_stop(self) -> list[str]: ...

    async def stand_down(self) -> dict[str, object]: ...

    async def stand_up(self, *, settle_s: float = 1.0) -> dict[str, object]: ...

    def status(self) -> dict[str, object]: ...

    def measure_home_position(
        self,
        home: dict[str, object],
    ) -> dict[str, object]: ...

    async def return_home_position(
        self,
        home: dict[str, object],
        *,
        forward_mps: float,
        arrival_tolerance_m: float,
        heading_gate_rad: float,
        maximum_yaw_rps: float,
        minimum_progress_m: float,
        stall_timeout_s: float,
        timeout_s: float,
    ) -> dict[str, object]: ...


@dataclass(frozen=True)
class FailureEpilogueReport:
    status: str
    reason: str
    attempted_return: bool
    exact_stop_confirmed: bool
    terminal_home_measurement: dict[str, object] | None = None
    return_evidence: dict[str, object] | None = None
    posture_evidence: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PositionOnlyFailureEpilogue:
    """Recover failed runs without changing their original outcome.

    The interface intentionally accepts no outbound pulse count, breadcrumb,
    fused pose, LiDAR sample, or desired final heading. Fresh measured Go2 pose
    is the only source of Home distance and one bounded attempt is the maximum.
    """

    def __init__(
        self,
        robot: PositionHomeRobot,
        *,
        arrival_tolerance_m: float = 0.10,
        forward_mps: float = 1.0,
        heading_gate_rad: float = 0.3490658503988659,
        maximum_yaw_rps: float = 0.50,
        minimum_progress_m: float = 0.03,
        stall_timeout_s: float = 2.0,
        timeout_s: float = 30.0,
        down_hold_s: float = 5.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._robot = robot
        self._arrival_tolerance_m = arrival_tolerance_m
        self._down_hold_s = down_hold_s
        self._sleep = sleep
        self._return_options = {
            "forward_mps": forward_mps,
            "arrival_tolerance_m": arrival_tolerance_m,
            "heading_gate_rad": heading_gate_rad,
            "maximum_yaw_rps": maximum_yaw_rps,
            "minimum_progress_m": minimum_progress_m,
            "stall_timeout_s": stall_timeout_s,
            "timeout_s": timeout_s,
        }

    async def recover(
        self,
        home: dict[str, Any] | None,
        *,
        original_reason: str,
        failed_phase: str,
        takeover_latched: bool,
    ) -> dict[str, object]:
        stop_errors = await self._robot.emergency_stop()
        if stop_errors:
            return self._report(
                "SKIPPED",
                "initial exact stop could not be confirmed: " + "; ".join(stop_errors),
                exact_stop_confirmed=False,
            )
        if takeover_latched:
            return self._report(
                "SKIPPED",
                "operator takeover is latched",
                exact_stop_confirmed=self._is_exactly_disarmed(),
            )
        if original_reason == "RETURN_HOME_FAILURE" or failed_phase in {
            "turn_toward_home",
            "return_home",
            "restore_heading",
        }:
            return self._report(
                "SKIPPED",
                "the failing operation was already a Home return",
                exact_stop_confirmed=self._is_exactly_disarmed(),
            )
        if home is None:
            return self._report(
                "SKIPPED",
                "Home was not captured",
                exact_stop_confirmed=self._is_exactly_disarmed(),
            )
        unsafe_reason = self._unsafe_reason()
        if unsafe_reason is not None:
            return self._report(
                "SKIPPED",
                unsafe_reason,
                exact_stop_confirmed=self._is_exactly_disarmed(),
            )
        try:
            before = self._robot.measure_home_position(home)
        except Exception as exc:  # noqa: BLE001 - hardware adapter seam
            return self._report(
                "SKIPPED",
                f"fresh Home pose is unavailable: {exc}",
                exact_stop_confirmed=self._is_exactly_disarmed(),
            )
        distance = before.get("home_distance_m")
        if (
            isinstance(distance, (int, float))
            and not isinstance(distance, bool)
            and float(distance) <= self._arrival_tolerance_m
        ):
            return self._report(
                "ALREADY_HOME",
                "fresh measured pose is already inside the Home gate",
                exact_stop_confirmed=self._is_exactly_disarmed(),
                terminal_home_measurement=before,
            )
        posture_evidence: dict[str, object]
        try:
            down = await self._robot.stand_down()
            await self._sleep(self._down_hold_s)
            up = await self._robot.stand_up(settle_s=1.0)
            posture_evidence = {
                "stand_down": down,
                "down_hold_s": self._down_hold_s,
                "stand_up": up,
                "bark_played": False,
            }
        except Exception as exc:  # noqa: BLE001 - preserve original failure
            stop_errors = await self._robot.emergency_stop()
            return self._report(
                "FAILED",
                f"failure posture sequence failed: {exc}",
                exact_stop_confirmed=(
                    not stop_errors and self._is_exactly_disarmed()
                ),
            )
        posture_stop_errors = await self._robot.emergency_stop()
        if posture_stop_errors or not self._is_exactly_disarmed():
            return self._report(
                "FAILED",
                "exact stop after failure posture could not be confirmed",
                exact_stop_confirmed=False,
                posture_evidence=posture_evidence,
            )
        try:
            self._robot.measure_home_position(home)
        except Exception as exc:  # noqa: BLE001 - moving on stale pose is unsafe
            return self._report(
                "FAILED",
                f"fresh Home pose after failure posture is unavailable: {exc}",
                exact_stop_confirmed=True,
                posture_evidence=posture_evidence,
            )
        try:
            returned = await self._robot.return_home_position(
                home,
                **self._return_options,
            )
        except Exception as exc:  # noqa: BLE001 - recovery must preserve failure
            stop_errors = await self._robot.emergency_stop()
            measurement = self._measure_if_safe(home)
            return self._report(
                "FAILED",
                f"bounded position-only return failed: {exc}",
                attempted_return=True,
                exact_stop_confirmed=(
                    not stop_errors and self._is_exactly_disarmed()
                ),
                terminal_home_measurement=measurement,
                posture_evidence=posture_evidence,
            )

        final_stop_errors = await self._robot.emergency_stop()
        exact_stop = not final_stop_errors and self._is_exactly_disarmed()
        measurement = self._measure_if_safe(home)
        if not exact_stop:
            return self._report(
                "FAILED",
                "final exact stop/disarm could not be confirmed",
                attempted_return=True,
                exact_stop_confirmed=False,
                terminal_home_measurement=measurement,
                return_evidence=returned,
                posture_evidence=posture_evidence,
            )
        if measurement is None:
            return self._report(
                "FAILED",
                "fresh terminal Home measurement is unavailable",
                attempted_return=True,
                exact_stop_confirmed=True,
                return_evidence=returned,
                posture_evidence=posture_evidence,
            )
        terminal_distance = measurement.get("home_distance_m")
        if not isinstance(terminal_distance, (int, float)) or isinstance(
            terminal_distance, bool
        ) or float(terminal_distance) > self._arrival_tolerance_m:
            return self._report(
                "FAILED",
                "bounded return ended outside the measured Home gate",
                attempted_return=True,
                exact_stop_confirmed=True,
                terminal_home_measurement=measurement,
                return_evidence=returned,
                posture_evidence=posture_evidence,
            )
        return self._report(
            "RETURNED_HOME",
            "one bounded position-only return reached the measured Home gate",
            attempted_return=True,
            exact_stop_confirmed=True,
            terminal_home_measurement=measurement,
            return_evidence=returned,
            posture_evidence=posture_evidence,
        )

    def _unsafe_reason(self) -> str | None:
        status = self._robot.status()
        if status.get("connected") is not True:
            return "robot hardware or fresh pose is not healthy"
        if status.get("fault") not in (None, ""):
            return f"robot hardware fault is present: {status.get('fault')}"
        if status.get("active_operation") is not None:
            return "another hardware operation is active"
        motion = status.get("motion")
        if not isinstance(motion, dict) or motion.get("initialized") is not True:
            return "motion client is not healthy"
        if status.get("posture") == "down":
            return "robot posture is not standing for recovery motion"
        if not self._is_exactly_disarmed(status):
            return "motion is not exactly stopped and disarmed"
        return None

    def _is_exactly_disarmed(self, status: dict[str, object] | None = None) -> bool:
        status = self._robot.status() if status is None else status
        motion = status.get("motion")
        if not isinstance(motion, dict) or motion.get("armed") is not False:
            return False
        command = motion.get("last_command")
        if command is None:
            return True
        return bool(
            isinstance(command, dict)
            and command.get("forward_mps") == 0.0
            and command.get("yaw_rps") == 0.0
        )

    def _measure_if_safe(
        self,
        home: dict[str, Any],
    ) -> dict[str, object] | None:
        try:
            return self._robot.measure_home_position(home)
        except Exception:  # noqa: BLE001 - terminal evidence is best effort
            return None

    @staticmethod
    def _report(
        status: str,
        reason: str,
        *,
        attempted_return: bool = False,
        exact_stop_confirmed: bool,
        terminal_home_measurement: dict[str, object] | None = None,
        return_evidence: dict[str, object] | None = None,
        posture_evidence: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return FailureEpilogueReport(
            status=status,
            reason=reason,
            attempted_return=attempted_return,
            exact_stop_confirmed=exact_stop_confirmed,
            terminal_home_measurement=terminal_home_measurement,
            return_evidence=return_evidence,
            posture_evidence=posture_evidence,
        ).to_dict()
