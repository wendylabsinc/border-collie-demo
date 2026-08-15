from __future__ import annotations

import asyncio

from border_collie_demo.failure_epilogue import PositionOnlyFailureEpilogue


class RecoveryRobot:
    def __init__(
        self,
        distances: list[float],
        *,
        connected: bool = True,
        fault: str | None = None,
        pose_error: str | None = None,
        return_error: str | None = None,
        turn_error: str | None = None,
        final_stop_arms: bool = False,
        posture: str = "standing",
    ) -> None:
        self.distances = iter(distances)
        self._last_distance = distances[-1]
        self.connected = connected
        self.fault = fault
        self.pose_error = pose_error
        self.return_error = return_error
        self.turn_error = turn_error
        self.final_stop_arms = final_stop_arms
        self.posture = posture
        self.armed = False
        self.last_command = {"forward_mps": 0.0, "yaw_rps": 0.0}
        self.stop_calls = 0
        self.return_calls = 0
        self.return_options: dict[str, float] | None = None
        self.turn_calls = 0
        self.posture_calls: list[object] = []
        self.events: list[str] = []

    async def emergency_stop(self) -> list[str]:
        self.stop_calls += 1
        self.armed = self.final_stop_arms and self.stop_calls > 1
        self.last_command = (
            {"forward_mps": 1.0, "yaw_rps": 0.0}
            if self.armed
            else {"forward_mps": 0.0, "yaw_rps": 0.0}
        )
        return []

    def status(self) -> dict[str, object]:
        return {
            "connected": self.connected,
            "fault": self.fault,
            "active_operation": None,
            "posture": self.posture,
            "motion": {
                "initialized": True,
                "armed": self.armed,
                "last_command": dict(self.last_command),
            },
        }

    async def stand_down(self) -> dict[str, object]:
        self.events.append("stand_down")
        self.posture_calls.append("stand_down")
        return {"posture": "stand_down"}

    async def stand_up(self, *, settle_s: float = 1.0) -> dict[str, object]:
        self.events.append("stand_up")
        self.posture_calls.append(("stand_up", settle_s))
        return {"posture": "balance_stand", "settle_s": settle_s}

    def measure_home_position(
        self,
        _home: dict[str, object],
    ) -> dict[str, object]:
        if self.pose_error is not None:
            raise RuntimeError(self.pose_error)
        self._last_distance = next(self.distances, self._last_distance)
        return {
            "home_distance_m": self._last_distance,
            "pose_age_s": 0.02,
            "pose_captured_monotonic_s": 42.0,
            "pose_source": "rt/sportmodestate",
        }

    async def turn_toward_home(
        self,
        _home: dict[str, object],
        **_options: float,
    ) -> dict[str, object]:
        self.turn_calls += 1
        self.events.append("turn_toward_home")
        if self.turn_error is not None:
            raise RuntimeError(self.turn_error)
        return {
            "home_distance_m": self._last_distance,
            "home_bearing_error_rad": 0.0,
            "motion_path": "sport_yaw",
        }

    async def return_home_position(
        self,
        _home: dict[str, object],
        **options: float,
    ) -> dict[str, object]:
        self.return_calls += 1
        self.return_options = options
        self.events.append("return_home_position")
        if self.return_error is not None:
            raise RuntimeError(self.return_error)
        return {
            "home_distance_m": 0.05,
            "heading_restoration_skipped": True,
        }


HOME = {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}


def recover(robot: RecoveryRobot, **overrides: object) -> dict[str, object]:
    arguments = {
        "original_reason": "ARRIVAL_FAILURE",
        "failed_phase": "approach_fruit",
        "takeover_latched": False,
    }
    arguments.update(overrides)
    async def no_wait(_duration_s: float) -> None:
        return None

    return asyncio.run(
        PositionOnlyFailureEpilogue(robot, sleep=no_wait).recover(HOME, **arguments)
    )


def test_failure_after_outbound_motion_gets_one_position_only_return() -> None:
    robot = RecoveryRobot([1.20, 0.05])

    report = recover(robot)

    assert report["status"] == "RETURNED_HOME"
    assert report["attempted_return"] is True
    assert report["attempted_alignment"] is True
    assert report["home_alignment_evidence"]["motion_path"] == "sport_yaw"
    assert report["exact_stop_confirmed"] is True
    assert report["terminal_home_measurement"]["home_distance_m"] == 0.05
    assert robot.return_calls == 1
    assert robot.return_options is not None
    assert robot.return_options["stall_timeout_s"] == 5.0
    assert robot.turn_calls == 1
    assert robot.events == [
        "stand_down",
        "stand_up",
        "turn_toward_home",
        "return_home_position",
    ]
    assert robot.stop_calls == 4
    assert robot.posture_calls == ["stand_down", ("stand_up", 1.0)]
    assert report["posture_evidence"]["bark_played"] is False
    assert robot.armed is False
    assert robot.last_command == {"forward_mps": 0.0, "yaw_rps": 0.0}


def test_failure_already_inside_home_gate_does_not_move() -> None:
    robot = RecoveryRobot([0.08])

    report = recover(robot)

    assert report["status"] == "ALREADY_HOME"
    assert report["attempted_return"] is False
    assert report["terminal_home_measurement"]["home_distance_m"] == 0.08
    assert robot.return_calls == 0
    assert robot.stop_calls == 1


def test_unsafe_hardware_skips_recovery() -> None:
    robot = RecoveryRobot([1.0], fault="motor posture unhealthy")

    report = recover(robot)

    assert report["status"] == "SKIPPED"
    assert "fault" in report["reason"]
    assert robot.return_calls == 0


def test_down_posture_skips_recovery_motion() -> None:
    robot = RecoveryRobot([1.0], posture="down")

    report = recover(robot)

    assert report["status"] == "SKIPPED"
    assert "posture" in report["reason"]
    assert robot.return_calls == 0


def test_takeover_stays_stopped_and_never_returns() -> None:
    robot = RecoveryRobot([1.0])

    report = recover(robot, takeover_latched=True)

    assert report["status"] == "SKIPPED"
    assert report["reason"] == "operator takeover is latched"
    assert robot.return_calls == 0
    assert robot.stop_calls == 1


def test_stale_pose_skips_recovery_fail_closed() -> None:
    robot = RecoveryRobot([1.0], pose_error="pose sample is stale")

    report = recover(robot)

    assert report["status"] == "SKIPPED"
    assert "pose sample is stale" in report["reason"]
    assert robot.return_calls == 0


def test_a_failure_in_return_home_does_not_recurse() -> None:
    robot = RecoveryRobot([1.0])

    report = recover(
        robot,
        original_reason="RETURN_HOME_FAILURE",
        failed_phase="return_home",
    )

    assert report["status"] == "SKIPPED"
    assert "already a Home return" in report["reason"]
    assert robot.return_calls == 0


def test_failed_bounded_return_preserves_a_fresh_terminal_measurement() -> None:
    robot = RecoveryRobot([1.0, 0.65], return_error="return stalled")

    report = recover(robot)

    assert report["status"] == "FAILED"
    assert report["attempted_return"] is True
    assert report["terminal_home_measurement"]["home_distance_m"] == 0.65
    assert robot.return_calls == 1
    assert robot.armed is False


def test_failed_home_alignment_stops_without_attempting_forward_return() -> None:
    robot = RecoveryRobot([1.0, 1.0], turn_error="sport yaw did not respond")

    report = recover(robot)

    assert report["status"] == "FAILED"
    assert report["attempted_alignment"] is True
    assert report["attempted_return"] is False
    assert report["reason"] == (
        "yaw-only Home alignment failed: sport yaw did not respond"
    )
    assert report["exact_stop_confirmed"] is True
    assert report["terminal_home_measurement"]["home_distance_m"] == 1.0
    assert robot.turn_calls == 1
    assert robot.return_calls == 0
    assert robot.armed is False


def test_recovery_fails_if_exact_final_disarm_is_not_confirmed() -> None:
    robot = RecoveryRobot([1.0, 0.05], final_stop_arms=True)

    report = recover(robot)

    assert report["status"] == "FAILED"
    assert report["reason"] == "exact stop after failure posture could not be confirmed"
    assert report["exact_stop_confirmed"] is False
    assert robot.return_calls == 0
