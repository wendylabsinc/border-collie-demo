from __future__ import annotations

import asyncio

import pytest

from border_collie_demo.go2_motion import (
    Go2Motion,
    LeaseMismatch,
    MotionConfig,
    MotionNotReady,
)
from border_collie_demo.models import VelocityCommand


class FakeSport:
    def __init__(self) -> None:
        self.timeout_s: float | None = None
        self.stop_calls = 0
        self.stop_result = 0
        self.stand_down_calls = 0
        self.stand_up_calls = 0
        self.balance_stand_calls = 0
        self.moves: list[tuple[float, float, float]] = []

    def SetTimeout(self, value: float) -> None:
        self.timeout_s = value

    def Init(self) -> None:
        return None

    def StopMove(self) -> int:
        self.stop_calls += 1
        return self.stop_result

    def Move(self, vx: float, vy: float, vyaw: float) -> int:
        self.moves.append((vx, vy, vyaw))
        return 0

    def StandDown(self) -> int:
        self.stand_down_calls += 1
        return 0

    def StandUp(self) -> int:
        self.stand_up_calls += 1
        return 0

    def BalanceStand(self) -> int:
        self.balance_stand_calls += 1
        return 0


class FakeAvoidance:
    def __init__(self) -> None:
        self.timeout_s: float | None = None
        self.enabled = False
        self.remote = False
        self.moves: list[tuple[float, float, float]] = []

    def SetTimeout(self, value: float) -> None:
        self.timeout_s = value

    def Init(self) -> None:
        return None

    def SwitchSet(self, enabled: bool) -> int:
        self.enabled = enabled
        return 0

    def SwitchGet(self) -> tuple[int, bool]:
        return (0, self.enabled)

    def UseRemoteCommandFromApi(self, enabled: bool) -> int:
        self.remote = enabled
        return 0

    def Move(self, vx: float, vy: float, vyaw: float) -> int:
        self.moves.append((vx, vy, vyaw))
        return 0


def test_factory_avoidance_motion_is_exclusive_and_stops_on_release() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = Go2Motion(
            sport,
            avoidance,
            MotionConfig(remote_api_settle_s=0.0),
        )
        await motion.initialize()
        lease = await motion.arm()

        sent = await motion.command(lease, VelocityCommand(0.50, 0.20, "test"))

        assert sent == VelocityCommand(0.50, 0.20, "test")
        assert avoidance.moves[-1] == (0.50, 0.0, 0.20)
        assert motion.armed is True
        await motion.release(lease)
        assert sport.stop_calls == 1
        assert avoidance.moves[-1] == (0.0, 0.0, 0.0)
        assert avoidance.remote is False
        assert avoidance.enabled is False
        assert motion.armed is False
        await motion.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "command,error",
    [
        (VelocityCommand(-0.01, 0.0), "non-negative"),
        (VelocityCommand(1.01, 0.0), "configured limit"),
        (VelocityCommand(0.0, 0.81), "configured limit"),
    ],
)
def test_unsafe_velocity_is_rejected_before_hardware(
    command: VelocityCommand, error: str
) -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = Go2Motion(
            sport,
            avoidance,
            MotionConfig(remote_api_settle_s=0.0),
        )
        await motion.initialize()
        lease = await motion.arm()
        moves_before = list(avoidance.moves)

        with pytest.raises(ValueError, match=error):
            await motion.command(lease, command)

        assert avoidance.moves == moves_before
        await motion.close()

    asyncio.run(scenario())


def test_step_back_suspends_avoidance_reverses_through_sport_then_restores() -> None:
    """The avoidance MODULE is robot-global and vetoes reverse from any client.

    Two supervised runs on 2026-08-08 measured -0.001 m over five accepted
    reverse commands — first through the avoidance client, then through the
    direct SportClient with the module still engaged. The bounded clearance
    step must therefore record the prior module state, switch the module
    off, reverse through the direct SportClient only, and always re-engage
    and confirm the module afterwards.
    """

    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = Go2Motion(
            sport,
            avoidance,
            MotionConfig(remote_api_settle_s=0.0, avoidance_switch_settle_s=0.0),
        )
        await motion.initialize()
        lease = await motion.arm()
        assert avoidance.enabled is True

        suspend = await motion.suspend_avoidance_for_step_back(lease)
        assert suspend["avoidance_prior_enabled"] is True
        assert avoidance.enabled is False
        assert motion.status()["avoidance_suspended"] is True

        held = await motion.step_back_hold(lease, 1.0, 0.02)
        reverse_moves = [move for move in sport.moves if move[0] < 0.0]
        assert reverse_moves == [(-1.0, 0.0, 0.0)]
        assert held["hold_pattern"] == "single_setpoint_hold"
        assert held["hold_move_commands"] == 1
        assert held["stop_issued_by"] == "window_end"
        assert held["stop_confirmed"] is True
        assert held["watchdog_renewed_without_resend"] is True
        assert sport.stop_calls == 1
        assert all(move[0] >= 0.0 for move in avoidance.moves)

        resume = await motion.resume_avoidance_after_step_back(lease)
        assert resume["avoidance_restored"] is True
        assert resume["avoidance_switched_off_s"] >= 0.0
        assert avoidance.enabled is True
        assert sport.stop_calls >= 2
        assert motion.status()["avoidance_suspended"] is False
        assert motion.armed is True

        await motion.release(lease)
        assert avoidance.moves[-1] == (0.0, 0.0, 0.0)
        assert all(move[0] >= 0.0 for move in avoidance.moves)
        assert motion.armed is False
        await motion.close()

    asyncio.run(scenario())


def test_step_back_reverse_requires_the_avoidance_off_window() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = Go2Motion(
            sport,
            avoidance,
            MotionConfig(remote_api_settle_s=0.0, avoidance_switch_settle_s=0.0),
        )
        await motion.initialize()
        lease = await motion.arm()

        with pytest.raises(MotionNotReady, match="suspended first"):
            await motion.step_back_hold(lease, 0.5, 0.1)

        assert sport.moves == []
        await motion.close()

    asyncio.run(scenario())


def test_step_back_hold_keeps_one_setpoint_and_the_watchdog_stays_quiet() -> None:
    """The hold must not re-send the setpoint, and the watchdog must not
    fire mid-hold.

    r5 (run 801b4a01) re-sent the reverse setpoint every 0.1 s and measured
    0.007 m: each re-send restarts gait initiation so the step never
    plants. The proven pattern is ONE Move, a silent hold, then StopMove.
    A silent hold longer than command_watchdog_s would normally trip the
    watchdog, so the hold renews the watchdog timer without emitting a new
    setpoint.
    """

    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = Go2Motion(
            sport,
            avoidance,
            MotionConfig(
                command_watchdog_s=0.06,
                remote_api_settle_s=0.0,
                avoidance_switch_settle_s=0.0,
            ),
        )
        await motion.initialize()
        lease = await motion.arm()
        await motion.suspend_avoidance_for_step_back(lease)
        moves_before = len(sport.moves)

        # Hold for 2.5x the watchdog window.
        held = await motion.step_back_hold(lease, 0.5, 0.15)

        reverse_moves = [move for move in sport.moves if move[0] < 0.0]
        assert len(sport.moves) - moves_before == 1
        assert reverse_moves == [(-0.5, 0.0, 0.0)]
        assert held["watchdog_renewals"] >= 3
        # Exactly one StopMove, issued at the window end - the watchdog did
        # not fire an emergency stop mid-hold and the lease survived.
        assert sport.stop_calls == 1
        assert held["stop_issued_by"] == "window_end"
        assert motion.armed is True
        await motion.close()

    asyncio.run(scenario())


def test_general_velocity_commands_are_blocked_while_avoidance_is_suspended() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = Go2Motion(
            sport,
            avoidance,
            MotionConfig(remote_api_settle_s=0.0, avoidance_switch_settle_s=0.0),
        )
        await motion.initialize()
        lease = await motion.arm()
        await motion.suspend_avoidance_for_step_back(lease)
        moves_before = list(avoidance.moves)

        with pytest.raises(MotionNotReady, match="blocked while avoidance"):
            await motion.command(lease, VelocityCommand(0.5, 0.0, "test"))

        assert avoidance.moves == moves_before
        await motion.close()

    asyncio.run(scenario())


def test_failed_avoidance_restore_is_a_hard_fault() -> None:
    class WedgedAvoidance(FakeAvoidance):
        wedged = False

        def SwitchSet(self, enabled: bool) -> int:
            if enabled and self.wedged:
                return 3203
            return super().SwitchSet(enabled)

    async def scenario() -> None:
        sport = FakeSport()
        avoidance = WedgedAvoidance()
        motion = Go2Motion(
            sport,
            avoidance,
            MotionConfig(remote_api_settle_s=0.0, avoidance_switch_settle_s=0.0),
        )
        await motion.initialize()
        lease = await motion.arm()
        await motion.suspend_avoidance_for_step_back(lease)
        await motion.step_back_hold(lease, 0.5, 0.01)
        avoidance.wedged = True

        with pytest.raises(MotionNotReady, match="avoidance restore failed"):
            await motion.resume_avoidance_after_step_back(lease)

        # The fault is latched: the adapter is disarmed and cannot be
        # re-armed without recovering from the fault.
        assert motion.armed is False
        with pytest.raises(MotionNotReady, match="avoidance restore failed"):
            await motion.arm()
        await motion.close()

    asyncio.run(scenario())


def test_step_back_rejects_unsafe_speeds_and_stale_leases() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = Go2Motion(
            sport,
            avoidance,
            MotionConfig(remote_api_settle_s=0.0),
        )
        await motion.initialize()
        lease = await motion.arm()
        avoidance_before = list(avoidance.moves)
        sport_before = list(sport.moves)

        with pytest.raises(ValueError, match="finite and positive"):
            await motion.step_back_hold(lease, 0.0, 0.1)
        with pytest.raises(ValueError, match="finite and positive"):
            await motion.step_back_hold(lease, -0.5, 0.1)
        with pytest.raises(ValueError, match="configured limit"):
            await motion.step_back_hold(lease, 1.01, 0.1)
        with pytest.raises(ValueError, match="duration must be positive"):
            await motion.step_back_hold(lease, 0.5, 0.0)
        with pytest.raises(LeaseMismatch):
            await motion.step_back_hold("not-the-lease", 0.5, 0.1)

        assert avoidance.moves == avoidance_before
        assert sport.moves == sport_before
        await motion.close()

    asyncio.run(scenario())


def test_stale_command_watchdog_brakes_and_revokes_lease() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = Go2Motion(
            sport,
            avoidance,
            MotionConfig(
                command_watchdog_s=0.03,
                remote_api_settle_s=0.0,
            ),
        )
        await motion.initialize()
        lease = await motion.arm()
        await motion.command(lease, VelocityCommand(0.50, 0.0, "test"))

        await asyncio.sleep(0.08)

        assert motion.armed is False
        assert sport.stop_calls >= 1
        assert avoidance.moves[-1] == (0.0, 0.0, 0.0)
        await motion.close()

    asyncio.run(scenario())


def test_stale_lease_cannot_command_motion() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = Go2Motion(
            sport,
            avoidance,
            MotionConfig(remote_api_settle_s=0.0),
        )
        await motion.initialize()
        await motion.arm()

        with pytest.raises(LeaseMismatch):
            await motion.command("not-the-lease", VelocityCommand(0.50, 0.0))

        assert all(move[0] == 0.0 for move in avoidance.moves)
        await motion.close()

    asyncio.run(scenario())


def test_stand_up_settles_before_and_after_balance_stand() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = Go2Motion(
            sport,
            avoidance,
            MotionConfig(remote_api_settle_s=0.0),
        )
        sleeps: list[float] = []

        async def record_sleep(duration_s: float) -> None:
            sleeps.append(duration_s)

        motion._sleep = record_sleep
        await motion.initialize()

        await motion.stand_up(settle_s=1.0)

        assert sport.stand_up_calls == 1
        assert sport.balance_stand_calls == 1
        assert sleeps == [1.0, 1.0]
        assert motion.armed is False
        await motion.close()

    asyncio.run(scenario())
