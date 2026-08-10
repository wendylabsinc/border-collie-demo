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
from border_collie_demo.motion_guardian import MotionAuthority


def authority(*, forward: bool = True, ttl_s: float = 2.0) -> MotionAuthority:
    return MotionAuthority(
        run_id="run-test",
        epoch="epoch-test",
        operation="test",
        allow_forward=forward,
        maximum_forward_mps=1.0,
        maximum_yaw_rps=1.0,
        ttl_s=ttl_s,
    )


class FakeSport:
    def __init__(self) -> None:
        self.timeout_s: float | None = None
        self.stop_calls = 0
        self.stop_result = 0
        self.stand_down_calls = 0
        self.stand_up_calls = 0
        self.balance_stand_calls = 0

    def SetTimeout(self, value: float) -> None:
        self.timeout_s = value

    def Init(self) -> None:
        return None

    def StopMove(self) -> int:
        self.stop_calls += 1
        return self.stop_result

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
        lease = await motion.arm(authority())

        sent = await motion.command(lease, VelocityCommand(0.55, 0.20, "test"))

        assert sent == VelocityCommand(0.55, 0.20, "test")
        assert avoidance.moves[-1] == (0.55, 0.0, 0.20)
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
        (VelocityCommand(0.54, 0.0), "minimum 0.55"),
        (VelocityCommand(1.01, 0.0), "configured limit"),
        (VelocityCommand(0.0, 1.01), "configured limit"),
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
        lease = await motion.arm(authority())
        moves_before = list(avoidance.moves)

        with pytest.raises(ValueError, match=error):
            await motion.command(lease, command)

        assert avoidance.moves == moves_before
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
        lease = await motion.arm(authority())
        await motion.command(lease, VelocityCommand(0.55, 0.0, "test"))

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
        await motion.arm(authority())

        with pytest.raises(LeaseMismatch):
            await motion.command("not-the-lease", VelocityCommand(0.55, 0.0))

        assert all(move[0] == 0.0 for move in avoidance.moves)
        await motion.close()

    asyncio.run(scenario())


def test_expired_authority_stops_hardware_before_rejecting_command() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = Go2Motion(
            sport,
            avoidance,
            MotionConfig(
                command_watchdog_s=1.0,
                remote_api_settle_s=0.0,
            ),
        )
        await motion.initialize()
        lease = await motion.arm(authority(ttl_s=0.01))
        await asyncio.sleep(0.02)

        with pytest.raises(MotionNotReady, match="authority expired"):
            await motion.command(lease, VelocityCommand(0.0, 0.1))

        assert motion.armed is False
        assert sport.stop_calls >= 1
        assert avoidance.moves[-1] == (0.0, 0.0, 0.0)
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
