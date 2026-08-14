from __future__ import annotations

import asyncio
import time

import pytest

from border_collie_demo.config import HardwareConfig
from border_collie_demo.go2_motion import (
    Go2Motion,
    LeaseMismatch,
    MotionConfig,
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
        self.switch_get_delay_s = 0.0

    def SetTimeout(self, value: float) -> None:
        self.timeout_s = value

    def Init(self) -> None:
        return None

    def SwitchSet(self, enabled: bool) -> int:
        self.enabled = enabled
        return 0

    def SwitchGet(self) -> tuple[int, bool]:
        if self.switch_get_delay_s:
            time.sleep(self.switch_get_delay_s)
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


def test_slow_avoidance_verification_is_flagged_and_pauses_stale_motion() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        diagnostics: list[dict[str, object]] = []
        motion = Go2Motion(
            sport,
            avoidance,
            MotionConfig(
                rpc_timeout_s=0.20,
                rpc_slow_threshold_s=0.03,
                avoidance_verify_interval_s=0.001,
                remote_api_settle_s=0.0,
            ),
            diagnostic_sink=diagnostics.append,
        )
        await motion.initialize()
        lease = await motion.arm()
        await asyncio.sleep(0.003)
        moves_before = list(avoidance.moves)
        avoidance.switch_get_delay_s = 0.06

        sent = await motion.command(lease, VelocityCommand(0.50, 0.0, "test"))

        assert sent == VelocityCommand(reason="slow_rpc_pause")
        assert avoidance.moves == moves_before
        assert sport.stop_calls == 1
        assert motion.armed is True
        assert diagnostics == [
            {
                "kind": "motion_rpc_slow",
                "method": "SwitchGet",
                "slow_threshold_s": 0.03,
                "timeout_s": 0.20,
            }
        ]
        assert motion.status()["rpc"]["slow_call_count"] == 1
        await motion.release(lease)
        await motion.close()

    asyncio.run(scenario())


def test_default_motion_rpc_allows_five_seconds_and_flags_after_one() -> None:
    config = MotionConfig()

    assert config.rpc_timeout_s == 5.0
    assert config.rpc_slow_threshold_s == 1.0


def test_motion_rpc_deadlines_are_runtime_environment_settings(monkeypatch) -> None:
    monkeypatch.setenv("BORDER_COLLIE_RPC_TIMEOUT_S", "4.5")
    monkeypatch.setenv("BORDER_COLLIE_RPC_SLOW_THRESHOLD_S", "0.9")

    config = HardwareConfig.from_env()

    assert config.rpc_timeout_s == 4.5
    assert config.rpc_slow_threshold_s == 0.9


def test_motion_rpc_slow_threshold_must_precede_timeout() -> None:
    with pytest.raises(ValueError, match="slow threshold"):
        MotionConfig(rpc_timeout_s=1.0, rpc_slow_threshold_s=1.0)


def test_regular_sports_yaw_lease_rejects_translation_and_never_enables_avoidance() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = Go2Motion(
            sport,
            avoidance,
            MotionConfig(remote_api_settle_s=0.0),
        )
        await motion.initialize()
        lease = await motion.arm_sport_yaw()

        sent = await motion.command(lease, VelocityCommand(0.0, 0.50, "home_turn"))

        assert sent == VelocityCommand(0.0, 0.50, "home_turn")
        assert sport.moves == [(0.0, 0.0, 0.50)]
        assert avoidance.enabled is False
        assert avoidance.remote is False
        assert motion.status()["mode"] == "sport_yaw"
        with pytest.raises(ValueError, match="yaw-only"):
            await motion.command(lease, VelocityCommand(0.50, 0.0, "unsafe"))
        await motion.release(lease)
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
