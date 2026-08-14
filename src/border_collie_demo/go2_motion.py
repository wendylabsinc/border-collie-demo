"""Exclusive, watchdog-protected Unitree motion boundary.

This is a deliberately reduced adaptation of the motion boundary physically
tested in ``wendylabsinc/collie-demo``. The clean foundation exposes only the
factory ``ObstaclesAvoidClient`` path. Direct SportClient translation remains
out of scope until return-home owns a qualified collision-planning contract.
"""

from __future__ import annotations

import asyncio
import math
import secrets
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from .models import VelocityCommand


class MotionError(RuntimeError):
    pass


class MotionNotReady(MotionError):
    pass


class LeaseMismatch(MotionError):
    pass


class SportClientProtocol(Protocol):
    def SetTimeout(self, timeout_s: float) -> Any: ...

    def Init(self) -> Any: ...

    def StandUp(self) -> int: ...

    def BalanceStand(self) -> int: ...

    def StandDown(self) -> int: ...

    def StopMove(self) -> int: ...

    def Move(self, vx: float, vy: float, vyaw: float) -> int: ...


class AvoidanceClientProtocol(Protocol):
    def SetTimeout(self, timeout_s: float) -> Any: ...

    def Init(self) -> Any: ...

    def SwitchSet(self, enabled: bool) -> int: ...

    def SwitchGet(self) -> tuple[int, bool]: ...

    def UseRemoteCommandFromApi(self, enabled: bool) -> int: ...

    def Move(self, vx: float, vy: float, vyaw: float) -> int: ...


@dataclass(frozen=True)
class MotionConfig:
    maximum_forward_mps: float = 1.0
    maximum_yaw_rps: float = 0.80
    command_watchdog_s: float = 0.35
    rpc_timeout_s: float = 5.0
    rpc_slow_threshold_s: float = 1.0
    client_timeout_s: float = 12.0
    avoidance_verify_interval_s: float = 0.30
    remote_api_settle_s: float = 0.50

    def __post_init__(self) -> None:
        for name in (
            "maximum_forward_mps",
            "maximum_yaw_rps",
            "command_watchdog_s",
            "rpc_timeout_s",
            "rpc_slow_threshold_s",
            "client_timeout_s",
            "avoidance_verify_interval_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.remote_api_settle_s < 0.0:
            raise ValueError("remote_api_settle_s must be non-negative")
        if self.rpc_slow_threshold_s >= self.rpc_timeout_s:
            raise ValueError("RPC slow threshold must be less than the RPC timeout")


class Go2Motion:
    def __init__(
        self,
        sport: SportClientProtocol,
        avoidance: AvoidanceClientProtocol,
        config: MotionConfig | None = None,
        *,
        diagnostic_sink: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.sport = sport
        self.avoidance = avoidance
        self.config = config or MotionConfig()
        self._diagnostic_sink = diagnostic_sink
        self._lock = asyncio.Lock()
        self._rpc_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="border-collie-sdk"
        )
        self._stop_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="border-collie-stop"
        )
        self._initialized = False
        self._closed = False
        self._fault: str | None = None
        self._lease: str | None = None
        self._mode: str | None = None
        self._avoidance_enabled = False
        self._remote_api_enabled = False
        self._last_verify_at: float | None = None
        self._last_command = VelocityCommand(reason="disarmed")
        self._watchdog: asyncio.Task[None] | None = None
        self._watchdog_generation = 0
        self._sleep = asyncio.sleep
        self._slow_call_count = 0
        self._last_rpc_diagnostic: dict[str, object] | None = None

    @property
    def armed(self) -> bool:
        common = bool(
            self._initialized
            and not self._closed
            and self._fault is None
            and self._lease
        )
        if self._mode == "factory_avoidance":
            return common and self._avoidance_enabled and self._remote_api_enabled
        if self._mode == "sport_yaw":
            return common and not self._avoidance_enabled and not self._remote_api_enabled
        return False

    def set_diagnostic_sink(
        self,
        sink: Callable[[dict[str, object]], None] | None,
    ) -> None:
        self._diagnostic_sink = sink

    def status(self) -> dict[str, object]:
        return {
            "initialized": self._initialized,
            "closed": self._closed,
            "armed": self.armed,
            "mode": self._mode if self._lease else None,
            "fault": self._fault,
            "avoidance_enabled": self._avoidance_enabled,
            "remote_api_enabled": self._remote_api_enabled,
            "watchdog_s": self.config.command_watchdog_s,
            "rpc": {
                "timeout_s": self.config.rpc_timeout_s,
                "slow_threshold_s": self.config.rpc_slow_threshold_s,
                "slow_call_count": self._slow_call_count,
                "last_diagnostic": (
                    dict(self._last_rpc_diagnostic)
                    if self._last_rpc_diagnostic is not None
                    else None
                ),
            },
            "limits": {
                "forward_mps": self.config.maximum_forward_mps,
                "yaw_rps": self.config.maximum_yaw_rps,
                "lateral_mps": 0.0,
                "reverse_allowed": False,
            },
            "last_command": self._last_command.to_dict(),
        }

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            if self._closed:
                raise MotionNotReady("motion adapter is closed")
            try:
                await self._call(self.sport.SetTimeout, self.config.client_timeout_s)
                await self._call(self.sport.Init)
                await self._call(
                    self.avoidance.SetTimeout, self.config.client_timeout_s
                )
                await self._call(self.avoidance.Init)
            except Exception as exc:
                self._fault = f"initialization failed: {exc}"
                raise MotionNotReady(self._fault) from exc
            self._initialized = True

    async def arm(self) -> str:
        async with self._lock:
            self._require_ready()
            if self._lease is not None:
                raise MotionNotReady("motion lease already active")
            try:
                await self._success(self.avoidance.SwitchSet, True)
                response = await self._call(self.avoidance.SwitchGet)
                if response != (0, True):
                    raise MotionError(f"avoidance not confirmed: {response!r}")
                self._avoidance_enabled = True
                self._last_verify_at = time.monotonic()
                await self._success(self.avoidance.UseRemoteCommandFromApi, True)
                self._remote_api_enabled = True
                if self.config.remote_api_settle_s:
                    await asyncio.sleep(self.config.remote_api_settle_s)
                await self._success(self.avoidance.Move, 0.0, 0.0, 0.0)
            except Exception as exc:
                await self._release_locked(use_stop=True)
                raise MotionNotReady(f"arm failed: {exc}") from exc
            self._lease = secrets.token_urlsafe(32)
            self._mode = "factory_avoidance"
            self._last_command = VelocityCommand(reason="armed_zero")
            return self._lease

    async def arm_sport_yaw(self) -> str:
        """Own a watchdog-protected regular SportClient yaw-only lease."""
        async with self._lock:
            self._require_ready()
            if self._lease is not None:
                raise MotionNotReady("motion lease already active")
            try:
                await self._disable_avoidance_locked()
                await self._idle_stop()
            except Exception as exc:
                await self._release_locked(use_stop=True)
                raise MotionNotReady(f"sport yaw arm failed: {exc}") from exc
            self._lease = secrets.token_urlsafe(32)
            self._mode = "sport_yaw"
            self._last_command = VelocityCommand(reason="sport_yaw_armed_zero")
            return self._lease

    async def command(self, lease: str, command: VelocityCommand) -> VelocityCommand:
        forward = self._bounded_forward(command.forward_mps)
        yaw = self._bounded_yaw(command.yaw_rps)
        async with self._lock:
            self._require_owner(lease)
            self._cancel_watchdog()
            try:
                if self._mode == "sport_yaw":
                    if forward != 0.0:
                        raise ValueError("regular SportClient lease is yaw-only")
                    await self._success(self.sport.Move, 0.0, 0.0, yaw)
                else:
                    verification_slow = False
                    if forward != 0.0 or yaw != 0.0:
                        verification_slow = await self._verify_avoidance_if_due()
                    if verification_slow:
                        forward = 0.0
                        yaw = 0.0
                        command = VelocityCommand(reason="slow_rpc_pause")
                    else:
                        await self._success(self.avoidance.Move, forward, 0.0, yaw)
            except ValueError:
                raise
            except Exception as exc:
                self._fault = f"velocity command failed: {exc}"
                await self._release_locked(use_stop=True)
                raise MotionNotReady(self._fault) from exc
            self._last_command = VelocityCommand(forward, yaw, command.reason)
            if forward != 0.0 or yaw != 0.0:
                self._arm_watchdog()
            return self._last_command

    async def release(self, lease: str) -> None:
        async with self._lock:
            self._require_owner(lease)
            errors = await self._release_locked(use_stop=True)
            if errors:
                raise MotionError("; ".join(errors))

    async def emergency_stop(self) -> list[str]:
        self._cancel_watchdog()
        errors: list[str] = []
        was_active = bool(
            self._lease is not None
            or self._last_command.forward_mps != 0.0
            or self._last_command.yaw_rps != 0.0
        )
        if self._initialized and not self._closed:
            try:
                result = await self._call_stop(self.sport.StopMove)
                already_stopped = result == -1 and not was_active
                if result != 0 and not already_stopped:
                    raise MotionError(f"StopMove returned {result!r}")
            except Exception as exc:  # noqa: BLE001 - SDK stop failures are untyped
                errors.append(f"StopMove: {exc}")
                if self._fault is None:
                    self._fault = f"StopMove failed: {exc}"
        async with self._lock:
            errors.extend(await self._release_locked(use_stop=False))
        return errors

    async def stand_down(self) -> None:
        await self._posture_action("StandDown", self.sport.StandDown)

    async def stand_up(self, *, settle_s: float = 1.0) -> None:
        settle = float(settle_s)
        if not math.isfinite(settle) or settle < 0.0:
            raise ValueError("StandUp settle time must be finite and non-negative")
        async with self._lock:
            self._require_ready()
            if self._lease is not None:
                raise MotionNotReady("motion lease already active")
            self._cancel_watchdog()
            try:
                await self._disable_avoidance_locked()
                await self._idle_stop()
                await self._success(
                    self.sport.StandUp,
                    timeout_s=self.config.client_timeout_s,
                )
                if settle:
                    await self._sleep(settle)
                await self._success(
                    self.sport.BalanceStand,
                    timeout_s=self.config.client_timeout_s,
                )
                if settle:
                    await self._sleep(settle)
            except Exception as exc:
                await self._release_locked(use_stop=True)
                raise MotionNotReady(f"StandUp failed: {exc}") from exc
            self._last_command = VelocityCommand(reason="stand_up_complete")

    async def close(self) -> list[str]:
        if self._closed:
            return []
        errors = await self.emergency_stop()
        self._closed = True
        self._rpc_executor.shutdown(wait=False, cancel_futures=False)
        self._stop_executor.shutdown(wait=False, cancel_futures=False)
        return errors

    async def _posture_action(self, label: str, method: Any) -> None:
        async with self._lock:
            self._require_ready()
            if self._lease is not None:
                raise MotionNotReady("motion lease already active")
            self._cancel_watchdog()
            try:
                await self._disable_avoidance_locked()
                await self._idle_stop()
                await self._success(method, timeout_s=self.config.client_timeout_s)
            except Exception as exc:
                await self._release_locked(use_stop=True)
                raise MotionNotReady(f"{label} failed: {exc}") from exc
            self._last_command = VelocityCommand(reason=f"{label.lower()}_complete")

    async def _disable_avoidance_locked(self) -> None:
        await self._success(self.avoidance.UseRemoteCommandFromApi, False)
        await self._success(self.avoidance.SwitchSet, False)
        self._avoidance_enabled = False
        self._remote_api_enabled = False
        self._last_verify_at = None

    async def _verify_avoidance_if_due(self) -> bool:
        if (
            self._last_verify_at is not None
            and time.monotonic() - self._last_verify_at
            < self.config.avoidance_verify_interval_s
        ):
            return False
        response, slow = await self._call_timed(
            self.avoidance.SwitchGet,
            on_slow=self._stop_for_slow_verification,
        )
        if response != (0, True):
            self._avoidance_enabled = False
            raise MotionError(f"avoidance switched off: {response!r}")
        self._last_verify_at = time.monotonic()
        return slow

    async def _stop_for_slow_verification(self) -> None:
        result = await self._call_stop(self.sport.StopMove)
        if result not in (0, -1):
            raise MotionError(f"StopMove returned {result!r}")

    async def _release_locked(self, *, use_stop: bool) -> list[str]:
        self._cancel_watchdog()
        errors: list[str] = []
        calls: list[tuple[str, Any, tuple[Any, ...], bool]] = []
        if self._initialized and not self._closed and use_stop:
            calls.append(("StopMove", self.sport.StopMove, (), True))
        if self._initialized and not self._closed:
            calls.extend(
                [
                    (
                        "avoidance zero",
                        self.avoidance.Move,
                        (0.0, 0.0, 0.0),
                        False,
                    ),
                    (
                        "remote API disable",
                        self.avoidance.UseRemoteCommandFromApi,
                        (False,),
                        False,
                    ),
                    (
                        "avoidance disable",
                        self.avoidance.SwitchSet,
                        (False,),
                        False,
                    ),
                ]
            )
        for label, method, args, emergency in calls:
            try:
                if emergency:
                    result = await self._call_stop(method, *args)
                else:
                    result = await self._call(method, *args)
                if result != 0:
                    raise MotionError(f"returned {result!r}")
            except Exception as exc:  # noqa: BLE001 - best-effort safety release
                errors.append(f"{label}: {exc}")
        self._lease = None
        self._mode = None
        self._avoidance_enabled = False
        self._remote_api_enabled = False
        self._last_verify_at = None
        self._last_command = VelocityCommand(reason="released")
        return errors

    def _require_ready(self) -> None:
        if not self._initialized:
            raise MotionNotReady("motion adapter is not initialized")
        if self._closed:
            raise MotionNotReady("motion adapter is closed")
        if self._fault is not None:
            raise MotionNotReady(self._fault)

    def _require_owner(self, lease: str) -> None:
        self._require_ready()
        if not self.armed:
            raise MotionNotReady("motion is disarmed")
        if (
            not isinstance(lease, str)
            or self._lease is None
            or not secrets.compare_digest(lease, self._lease)
        ):
            raise LeaseMismatch("motion lease is stale or does not match")

    def _arm_watchdog(self) -> None:
        self._cancel_watchdog()
        generation = self._watchdog_generation

        async def expire() -> None:
            try:
                await asyncio.sleep(self.config.command_watchdog_s)
                if generation == self._watchdog_generation:
                    await self.emergency_stop()
            except asyncio.CancelledError:
                return

        self._watchdog = asyncio.create_task(expire())

    def _cancel_watchdog(self) -> None:
        self._watchdog_generation += 1
        task, self._watchdog = self._watchdog, None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _success(
        self,
        method: Any,
        *args: Any,
        timeout_s: float | None = None,
    ) -> None:
        result = await self._call(method, *args, timeout_s=timeout_s)
        if result != 0:
            raise MotionError(f"{method.__name__} returned {result!r}")

    async def _idle_stop(self) -> None:
        if self._lease is not None:
            raise MotionError("idle StopMove requested while motion is active")
        result = await self._call_stop(self.sport.StopMove)
        if result not in (0, -1):
            raise MotionError(f"StopMove returned {result!r}")

    async def _call(
        self,
        method: Any,
        *args: Any,
        timeout_s: float | None = None,
    ) -> Any:
        result, _slow = await self._call_timed(
            method,
            *args,
            timeout_s=timeout_s,
        )
        return result

    async def _call_timed(
        self,
        method: Any,
        *args: Any,
        timeout_s: float | None = None,
        on_slow: Callable[[], Any] | None = None,
    ) -> tuple[Any, bool]:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._rpc_executor, method, *args)
        timeout = self.config.rpc_timeout_s if timeout_s is None else timeout_s
        started = time.monotonic()
        try:
            done, _pending = await asyncio.wait(
                (future,),
                timeout=min(self.config.rpc_slow_threshold_s, timeout),
            )
            if done:
                return await asyncio.shield(future), False
            diagnostic = {
                "kind": "motion_rpc_slow",
                "method": method.__name__,
                "slow_threshold_s": self.config.rpc_slow_threshold_s,
                "timeout_s": timeout,
            }
            self._slow_call_count += 1
            self._last_rpc_diagnostic = diagnostic
            if self._diagnostic_sink is not None:
                try:
                    self._diagnostic_sink(dict(diagnostic))
                except Exception:  # noqa: BLE001, S110 - diagnostics cannot stop motion
                    pass
            if on_slow is not None:
                await on_slow()
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0.0:
                raise asyncio.TimeoutError
            return (
                await asyncio.wait_for(asyncio.shield(future), remaining),
                True,
            )
        except asyncio.TimeoutError as exc:
            raise MotionNotReady(f"{method.__name__} timed out") from exc

    async def _call_stop(self, method: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._stop_executor, method, *args)
        try:
            return await asyncio.wait_for(
                asyncio.shield(future), self.config.rpc_timeout_s
            )
        except asyncio.TimeoutError as exc:
            raise MotionNotReady(f"{method.__name__} timed out") from exc

    def _bounded_forward(self, value: float) -> float:
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError("forward speed must be finite and non-negative")
        if number > self.config.maximum_forward_mps:
            raise ValueError("forward speed exceeds the configured limit")
        return number

    def _bounded_yaw(self, value: float) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("yaw speed must be finite")
        if abs(number) > self.config.maximum_yaw_rps:
            raise ValueError("yaw speed exceeds the configured limit")
        return number


def initialize_dds(network_interface: str | None) -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    if network_interface:
        ChannelFactoryInitialize(0, network_interface)
    else:
        ChannelFactoryInitialize(0)


def create_go2_motion(config: MotionConfig | None = None) -> Go2Motion:
    from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import (
        ObstaclesAvoidClient,
    )
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    return Go2Motion(SportClient(), ObstaclesAvoidClient(), config)
