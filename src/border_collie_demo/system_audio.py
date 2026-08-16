"""Own the Go2 speaker as muted-by-default demo infrastructure."""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .media import BarkFailure


class VuiClientProtocol(Protocol):
    def SetTimeout(self, timeout_s: float) -> Any: ...

    def Init(self) -> Any: ...

    def GetVolume(self) -> tuple[int, int]: ...

    def SetVolume(self, volume: int) -> int: ...


class BarkPort(Protocol):
    def status(self) -> dict[str, object]: ...

    async def bark(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class SystemAudioConfig:
    enabled: bool = True
    bark_volume: int = 6
    bark_audible_s: float = 2.0
    rpc_timeout_s: float = 3.0
    restore_original_on_close: bool = False
    volume_settle_s: float = 0.25
    volume_verify_attempts: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.bark_volume, bool) or not 0 <= self.bark_volume <= 10:
            raise ValueError("bark_volume must be an integer from 0 through 10")
        if (
            not math.isfinite(self.bark_audible_s)
            or not 0.25 <= self.bark_audible_s <= 10.0
        ):
            raise ValueError("bark_audible_s must be between 0.25 and 10 seconds")
        if not math.isfinite(self.rpc_timeout_s) or self.rpc_timeout_s <= 0.0:
            raise ValueError("rpc_timeout_s must be finite and positive")

    @classmethod
    def from_env(cls) -> SystemAudioConfig:
        policy = (
            os.environ.get("BORDER_COLLIE_SYSTEM_AUDIO_POLICY", "muted_except_bark")
            .strip()
            .casefold()
        )
        if policy not in {"normal", "muted_except_bark"}:
            raise ValueError(
                "BORDER_COLLIE_SYSTEM_AUDIO_POLICY must be normal or muted_except_bark"
            )
        return cls(
            enabled=policy == "muted_except_bark",
            bark_volume=int(os.environ.get("BORDER_COLLIE_BARK_VOLUME", "6")),
            bark_audible_s=float(os.environ.get("BORDER_COLLIE_BARK_AUDIBLE_S", "2.0")),
            rpc_timeout_s=float(os.environ.get("BORDER_COLLIE_VUI_TIMEOUT_S", "3.0")),
            restore_original_on_close=os.environ.get(
                "BORDER_COLLIE_RESTORE_SPEAKER_ON_CLOSE", "0"
            )
            .strip()
            .casefold()
            in {"1", "true", "yes", "on"},
        )


class SystemAudioPolicy:
    """Keep the robot speaker muted except for one bounded bark operation."""

    def __init__(
        self,
        vui: VuiClientProtocol | None,
        bark: BarkPort,
        config: SystemAudioConfig | None = None,
        *,
        vui_factory: Callable[[], VuiClientProtocol] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.config = config or SystemAudioConfig()
        self._vui = vui
        self._vui_factory = vui_factory
        self._bark = bark
        self._sleep = sleep
        self._lock = asyncio.Lock()
        self._started = False
        self._ready = False
        self._muted = False
        self._original_volume: int | None = None
        self._error: str | None = None
        self._warning: str | None = None

    async def start_muted(self) -> None:
        if self._started:
            return
        self._started = True
        if not self.config.enabled:
            self._ready = True
            return
        if self._vui is None and self._vui_factory is None:
            self._error = "Go2 VUI client is unavailable"
            return
        try:
            if self._vui is None:
                assert self._vui_factory is not None
                self._vui = self._vui_factory()
            await self._call(self._vui.SetTimeout, self.config.rpc_timeout_s)
            await self._call(self._vui.Init)
            self._original_volume = await self._get_volume()
            await self._set_and_verify(0)
            self._muted = True
            self._ready = True
            self._error = None
        except Exception as exc:  # noqa: BLE001 - SDK errors are untyped
            self._ready = False
            self._error = f"speaker mute failed: {exc}"

    def status(self) -> dict[str, object]:
        bark = self._bark.status()
        ready = bool(bark.get("ready")) and self._ready
        return {
            **bark,
            "ready": ready,
            "detail": (
                str(bark.get("detail") or "Go2 bark sidecar is not ready")
                if not bark.get("ready")
                else (
                    "Go2 speaker is muted and bark is ready"
                    if ready
                    else self._error or "Go2 speaker policy is not ready"
                )
            ),
            "speaker_policy": (
                "muted_except_bark" if self.config.enabled else "normal"
            ),
            "speaker_muted": self._muted,
            "speaker_error": self._error,
            "speaker_warning": self._warning,
        }

    async def bark(self) -> dict[str, object]:
        if not self.config.enabled:
            return await self._bark.bark()
        if not self._ready or self._vui is None:
            raise BarkFailure(self._error or "Go2 speaker policy is not ready")
        async with self._lock:
            unmute_error: Exception | None = None
            bark_error: Exception | None = None
            result: dict[str, object] | None = None
            try:
                await self._set_and_verify(self.config.bark_volume)
                self._muted = False
                self._warning = None
            except Exception as exc:  # noqa: BLE001 - silent bark is non-terminal
                unmute_error = exc
                self._warning = f"bark remained muted: {exc}"
            try:
                result = await self._bark.bark()
                if unmute_error is None:
                    await self._sleep(self.config.bark_audible_s)
            except Exception as exc:  # noqa: BLE001 - remute must always run
                bark_error = exc
            remute_error: Exception | None = None
            try:
                await self._set_and_verify(0)
                self._muted = True
            except Exception as exc:  # noqa: BLE001 - fail loud, not silent
                # A failed remute leaves the speaker audible, which is the safe
                # direction: the sound the caller asked for did play. Latching
                # not-ready here used to disable every later bark and alarm
                # until restart, turning one stale read into permanent silence.
                remute_error = exc
                self._muted = False
                self._warning = f"speaker remained audible: {exc}"
            if bark_error is not None:
                if isinstance(bark_error, BarkFailure):
                    raise bark_error
                raise BarkFailure(f"bark audio failed: {bark_error}") from bark_error
            assert result is not None
            response: dict[str, object] = {
                **result,
                "speaker_policy": "muted_except_bark",
                "bark_volume": self.config.bark_volume,
                "bark_audible_s": self.config.bark_audible_s,
                "speaker_remuted": remute_error is None,
                "audio_trace": [
                    {
                        "state": "audible",
                        "volume": self.config.bark_volume,
                    },
                    {"state": "bark_requested"},
                    {"state": "muted", "volume": 0},
                ],
            }
            if unmute_error is not None:
                response.update(
                    {
                        "speaker_audible": False,
                        "speaker_warning": self._warning,
                        "audio_trace": [
                            {
                                "state": "audibility_degraded",
                                "requested_volume": self.config.bark_volume,
                                "error": str(unmute_error),
                            },
                            {"state": "bark_requested"},
                            {"state": "muted", "volume": 0},
                        ],
                    }
                )
            return response

    async def close(self) -> list[str]:
        if not self.config.enabled or self._vui is None or not self._started:
            return []
        target = (
            self._original_volume
            if self.config.restore_original_on_close
            and self._original_volume is not None
            else 0
        )
        try:
            await self._set_and_verify(target)
            self._muted = target == 0
            return []
        except Exception as exc:  # noqa: BLE001 - shutdown reports exact failure
            return [f"speaker shutdown volume failed: {exc}"]

    async def _get_volume(self) -> int:
        assert self._vui is not None
        response = await self._call(self._vui.GetVolume)
        if (
            not isinstance(response, tuple)
            or len(response) != 2
            or response[0] != 0
            or isinstance(response[1], bool)
            or not isinstance(response[1], int)
            or not 0 <= response[1] <= 10
        ):
            raise RuntimeError(f"invalid GetVolume response: {response!r}")
        return response[1]

    async def _set_and_verify(self, volume: int) -> None:
        assert self._vui is not None
        result = await self._call(self._vui.SetVolume, volume)
        if result != 0:
            raise RuntimeError(f"SetVolume({volume}) returned {result!r}")
        # The Go2 acks SetVolume before GetVolume reflects it. Reading straight
        # back returned the previous level on hardware, so give it a settle and
        # one retry before calling it a mismatch.
        observed = None
        for attempt in range(self.config.volume_verify_attempts):
            if attempt:
                await self._sleep(self.config.volume_settle_s)
            observed = await self._get_volume()
            if observed == volume:
                return
        raise RuntimeError(
            f"speaker volume verification expected {volume}, got {observed}"
        )

    async def _call(self, method: Callable[..., Any], *args: Any) -> Any:
        return await asyncio.wait_for(
            asyncio.to_thread(method, *args),
            timeout=self.config.rpc_timeout_s,
        )


def create_system_audio_policy(
    bark: BarkPort,
    config: SystemAudioConfig | None = None,
    *,
    vui_factory: Callable[[], VuiClientProtocol] | None = None,
) -> SystemAudioPolicy:
    resolved = config or SystemAudioConfig.from_env()
    if not resolved.enabled:
        return SystemAudioPolicy(None, bark, resolved)
    if vui_factory is None:
        # Imported when the client is built, not when the policy is, so the
        # app can be constructed off-device where the SDK is absent.
        def vui_factory() -> VuiClientProtocol:
            from unitree_sdk2py.go2.vui.vui_client import VuiClient

            return VuiClient()

    return SystemAudioPolicy(None, bark, resolved, vui_factory=vui_factory)
