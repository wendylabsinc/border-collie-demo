"""Small fail-closed adapter for the trusted Go2 bark sidecar."""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen

from .config import env_bool


class BarkFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class BarkConfig:
    enabled: bool = False
    url: str = "http://127.0.0.1:8111/api/bark"
    thermal_beep_url: str = "http://127.0.0.1:8111/api/thermal/beep"
    status_url: str = "http://127.0.0.1:8111/status"
    timeout_s: float = 2.0

    def __post_init__(self) -> None:
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("bark URL must use http or https")
        if not self.thermal_beep_url.startswith(("http://", "https://")):
            raise ValueError("thermal beep URL must use http or https")
        if not self.status_url.startswith(("http://", "https://")):
            raise ValueError("bark status URL must use http or https")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("bark timeout must be finite and positive")

    @classmethod
    def from_env(cls) -> BarkConfig:
        return cls(
            enabled=env_bool("BORDER_COLLIE_BARK_ENABLED"),
            url=os.environ.get(
                "BORDER_COLLIE_BARK_URL",
                "http://127.0.0.1:8111/api/bark",
            ).strip(),
            thermal_beep_url=os.environ.get(
                "BORDER_COLLIE_THERMAL_BEEP_URL",
                "http://127.0.0.1:8111/api/thermal/beep",
            ).strip(),
            status_url=os.environ.get(
                "BORDER_COLLIE_BARK_STATUS_URL",
                "http://127.0.0.1:8111/status",
            ).strip(),
            timeout_s=float(os.environ.get("BORDER_COLLIE_BARK_TIMEOUT_S", "2.0")),
        )


BarkPoster = Callable[[str, float], dict[str, Any]]


class BarkClient:
    def __init__(
        self,
        config: BarkConfig | None = None,
        *,
        poster: BarkPoster | None = None,
        fetcher: BarkPoster | None = None,
    ) -> None:
        self.config = config or BarkConfig()
        self._poster = poster or _post_json
        self._fetcher = fetcher or _fetch_json

    def status(self) -> dict[str, object]:
        if not self.config.enabled:
            return {"ready": False, "detail": "Go2 bark sidecar is disabled"}
        try:
            payload = self._fetcher(self.config.status_url, self.config.timeout_s)
        except Exception as exc:  # noqa: BLE001 - sidecar failures are untyped
            return {"ready": False, "detail": f"Go2 bark status unavailable: {exc}"}
        ready = payload.get("bark_ready") is True
        return {
            "ready": ready,
            "detail": (
                "Go2 bark sidecar is ready"
                if ready
                else str(payload.get("error") or "Go2 bark sidecar is not ready")
            ),
        }

    async def bark(self) -> dict[str, object]:
        if not self.config.enabled:
            raise BarkFailure("Go2 bark sidecar is disabled")
        try:
            payload = await asyncio.to_thread(
                self._poster,
                self.config.url,
                self.config.timeout_s,
            )
        except Exception as exc:
            raise BarkFailure(f"Go2 bark request failed: {exc}") from exc
        if payload.get("ok") is not True:
            raise BarkFailure(str(payload.get("error") or "bark was not acknowledged"))
        return {
            "bark_played": True,
            "bark_uuid": payload.get("uuid"),
            "bark_sound": payload.get("sound"),
            "bark_source": payload.get("source"),
        }

    async def thermal_beep(self) -> dict[str, object]:
        if not self.config.enabled:
            raise BarkFailure("Go2 audio sidecar is disabled")
        try:
            payload = await asyncio.to_thread(
                self._poster,
                self.config.thermal_beep_url,
                self.config.timeout_s,
            )
        except Exception as exc:
            raise BarkFailure(f"thermal beep request failed: {exc}") from exc
        if payload.get("ok") is not True:
            raise BarkFailure(
                str(payload.get("error") or "thermal beep was not acknowledged")
            )
        return {
            "thermal_beep_played": True,
            "thermal_beep_uuid": payload.get("uuid"),
        }


def _post_json(url: str, timeout_s: float) -> dict[str, Any]:
    request = Request(url, data=b"{}", method="POST", headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout_s) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise TypeError("bark response must be a JSON object")
    return payload


def _fetch_json(url: str, timeout_s: float) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout_s) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise TypeError("bark status response must be a JSON object")
    return payload
