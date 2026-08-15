"""One app-side recording interface with a stable cross-service Home journal."""

from __future__ import annotations

import json
import os
import queue
import threading
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import UUID

from .black_box import RunBlackBox

HOME_RECORDING_SCHEMA_VERSION = 1
HOME_PHASES = frozenset(
    {
        "capture_home",
        "turn_toward_home",
        "return_home",
        # Durable mission schemas retain this historical name. It is now a
        # no-motion acknowledgement of settled position verification.
        "restore_heading",
        "complete",
        "failed",
        "failure_epilogue",
    }
)
HOME_KINDS = frozenset(
    {
        "home_captured",
        "home_pose_sample",
        "home_motion_state",
        "home_settled_sample",
        "home_settled_window",
        "home_position_retry",
        "home_verification_terminal",
        "run_sealed",
        # The passive recorder needs every command phase to correlate its
        # high-rate pose stream with yaw-only drift. It remains read-only and
        # never feeds observations back into control authority.
        "motion_command",
        "mission_event",
        "stage_result",
    }
)


def _redacted(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if any(
                    marker in str(key).casefold()
                    for marker in ("token", "password", "secret", "credential")
                )
                else _redacted(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redacted(item) for item in value]
    return deepcopy(value)


class SharedHomeEventJournal:
    """Non-blocking, bounded event handoff to the passive recorder service."""

    def __init__(self, root: Path, *, maximum_bytes_per_run: int = 8 * 1024 * 1024):
        if maximum_bytes_per_run < 1024:
            raise ValueError("Home journal limit must be at least 1024 bytes")
        self.root = root.resolve()
        self.maximum_bytes_per_run = maximum_bytes_per_run
        self._pending: queue.Queue[tuple[Path, dict[str, object]] | None] = (
            queue.Queue(maxsize=4096)
        )
        self._last_error: str | None = None
        self._dropped_events = 0
        self._closed = False
        self._writer = threading.Thread(
            target=self._write_loop,
            name="border-collie-home-journal",
            daemon=True,
        )
        self._writer.start()

    def record(self, event: dict[str, Any]) -> None:
        if self._closed:
            self._dropped_events += 1
            return
        try:
            run_id = str(UUID(str(event["run_id"])))
            queued = {
                "schema_version": HOME_RECORDING_SCHEMA_VERSION,
                "run_id": run_id,
                "sequence": int(event["sequence"]),
                "recorded_at_utc": event["recorded_at_utc"],
                "recorded_monotonic_s": float(event["recorded_monotonic_s"]),
                "kind": str(event["kind"]),
                "phase": event.get("phase"),
                "payload": _redacted(event.get("payload", {})),
            }
            self._pending.put_nowait((self.root / f"{run_id}.ndjson", queued))
        except (KeyError, TypeError, ValueError, queue.Full) as exc:
            self._last_error = str(exc)
            self._dropped_events += 1

    def flush(self) -> None:
        self._pending.join()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._pending.put(None)
        self._writer.join(timeout=2.0)
        self._closed = True

    def status(self) -> dict[str, object]:
        return {
            "schema_version": HOME_RECORDING_SCHEMA_VERSION,
            "available": self._last_error is None,
            "last_error": self._last_error,
            "dropped_events": self._dropped_events,
            "maximum_bytes_per_run": self.maximum_bytes_per_run,
            "root": str(self.root),
        }

    def _write_loop(self) -> None:
        while True:
            item = self._pending.get()
            try:
                if item is None:
                    return
                path, event = item
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and path.stat().st_size >= self.maximum_bytes_per_run:
                    self._dropped_events += 1
                    continue
                with path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(event, separators=(",", ":")) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
            except Exception as exc:  # noqa: BLE001 - never blocks motion safety
                self._last_error = str(exc)
                self._dropped_events += 1
            finally:
                self._pending.task_done()


class HomeRecordingHub:
    """Preserve the full black box and fan Home facts into one stable section."""

    def __init__(
        self,
        primary: RunBlackBox,
        shared: SharedHomeEventJournal,
    ) -> None:
        self._primary = primary
        self._shared = shared

    def record(
        self,
        run_id: str,
        kind: str,
        *,
        phase: str | None,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        event = self._primary.record(
            run_id,
            kind,
            phase=phase,
            payload=payload,
        )
        if kind in HOME_KINDS or (
            phase in HOME_PHASES
            and kind in {"motion_command", "motion_rpc", "stage_result"}
        ):
            self._shared.record({**event, "run_id": str(UUID(run_id))})
        return event

    def read(self, run_id: str) -> list[dict[str, Any]]:
        return self._primary.read(run_id)

    def path(self, run_id: str) -> Path:
        return self._primary.path(run_id)

    def flush(self) -> None:
        self._primary.flush()
        self._shared.flush()

    def close(self) -> None:
        self._shared.close()
        self._primary.close()

    def status(self) -> dict[str, object]:
        shared = self._shared.status()
        return {
            "schema_version": HOME_RECORDING_SCHEMA_VERSION,
            "shared_available": shared["available"],
            "shared_last_error": shared["last_error"],
            "shared_dropped_events": shared["dropped_events"],
            "maximum_bytes_per_run": shared["maximum_bytes_per_run"],
        }


class HomeRecordingStatus:
    """Combine the app journal and passive service without affecting safety."""

    def __init__(
        self,
        hub: HomeRecordingHub,
        *,
        passive_status_url: str,
        timeout_s: float = 0.10,
    ) -> None:
        self.hub = hub
        self.passive_status_url = passive_status_url
        self.timeout_s = timeout_s

    def __call__(self) -> dict[str, object]:
        status = self.hub.status()
        try:
            with urllib.request.urlopen(
                self.passive_status_url,
                timeout=self.timeout_s,
            ) as response:
                passive = json.loads(response.read().decode("utf-8"))
            if not isinstance(passive, dict):
                raise TypeError("passive recorder status must be an object")
        except (OSError, TypeError, ValueError, urllib.error.URLError) as exc:
            passive = {
                "ready": False,
                "strictly_passive": True,
                "detail": f"passive Home recorder unavailable: {exc}",
            }
        return {**status, "passive": passive}
