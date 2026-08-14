"""Passive DDS and app-event recorder for Home return diagnostics.

This service owns no robot command client, writer, or mutation route. It reads
SportModeState and the shared, versioned app event journal only.
"""

from __future__ import annotations

import json
import math
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import UUID

SCHEMA_VERSION = 1
RECORDED_HOME_PHASES = frozenset(
    {
        "turn_toward_home",
        "return_home",
        "restore_heading",
        "failure_epilogue",
        "complete",
        "failed",
    }
)


class HomeTimelineRecorder:
    def __init__(self, root: Path, *, maximum_events_per_run: int = 10_000) -> None:
        if maximum_events_per_run < 3:
            raise ValueError("maximum_events_per_run must be at least 3")
        self.root = root.resolve()
        self.maximum_events_per_run = maximum_events_per_run
        self._lock = threading.Lock()
        self._runs: dict[str, dict[str, Any]] = {}
        self._counts: dict[str, int] = {}
        self._pose_sequence = 0
        self._dropped_events = 0
        self._last_error: str | None = None

    def process_app_event(self, event: dict[str, Any]) -> None:
        if event.get("schema_version") != SCHEMA_VERSION:
            self._last_error = "unsupported app event schema"
            return
        try:
            run_id = str(UUID(str(event["run_id"])))
            kind = str(event["kind"])
            phase = str(event.get("phase") or "unknown")
            payload = event.get("payload")
            if not isinstance(payload, dict):
                raise TypeError("app event payload must be an object")
        except (KeyError, TypeError, ValueError) as exc:
            self._last_error = str(exc)
            return

        with self._lock:
            if kind == "home_captured":
                try:
                    self._runs[run_id] = {
                        "home": {
                            "x_m": float(payload["x_m"]),
                            "y_m": float(payload["y_m"]),
                            "yaw_rad": float(payload["yaw_rad"]),
                        },
                        "odometry_epoch": str(payload["odometry_epoch"]),
                        "stage": phase,
                        "armed": False,
                        "mode": None,
                        "command": {
                            "motion_path": None,
                            "forward_mps": 0.0,
                            "yaw_rps": 0.0,
                        },
                        "recording": False,
                    }
                except (KeyError, TypeError, ValueError) as exc:
                    self._last_error = f"invalid Home event: {exc}"
                    return
                self._counts.setdefault(run_id, 0)
            context = self._runs.get(run_id)
            if context is None:
                return
            context["stage"] = phase
            if phase in RECORDED_HOME_PHASES:
                context["recording"] = True
            if kind == "motion_command":
                context["command"] = {
                    "motion_path": payload.get("motion_path"),
                    "forward_mps": float(payload.get("forward_mps", 0.0)),
                    "yaw_rps": float(payload.get("yaw_rps", 0.0)),
                }
                context["armed"] = True
                context["mode"] = payload.get("motion_path")
            elif kind == "home_motion_state":
                if isinstance(payload.get("armed"), bool):
                    context["armed"] = payload["armed"]
                context["mode"] = payload.get("motion_path", context["mode"])
                if payload.get("forward_mps") == 0.0 and payload.get("yaw_rps") == 0.0:
                    context["command"] = {
                        "motion_path": context["mode"],
                        "forward_mps": 0.0,
                        "yaw_rps": 0.0,
                    }
            row = {
                "schema_version": SCHEMA_VERSION,
                "kind": "app_event",
                "run_id": run_id,
                "app_sequence": event.get("sequence"),
                "recorded_monotonic_s": event.get("recorded_monotonic_s"),
                "stage": phase,
                "event_kind": kind,
                "payload": payload,
                "armed": context["armed"],
                "mode": context["mode"],
                "command": dict(context["command"]),
                "terminal_reason": (
                    payload.get("reason") if kind == "run_sealed" else None
                ),
            }
            self._append(run_id, row)
            if kind == "run_sealed":
                self._runs.pop(run_id, None)

    def record_pose(
        self,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        *,
        captured_monotonic_s: float,
    ) -> None:
        values = (x_m, y_m, yaw_rad, captured_monotonic_s)
        if not all(math.isfinite(value) for value in values):
            self._last_error = "non-finite pose sample"
            return
        with self._lock:
            self._pose_sequence += 1
            for run_id, context in tuple(self._runs.items()):
                if not context["recording"]:
                    continue
                home = context["home"]
                delta_x = x_m - home["x_m"]
                delta_y = y_m - home["y_m"]
                self._append(
                    run_id,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "pose",
                        "run_id": run_id,
                        "pose_sequence": self._pose_sequence,
                        "captured_monotonic_s": captured_monotonic_s,
                        "source_age_s": 0.0,
                        "raw_pose": {
                            "x_m": x_m,
                            "y_m": y_m,
                            "yaw_rad": yaw_rad,
                        },
                        "home_delta": {
                            "x_m": round(delta_x, 6),
                            "y_m": round(delta_y, 6),
                        },
                        "home_distance_m": round(math.hypot(delta_x, delta_y), 6),
                        "odometry_epoch": context["odometry_epoch"],
                        "stage": context["stage"],
                        "armed": context["armed"],
                        "mode": context["mode"],
                        "command": dict(context["command"]),
                    },
                )

    def path(self, run_id: str) -> Path:
        return self.root / str(UUID(run_id)) / "home-deep.ndjson"

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "schema_version": SCHEMA_VERSION,
                "ready": self._last_error is None and self._pose_sequence > 0,
                "strictly_passive": True,
                "source": "rt/sportmodestate",
                "source_available": self._pose_sequence > 0,
                "active_run_ids": sorted(self._runs),
                "pose_sequence": self._pose_sequence,
                "dropped_events": self._dropped_events,
                "maximum_events_per_run": self.maximum_events_per_run,
                "last_error": self._last_error,
            }

    def set_error(self, error: str | None) -> None:
        with self._lock:
            self._last_error = error

    def _append(self, run_id: str, row: dict[str, object]) -> None:
        count = self._counts.get(run_id, 0)
        if count >= self.maximum_events_per_run:
            self._dropped_events += 1
            return
        try:
            path = self.path(run_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
                output.flush()
                if row.get("terminal_reason") is not None:
                    os.fsync(output.fileno())
            self._counts[run_id] = count + 1
        except OSError as exc:
            self._last_error = str(exc)
            self._dropped_events += 1


class SharedEventTailer:
    def __init__(self, root: Path, recorder: HomeTimelineRecorder) -> None:
        self.root = root.resolve()
        self.recorder = recorder
        self._offsets: dict[Path, int] = {}

    def poll(self) -> None:
        try:
            paths = sorted(self.root.glob("*.ndjson"))
        except OSError as exc:
            self.recorder.set_error(f"event journal unavailable: {exc}")
            return
        for path in paths:
            offset = self._offsets.get(path, 0)
            try:
                with path.open("r", encoding="utf-8") as source:
                    source.seek(offset)
                    for line in source:
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError as exc:
                            self.recorder.set_error(f"invalid app event: {exc}")
                            continue
                        self.recorder.process_app_event(event)
                    self._offsets[path] = source.tell()
            except OSError as exc:
                self.recorder.set_error(f"event journal read failed: {exc}")


def start_pose_subscriber(recorder: HomeTimelineRecorder, interface: str | None):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

    ChannelFactoryInitialize(0, interface)

    def on_state(message: Any) -> None:
        try:
            recorder.record_pose(
                float(message.position[0]),
                float(message.position[1]),
                float(message.imu_state.rpy[2]),
                captured_monotonic_s=time.monotonic(),
            )
        except Exception as exc:  # noqa: BLE001 - dynamic DDS input
            recorder.set_error(f"invalid pose sample: {exc}")

    subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
    subscriber.Init(on_state, 1)
    return subscriber


def status_server(recorder: HomeTimelineRecorder, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/status":
                self.send_error(404)
                return
            body = json.dumps(recorder.status(), separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    state_root = Path(os.environ.get("HOME_RECORDER_STATE_DIR", "/state/home-recorder"))
    event_root = Path(
        os.environ.get("HOME_RECORDER_EVENTS_DIR", "/state/home-recorder/events")
    )
    recorder = HomeTimelineRecorder(
        state_root / "runs",
        maximum_events_per_run=int(
            os.environ.get("HOME_RECORDER_MAX_EVENTS_PER_RUN", "10000")
        ),
    )
    tailer = SharedEventTailer(event_root, recorder)
    subscriber = start_pose_subscriber(
        recorder, os.environ.get("GO2_NETWORK_INTERFACE") or None
    )
    server = status_server(recorder, int(os.environ.get("HOME_RECORDER_PORT", "8112")))
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_args: stopped.set())
    signal.signal(signal.SIGINT, lambda *_args: stopped.set())
    while not stopped.wait(0.05):
        tailer.poll()
    server.shutdown()
    close = getattr(subscriber, "Close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
