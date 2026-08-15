"""Passive DDS and app-event recorder for Home return diagnostics.

This service owns no robot command client, writer, or mutation route. It reads
SportModeState and the shared, versioned app event journal only.
"""

from __future__ import annotations

import json
import logging
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
POSE_RECOVERY_CONFIRMATIONS = 3
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
    def __init__(
        self,
        root: Path,
        *,
        maximum_events_per_run: int = 10_000,
        yaw_drift_threshold_m: float = 0.03,
        command_active_s: float = 0.50,
        logger: Any | None = None,
    ) -> None:
        if maximum_events_per_run < 3:
            raise ValueError("maximum_events_per_run must be at least 3")
        if (
            not math.isfinite(yaw_drift_threshold_m)
            or not 0.005 <= yaw_drift_threshold_m <= 0.50
        ):
            raise ValueError("yaw_drift_threshold_m must be between 0.005 and 0.50")
        if not math.isfinite(command_active_s) or not 0.10 <= command_active_s <= 2.0:
            raise ValueError("command_active_s must be between 0.10 and 2.0")
        self.root = root.resolve()
        self.maximum_events_per_run = maximum_events_per_run
        self.yaw_drift_threshold_m = yaw_drift_threshold_m
        self.command_active_s = command_active_s
        self._logger = logger or logging.getLogger("home-recorder")
        self._lock = threading.Lock()
        self._runs: dict[str, dict[str, Any]] = {}
        self._counts: dict[str, int] = {}
        self._pose_sequence = 0
        self._rejected_pose_samples = 0
        self._consecutive_finite_pose_samples = 0
        self._last_finite_pose_monotonic_s: float | None = None
        self._pose_error: str | None = None
        self._dropped_events = 0
        self._drift_detection_count = 0
        self._alert_error: str | None = None
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
                        "command_recorded_monotonic_s": None,
                        "last_pose": None,
                        "yaw_episode": None,
                        "recording": True,
                    }
                except (KeyError, TypeError, ValueError) as exc:
                    self._last_error = f"invalid Home event: {exc}"
                    return
                self._counts.setdefault(run_id, 0)
            context = self._runs.get(run_id)
            if context is None:
                return
            previous_stage = context["stage"]
            context["stage"] = phase
            if phase in RECORDED_HOME_PHASES:
                context["recording"] = True
            if kind == "mission_event" and phase != previous_stage:
                context["armed"] = False
                context["mode"] = None
                context["command"] = {
                    "motion_path": None,
                    "forward_mps": 0.0,
                    "yaw_rps": 0.0,
                }
                context["command_recorded_monotonic_s"] = None
                context["yaw_episode"] = None
            if kind == "motion_command":
                context["command"] = {
                    "motion_path": payload.get("motion_path"),
                    "forward_mps": float(payload.get("forward_mps", 0.0)),
                    "yaw_rps": float(payload.get("yaw_rps", 0.0)),
                }
                context["armed"] = True
                context["mode"] = payload.get("motion_path")
                command_recorded_monotonic_s = float(
                    payload.get(
                        "recorded_monotonic_s",
                        event.get("recorded_monotonic_s", 0.0),
                    )
                )
                context["command_recorded_monotonic_s"] = (
                    command_recorded_monotonic_s
                )
                command = context["command"]
                yaw_only = (
                    command["forward_mps"] == 0.0
                    and command["yaw_rps"] != 0.0
                )
                if yaw_only:
                    episode = context.get("yaw_episode")
                    last_command_at = (
                        None if episode is None else episode["last_command_monotonic_s"]
                    )
                    if (
                        episode is None
                        or last_command_at is None
                        or command_recorded_monotonic_s - last_command_at
                        > self.command_active_s
                    ):
                        last_pose = context.get("last_pose")
                        context["yaw_episode"] = {
                            "origin": last_pose,
                            "last_command_monotonic_s": command_recorded_monotonic_s,
                            "alerted": False,
                        }
                    else:
                        episode["last_command_monotonic_s"] = (
                            command_recorded_monotonic_s
                        )
                else:
                    context["yaw_episode"] = None
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
                    context["command_recorded_monotonic_s"] = None
                    context["yaw_episode"] = None
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
            with self._lock:
                self._rejected_pose_samples += 1
                self._consecutive_finite_pose_samples = 0
                self._pose_error = "non-finite pose sample"
            return
        with self._lock:
            if (
                self._last_finite_pose_monotonic_s is None
                or captured_monotonic_s > self._last_finite_pose_monotonic_s
            ):
                self._last_finite_pose_monotonic_s = captured_monotonic_s
                self._consecutive_finite_pose_samples = min(
                    POSE_RECOVERY_CONFIRMATIONS,
                    self._consecutive_finite_pose_samples + 1,
                )
                if (
                    self._pose_error is not None
                    and self._consecutive_finite_pose_samples
                    >= POSE_RECOVERY_CONFIRMATIONS
                ):
                    self._pose_error = None
            self._pose_sequence += 1
            for run_id, context in tuple(self._runs.items()):
                if not context["recording"]:
                    continue
                current_pose = {
                    "x_m": x_m,
                    "y_m": y_m,
                    "yaw_rad": yaw_rad,
                    "captured_monotonic_s": captured_monotonic_s,
                }
                context["last_pose"] = current_pose
                command_recorded_monotonic_s = context.get(
                    "command_recorded_monotonic_s"
                )
                if (
                    command_recorded_monotonic_s is not None
                    and captured_monotonic_s - command_recorded_monotonic_s
                    > self.command_active_s
                ):
                    context["armed"] = False
                    context["mode"] = None
                    context["command"] = {
                        "motion_path": None,
                        "forward_mps": 0.0,
                        "yaw_rps": 0.0,
                    }
                    context["command_recorded_monotonic_s"] = None
                    context["yaw_episode"] = None
                home = context["home"]
                delta_x = x_m - home["x_m"]
                delta_y = y_m - home["y_m"]
                yaw_episode = context.get("yaw_episode")
                yaw_drift_m: float | None = None
                drift_detected = False
                if yaw_episode is not None:
                    origin = yaw_episode.get("origin")
                    if origin is None:
                        yaw_episode["origin"] = current_pose
                    else:
                        yaw_drift_m = round(
                            math.hypot(
                                x_m - float(origin["x_m"]),
                                y_m - float(origin["y_m"]),
                            ),
                            6,
                        )
                        drift_detected = yaw_drift_m >= self.yaw_drift_threshold_m
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
                        "yaw_only_drift_m": yaw_drift_m,
                        "yaw_drift_threshold_m": self.yaw_drift_threshold_m,
                        "drift_detected": drift_detected,
                    },
                )
                if (
                    drift_detected
                    and yaw_episode is not None
                    and not yaw_episode["alerted"]
                ):
                    yaw_episode["alerted"] = True
                    self._drift_detection_count += 1
                    drift_event = {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "drift_detected",
                        "run_id": run_id,
                        "pose_sequence": self._pose_sequence,
                        "captured_monotonic_s": captured_monotonic_s,
                        "stage": context["stage"],
                        "motion_path": context["command"]["motion_path"],
                        "drift_m": yaw_drift_m,
                        "threshold_m": self.yaw_drift_threshold_m,
                        "origin_pose": dict(yaw_episode["origin"]),
                        "raw_pose": dict(current_pose),
                        "command": dict(context["command"]),
                        "observability_only": True,
                    }
                    self._append(run_id, drift_event)
                    alert = {
                        "run_id": run_id,
                        "stage": context["stage"],
                        "motion_path": context["command"]["motion_path"],
                        "drift_m": yaw_drift_m,
                        "threshold_m": self.yaw_drift_threshold_m,
                        "forward_mps": context["command"]["forward_mps"],
                        "yaw_rps": context["command"]["yaw_rps"],
                    }
                    try:
                        self._logger.warning(
                            "DRIFT DETECTED %s",
                            json.dumps(alert, separators=(",", ":")),
                        )
                    except Exception as exc:  # noqa: BLE001 - observation only
                        self._alert_error = str(exc)

    def path(self, run_id: str) -> Path:
        return self.root / str(UUID(run_id)) / "home-deep.ndjson"

    def status(self) -> dict[str, object]:
        with self._lock:
            last_error = self._last_error or self._pose_error
            return {
                "schema_version": SCHEMA_VERSION,
                "ready": last_error is None and self._pose_sequence > 0,
                "strictly_passive": True,
                "source": "rt/sportmodestate",
                "source_available": self._pose_sequence > 0,
                "active_run_ids": sorted(self._runs),
                "pose_sequence": self._pose_sequence,
                "rejected_pose_samples": self._rejected_pose_samples,
                "consecutive_finite_pose_samples": (
                    self._consecutive_finite_pose_samples
                ),
                "pose_recovery_confirmations_required": POSE_RECOVERY_CONFIRMATIONS,
                "dropped_events": self._dropped_events,
                "drift_detection_count": self._drift_detection_count,
                "alert_error": self._alert_error,
                "yaw_drift_threshold_m": self.yaw_drift_threshold_m,
                "command_active_s": self.command_active_s,
                "maximum_events_per_run": self.maximum_events_per_run,
                "last_error": last_error,
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [home-recorder] %(message)s",
    )
    state_root = Path(os.environ.get("HOME_RECORDER_STATE_DIR", "/state/home-recorder"))
    event_root = Path(
        os.environ.get("HOME_RECORDER_EVENTS_DIR", "/state/home-recorder/events")
    )
    recorder = HomeTimelineRecorder(
        state_root / "runs",
        maximum_events_per_run=int(
            os.environ.get("HOME_RECORDER_MAX_EVENTS_PER_RUN", "10000")
        ),
        yaw_drift_threshold_m=float(
            os.environ.get("HOME_RECORDER_YAW_DRIFT_THRESHOLD_M", "0.03")
        ),
        command_active_s=float(
            os.environ.get("HOME_RECORDER_COMMAND_ACTIVE_S", "0.50")
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
