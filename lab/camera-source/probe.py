"""Read-only Unitree WebRTC decoded-frame identity measurement."""

from __future__ import annotations

import asyncio
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import statistics
import time
from typing import Any

from unitree_webrtc_connect import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)


ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "192.168.123.161")
DURATION_S = float(os.environ.get("CAMERA_PROBE_DURATION_S", "30"))
STARTUP_TIMEOUT_S = float(
    os.environ.get("CAMERA_PROBE_STARTUP_TIMEOUT_S", "15")
)
SNAPSHOT_PATH = os.environ.get("CAMERA_PROBE_SNAPSHOT_PATH", "snapshot.jpg")
SNAPSHOT_PORT = int(os.environ.get("CAMERA_PROBE_SNAPSHOT_PORT", "8122"))


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "minimum": round(ordered[0], 6),
        "mean": round(statistics.fmean(ordered), 6),
        "p95": round(ordered[p95_index], 6),
        "maximum": round(ordered[-1], 6),
    }


class Measurement:
    def __init__(self) -> None:
        self.started_monotonic_s = time.monotonic()
        self.first_frame = asyncio.Event()
        self.stop = asyncio.Event()
        self.frames = 0
        self.pts_missing = 0
        self.pts_repeated = 0
        self.pts_regressed = 0
        self.time_base_missing = 0
        self.time_base_changes = 0
        self.content_repeated = 0
        self.callback_error = ""
        self.snapshot_error = ""
        self.snapshot_written = False
        self.width: int | None = None
        self.height: int | None = None
        self.last_pts: int | None = None
        self.last_time_base: str | None = None
        self.last_digest: str | None = None
        self.last_receipt_s: float | None = None
        self.callback_intervals_s: list[float] = []
        self.presentation_deltas_s: list[float] = []
        self.last_frame: Any | None = None

    async def consume(self, track: Any) -> None:
        try:
            while not self.stop.is_set():
                frame = await track.recv()
                received_s = time.monotonic()
                pts = None if frame.pts is None else int(frame.pts)
                time_base = (
                    None if frame.time_base is None else str(frame.time_base)
                )
                presentation_s = (
                    None
                    if pts is None or frame.time_base is None
                    else float(pts * frame.time_base)
                )
                digest = hashlib.sha256(bytes(frame.planes[0])).hexdigest()

                if self.last_receipt_s is not None:
                    self.callback_intervals_s.append(received_s - self.last_receipt_s)
                if pts is None:
                    self.pts_missing += 1
                if time_base is None:
                    self.time_base_missing += 1
                if self.last_time_base is not None and time_base != self.last_time_base:
                    self.time_base_changes += 1
                if pts is not None and self.last_pts is not None:
                    if pts == self.last_pts:
                        self.pts_repeated += 1
                    elif pts < self.last_pts:
                        self.pts_regressed += 1
                    elif (
                        presentation_s is not None
                        and self.last_time_base is not None
                        and time_base == self.last_time_base
                    ):
                        previous_s = float(self.last_pts * frame.time_base)
                        self.presentation_deltas_s.append(
                            presentation_s - previous_s
                        )
                if digest == self.last_digest:
                    self.content_repeated += 1

                self.frames += 1
                self.width = int(frame.width)
                self.height = int(frame.height)
                self.last_pts = pts
                self.last_time_base = time_base
                self.last_digest = digest
                self.last_receipt_s = received_s
                self.last_frame = frame
                self.first_frame.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self.stop.is_set():
                self.callback_error = f"{type(exc).__name__}: {exc}"
                self.first_frame.set()

    def write_snapshot(self) -> None:
        if self.last_frame is None:
            self.snapshot_error = "no decoded frame available"
            return
        try:
            self.last_frame.to_image().save(
                SNAPSHOT_PATH,
                format="JPEG",
                quality=85,
            )
            self.snapshot_written = True
        except Exception as exc:
            self.snapshot_error = f"{type(exc).__name__}: {exc}"

    def result(self) -> dict[str, object]:
        elapsed_s = time.monotonic() - self.started_monotonic_s
        identity_complete = bool(
            self.frames > 1
            and self.pts_missing == 0
            and self.time_base_missing == 0
        )
        strictly_advancing = bool(
            identity_complete
            and self.pts_repeated == 0
            and self.pts_regressed == 0
            and self.time_base_changes == 0
        )
        return {
            "test_id": "CAMERA-SOURCE-001",
            "mode": "read_only_no_motion_imports",
            "robot_ip": ROBOT_IP,
            "requested_duration_s": DURATION_S,
            "observed_duration_s": round(elapsed_s, 3),
            "frames": self.frames,
            "dimensions": {"width": self.width, "height": self.height},
            "source_identity": {
                "pts_missing": self.pts_missing,
                "pts_repeated": self.pts_repeated,
                "pts_regressed": self.pts_regressed,
                "time_base_missing": self.time_base_missing,
                "time_base_changes": self.time_base_changes,
                "last_pts": self.last_pts,
                "last_time_base": self.last_time_base,
                "complete": identity_complete,
                "strictly_advancing": strictly_advancing,
            },
            "callback_interval_s": _stats(self.callback_intervals_s),
            "presentation_delta_s": _stats(self.presentation_deltas_s),
            "identical_luma_frames": self.content_repeated,
            "callback_error": self.callback_error,
            "snapshot": {
                "path": SNAPSHOT_PATH,
                "port": SNAPSHOT_PORT,
                "written": self.snapshot_written,
                "error": self.snapshot_error,
            },
            "measurement_complete": bool(
                self.frames > 1
                and not self.callback_error
                and self.snapshot_written
            ),
            "motion_commands_sent": False,
        }


async def run() -> dict[str, object]:
    measurement = Measurement()
    connection = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip=ROBOT_IP,
    )
    connected = False
    try:
        await connection.connect()
        connected = True
        connection.video.add_track_callback(measurement.consume)
        connection.video.switchVideoChannel(True)
        await asyncio.wait_for(
            measurement.first_frame.wait(),
            timeout=STARTUP_TIMEOUT_S,
        )
        if measurement.callback_error:
            raise RuntimeError(measurement.callback_error)
        await asyncio.sleep(DURATION_S)
        measurement.write_snapshot()
    finally:
        measurement.stop.set()
        if connected:
            await connection.disconnect()
    return measurement.result()


def main() -> None:
    try:
        result = asyncio.run(run())
    except Exception as exc:
        result = {
            "test_id": "CAMERA-SOURCE-001",
            "mode": "read_only_no_motion_imports",
            "robot_ip": ROBOT_IP,
            "measurement_complete": False,
            "motion_commands_sent": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(f"CAMERA_SOURCE_RESULT={json.dumps(result, sort_keys=True)}", flush=True)
    print(
        f"CAMERA_SOURCE_SNAPSHOT=http://woof.local:{SNAPSHOT_PORT}/{SNAPSHOT_PATH}",
        flush=True,
    )
    ThreadingHTTPServer(
        ("0.0.0.0", SNAPSHOT_PORT),
        SimpleHTTPRequestHandler,
    ).serve_forever()


if __name__ == "__main__":
    main()
