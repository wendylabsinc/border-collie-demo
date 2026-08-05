"""Read-only pear recognition evidence probe for Woof."""

from __future__ import annotations

import asyncio
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any

from unitree_webrtc_connect import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from ultralytics import YOLO


ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "192.168.123.161")
MODEL_PATH = Path(os.environ.get("PEAR_MODEL_PATH", "/probe/model.engine"))
DURATION_S = float(os.environ.get("PEAR_PROBE_DURATION_S", "60"))
STARTUP_TIMEOUT_S = float(os.environ.get("PEAR_STARTUP_TIMEOUT_S", "20"))
THRESHOLD = float(os.environ.get("PEAR_CONFIDENCE_THRESHOLD", "0.35"))
REQUIRED_RATIO = float(os.environ.get("PEAR_REQUIRED_MATCH_RATIO", "0.80"))
MINIMUM_SAMPLES = int(os.environ.get("PEAR_MINIMUM_SAMPLES", "30"))
SNAPSHOT_PATH = Path(os.environ.get("PEAR_SNAPSHOT_PATH", "snapshot.jpg"))
SNAPSHOT_PORT = int(os.environ.get("PEAR_SNAPSHOT_PORT", "8123"))


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
        self.started_s = time.monotonic()
        self.first_frame = asyncio.Event()
        self.stop = asyncio.Event()
        self.frames: asyncio.Queue[tuple[Any, float]] = asyncio.Queue(maxsize=1)
        self.decoded_frames = 0
        self.dropped_for_inference = 0
        self.pts_missing = 0
        self.pts_repeated = 0
        self.pts_regressed = 0
        self.time_base_missing = 0
        self.time_base_changes = 0
        self.identical_luma_frames = 0
        self.last_pts: int | None = None
        self.last_time_base: str | None = None
        self.last_digest: str | None = None
        self.camera_error = ""
        self.detector_error = ""
        self.detector_samples = 0
        self.matches = 0
        self.current_miss_run = 0
        self.maximum_miss_run = 0
        self.confidences: list[float] = []
        self.inference_s: list[float] = []
        self.detection_age_s: list[float] = []
        self.best_confidence = -1.0
        self.best_annotated: Any | None = None
        self.latest_bgr: Any | None = None
        self.snapshot_written = False
        self.snapshot_error = ""
        self.model = YOLO(str(MODEL_PATH), task="segment")
        self.pear_class_id = self._pear_class_id()

    def _pear_class_id(self) -> int:
        names = getattr(self.model, "names", {})
        items = names.items() if isinstance(names, dict) else enumerate(names)
        for class_id, label in items:
            if str(label).casefold().strip() == "pear":
                return int(class_id)
        raise RuntimeError(f"pear class missing from model names: {names!r}")

    async def consume_camera(self, track: Any) -> None:
        try:
            while not self.stop.is_set():
                frame = await track.recv()
                received_s = time.monotonic()
                pts = None if frame.pts is None else int(frame.pts)
                time_base = (
                    None if frame.time_base is None else str(frame.time_base)
                )
                digest = hashlib.sha256(bytes(frame.planes[0])).hexdigest()

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
                if digest == self.last_digest:
                    self.identical_luma_frames += 1

                self.decoded_frames += 1
                self.last_pts = pts
                self.last_time_base = time_base
                self.last_digest = digest
                if self.frames.full():
                    self.frames.get_nowait()
                    self.dropped_for_inference += 1
                self.frames.put_nowait((frame, received_s))
                self.first_frame.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self.stop.is_set():
                self.camera_error = f"{type(exc).__name__}: {exc}"
                self.first_frame.set()

    async def detect(self) -> None:
        while not self.stop.is_set() or not self.frames.empty():
            try:
                frame, received_s = await asyncio.wait_for(
                    self.frames.get(), timeout=0.25
                )
            except TimeoutError:
                continue
            try:
                await asyncio.to_thread(self._detect_frame, frame, received_s)
            except Exception as exc:
                self.detector_error = f"{type(exc).__name__}: {exc}"
                self.stop.set()
                return

    def _detect_frame(self, frame: Any, received_s: float) -> None:
        bgr = frame.to_ndarray(format="bgr24")
        started_s = time.monotonic()
        results = self.model.predict(
            source=bgr,
            conf=0.01,
            classes=[self.pear_class_id],
            device=0,
            verbose=False,
        )
        completed_s = time.monotonic()
        self.detector_samples += 1
        self.inference_s.append(completed_s - started_s)
        self.detection_age_s.append(completed_s - received_s)
        self.latest_bgr = bgr

        result = results[0] if results else None
        boxes = None if result is None else getattr(result, "boxes", None)
        scores: list[float] = []
        if boxes is not None and len(boxes):
            scores = [float(value) for value in boxes.conf.detach().cpu().tolist()]
        best = max(scores, default=0.0)
        if best >= THRESHOLD:
            self.matches += 1
            self.confidences.append(best)
            self.current_miss_run = 0
            if best > self.best_confidence:
                self.best_confidence = best
                self.best_annotated = result.plot()
        else:
            self.current_miss_run += 1
            self.maximum_miss_run = max(
                self.maximum_miss_run, self.current_miss_run
            )

    def write_snapshot(self) -> None:
        image = self.best_annotated if self.best_annotated is not None else self.latest_bgr
        if image is None:
            self.snapshot_error = "no inference frame available"
            return
        try:
            from PIL import Image

            Image.fromarray(image[:, :, ::-1]).save(
                SNAPSHOT_PATH,
                format="JPEG",
                quality=85,
            )
            self.snapshot_written = True
        except Exception as exc:
            self.snapshot_error = f"{type(exc).__name__}: {exc}"

    def result(self) -> dict[str, object]:
        ratio = self.matches / self.detector_samples if self.detector_samples else 0.0
        source_strict = bool(
            self.decoded_frames > 1
            and self.pts_missing == 0
            and self.time_base_missing == 0
            and self.pts_repeated == 0
            and self.pts_regressed == 0
            and self.time_base_changes == 0
        )
        passed = bool(
            source_strict
            and not self.camera_error
            and not self.detector_error
            and self.detector_samples >= MINIMUM_SAMPLES
            and ratio >= REQUIRED_RATIO
            and self.snapshot_written
        )
        return {
            "test_id": "PEAR-EVIDENCE-001",
            "mode": "read_only_no_motion_imports",
            "robot_ip": ROBOT_IP,
            "requested_duration_s": DURATION_S,
            "observed_duration_s": round(time.monotonic() - self.started_s, 3),
            "model": {
                "path": str(MODEL_PATH),
                "pear_class_id": self.pear_class_id,
                "confidence_threshold": THRESHOLD,
            },
            "camera": {
                "decoded_frames": self.decoded_frames,
                "dropped_for_inference": self.dropped_for_inference,
                "pts_missing": self.pts_missing,
                "pts_repeated": self.pts_repeated,
                "pts_regressed": self.pts_regressed,
                "time_base_missing": self.time_base_missing,
                "time_base_changes": self.time_base_changes,
                "last_time_base": self.last_time_base,
                "identical_luma_frames": self.identical_luma_frames,
                "strict_source_identity": source_strict,
                "error": self.camera_error,
            },
            "recognition": {
                "eligible_samples": self.detector_samples,
                "matches": self.matches,
                "misses": self.detector_samples - self.matches,
                "match_ratio": round(ratio, 6),
                "required_ratio": REQUIRED_RATIO,
                "minimum_samples": MINIMUM_SAMPLES,
                "maximum_consecutive_misses": self.maximum_miss_run,
                "confidence": _stats(self.confidences),
            },
            "inference_s": _stats(self.inference_s),
            "detection_age_s": _stats(self.detection_age_s),
            "detector_error": self.detector_error,
            "snapshot": {
                "path": str(SNAPSHOT_PATH),
                "port": SNAPSHOT_PORT,
                "written": self.snapshot_written,
                "error": self.snapshot_error,
            },
            "passed": passed,
            "measurement_complete": bool(
                self.detector_samples >= MINIMUM_SAMPLES
                and not self.camera_error
                and not self.detector_error
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
    detector_task: asyncio.Task[None] | None = None
    try:
        await connection.connect()
        connected = True
        connection.video.add_track_callback(measurement.consume_camera)
        connection.video.switchVideoChannel(True)
        await asyncio.wait_for(
            measurement.first_frame.wait(),
            timeout=STARTUP_TIMEOUT_S,
        )
        if measurement.camera_error:
            raise RuntimeError(measurement.camera_error)
        detector_task = asyncio.create_task(measurement.detect())
        await asyncio.sleep(DURATION_S)
    finally:
        measurement.stop.set()
        if detector_task is not None:
            await asyncio.wait_for(detector_task, timeout=20)
        measurement.write_snapshot()
        if connected:
            await connection.disconnect()
    return measurement.result()


def main() -> None:
    try:
        result = asyncio.run(run())
    except Exception as exc:
        result = {
            "test_id": "PEAR-EVIDENCE-001",
            "mode": "read_only_no_motion_imports",
            "robot_ip": ROBOT_IP,
            "measurement_complete": False,
            "passed": False,
            "motion_commands_sent": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(f"PEAR_EVIDENCE_RESULT={json.dumps(result, sort_keys=True)}", flush=True)
    print(
        f"PEAR_EVIDENCE_SNAPSHOT=http://woof.local:{SNAPSHOT_PORT}/{SNAPSHOT_PATH}",
        flush=True,
    )
    ThreadingHTTPServer(
        ("0.0.0.0", SNAPSHOT_PORT),
        SimpleHTTPRequestHandler,
    ).serve_forever()


if __name__ == "__main__":
    main()
