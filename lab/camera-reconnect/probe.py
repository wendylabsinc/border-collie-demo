"""Read-only WebRTC reconnect and pear-evidence probe for Woof."""

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
import uuid

from unitree_webrtc_connect import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from ultralytics import YOLO


ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "192.168.123.161")
MODEL_PATH = Path(os.environ.get("PEAR_MODEL_PATH", "/probe/model.engine"))
THRESHOLD = float(os.environ.get("PEAR_CONFIDENCE_THRESHOLD", "0.35"))
MINIMUM_SAMPLES = int(os.environ.get("RECONNECT_MINIMUM_SAMPLES", "30"))
REQUIRED_RATIO = float(
    os.environ.get("RECONNECT_REQUIRED_MATCH_RATIO", "0.80")
)
GENERATION_TIMEOUT_S = float(
    os.environ.get("RECONNECT_GENERATION_TIMEOUT_S", "30")
)
OFFLINE_HOLD_S = float(os.environ.get("RECONNECT_OFFLINE_HOLD_S", "2"))
SNAPSHOT_PATH = Path(os.environ.get("RECONNECT_SNAPSHOT_PATH", "snapshot.jpg"))
SNAPSHOT_PORT = int(os.environ.get("RECONNECT_SNAPSHOT_PORT", "8124"))


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


class Generation:
    def __init__(self, label: str, model: YOLO, pear_class_id: int) -> None:
        self.label = label
        self.id = uuid.uuid4().hex
        self.model = model
        self.pear_class_id = pear_class_id
        self.connect_attempt_s = time.monotonic()
        self.connect_completed_s: float | None = None
        self.first_frame_s: float | None = None
        self.first_match_s: float | None = None
        self.preflight_ready_s: float | None = None
        self.disconnect_started_s: float | None = None
        self.disconnect_completed_s: float | None = None
        self.first_frame = asyncio.Event()
        self.stop = asyncio.Event()
        self.expected_disconnect = False
        self.frames: asyncio.Queue[tuple[Any, float, int | None, str | None]] = (
            asyncio.Queue(maxsize=1)
        )
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
        self.first_match_marker: dict[str, object] | None = None
        self.best_confidence = -1.0
        self.best_annotated: Any | None = None
        self.latest_bgr: Any | None = None

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

                if self.first_frame_s is None:
                    self.first_frame_s = received_s
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
                self.frames.put_nowait((frame, received_s, pts, time_base))
                self.first_frame.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self.stop.is_set() and not self.expected_disconnect:
                self.camera_error = f"{type(exc).__name__}: {exc}"
                self.first_frame.set()

    async def detect(self) -> None:
        while not self.stop.is_set() or not self.frames.empty():
            try:
                frame, received_s, pts, time_base = await asyncio.wait_for(
                    self.frames.get(), timeout=0.25
                )
            except TimeoutError:
                continue
            try:
                await asyncio.to_thread(
                    self._detect_frame,
                    frame,
                    received_s,
                    pts,
                    time_base,
                )
            except Exception as exc:
                self.detector_error = f"{type(exc).__name__}: {exc}"
                self.stop.set()
                return

    def _detect_frame(
        self,
        frame: Any,
        received_s: float,
        pts: int | None,
        time_base: str | None,
    ) -> None:
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
            if self.first_match_s is None:
                self.first_match_s = completed_s
                self.first_match_marker = {
                    "generation": self.id,
                    "pts": pts,
                    "time_base": time_base,
                }
            if best > self.best_confidence:
                self.best_confidence = best
                self.best_annotated = result.plot()
        else:
            self.current_miss_run += 1
            self.maximum_miss_run = max(
                self.maximum_miss_run,
                self.current_miss_run,
            )

    def source_is_strict(self) -> bool:
        return bool(
            self.decoded_frames > 1
            and self.pts_missing == 0
            and self.time_base_missing == 0
            and self.pts_repeated == 0
            and self.pts_regressed == 0
            and self.time_base_changes == 0
        )

    def match_ratio(self) -> float:
        return self.matches / self.detector_samples if self.detector_samples else 0.0

    def is_ready(self) -> bool:
        return bool(
            self.source_is_strict()
            and not self.camera_error
            and not self.detector_error
            and self.detector_samples >= MINIMUM_SAMPLES
            and self.match_ratio() >= REQUIRED_RATIO
            and self.first_match_marker is not None
        )

    def result(self, test_started_s: float) -> dict[str, object]:
        def since_test(value: float | None) -> float | None:
            return None if value is None else round(value - test_started_s, 6)

        def since_connect(value: float | None) -> float | None:
            return None if value is None else round(value - self.connect_attempt_s, 6)

        return {
            "label": self.label,
            "generation": self.id,
            "connect_attempt_at_s": since_test(self.connect_attempt_s),
            "connect_completed_after_s": since_connect(self.connect_completed_s),
            "first_frame_after_s": since_connect(self.first_frame_s),
            "first_match_after_s": since_connect(self.first_match_s),
            "preflight_ready_after_s": since_connect(self.preflight_ready_s),
            "disconnect_started_at_s": since_test(self.disconnect_started_s),
            "disconnect_duration_s": (
                None
                if self.disconnect_started_s is None
                or self.disconnect_completed_s is None
                else round(
                    self.disconnect_completed_s - self.disconnect_started_s,
                    6,
                )
            ),
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
                "strict_source_identity": self.source_is_strict(),
                "error": self.camera_error,
            },
            "recognition": {
                "eligible_samples": self.detector_samples,
                "matches": self.matches,
                "misses": self.detector_samples - self.matches,
                "match_ratio": round(self.match_ratio(), 6),
                "confidence": _stats(self.confidences),
                "maximum_consecutive_misses": self.maximum_miss_run,
                "first_match_marker": self.first_match_marker,
            },
            "inference_s": _stats(self.inference_s),
            "detection_age_s": _stats(self.detection_age_s),
            "detector_error": self.detector_error,
            "preflight_passed": self.is_ready(),
        }


def _pear_class_id(model: YOLO) -> int:
    names = getattr(model, "names", {})
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for class_id, label in items:
        if str(label).casefold().strip() == "pear":
            return int(class_id)
    raise RuntimeError(f"pear class missing from model names: {names!r}")


async def _run_generation(
    label: str,
    model: YOLO,
    pear_class_id: int,
) -> Generation:
    generation = Generation(label, model, pear_class_id)
    connection = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip=ROBOT_IP,
    )
    connected = False
    detector_task: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(connection.connect(), timeout=GENERATION_TIMEOUT_S)
        connected = True
        generation.connect_completed_s = time.monotonic()
        connection.video.add_track_callback(generation.consume_camera)
        connection.video.switchVideoChannel(True)
        await asyncio.wait_for(
            generation.first_frame.wait(),
            timeout=GENERATION_TIMEOUT_S,
        )
        if generation.camera_error:
            raise RuntimeError(generation.camera_error)
        detector_task = asyncio.create_task(generation.detect())
        deadline_s = generation.connect_attempt_s + GENERATION_TIMEOUT_S
        while time.monotonic() < deadline_s:
            if generation.camera_error or generation.detector_error:
                break
            if generation.is_ready():
                generation.preflight_ready_s = time.monotonic()
                break
            await asyncio.sleep(0.05)
    finally:
        generation.stop.set()
        if detector_task is not None:
            await asyncio.wait_for(detector_task, timeout=20)
        generation.expected_disconnect = True
        if connected:
            generation.disconnect_started_s = time.monotonic()
            await connection.disconnect()
            generation.disconnect_completed_s = time.monotonic()
    return generation


def _write_snapshot(generation: Generation) -> tuple[bool, str]:
    image = (
        generation.best_annotated
        if generation.best_annotated is not None
        else generation.latest_bgr
    )
    if image is None:
        return False, "no replacement-generation inference frame available"
    try:
        from PIL import Image

        Image.fromarray(image[:, :, ::-1]).save(
            SNAPSHOT_PATH,
            format="JPEG",
            quality=85,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


async def run() -> dict[str, object]:
    test_started_s = time.monotonic()
    model = YOLO(str(MODEL_PATH), task="segment")
    pear_class_id = _pear_class_id(model)
    baseline = await _run_generation("baseline", model, pear_class_id)
    outage_started_s = baseline.disconnect_started_s or time.monotonic()
    await asyncio.sleep(OFFLINE_HOLD_S)
    replacement = await _run_generation("replacement", model, pear_class_id)
    snapshot_written, snapshot_error = _write_snapshot(replacement)
    generations_are_distinct = baseline.id != replacement.id
    replacement_marker_is_bound = bool(
        replacement.first_match_marker
        and replacement.first_match_marker.get("generation") == replacement.id
    )
    passed = bool(
        baseline.is_ready()
        and replacement.is_ready()
        and generations_are_distinct
        and replacement_marker_is_bound
        and snapshot_written
    )
    return {
        "test_id": "CAMERA-RECONNECT-001",
        "mode": "read_only_no_motion_imports",
        "robot_ip": ROBOT_IP,
        "model": {
            "path": str(MODEL_PATH),
            "pear_class_id": pear_class_id,
            "confidence_threshold": THRESHOLD,
        },
        "acceptance": {
            "minimum_samples_per_generation": MINIMUM_SAMPLES,
            "required_match_ratio": REQUIRED_RATIO,
            "generation_timeout_s": GENERATION_TIMEOUT_S,
        },
        "planned_disconnect": {
            "type": "client_initiated_webrtc_disconnect",
            "offline_hold_requested_s": OFFLINE_HOLD_S,
            "outage_to_reconnect_attempt_s": round(
                replacement.connect_attempt_s - outage_started_s,
                6,
            ),
            "outage_to_first_replacement_frame_s": (
                None
                if replacement.first_frame_s is None
                else round(replacement.first_frame_s - outage_started_s, 6)
            ),
            "outage_to_replacement_preflight_s": (
                None
                if replacement.preflight_ready_s is None
                else round(replacement.preflight_ready_s - outage_started_s, 6)
            ),
        },
        "baseline": baseline.result(test_started_s),
        "replacement": replacement.result(test_started_s),
        "generation_boundary": {
            "distinct": generations_are_distinct,
            "replacement_detection_bound_to_replacement_generation": (
                replacement_marker_is_bound
            ),
            "active_run_resumed": False,
        },
        "snapshot": {
            "path": str(SNAPSHOT_PATH),
            "port": SNAPSHOT_PORT,
            "generation": replacement.id,
            "written": snapshot_written,
            "error": snapshot_error,
        },
        "observed_duration_s": round(time.monotonic() - test_started_s, 3),
        "passed": passed,
        "measurement_complete": True,
        "motion_commands_sent": False,
    }


def main() -> None:
    try:
        result = asyncio.run(run())
    except Exception as exc:
        result = {
            "test_id": "CAMERA-RECONNECT-001",
            "mode": "read_only_no_motion_imports",
            "robot_ip": ROBOT_IP,
            "measurement_complete": False,
            "passed": False,
            "motion_commands_sent": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(
        f"CAMERA_RECONNECT_RESULT={json.dumps(result, sort_keys=True)}",
        flush=True,
    )
    print(
        f"CAMERA_RECONNECT_SNAPSHOT=http://woof.local:{SNAPSHOT_PORT}/{SNAPSHOT_PATH}",
        flush=True,
    )
    ThreadingHTTPServer(
        ("0.0.0.0", SNAPSHOT_PORT),
        SimpleHTTPRequestHandler,
    ).serve_forever()


if __name__ == "__main__":
    main()
