"""Read-only Go2 camera, pear inference, and bark sidecar.

The process intentionally owns no Unitree motion client. It exposes only fresh
source/detection evidence and the preloaded bark action.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response


def configure_media_logging() -> None:
    """Prevent recoverable decoder packet errors from flooding device logs."""
    logging.getLogger("aiortc.codecs.h264").setLevel(logging.ERROR)


class PerceptionEvidence:
    def __init__(self, *, generation: str) -> None:
        self.generation = generation
        self._lock = Lock()
        self._source: dict[str, object] = {}
        self._detection: dict[str, object] = {}
        self._source_count = 0
        self._detection_count = 0
        self._last_source_pts: int | None = None
        self._last_detection_pts: int | None = None
        self._last_source_received_s: float | None = None
        self._time_base: str | None = None
        self._error: str | None = None

    def note_source(
        self,
        *,
        pts: int,
        time_base: str,
        received_monotonic_s: float,
        width: int,
        height: int,
    ) -> None:
        with self._lock:
            advances = (
                self._last_source_pts is None
                or (
                    pts > self._last_source_pts
                    and time_base == self._time_base
                    and self._last_source_received_s is not None
                    and 0.0
                    < received_monotonic_s - self._last_source_received_s
                    <= 0.350
                )
            )
            self._source_count = self._source_count + 1 if advances else 1
            if not advances:
                self._detection_count = 0
                self._detection = {}
            self._last_source_pts = pts
            self._last_source_received_s = received_monotonic_s
            self._time_base = time_base
            self._source = {
                "pts": pts,
                "time_base": time_base,
                "received_monotonic_s": received_monotonic_s,
                "consecutive_frames": self._source_count,
                "width": width,
                "height": height,
            }

    def note_detection(
        self,
        *,
        pts: int,
        label: str,
        confidence: float,
        bbox_xyxy: tuple[int, int, int, int],
        inference_s: float,
        completed_monotonic_s: float,
    ) -> None:
        with self._lock:
            advances = self._last_detection_pts is None or pts > self._last_detection_pts
            qualified = label.casefold().strip() == "pear" and confidence >= 0.65
            self._detection_count = (
                self._detection_count + 1 if advances and qualified else 0
            )
            self._last_detection_pts = pts
            self._detection = {
                "source_pts": pts,
                "source_time_base": self._time_base,
                "generation": self.generation,
                "label": label,
                "confidence": confidence,
                "consecutive_detections": self._detection_count,
                "inference_s": inference_s,
                "completed_monotonic_s": completed_monotonic_s,
                "bbox_xyxy": list(bbox_xyxy),
            }

    def note_miss(self) -> None:
        with self._lock:
            self._detection_count = 0
            self._detection = {}

    def fail(self, error: str) -> None:
        with self._lock:
            self._error = error

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "generation": self.generation,
                "source": dict(self._source),
                "detection": dict(self._detection),
                "error": self._error,
            }


class PerceptionRuntime:
    def __init__(self) -> None:
        self.robot_ip = os.environ.get("UNITREE_ROBOT_IP", "192.168.123.161")
        self.model_path = os.environ.get("PEAR_MODEL_PATH", "/media/model.engine")
        self.bark_uuid = os.environ.get(
            "BORDER_COLLIE_BARK_UUID",
            "161387de-21ab-4f0b-b4e9-97124b000d06",
        )
        self.evidence = PerceptionEvidence(generation=uuid4().hex)
        self._frames: asyncio.Queue[tuple[Any, float, int]] = asyncio.Queue(maxsize=1)
        self._connection: Any | None = None
        self._audiohub: Any | None = None
        self._detector_task: asyncio.Task[None] | None = None
        self._model: Any | None = None
        self._pear_class_id: int | None = None
        self._preview_lock = Lock()
        self._preview_jpeg: bytes | None = None

    async def start(self) -> None:
        configure_media_logging()
        from ultralytics import YOLO
        from unitree_webrtc_connect import (
            UnitreeWebRTCConnection,
            WebRTCConnectionMethod,
        )
        from unitree_webrtc_connect.webrtc_audiohub import WebRTCAudioHub

        self._model = await asyncio.to_thread(YOLO, self.model_path, task="segment")
        names = getattr(self._model, "names", {})
        items = names.items() if isinstance(names, dict) else enumerate(names)
        self._pear_class_id = next(
            int(class_id)
            for class_id, label in items
            if str(label).casefold().strip() == "pear"
        )
        self._connection = UnitreeWebRTCConnection(
            WebRTCConnectionMethod.LocalSTA,
            ip=self.robot_ip,
        )
        await self._connection.connect()
        self._audiohub = WebRTCAudioHub(self._connection)
        self._connection.video.add_track_callback(self._consume_camera)
        self._connection.video.switchVideoChannel(True)
        self._detector_task = asyncio.create_task(self._detect())

    async def close(self) -> None:
        if self._detector_task is not None:
            self._detector_task.cancel()
            await asyncio.gather(self._detector_task, return_exceptions=True)
        if self._connection is not None:
            await self._connection.disconnect()

    async def bark(self) -> dict[str, object]:
        if self._audiohub is None:
            raise RuntimeError("Go2 AudioHub is not connected")
        await self._audiohub.play_by_uuid(self.bark_uuid)
        return {"ok": True, "uuid": self.bark_uuid}

    def status(self) -> dict[str, object]:
        return {
            **self.evidence.status(),
            "bark_ready": self._audiohub is not None and bool(self.bark_uuid),
        }

    def camera_frame(self) -> bytes:
        with self._preview_lock:
            if self._preview_jpeg is None:
                raise RuntimeError("annotated camera preview is not ready")
            return self._preview_jpeg

    async def _consume_camera(self, track: Any) -> None:
        try:
            while True:
                frame = await track.recv()
                received = time.monotonic()
                if frame.pts is None or frame.time_base is None:
                    self.evidence.fail("camera frame has no PTS or time base")
                    continue
                pts = int(frame.pts)
                width = int(frame.width)
                height = int(frame.height)
                self.evidence.note_source(
                    pts=pts,
                    time_base=str(frame.time_base),
                    received_monotonic_s=received,
                    width=width,
                    height=height,
                )
                if self._frames.full():
                    self._frames.get_nowait()
                self._frames.put_nowait((frame, received, pts))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - WebRTC errors are untyped
            self.evidence.fail(f"camera failure: {type(exc).__name__}: {exc}")

    async def _detect(self) -> None:
        assert self._model is not None and self._pear_class_id is not None
        while True:
            frame, _received, pts = await self._frames.get()
            bgr = frame.to_ndarray(format="bgr24")
            started = time.monotonic()
            try:
                results = await asyncio.to_thread(
                    self._model.predict,
                    source=bgr,
                    conf=0.01,
                    classes=[self._pear_class_id],
                    device=0,
                    verbose=False,
                )
            except Exception as exc:  # noqa: BLE001 - detector errors are untyped
                self.evidence.fail(f"detector failure: {type(exc).__name__}: {exc}")
                self._publish_preview(bgr, message="MODEL ERROR", color=(0, 0, 255))
                continue
            completed = time.monotonic()
            result = results[0] if results else None
            boxes = None if result is None else getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                self.evidence.note_miss()
                self._publish_preview(
                    bgr,
                    message="SEARCHING FOR PEAR",
                    color=(0, 191, 255),
                )
                continue
            confidences = boxes.conf.detach().cpu().tolist()
            best_index = max(range(len(confidences)), key=confidences.__getitem__)
            confidence = float(confidences[best_index])
            coordinates = boxes.xyxy[best_index].detach().cpu().tolist()
            bbox = tuple(round(float(value)) for value in coordinates)
            self.evidence.note_detection(
                pts=pts,
                label="pear",
                confidence=confidence,
                bbox_xyxy=bbox,
                inference_s=completed - started,
                completed_monotonic_s=completed,
            )
            self._publish_preview(
                bgr,
                bbox_xyxy=bbox,
                message=(
                    f"PEAR {confidence:.0%} | "
                    f"{self.evidence.status()['detection']['consecutive_detections']}/5"
                ),
                color=(0, 200, 0),
            )

    def _publish_preview(
        self,
        bgr: Any,
        *,
        message: str,
        color: tuple[int, int, int],
        bbox_xyxy: tuple[int, int, int, int] | None = None,
    ) -> None:
        import cv2

        preview = bgr.copy()
        if bbox_xyxy is not None:
            x1, y1, x2, y2 = bbox_xyxy
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 4)
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 92), (20, 20, 20), -1)
        cv2.putText(
            preview,
            "YOLO PEAR MODEL: LIVE",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            message,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )
        encoded, jpeg = cv2.imencode(
            ".jpg",
            preview,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )
        if not encoded:
            self.evidence.fail("camera preview JPEG encoding failed")
            return
        with self._preview_lock:
            self._preview_jpeg = jpeg.tobytes()


def create_app(runtime: PerceptionRuntime | None = None) -> FastAPI:
    media = runtime or PerceptionRuntime()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await media.start()
        try:
            yield
        finally:
            await media.close()

    app = FastAPI(title="Border Collie Media", lifespan=lifespan)

    @app.get("/status")
    async def status() -> dict[str, object]:
        return media.status()

    @app.post("/api/bark")
    async def bark() -> dict[str, object]:
        try:
            return await media.bark()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/camera/frame.jpg")
    async def camera_frame() -> Response:
        try:
            jpeg = media.camera_frame()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    return app


app = create_app()
