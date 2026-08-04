"""Read-only Go2 camera, pear inference, and bark sidecar.

The process intentionally owns no Unitree motion client. It exposes only fresh
source/detection evidence and the preloaded bark action.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
import time
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from functools import partial
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response


@dataclass(frozen=True)
class CropConfirmConfig:
    enabled: bool = True
    minimum_candidate_confidence: float = 0.35
    uncertain_below_confidence: float = 0.65
    small_bbox_area_ratio: float = 0.005
    minimum_crop_side_px: int = 256
    context_scale: float = 6.0
    minimum_confirmation_confidence: float = 0.55
    minimum_agreement_iou: float = 0.10

    def __post_init__(self) -> None:
        confidences = (
            self.minimum_candidate_confidence,
            self.uncertain_below_confidence,
            self.minimum_confirmation_confidence,
            self.minimum_agreement_iou,
        )
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in confidences):
            raise ValueError("crop-confirm confidence and IoU values must be in [0, 1]")
        if not (
            math.isfinite(self.small_bbox_area_ratio)
            and 0.0 < self.small_bbox_area_ratio <= 1.0
        ):
            raise ValueError("crop-confirm small-area ratio must be in (0, 1]")
        if self.minimum_crop_side_px < 1:
            raise ValueError("crop-confirm minimum crop side must be positive")
        if not math.isfinite(self.context_scale) or self.context_scale <= 1.0:
            raise ValueError("crop-confirm context scale must be greater than one")

    @classmethod
    def from_env(cls) -> CropConfirmConfig:
        return cls(
            enabled=os.environ.get("PEAR_CROP_CONFIRM_ENABLED", "1")
            .strip()
            .casefold()
            in {"1", "true", "yes", "on"},
            minimum_candidate_confidence=float(
                os.environ.get("PEAR_CROP_CONFIRM_MIN_CANDIDATE_CONFIDENCE", "0.35")
            ),
            uncertain_below_confidence=float(
                os.environ.get("PEAR_CROP_CONFIRM_UNCERTAIN_BELOW_CONFIDENCE", "0.65")
            ),
            small_bbox_area_ratio=float(
                os.environ.get("PEAR_CROP_CONFIRM_SMALL_AREA_RATIO", "0.005")
            ),
            minimum_crop_side_px=int(
                os.environ.get("PEAR_CROP_CONFIRM_MIN_SIDE_PX", "256")
            ),
            context_scale=float(
                os.environ.get("PEAR_CROP_CONFIRM_CONTEXT_SCALE", "6.0")
            ),
            minimum_confirmation_confidence=float(
                os.environ.get("PEAR_CROP_CONFIRM_MIN_CONFIDENCE", "0.55")
            ),
            minimum_agreement_iou=float(
                os.environ.get("PEAR_CROP_CONFIRM_MIN_IOU", "0.10")
            ),
        )


@dataclass(frozen=True)
class PearCandidate:
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


def configure_media_logging() -> None:
    """Prevent recoverable decoder packet errors from flooding device logs."""
    logging.getLogger("aiortc.codecs.h264").setLevel(logging.ERROR)


class EvidenceFrameBuffer:
    """Bounded raw-frame history for review and Fieldmark labeling."""

    def __init__(self, *, generation: str, maximum_frames: int = 40) -> None:
        if maximum_frames < 1:
            raise ValueError("maximum_frames must be positive")
        self.generation = generation
        self._frames: deque[dict[str, object]] = deque(maxlen=maximum_frames)
        self._lock = Lock()

    def record(
        self,
        *,
        raw_jpeg: bytes,
        annotated_jpeg: bytes,
        pts: int,
        time_base: str,
        received_monotonic_s: float,
        width: int,
        height: int,
        detection: dict[str, object],
    ) -> None:
        frame = {
            "raw_jpeg": bytes(raw_jpeg),
            "annotated_jpeg": bytes(annotated_jpeg),
            "source_pts": int(pts),
            "source_time_base": str(time_base),
            "received_monotonic_s": float(received_monotonic_s),
            "width": int(width),
            "height": int(height),
            "detection": dict(detection),
        }
        with self._lock:
            self._frames.append(frame)

    def raw_camera_frame(self) -> bytes:
        with self._lock:
            if not self._frames:
                raise RuntimeError("raw camera frame is not ready")
            return bytes(self._frames[-1]["raw_jpeg"])

    def archive(self) -> bytes:
        with self._lock:
            frames = [dict(frame) for frame in self._frames]
        if not frames:
            raise RuntimeError("camera evidence history is not ready")

        manifest_frames: list[dict[str, object]] = []
        closest: dict[str, object] | None = None
        maximum_confidence: float | None = None
        pear_candidates = 0
        for index, frame in enumerate(frames, start=1):
            detection = frame["detection"]
            assert isinstance(detection, dict)
            confidence = _finite_number(detection.get("confidence"))
            bbox = _valid_bbox(
                detection.get("bbox_xyxy"),
                int(frame["width"]),
                int(frame["height"]),
            )
            bbox_area_ratio = None
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                bbox_area_ratio = ((x2 - x1) * (y2 - y1)) / (
                    int(frame["width"]) * int(frame["height"])
                )
            if str(detection.get("label") or "").casefold() == "pear":
                pear_candidates += 1
            if confidence is not None:
                maximum_confidence = (
                    confidence
                    if maximum_confidence is None
                    else max(maximum_confidence, confidence)
                )
            if bbox_area_ratio is not None and (
                closest is None
                or bbox_area_ratio > float(closest["bbox_area_ratio"])
            ):
                closest = {
                    "source_pts": frame["source_pts"],
                    "confidence": confidence,
                    "bbox_xyxy": list(bbox) if bbox is not None else None,
                    "bbox_area_ratio": bbox_area_ratio,
                }
            manifest_frames.append(
                {
                    "filename": f"frames/{index:06d}.jpg",
                    "source_pts": frame["source_pts"],
                    "source_time_base": frame["source_time_base"],
                    "received_monotonic_s": frame["received_monotonic_s"],
                    "width": frame["width"],
                    "height": frame["height"],
                    "detection": {
                        **detection,
                        "bbox_area_ratio": bbox_area_ratio,
                    },
                }
            )

        manifest = {
            "schema_version": 1,
            "generation": self.generation,
            "format": "fieldmark-image-sequence",
            "frames": manifest_frames,
            "summary": {
                "sample_count": len(frames),
                "pear_candidate_samples": pear_candidates,
                "maximum_confidence": maximum_confidence,
                "maximum_bbox_area_ratio": (
                    None if closest is None else closest["bbox_area_ratio"]
                ),
                "closest_detection": closest,
            },
        }
        destination = io.BytesIO()
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2))
            for index, frame in enumerate(frames, start=1):
                archive.writestr(f"frames/{index:06d}.jpg", frame["raw_jpeg"])
            archive.writestr("terminal/annotated.jpg", frames[-1]["annotated_jpeg"])
        return destination.getvalue()


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _valid_bbox(
    value: object,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    values = tuple(_finite_number(item) for item in value)
    if any(item is None for item in values):
        return None
    x1, y1, x2, y2 = (float(item) for item in values if item is not None)
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        return None
    return x1, y1, x2, y2


def _best_candidate(
    results: object,
    *,
    x_offset: int = 0,
    y_offset: int = 0,
) -> PearCandidate | None:
    if not isinstance(results, (list, tuple)) or not results:
        return None
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    confidences = boxes.conf.detach().cpu().tolist()
    best_index = max(range(len(confidences)), key=confidences.__getitem__)
    confidence = float(confidences[best_index])
    coordinates = boxes.xyxy[best_index].detach().cpu().tolist()
    x1, y1, x2, y2 = (round(float(value)) for value in coordinates)
    return PearCandidate(
        confidence=confidence,
        bbox_xyxy=(
            x1 + x_offset,
            y1 + y_offset,
            x2 + x_offset,
            y2 + y_offset,
        ),
    )


def _bbox_area_ratio(
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> float:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1) / (width * height)


def _expanded_square_crop(
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    minimum_side_px: int,
    context_scale: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    desired_side = max(
        float(minimum_side_px),
        max(x2 - x1, y2 - y1) * context_scale,
    )
    side = max(1, min(round(desired_side), width, height))
    crop_x1 = round(center_x - side / 2.0)
    crop_y1 = round(center_y - side / 2.0)
    crop_x1 = max(0, min(crop_x1, width - side))
    crop_y1 = max(0, min(crop_y1, height - side))
    return crop_x1, crop_y1, crop_x1 + side, crop_y1 + side


def _bbox_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(
        0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


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
        details: dict[str, object] | None = None,
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
                **(details or {}),
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
        self._frames: asyncio.Queue[tuple[Any, float, int, str]] = asyncio.Queue(
            maxsize=1
        )
        self._evidence_frames = EvidenceFrameBuffer(
            generation=self.evidence.generation,
            maximum_frames=int(os.environ.get("EVIDENCE_MAXIMUM_FRAMES", "40")),
        )
        self._evidence_interval_s = float(
            os.environ.get("EVIDENCE_FRAME_INTERVAL_S", "0.5")
        )
        self._crop_confirm = CropConfirmConfig.from_env()
        self._last_evidence_capture_s: float | None = None
        self._connection: Any | None = None
        self._audiohub: Any | None = None
        self._detector_task: asyncio.Task[None] | None = None
        self._model: Any | None = None
        self._pear_class_id: int | None = None
        self._inference_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="border-collie-inference",
        )
        self._inference_executor_closed = False
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

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(
            self._inference_executor,
            partial(YOLO, self.model_path, task="segment"),
        )
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
        if not self._inference_executor_closed:
            await asyncio.to_thread(
                self._inference_executor.shutdown,
                wait=True,
                cancel_futures=True,
            )
            self._inference_executor_closed = True

    async def bark(self) -> dict[str, object]:
        if self._audiohub is None:
            raise RuntimeError("Go2 AudioHub is not connected")
        await self._audiohub.play_by_uuid(self.bark_uuid)
        return {"ok": True, "uuid": self.bark_uuid}

    def status(self) -> dict[str, object]:
        return {
            **self.evidence.status(),
            "bark_ready": self._audiohub is not None and bool(self.bark_uuid),
            "crop_confirm": asdict(self._crop_confirm),
        }

    def camera_frame(self) -> bytes:
        with self._preview_lock:
            if self._preview_jpeg is None:
                raise RuntimeError("annotated camera preview is not ready")
            return self._preview_jpeg

    def raw_camera_frame(self) -> bytes:
        return self._evidence_frames.raw_camera_frame()

    def evidence_archive(self) -> bytes:
        return self._evidence_frames.archive()

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
                self._frames.put_nowait((frame, received, pts, str(frame.time_base)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - WebRTC errors are untyped
            self.evidence.fail(f"camera failure: {type(exc).__name__}: {exc}")

    async def _detect(self) -> None:
        assert self._model is not None and self._pear_class_id is not None
        loop = asyncio.get_running_loop()
        while True:
            frame, received, pts, time_base = await self._frames.get()
            await loop.run_in_executor(
                self._inference_executor,
                self._process_frame,
                frame,
                received,
                pts,
                time_base,
            )

    def _process_frame(
        self,
        frame: Any,
        received_monotonic_s: float,
        pts: int,
        time_base: str,
    ) -> None:
        """Own all frame conversion, model, postprocess, and preview work."""
        assert self._model is not None and self._pear_class_id is not None
        bgr = frame.to_ndarray(format="bgr24")
        started = time.monotonic()
        try:
            results = self._model.predict(
                source=bgr,
                conf=0.01,
                classes=[self._pear_class_id],
                device=0,
                verbose=False,
            )
            candidate = _best_candidate(results)
            inference_passes = 1
            crop_confirmation: dict[str, object] = {
                "attempted": False,
                "promoted": False,
                "full_frame_confidence": (
                    None if candidate is None else candidate.confidence
                ),
                "crop_confidence": None,
                "crop_xyxy": None,
                "agreement_iou": None,
            }
            if candidate is not None and self._should_crop_confirm(
                candidate,
                width=int(bgr.shape[1]),
                height=int(bgr.shape[0]),
            ):
                crop_xyxy = _expanded_square_crop(
                    candidate.bbox_xyxy,
                    width=int(bgr.shape[1]),
                    height=int(bgr.shape[0]),
                    minimum_side_px=self._crop_confirm.minimum_crop_side_px,
                    context_scale=self._crop_confirm.context_scale,
                )
                crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy
                cropped_bgr = bgr[crop_y1:crop_y2, crop_x1:crop_x2]
                crop_results = self._model.predict(
                    source=cropped_bgr,
                    conf=0.01,
                    classes=[self._pear_class_id],
                    device=0,
                    verbose=False,
                )
                inference_passes = 2
                crop_candidate = _best_candidate(
                    crop_results,
                    x_offset=crop_x1,
                    y_offset=crop_y1,
                )
                agreement_iou = (
                    None
                    if crop_candidate is None
                    else _bbox_iou(
                        candidate.bbox_xyxy,
                        crop_candidate.bbox_xyxy,
                    )
                )
                promoted = bool(
                    crop_candidate is not None
                    and crop_candidate.confidence
                    >= self._crop_confirm.minimum_confirmation_confidence
                    and crop_candidate.confidence > candidate.confidence
                    and agreement_iou is not None
                    and agreement_iou >= self._crop_confirm.minimum_agreement_iou
                )
                crop_confirmation = {
                    "attempted": True,
                    "promoted": promoted,
                    "full_frame_confidence": candidate.confidence,
                    "crop_confidence": (
                        None if crop_candidate is None else crop_candidate.confidence
                    ),
                    "crop_xyxy": list(crop_xyxy),
                    "agreement_iou": agreement_iou,
                }
                if promoted:
                    assert crop_candidate is not None
                    candidate = crop_candidate
        except Exception as exc:  # noqa: BLE001 - detector errors are untyped
            self.evidence.fail(f"detector failure: {type(exc).__name__}: {exc}")
            self._publish_preview(
                bgr,
                message="MODEL ERROR",
                color=(0, 0, 255),
                pts=pts,
                time_base=time_base,
                received_monotonic_s=received_monotonic_s,
                detection={},
            )
            return
        completed = time.monotonic()
        if candidate is None:
            self.evidence.note_miss()
            self._publish_preview(
                bgr,
                message="SEARCHING FOR PEAR",
                color=(0, 191, 255),
                pts=pts,
                time_base=time_base,
                received_monotonic_s=received_monotonic_s,
                detection={},
            )
            return
        confidence = candidate.confidence
        bbox = candidate.bbox_xyxy
        self.evidence.note_detection(
            pts=pts,
            label="pear",
            confidence=confidence,
            bbox_xyxy=bbox,
            inference_s=completed - started,
            completed_monotonic_s=completed,
            details={
                "inference_passes": inference_passes,
                "crop_confirmation": crop_confirmation,
            },
        )
        crop_message = (
            " | CROP CONFIRMED"
            if crop_confirmation["promoted"]
            else " | CROP UNCONFIRMED"
            if crop_confirmation["attempted"]
            else ""
        )
        self._publish_preview(
            bgr,
            bbox_xyxy=bbox,
            message=(
                f"PEAR {confidence:.0%} | "
                f"{self.evidence.status()['detection']['consecutive_detections']}/5"
                f"{crop_message}"
            ),
            color=(0, 200, 0),
            pts=pts,
            time_base=time_base,
            received_monotonic_s=received_monotonic_s,
            detection=dict(self.evidence.status()["detection"]),
        )

    def _should_crop_confirm(
        self,
        candidate: PearCandidate,
        *,
        width: int,
        height: int,
    ) -> bool:
        if not self._crop_confirm.enabled:
            return False
        if candidate.confidence < self._crop_confirm.minimum_candidate_confidence:
            return False
        area_ratio = _bbox_area_ratio(
            candidate.bbox_xyxy,
            width=width,
            height=height,
        )
        return (
            candidate.confidence < self._crop_confirm.uncertain_below_confidence
            or area_ratio <= self._crop_confirm.small_bbox_area_ratio
        )

    def _publish_preview(
        self,
        bgr: Any,
        *,
        message: str,
        color: tuple[int, int, int],
        pts: int,
        time_base: str,
        received_monotonic_s: float,
        detection: dict[str, object],
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
        if (
            self._last_evidence_capture_s is not None
            and received_monotonic_s - self._last_evidence_capture_s
            < self._evidence_interval_s
        ):
            return
        raw_encoded, raw_jpeg = cv2.imencode(
            ".jpg",
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )
        if not raw_encoded:
            self.evidence.fail("raw camera evidence JPEG encoding failed")
            return
        self._evidence_frames.record(
            raw_jpeg=raw_jpeg.tobytes(),
            annotated_jpeg=jpeg.tobytes(),
            pts=pts,
            time_base=time_base,
            received_monotonic_s=received_monotonic_s,
            width=int(bgr.shape[1]),
            height=int(bgr.shape[0]),
            detection=detection,
        )
        self._last_evidence_capture_s = received_monotonic_s


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

    @app.get("/api/camera/raw.jpg")
    async def raw_camera_frame() -> Response:
        try:
            jpeg = media.raw_camera_frame()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/evidence/clip.zip")
    async def evidence_archive() -> Response:
        try:
            archive = await asyncio.to_thread(media.evidence_archive)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(
            content=archive,
            media_type="application/zip",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": (
                    'attachment; filename="border-collie-evidence.zip"'
                ),
            },
        )

    return app


app = create_app()
