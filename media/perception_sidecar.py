"""Read-only Go2 camera, fruit inference, and bark sidecar.

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
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from media.model_router import FruitCandidate, FruitModelRouter, RoutedPrediction
from media.service_supervision import (
    ServiceState,
    ServiceSupervisionConfig,
    ServiceSupervisor,
)
from media.visual_odometry import SparseVisualOdometry, VisualOdometryConfig

FRUIT_ACQUISITION_CONFIDENCE = {
    "apple": 0.65,
    "banana": 0.20,
    "pear": 0.65,
}
SUPPORTED_FRUITS = tuple(FRUIT_ACQUISITION_CONFIDENCE)
SEARCH_CROP_FRUITS = frozenset({"apple", "banana"})
LOGGER = logging.getLogger(__name__)


def release_cohort_status() -> dict[str, object] | None:
    release_id = os.environ.get("BORDER_COLLIE_RELEASE_ID", "").strip()
    if not release_id:
        return None
    return {
        "release_id": release_id,
        "config_schema": int(os.environ.get("BORDER_COLLIE_CONFIG_SCHEMA", "1")),
        "service": "media",
    }


class TargetFruitRequest(BaseModel):
    target_fruit: Literal["apple", "banana", "pear"]


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
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for value in confidences
        ):
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
            enabled=os.environ.get("PEAR_CROP_CONFIRM_ENABLED", "1").strip().casefold()
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


def configure_media_logging() -> None:
    """Prevent recoverable decoder packet errors from flooding device logs."""
    logging.getLogger("aiortc.codecs.h264").setLevel(logging.ERROR)


def _positive_env_float(name: str, default: str) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


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
                closest is None or bbox_area_ratio > float(closest["bbox_area_ratio"])
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
        with zipfile.ZipFile(
            destination, "w", compression=zipfile.ZIP_STORED
        ) as archive:
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


def _lower_center_search_crop(
    *,
    width: int,
    height: int,
    maximum_side_px: int = 512,
) -> tuple[int, int, int, int]:
    """Return one bounded floor-facing crop for small optional targets."""
    side = max(1, min(maximum_side_px, width, height))
    crop_x1 = max(0, (width - side) // 2)
    crop_y1 = max(0, height - side)
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
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


class PerceptionEvidence:
    def __init__(self, *, generation: str, target_fruit: str = "pear") -> None:
        self.generation = generation
        self._lock = Lock()
        self._target_fruit = target_fruit
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
            advances = self._last_source_pts is None or (
                pts > self._last_source_pts
                and time_base == self._time_base
                and self._last_source_received_s is not None
                and 0.0 < received_monotonic_s - self._last_source_received_s <= 0.350
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
    ) -> dict[str, object] | None:
        with self._lock:
            normalized_label = label.casefold().strip()
            if normalized_label != self._target_fruit:
                return None
            advances = (
                self._last_detection_pts is None or pts > self._last_detection_pts
            )
            qualified = confidence >= FRUIT_ACQUISITION_CONFIDENCE[self._target_fruit]
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
            return dict(self._detection)

    def note_miss(self, target_fruit: str) -> None:
        with self._lock:
            if target_fruit.casefold().strip() != self._target_fruit:
                return
            self._detection_count = 0
            self._detection = {}

    def select_target(self, target_fruit: str) -> None:
        normalized = target_fruit.casefold().strip()
        if normalized not in FRUIT_ACQUISITION_CONFIDENCE:
            raise ValueError(f"unsupported Target Fruit: {target_fruit}")
        with self._lock:
            if normalized == self._target_fruit:
                return
            self._target_fruit = normalized
            self._detection_count = 0
            self._last_detection_pts = None
            self._detection = {}

    def fail(self, error: str) -> None:
        with self._lock:
            self._error = error

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "generation": self.generation,
                "target_fruit": self._target_fruit,
                "supported_fruits": list(SUPPORTED_FRUITS),
                "source": dict(self._source),
                "detection": dict(self._detection),
                "error": self._error,
            }


class PerceptionRuntime:
    def __init__(self) -> None:
        self.robot_ip = os.environ.get("UNITREE_ROBOT_IP", "192.168.123.161")
        self.model_path = os.environ.get("PEAR_MODEL_PATH", "/media/model.engine")
        self.banana_specialist_model_path = os.environ.get(
            "BANANA_SPECIALIST_MODEL_PATH", ""
        ).strip()
        self.banana_specialist_minimum_confidence = float(
            os.environ.get("BANANA_SPECIALIST_MIN_CONFIDENCE", "0.55")
        )
        self.banana_specialist_minimum_agreement_iou = float(
            os.environ.get("BANANA_SPECIALIST_MIN_IOU", "0.10")
        )
        self.bark_uuid = os.environ.get(
            "BORDER_COLLIE_BARK_UUID",
            "161387de-21ab-4f0b-b4e9-97124b000d06",
        )
        self.evidence = PerceptionEvidence(generation=uuid4().hex)
        self._frames: asyncio.Queue[tuple[str, Any, float, int, str]] = asyncio.Queue(
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
        self._supervisor_task: asyncio.Task[None] | None = None
        self._session_failures: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._closing = False
        self._connection_class: Any | None = None
        self._connection_method: Any | None = None
        self._audiohub_class: Any | None = None
        self._connect_timeout_s = _positive_env_float(
            "MEDIA_CONNECT_TIMEOUT_S", "15.0"
        )
        self._cleanup_timeout_s = _positive_env_float(
            "MEDIA_CLEANUP_TIMEOUT_S", "5.0"
        )
        self._monitor_interval_s = _positive_env_float(
            "MEDIA_SUPERVISION_INTERVAL_S", "0.10"
        )
        self._supervision = ServiceSupervisor(
            ServiceSupervisionConfig(
                stable_frame_count=int(
                    os.environ.get("MEDIA_STABLE_FRAME_COUNT", "10")
                ),
                frame_stall_timeout_s=float(
                    os.environ.get("MEDIA_FRAME_STALL_TIMEOUT_S", "0.75")
                ),
                first_frame_timeout_s=float(
                    os.environ.get("MEDIA_FIRST_FRAME_TIMEOUT_S", "3.0")
                ),
                restart_budget=int(
                    os.environ.get("MEDIA_RESTART_BUDGET", "5")
                ),
                initial_backoff_s=float(
                    os.environ.get("MEDIA_INITIAL_BACKOFF_S", "8.0")
                ),
                maximum_backoff_s=float(
                    os.environ.get("MEDIA_MAXIMUM_BACKOFF_S", "30.0")
                ),
            )
        )
        self._model: Any | None = None
        self._banana_specialist_model: Any | None = None
        self._model_router: FruitModelRouter | None = None
        self._fruit_class_ids: dict[str, int] = {}
        self._target_lock = Lock()
        self._target_fruit = "pear"
        self._inference_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="border-collie-inference",
        )
        self._inference_executor_closed = False
        self._preview_lock = Lock()
        self._preview_jpeg: bytes | None = None
        self._visual_odometry = SparseVisualOdometry(
            VisualOdometryConfig.from_env(dict(os.environ))
        )
        self._visual_odometry.reset(self.evidence.generation)

    async def start(self) -> None:
        configure_media_logging()
        from ultralytics import YOLO
        from unitree_webrtc_connect import (
            UnitreeWebRTCConnection,
            WebRTCConnectionMethod,
        )
        from unitree_webrtc_connect.webrtc_audiohub import WebRTCAudioHub

        self._connection_class = UnitreeWebRTCConnection
        self._connection_method = WebRTCConnectionMethod.LocalSTA
        self._audiohub_class = WebRTCAudioHub

        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(
            self._inference_executor,
            partial(YOLO, self.model_path, task="segment"),
        )
        names = getattr(self._model, "names", {})
        items = names.items() if isinstance(names, dict) else enumerate(names)
        available = {
            str(label).casefold().strip(): int(class_id) for class_id, label in items
        }
        self._fruit_class_ids = {
            fruit: available[fruit] for fruit in SUPPORTED_FRUITS if fruit in available
        }
        missing = sorted(set(SUPPORTED_FRUITS) - set(self._fruit_class_ids))
        if missing:
            raise RuntimeError(
                "fruit model is missing required classes: " + ", ".join(missing)
            )
        banana_specialist_class_id = None
        if self.banana_specialist_model_path:
            self._banana_specialist_model = await loop.run_in_executor(
                self._inference_executor,
                partial(YOLO, self.banana_specialist_model_path, task="detect"),
            )
            specialist_names = getattr(self._banana_specialist_model, "names", {})
            specialist_items = (
                specialist_names.items()
                if isinstance(specialist_names, dict)
                else enumerate(specialist_names)
            )
            specialist_classes = {
                str(label).casefold().strip(): int(class_id)
                for class_id, label in specialist_items
            }
            try:
                banana_specialist_class_id = specialist_classes["banana"]
            except KeyError as exc:
                raise RuntimeError(
                    "banana specialist model is missing required banana class"
                ) from exc
        self._model_router = FruitModelRouter(
            general_model=self._model,
            general_class_ids=self._fruit_class_ids,
            banana_specialist_model=self._banana_specialist_model,
            banana_specialist_class_id=banana_specialist_class_id,
            banana_minimum_confidence=self.banana_specialist_minimum_confidence,
            banana_minimum_agreement_iou=(
                self.banana_specialist_minimum_agreement_iou
            ),
        )
        self._detector_task = asyncio.create_task(self._detect())
        self._supervisor_task = asyncio.create_task(self._supervise_sessions())

    async def close(self) -> None:
        self._closing = True
        if self._supervisor_task is not None:
            self._supervisor_task.cancel()
            await asyncio.gather(self._supervisor_task, return_exceptions=True)
            self._supervisor_task = None
        await self._cleanup_session()
        if self._detector_task is not None:
            self._detector_task.cancel()
            await asyncio.gather(self._detector_task, return_exceptions=True)
            self._detector_task = None
        if not self._inference_executor_closed:
            await asyncio.to_thread(
                self._inference_executor.shutdown,
                wait=True,
                cancel_futures=True,
            )
            self._inference_executor_closed = True

    async def bark(self) -> dict[str, object]:
        if self._audiohub is None or not self._supervision.ready:
            raise RuntimeError("Go2 AudioHub is not connected")
        await self._audiohub.play_by_uuid(self.bark_uuid)
        return {"ok": True, "uuid": self.bark_uuid}

    def status(self) -> dict[str, object]:
        self._supervision.check_health()
        supervision = self._supervision.status()
        return {
            **self.evidence.status(),
            "release": release_cohort_status(),
            "supervision": supervision,
            "bark_ready": (
                supervision["ready"] is True
                and self._audiohub is not None
                and bool(self.bark_uuid)
            ),
            "crop_confirm": asdict(self._crop_confirm),
            "model_router": (
                self._model_router.status()
                if self._model_router is not None
                else {
                    "mode": "loading"
                    if self.banana_specialist_model_path
                    else "general_only",
                    "banana_specialist_loaded": False,
                    "banana_minimum_confidence": (
                        self.banana_specialist_minimum_confidence
                    ),
                    "banana_minimum_agreement_iou": (
                        self.banana_specialist_minimum_agreement_iou
                    ),
                }
            ),
            "visual_odometry": self._visual_odometry.status(),
        }

    def select_target(self, target_fruit: str) -> dict[str, object]:
        normalized = target_fruit.casefold().strip()
        if normalized not in FRUIT_ACQUISITION_CONFIDENCE:
            raise ValueError(f"unsupported Target Fruit: {target_fruit}")
        if self._fruit_class_ids and normalized not in self._fruit_class_ids:
            raise RuntimeError(f"model class is unavailable for {normalized}")
        with self._target_lock:
            self._target_fruit = normalized
            self.evidence.select_target(normalized)
        return {
            "target_fruit": normalized,
            "supported_fruits": list(SUPPORTED_FRUITS),
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

    async def _supervise_sessions(self) -> None:
        while not self._closing and self._supervision.state is not ServiceState.FAILED:
            now = time.monotonic()
            if not self._supervision.begin_attempt(now_s=now):
                retry_at = self._supervision.next_retry_s
                delay_s = self._monitor_interval_s
                if retry_at is not None:
                    delay_s = max(0.0, min(delay_s, retry_at - now))
                await asyncio.sleep(delay_s)
                continue

            await self._cleanup_session()
            generation = uuid4().hex
            self._begin_generation(generation)
            self._supervision.session_started(generation)
            LOGGER.info(
                "media session connection attempt %s generation=%s",
                self._supervision.status()["attempts"],
                generation,
            )
            try:
                await asyncio.wait_for(
                    self._open_session(generation),
                    timeout=self._connect_timeout_s,
                )
                self._supervision.session_connected(generation)
                LOGGER.info("media WebRTC session connected generation=%s", generation)
                while not self._closing:
                    await asyncio.sleep(self._monitor_interval_s)
                    failure = self._session_failure_for(generation)
                    if failure is not None:
                        raise RuntimeError(failure)
                    self._supervision.check_health()
                    if (
                        self._supervision.restart_required
                        or self._supervision.state is ServiceState.FAILED
                    ):
                        status = self._supervision.status()
                        LOGGER.warning(
                            "media session unhealthy state=%s generation=%s "
                            "restarts=%s/%s retry_in_s=%s error=%s",
                            status["state"],
                            generation,
                            status["failures_since_ready"],
                            status["restart_budget"],
                            status["next_retry_in_s"],
                            status["last_error"],
                        )
                        break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - WebRTC failures are untyped
                if not self._supervision.restart_required:
                    self._supervision.session_failed(
                        f"{type(exc).__name__}: {exc}"
                    )
                status = self._supervision.status()
                LOGGER.warning(
                    "media session failed state=%s generation=%s "
                    "restarts=%s/%s retry_in_s=%s error=%s",
                    status["state"],
                    generation,
                    status["failures_since_ready"],
                    status["restart_budget"],
                    status["next_retry_in_s"],
                    status["last_error"],
                )
            finally:
                await self._cleanup_session()

    async def _open_session(self, generation: str) -> None:
        if (
            self._connection_class is None
            or self._connection_method is None
            or self._audiohub_class is None
        ):
            raise RuntimeError("media session SDK adapters are not configured")
        connection = self._connection_class(
            self._connection_method,
            ip=self.robot_ip,
        )
        # Store before connect so a partial or timed-out startup is still cleaned.
        self._connection = connection
        await connection.connect()
        self._audiohub = self._audiohub_class(connection)
        connection.video.add_track_callback(
            partial(self._consume_camera, generation=generation)
        )
        connection.video.switchVideoChannel(True)

    async def _cleanup_session(self) -> None:
        connection = self._connection
        self._connection = None
        self._audiohub = None
        while not self._frames.empty():
            try:
                self._frames.get_nowait()
            except asyncio.QueueEmpty:
                break
        if connection is None:
            return
        try:
            connection.video.switchVideoChannel(False)
        except Exception:
            LOGGER.debug("failed to disable stale video channel", exc_info=True)
        try:
            await asyncio.wait_for(
                connection.disconnect(),
                timeout=self._cleanup_timeout_s,
            )
        except Exception:
            LOGGER.debug("failed to disconnect stale media session", exc_info=True)

    def _begin_generation(self, generation: str) -> None:
        with self._target_lock:
            target_fruit = self._target_fruit
        self.evidence = PerceptionEvidence(
            generation=generation,
            target_fruit=target_fruit,
        )
        self._evidence_frames = EvidenceFrameBuffer(
            generation=generation,
            maximum_frames=int(os.environ.get("EVIDENCE_MAXIMUM_FRAMES", "40")),
        )
        self._last_evidence_capture_s = None
        self._visual_odometry.reset(generation)

    def _session_failure_for(self, generation: str) -> str | None:
        while not self._session_failures.empty():
            try:
                failed_generation, error = self._session_failures.get_nowait()
            except asyncio.QueueEmpty:
                return None
            if failed_generation == generation:
                return error
        return None

    async def _consume_camera(self, track: Any, *, generation: str | None = None) -> None:
        active_generation = generation or self.evidence.generation
        try:
            while True:
                frame = await track.recv()
                received = time.monotonic()
                if active_generation != self.evidence.generation:
                    return
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
                self._supervision.note_frame(
                    active_generation,
                    pts,
                    now_s=received,
                )
                if self._frames.full():
                    self._frames.get_nowait()
                self._frames.put_nowait(
                    (
                        active_generation,
                        frame,
                        received,
                        pts,
                        str(frame.time_base),
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - WebRTC errors are untyped
            error = f"camera failure: {type(exc).__name__}: {exc}"
            if active_generation == self.evidence.generation:
                self.evidence.fail(error)
                self._session_failures.put_nowait((active_generation, error))

    async def _detect(self) -> None:
        assert self._model is not None and self._fruit_class_ids
        loop = asyncio.get_running_loop()
        while True:
            generation, frame, received, pts, time_base = await self._frames.get()
            if generation != self.evidence.generation:
                continue
            await loop.run_in_executor(
                self._inference_executor,
                self._process_frame,
                frame,
                received,
                pts,
                time_base,
                generation,
            )

    def _predict_candidate(
        self,
        *,
        source: Any,
        target_fruit: str,
        x_offset: int = 0,
        y_offset: int = 0,
    ) -> RoutedPrediction:
        assert self._model is not None and self._fruit_class_ids
        router = self._model_router or FruitModelRouter(
            general_model=self._model,
            general_class_ids=self._fruit_class_ids,
        )
        return router.predict(
            source=source,
            target_fruit=target_fruit,
            device=0,
            x_offset=x_offset,
            y_offset=y_offset,
        )

    def _process_frame(
        self,
        frame: Any,
        received_monotonic_s: float,
        pts: int,
        time_base: str,
        generation: str | None = None,
    ) -> None:
        """Own all frame conversion, model, postprocess, and preview work."""
        if generation is not None and generation != self.evidence.generation:
            return
        assert self._model is not None and self._fruit_class_ids
        with self._target_lock:
            target_fruit = self._target_fruit
        bgr = frame.to_ndarray(format="bgr24")
        started = time.monotonic()
        try:
            full_frame_prediction = self._predict_candidate(
                source=bgr,
                target_fruit=target_fruit,
            )
            candidate = full_frame_prediction.candidate
            inference_passes = full_frame_prediction.inference_passes
            model_route: dict[str, object] = {
                "full_frame": full_frame_prediction.route,
                "search_crop": None,
                "crop_confirmation": None,
            }
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
            search_crop: dict[str, object] = {
                "attempted": False,
                "crop_xyxy": None,
            }
            if (
                candidate is None
                and target_fruit in SEARCH_CROP_FRUITS
                and not bool(full_frame_prediction.route.get("triggered"))
            ):
                search_crop_xyxy = _lower_center_search_crop(
                    width=int(bgr.shape[1]),
                    height=int(bgr.shape[0]),
                )
                crop_x1, crop_y1, crop_x2, crop_y2 = search_crop_xyxy
                cropped_bgr = bgr[crop_y1:crop_y2, crop_x1:crop_x2]
                search_prediction = self._predict_candidate(
                    source=cropped_bgr,
                    target_fruit=target_fruit,
                    x_offset=crop_x1,
                    y_offset=crop_y1,
                )
                inference_passes += search_prediction.inference_passes
                candidate = search_prediction.candidate
                model_route["search_crop"] = search_prediction.route
                search_crop = {
                    "attempted": True,
                    "crop_xyxy": list(search_crop_xyxy),
                }
            if (
                candidate is not None
                and not search_crop["attempted"]
                and not bool(full_frame_prediction.route.get("triggered"))
                and self._should_crop_confirm(
                    candidate,
                    width=int(bgr.shape[1]),
                    height=int(bgr.shape[0]),
                )
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
                crop_prediction = self._predict_candidate(
                    source=cropped_bgr,
                    target_fruit=target_fruit,
                    x_offset=crop_x1,
                    y_offset=crop_y1,
                )
                inference_passes += crop_prediction.inference_passes
                crop_candidate = crop_prediction.candidate
                model_route["crop_confirmation"] = crop_prediction.route
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
            if generation is not None and generation != self.evidence.generation:
                return
            self.evidence.fail(f"detector failure: {type(exc).__name__}: {exc}")
            self._publish_preview(
                bgr,
                message="MODEL ERROR",
                color=(0, 0, 255),
                pts=pts,
                time_base=time_base,
                received_monotonic_s=received_monotonic_s,
                detection={},
                generation=generation,
            )
            return
        completed = time.monotonic()
        if generation is not None and generation != self.evidence.generation:
            return
        self._visual_odometry.observe(
            bgr,
            captured_monotonic_s=received_monotonic_s,
            source_pts=pts,
            generation=generation or self.evidence.generation,
        )
        if candidate is None:
            self.evidence.note_miss(target_fruit)
            self._publish_preview(
                bgr,
                message=f"SEARCHING FOR {target_fruit.upper()}",
                color=(0, 191, 255),
                pts=pts,
                time_base=time_base,
                received_monotonic_s=received_monotonic_s,
                detection={},
                generation=generation,
            )
            return
        # Published raw by contract: consumers own their confidence floors.
        # See docs/camera-perception-contract.md, "Published detection
        # confidence is raw by design" - a floor here would starve the
        # close-range continuation path (floors as low as 0.10).
        confidence = candidate.confidence
        bbox = candidate.bbox_xyxy
        detection = self.evidence.note_detection(
            pts=pts,
            label=target_fruit,
            confidence=confidence,
            bbox_xyxy=bbox,
            inference_s=completed - started,
            completed_monotonic_s=completed,
            details={
                "inference_passes": inference_passes,
                "model_route": model_route,
                "crop_confirmation": crop_confirmation,
                "search_crop": search_crop,
            },
        )
        if detection is None:
            return
        crop_message = (
            " | CROP CONFIRMED"
            if crop_confirmation["promoted"]
            else " | CROP UNCONFIRMED"
            if crop_confirmation["attempted"]
            else ""
        )
        search_crop_message = " | SEARCH CROP" if search_crop["attempted"] else ""
        self._publish_preview(
            bgr,
            bbox_xyxy=bbox,
            message=(
                f"{target_fruit.upper()} {confidence:.0%} | "
                f"{detection['consecutive_detections']}/5"
                f"{crop_message}{search_crop_message}"
            ),
            color=(0, 200, 0),
            pts=pts,
            time_base=time_base,
            received_monotonic_s=received_monotonic_s,
            detection=detection,
            generation=generation,
        )

    def _should_crop_confirm(
        self,
        candidate: FruitCandidate,
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
        generation: str | None = None,
    ) -> None:
        import cv2

        preview = bgr.copy()
        if bbox_xyxy is not None:
            x1, y1, x2, y2 = bbox_xyxy
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 4)
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 92), (20, 20, 20), -1)
        cv2.putText(
            preview,
            "YOLO FRUIT MODEL: LIVE",
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
        if generation is not None and generation != self.evidence.generation:
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

    @app.post("/api/target")
    async def select_target(request: TargetFruitRequest) -> dict[str, object]:
        try:
            return media.select_target(request.target_fruit)
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

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
