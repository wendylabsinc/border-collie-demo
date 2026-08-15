"""Read-only, opt-in COCO confidence sampler for choosing demo props."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any

from media.fruit_color import classify_bbox_color


@dataclass
class _ClassStats:
    class_id: int
    frames_detected: int = 0
    confidence_sum: float = 0.0
    maximum_confidence: float = 0.0
    latest_confidence: float | None = None
    latest_bbox_xyxy: tuple[float, float, float, float] | None = None
    latest_color: dict[str, object] | None = None


class CocoTester:
    """Sample a stock COCO detector without participating in motion evidence."""

    def __init__(
        self,
        *,
        model_path: str = "/models/yolo11n.pt",
        model_loader: Callable[[str], Any] | None = None,
        interval_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(interval_s) or not 0.1 <= interval_s <= 5.0:
            raise ValueError("COCO test interval must be within 0.1..5.0 seconds")
        self._model_path = model_path
        self._model_loader = model_loader or self._load_yolo
        self._interval_s = interval_s
        self._clock = clock
        self._lock = Lock()
        self._model: Any | None = None
        self._names: dict[int, str] = {}
        self._enabled = False
        self._minimum_confidence = 0.05
        self._last_sample_s: float | None = None
        self._frames_processed = 0
        self._frames_skipped = 0
        self._latest_pts: int | None = None
        self._latest_inference_ms: float | None = None
        self._source_width: int | None = None
        self._source_height: int | None = None
        self._total_inference_ms = 0.0
        self._stats: dict[str, _ClassStats] = {}
        self._error: str | None = None

    @staticmethod
    def _load_yolo(path: str) -> Any:
        from ultralytics import YOLO

        return YOLO(path, task="detect")

    def configure(
        self,
        *,
        enabled: bool,
        minimum_confidence: float | None = None,
        reset: bool = False,
    ) -> dict[str, object]:
        if minimum_confidence is not None and (
            not math.isfinite(minimum_confidence)
            or not 0.01 <= minimum_confidence <= 0.95
        ):
            raise ValueError("minimum confidence must be within 0.01..0.95")
        if enabled and self._model is None:
            model = self._model_loader(self._model_path)
            raw_names = getattr(model, "names", {})
            items = (
                raw_names.items()
                if isinstance(raw_names, dict)
                else enumerate(raw_names)
            )
            names = {
                int(class_id): str(label).strip()
                for class_id, label in items
                if str(label).strip()
            }
            if not names:
                raise RuntimeError("COCO test model did not expose class names")
            self._model = model
            self._names = names
        with self._lock:
            if minimum_confidence is not None:
                self._minimum_confidence = minimum_confidence
            if reset:
                self._reset_locked()
            self._enabled = enabled
            return self._status_locked()

    def observe(self, source: Any, *, pts: int, now_s: float | None = None) -> bool:
        sampled_at = self._clock() if now_s is None else now_s
        with self._lock:
            if not self._enabled or self._model is None:
                return False
            if (
                self._last_sample_s is not None
                and sampled_at - self._last_sample_s < self._interval_s
            ):
                self._frames_skipped += 1
                return False
            self._last_sample_s = sampled_at
            model = self._model
            minimum_confidence = self._minimum_confidence

        started = self._clock()
        try:
            results = model.predict(
                source=source,
                conf=minimum_confidence,
                device=0,
                verbose=False,
            )
            best_by_class = self._best_by_class(results)
        except Exception as exc:  # noqa: BLE001 - model runtimes are untyped
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
            return True
        inference_ms = max(0.0, (self._clock() - started) * 1000.0)
        with self._lock:
            self._frames_processed += 1
            self._latest_pts = pts
            shape = getattr(source, "shape", None)
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                height, width = shape[:2]
                if isinstance(width, int) and isinstance(height, int):
                    self._source_width = width
                    self._source_height = height
            self._latest_inference_ms = inference_ms
            self._total_inference_ms += inference_ms
            self._error = None
            for stats in self._stats.values():
                stats.latest_confidence = None
                stats.latest_bbox_xyxy = None
                stats.latest_color = None
            for class_id, (confidence, bbox_xyxy) in best_by_class.items():
                label = self._names.get(class_id, f"class-{class_id}")
                stats = self._stats.setdefault(label, _ClassStats(class_id=class_id))
                stats.frames_detected += 1
                stats.confidence_sum += confidence
                stats.maximum_confidence = max(stats.maximum_confidence, confidence)
                stats.latest_confidence = confidence
                stats.latest_bbox_xyxy = bbox_xyxy
                stats.latest_color = classify_bbox_color(source, bbox_xyxy)
        return True

    def status(self) -> dict[str, object]:
        with self._lock:
            return self._status_locked()

    def _best_by_class(
        self, results: object
    ) -> dict[int, tuple[float, tuple[float, float, float, float] | None]]:
        if not isinstance(results, (list, tuple)) or not results:
            return {}
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return {}
        confidences = boxes.conf.detach().cpu().tolist()
        class_ids = boxes.cls.detach().cpu().tolist()
        raw_boxes = getattr(boxes, "xyxy", None)
        if hasattr(raw_boxes, "detach"):
            box_values = raw_boxes.detach().cpu().tolist()
        elif isinstance(raw_boxes, (list, tuple)):
            box_values = [
                item.detach().cpu().tolist() if hasattr(item, "detach") else item
                for item in raw_boxes
            ]
        else:
            box_values = [None] * len(confidences)
        best: dict[
            int, tuple[float, tuple[float, float, float, float] | None]
        ] = {}
        for raw_class_id, raw_confidence, raw_box in zip(
            class_ids, confidences, box_values, strict=True
        ):
            class_id = int(raw_class_id)
            confidence = float(raw_confidence)
            if not math.isfinite(confidence):
                continue
            bbox_xyxy = _valid_bbox(raw_box)
            current = best.get(class_id)
            if current is None or confidence > current[0]:
                best[class_id] = (confidence, bbox_xyxy)
        return best

    def _reset_locked(self) -> None:
        self._last_sample_s = None
        self._frames_processed = 0
        self._frames_skipped = 0
        self._latest_pts = None
        self._latest_inference_ms = None
        self._source_width = None
        self._source_height = None
        self._total_inference_ms = 0.0
        self._stats = {}
        self._error = None

    def _status_locked(self) -> dict[str, object]:
        frames = self._frames_processed
        classes = []
        for label, stats in self._stats.items():
            classes.append(
                {
                    "label": label,
                    "class_id": stats.class_id,
                    "latest_confidence": stats.latest_confidence,
                    "mean_detected_confidence": (
                        stats.confidence_sum / stats.frames_detected
                    ),
                    "all_frame_score": stats.confidence_sum / frames if frames else 0.0,
                    "maximum_confidence": stats.maximum_confidence,
                    "detection_rate": stats.frames_detected / frames if frames else 0.0,
                    "frames_detected": stats.frames_detected,
                    "latest_bbox_xyxy": (
                        None
                        if stats.latest_bbox_xyxy is None
                        else list(stats.latest_bbox_xyxy)
                    ),
                    "latest_color": stats.latest_color,
                }
            )
        classes.sort(
            key=lambda item: (
                float(item["all_frame_score"]),
                float(item["maximum_confidence"]),
            ),
            reverse=True,
        )
        return {
            "enabled": self._enabled,
            "strictly_read_only": True,
            "model": self._model_path.rsplit("/", 1)[-1],
            "class_count": len(self._names),
            "minimum_confidence": self._minimum_confidence,
            "sample_interval_s": self._interval_s,
            "frames_processed": frames,
            "frames_skipped_by_interval": self._frames_skipped,
            "latest_source_pts": self._latest_pts,
            "latest_inference_ms": self._latest_inference_ms,
            "source_width": self._source_width,
            "source_height": self._source_height,
            "mean_inference_ms": (
                self._total_inference_ms / frames if frames else None
            ),
            "classes": classes,
            "error": self._error,
        }


def _valid_bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)):
        return None
    if x1 < 0.0 or y1 < 0.0 or x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2
