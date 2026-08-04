"""Fail-closed reader for qualified camera and pear evidence."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from typing import Any
from urllib.request import Request, urlopen

from .config import PerceptionConfig

SOURCE_MAXIMUM_AGE_S = 0.350
SOURCE_MINIMUM_CONSECUTIVE_FRAMES = 10
PEAR_MINIMUM_CONFIDENCE = 0.65
PEAR_MINIMUM_CONSECUTIVE_DETECTIONS = 5
INFERENCE_MAXIMUM_S = 0.200
DETECTION_MAXIMUM_AGE_S = 0.250

StatusFetcher = Callable[[str, float], dict[str, Any]]
Clock = Callable[[], float]


class PerceptionStatusClient:
    """Expose one small readiness interface over sidecar evidence."""

    def __init__(
        self,
        config: PerceptionConfig | None = None,
        *,
        fetcher: StatusFetcher | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self.config = config or PerceptionConfig()
        self._fetcher = fetcher or _fetch_status
        self._clock = clock

    def status(self) -> dict[str, object]:
        if not self.config.enabled:
            return {
                "ready": False,
                "detail": "production camera/perception adapter is disabled",
            }
        try:
            payload = self._fetcher(
                self.config.status_url,
                self.config.timeout_s,
            )
            return evaluate_perception_evidence(payload, now_s=self._clock())
        except Exception as exc:  # noqa: BLE001 - remote evidence is untrusted
            return {
                "ready": False,
                "detail": f"camera/perception status unavailable: {exc}",
            }


def evaluate_perception_evidence(
    payload: dict[str, Any],
    *,
    now_s: float,
) -> dict[str, object]:
    violations: list[str] = []
    camera_violations: list[str] = []
    generation = payload.get("generation")
    source = payload.get("source")
    detection = payload.get("detection")
    if not isinstance(generation, str) or not generation.strip():
        camera_violations.append("connection generation is missing")
    if not isinstance(source, dict):
        source = {}
        camera_violations.append("source evidence is missing")
    if not isinstance(detection, dict):
        detection = {}
        violations.append("pear detection evidence is missing")

    pts = source.get("pts")
    time_base = source.get("time_base")
    source_received_s = _finite_number(source.get("received_monotonic_s"))
    source_frames = _whole_number(source.get("consecutive_frames"))
    source_width = _whole_number(source.get("width"))
    source_height = _whole_number(source.get("height"))
    source_age_s = _age(now_s, source_received_s)
    if not isinstance(pts, int) or isinstance(pts, bool):
        camera_violations.append("source PTS is missing")
    if not isinstance(time_base, str) or not time_base:
        camera_violations.append("source time base is missing")
    if source_frames is None or source_frames < SOURCE_MINIMUM_CONSECUTIVE_FRAMES:
        camera_violations.append("fewer than 10 consecutive source frames")
    if source_age_s is None or source_age_s > SOURCE_MAXIMUM_AGE_S:
        camera_violations.append("source progress is stale")
    if source_width is None or source_width <= 0 or source_height is None or source_height <= 0:
        camera_violations.append("source dimensions are missing")

    violations.extend(camera_violations)

    label = detection.get("label")
    detection_generation = detection.get("generation")
    detection_source_pts = _whole_number(detection.get("source_pts"))
    detection_time_base = detection.get("source_time_base")
    confidence = _finite_number(detection.get("confidence"))
    detection_count = _whole_number(detection.get("consecutive_detections"))
    inference_s = _finite_number(detection.get("inference_s"))
    detection_completed_s = _finite_number(detection.get("completed_monotonic_s"))
    detection_age_s = _age(now_s, detection_completed_s)
    bbox = _bounding_box(detection.get("bbox_xyxy"), source_width, source_height)
    if not isinstance(label, str) or label.casefold().strip() != "pear":
        violations.append("qualifying pear detection is missing")
    if detection_generation != generation:
        violations.append("pear detection generation does not match camera generation")
    if detection_time_base != time_base:
        violations.append("pear detection time base does not match camera source")
    if (
        detection_source_pts is None
        or not isinstance(pts, int)
        or detection_source_pts > pts
    ):
        violations.append("pear detection source PTS is missing or invalid")
    if confidence is None or confidence < PEAR_MINIMUM_CONFIDENCE:
        violations.append("pear confidence is below 0.65")
    if (
        detection_count is None
        or detection_count < PEAR_MINIMUM_CONSECUTIVE_DETECTIONS
    ):
        violations.append("fewer than 5 consecutive pear detections")
    if inference_s is None or inference_s < 0.0 or inference_s > INFERENCE_MAXIMUM_S:
        violations.append("detector execution exceeds 0.200 seconds")
    if detection_age_s is None or detection_age_s > DETECTION_MAXIMUM_AGE_S:
        violations.append("pear detection is stale")
    if bbox is None:
        violations.append("pear bounding box is missing or invalid")
    if payload.get("error"):
        violations.append(f"sidecar error: {payload['error']}")

    ready = not violations
    camera_healthy = not camera_violations and not payload.get("error")
    target_ready = ready
    center_x_ratio = None
    center_y_ratio = None
    bottom_ratio = None
    if bbox is not None and source_width and source_height:
        x1, y1, x2, y2 = bbox
        center_x_ratio = ((x1 + x2) / 2.0) / source_width
        center_y_ratio = ((y1 + y2) / 2.0) / source_height
        bottom_ratio = y2 / source_height
    return {
        "ready": ready,
        "camera_healthy": camera_healthy,
        "target_ready": target_ready,
        "detail": (
            "camera generation and pear detector passed qualified preflight"
            if ready
            else "; ".join(violations)
        ),
        "generation": generation,
        "source": {
            "pts": pts,
            "time_base": time_base,
            "received_monotonic_s": source_received_s,
            "age_s": source_age_s,
            "consecutive_frames": source_frames,
            "width": source_width,
            "height": source_height,
        },
        "detection": {
            "label": label,
            "generation": detection_generation,
            "source_pts": detection_source_pts,
            "source_time_base": detection_time_base,
            "confidence": confidence,
            "consecutive_detections": detection_count,
            "inference_s": inference_s,
            "completed_monotonic_s": detection_completed_s,
            "age_s": detection_age_s,
            "bbox_xyxy": None if bbox is None else list(bbox),
            "center_x_ratio": center_x_ratio,
            "center_y_ratio": center_y_ratio,
            "bottom_ratio": bottom_ratio,
        },
        "thresholds": {
            "source_maximum_age_s": SOURCE_MAXIMUM_AGE_S,
            "source_minimum_consecutive_frames": SOURCE_MINIMUM_CONSECUTIVE_FRAMES,
            "pear_minimum_confidence": PEAR_MINIMUM_CONFIDENCE,
            "pear_minimum_consecutive_detections": (
                PEAR_MINIMUM_CONSECUTIVE_DETECTIONS
            ),
            "inference_maximum_s": INFERENCE_MAXIMUM_S,
            "detection_maximum_age_s": DETECTION_MAXIMUM_AGE_S,
        },
    }


def _fetch_status(url: str, timeout_s: float) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout_s) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise TypeError("camera/perception status must be a JSON object")
    return payload


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _whole_number(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _age(now_s: float, recorded_s: float | None) -> float | None:
    if not math.isfinite(now_s) or recorded_s is None or recorded_s > now_s:
        return None
    return now_s - recorded_s


def _bounding_box(
    value: object,
    width: int | None,
    height: int | None,
) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4 or not width or not height:
        return None
    values = tuple(_finite_number(item) for item in value)
    if any(item is None for item in values):
        return None
    x1, y1, x2, y2 = (float(item) for item in values if item is not None)
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        return None
    return x1, y1, x2, y2
