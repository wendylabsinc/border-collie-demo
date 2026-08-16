"""Fail-closed reader for qualified camera and Target Fruit evidence."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from typing import Any
from urllib.request import Request, urlopen

from .config import PerceptionConfig
from .fruits import (
    SUPPORTED_FRUITS,
    fruit_policy,
    mango_color_confidence,
    mango_raw_confidence,
)

SOURCE_MAXIMUM_AGE_S = 0.350
SOURCE_MINIMUM_CONSECUTIVE_FRAMES = 10
PEAR_MINIMUM_CONFIDENCE = 0.65
PEAR_MINIMUM_CONSECUTIVE_DETECTIONS = 5
INFERENCE_MAXIMUM_S = 0.200
DETECTION_MAXIMUM_AGE_S = 0.250

StatusFetcher = Callable[[str, float], dict[str, Any]]
TargetPoster = Callable[[str, str, float], dict[str, Any]]
JsonPoster = Callable[[str, dict[str, object], float], dict[str, Any]]
Clock = Callable[[], float]


class PerceptionStatusClient:
    """Expose one small readiness interface over sidecar evidence."""

    def __init__(
        self,
        config: PerceptionConfig | None = None,
        *,
        fetcher: StatusFetcher | None = None,
        target_poster: TargetPoster | None = None,
        json_poster: JsonPoster | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self.config = config or PerceptionConfig()
        self._fetcher = fetcher or _fetch_status
        self._target_poster = target_poster or _post_target
        self._json_poster = json_poster or _post_json
        self._clock = clock

    def select_target(self, target_fruit: str) -> dict[str, object]:
        if not self.config.enabled:
            raise RuntimeError("production camera/perception adapter is disabled")
        fruit_policy(target_fruit)
        payload = self._target_poster(
            self.config.target_url,
            target_fruit,
            self.config.timeout_s,
        )
        selected = payload.get("target_fruit")
        if (
            not isinstance(selected, str)
            or selected.casefold() != target_fruit.casefold()
        ):
            raise RuntimeError("perception sidecar did not acknowledge Target Fruit")
        return payload

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

    def coco_test_status(self) -> dict[str, object]:
        if not self.config.enabled:
            raise RuntimeError("production camera/perception adapter is disabled")
        return self._fetcher(self.config.coco_test_url, self.config.timeout_s)

    def configure_coco_test(
        self, payload: dict[str, object]
    ) -> dict[str, object]:
        if not self.config.enabled:
            raise RuntimeError("production camera/perception adapter is disabled")
        return self._json_poster(
            self.config.coco_test_url,
            payload,
            max(self.config.timeout_s, 10.0),
        )

    def camera_frame(self) -> bytes:
        return self._camera_frame(self.config.frame_url)

    def raw_camera_frame(self) -> bytes:
        return self._camera_frame(self.config.raw_frame_url)

    def _camera_frame(self, url: str) -> bytes:
        if not self.config.enabled:
            raise RuntimeError("production camera/perception adapter is disabled")
        request = Request(
            url,
            headers={"Accept": "image/jpeg"},
        )
        with urlopen(request, timeout=self.config.timeout_s) as response:
            jpeg = response.read()
        if (
            len(jpeg) < 4
            or not jpeg.startswith(b"\xff\xd8")
            or not jpeg.endswith(b"\xff\xd9")
        ):
            raise ValueError("camera preview is not a complete JPEG")
        return jpeg


def evaluate_perception_evidence(
    payload: dict[str, Any],
    *,
    now_s: float,
) -> dict[str, object]:
    violations: list[str] = []
    camera_violations: list[str] = []
    raw_target_fruit = payload.get("target_fruit", "pear")
    target_fruit = (
        raw_target_fruit.casefold().strip() if isinstance(raw_target_fruit, str) else ""
    )
    try:
        target_policy = fruit_policy(target_fruit)
    except ValueError:
        target_policy = fruit_policy("pear")
        violations.append("selected Target Fruit is unsupported")
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
        violations.append(
            f"{target_fruit or 'target fruit'} detection evidence is missing"
        )

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
    if (
        source_width is None
        or source_width <= 0
        or source_height is None
        or source_height <= 0
    ):
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
    inference_passes = _whole_number(detection.get("inference_passes"))
    color_identity = detection.get("color_identity")
    if color_identity not in {"red_apple", "orange", "unknown"}:
        color_identity = "unknown"
    color_confidence = _finite_number(detection.get("color_confidence"))
    raw_label = detection.get("raw_label")
    raw_confidence = _finite_number(detection.get("raw_confidence"))
    raw_bbox = _bounding_box(
        detection.get("raw_bbox_xyxy"), source_width, source_height
    )
    derived_identity = detection.get("derived_identity")
    derived_confidence = _finite_number(detection.get("derived_confidence"))
    raw_crop_confirmation = detection.get("crop_confirmation")
    if isinstance(raw_crop_confirmation, dict):
        crop_confirmation: dict[str, object] | None = {
            "attempted": raw_crop_confirmation.get("attempted") is True,
            "promoted": raw_crop_confirmation.get("promoted") is True,
            "full_frame_confidence": _finite_number(
                raw_crop_confirmation.get("full_frame_confidence")
            ),
            "crop_confidence": _finite_number(
                raw_crop_confirmation.get("crop_confidence")
            ),
            "crop_xyxy": (
                None
                if (
                    crop_box := _bounding_box(
                        raw_crop_confirmation.get("crop_xyxy"),
                        source_width,
                        source_height,
                    )
                )
                is None
                else list(crop_box)
            ),
            "agreement_iou": _finite_number(raw_crop_confirmation.get("agreement_iou")),
        }
    else:
        crop_confirmation = None
    if not isinstance(label, str) or label.casefold().strip() != target_fruit:
        violations.append(f"qualifying {target_fruit} detection is missing")
    if target_fruit == "mango" and (
        raw_label != "bowl"
        or raw_confidence is None
        or raw_confidence <= mango_raw_confidence()
        or raw_bbox is None
        or derived_identity != "mango"
        or derived_confidence is None
        or derived_confidence < mango_color_confidence()
    ):
        violations.append("derived Mango identity evidence is missing or unqualified")
    if detection_generation != generation:
        violations.append(
            f"{target_fruit} detection generation does not match camera generation"
        )
    if detection_time_base != time_base:
        violations.append(
            f"{target_fruit} detection time base does not match camera source"
        )
    if (
        detection_source_pts is None
        or not isinstance(pts, int)
        or detection_source_pts > pts
    ):
        violations.append(f"{target_fruit} detection source PTS is missing or invalid")
    if confidence is None or confidence < target_policy.acquisition_confidence:
        violations.append(
            f"{target_fruit} confidence is below "
            f"{target_policy.acquisition_confidence:.2f}"
        )
    if detection_count is None or detection_count < PEAR_MINIMUM_CONSECUTIVE_DETECTIONS:
        violations.append(f"fewer than 5 consecutive {target_fruit} detections")
    if inference_s is None or inference_s < 0.0 or inference_s > INFERENCE_MAXIMUM_S:
        violations.append("detector execution exceeds 0.200 seconds")
    if detection_age_s is None or detection_age_s > DETECTION_MAXIMUM_AGE_S:
        violations.append(f"{target_fruit} detection is stale")
    if bbox is None:
        violations.append(f"{target_fruit} bounding box is missing or invalid")
    if payload.get("error"):
        violations.append(f"sidecar error: {payload['error']}")

    ready = not violations
    camera_healthy = not camera_violations and not payload.get("error")
    target_ready = ready
    center_x_ratio = None
    center_y_ratio = None
    bottom_ratio = None
    bbox_width_ratio = None
    bbox_height_ratio = None
    bbox_area_ratio = None
    if bbox is not None and source_width and source_height:
        x1, y1, x2, y2 = bbox
        center_x_ratio = ((x1 + x2) / 2.0) / source_width
        center_y_ratio = ((y1 + y2) / 2.0) / source_height
        bottom_ratio = y2 / source_height
        bbox_width_ratio = (x2 - x1) / source_width
        bbox_height_ratio = (y2 - y1) / source_height
        bbox_area_ratio = bbox_width_ratio * bbox_height_ratio
    return {
        "ready": ready,
        "camera_healthy": camera_healthy,
        # Published separately from "detail", which mixes camera and target
        # violations. A camera failure stops the run, so the reason it stopped
        # has to survive into the run record on its own.
        "camera_violations": list(camera_violations),
        "target_ready": target_ready,
        "detail": (
            f"camera generation and {target_fruit} detector passed recognition check"
            if ready
            else "; ".join(violations)
        ),
        "target_fruit": target_fruit,
        "supported_fruits": list(SUPPORTED_FRUITS),
        "motion_qualified": target_policy.motion_qualified,
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
            "inference_passes": inference_passes,
            "color_identity": color_identity,
            "color_confidence": color_confidence,
            "raw_label": raw_label,
            "raw_confidence": raw_confidence,
            "raw_bbox_xyxy": None if raw_bbox is None else list(raw_bbox),
            "derived_identity": derived_identity,
            "derived_confidence": derived_confidence,
            "crop_confirmation": crop_confirmation,
            "completed_monotonic_s": detection_completed_s,
            "age_s": detection_age_s,
            "bbox_xyxy": None if bbox is None else list(bbox),
            "center_x_ratio": center_x_ratio,
            "center_y_ratio": center_y_ratio,
            "bottom_ratio": bottom_ratio,
            "bbox_width_ratio": bbox_width_ratio,
            "bbox_height_ratio": bbox_height_ratio,
            "bbox_area_ratio": bbox_area_ratio,
        },
        "thresholds": {
            "source_maximum_age_s": SOURCE_MAXIMUM_AGE_S,
            "source_minimum_consecutive_frames": SOURCE_MINIMUM_CONSECUTIVE_FRAMES,
            "pear_minimum_confidence": PEAR_MINIMUM_CONFIDENCE,
            "target_minimum_confidence": target_policy.acquisition_confidence,
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


def _post_target(url: str, target_fruit: str, timeout_s: float) -> dict[str, Any]:
    return _post_json(url, {"target_fruit": target_fruit}, timeout_s)


def _post_json(
    url: str, payload_body: dict[str, object], timeout_s: float
) -> dict[str, Any]:
    body = json.dumps(payload_body).encode("utf-8")
    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=timeout_s) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise TypeError("perception response must be a JSON object")
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
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or not width
        or not height
    ):
        return None
    values = tuple(_finite_number(item) for item in value)
    if any(item is None for item in values):
        return None
    x1, y1, x2, y2 = (float(item) for item in values if item is not None)
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        return None
    return x1, y1, x2, y2
