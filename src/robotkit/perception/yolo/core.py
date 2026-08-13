"""Pure fruit detection interpretation.

Keeping this module free of model and middleware imports makes the audited
post-processing behavior cheap to exercise with ordinary unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from robotkit.fruits import SUPPORTED_FRUITS

FRUIT_CLASSES = SUPPORTED_FRUITS
# Public compatibility alias from the original COCO-only implementation.
COCO_FRUIT_CLASSES = FRUIT_CLASSES


@dataclass(frozen=True)
class Detection:
    """One raw object detection in pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class FruitInterpretation:
    """World-state-ready interpretation of one complete image."""

    stream: str
    observation_type: str
    confidence: float | None
    payload: dict[str, Any]


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def interpret_detections(
    detections: Iterable[Detection],
    *,
    image_width: int,
    image_height: int,
    allowed_classes: frozenset[str] = FRUIT_CLASSES,
    min_confidence: float = 0.25,
) -> FruitInterpretation:
    """Filter fruit detections and produce a deterministic frame snapshot.

    Bounding boxes are clamped to the image and represented in both pixel and
    normalized coordinates. Degenerate boxes, disallowed labels, and detections
    below the threshold are discarded. An empty interpretation is intentional:
    it tells consumers that the latest processed frame contained no fruit.
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")

    normalized_allowed = {name.strip().casefold() for name in allowed_classes}
    fruits: list[dict[str, Any]] = []
    for item in detections:
        name = item.class_name.strip().casefold()
        confidence = float(item.confidence)
        if name not in normalized_allowed or confidence < min_confidence:
            continue
        if not 0.0 <= confidence <= 1.0:
            continue

        x1, y1, x2, y2 = item.xyxy
        x1 = _clamp(x1, 0.0, float(image_width))
        y1 = _clamp(y1, 0.0, float(image_height))
        x2 = _clamp(x2, 0.0, float(image_width))
        y2 = _clamp(y2, 0.0, float(image_height))
        if x2 <= x1 or y2 <= y1:
            continue

        fruits.append(
            {
                "class_id": int(item.class_id),
                "class_name": name,
                "confidence": round(confidence, 6),
                "bbox_xyxy_px": [round(v, 3) for v in (x1, y1, x2, y2)],
                "bbox_xyxy_normalized": [
                    round(x1 / image_width, 6),
                    round(y1 / image_height, 6),
                    round(x2 / image_width, 6),
                    round(y2 / image_height, 6),
                ],
            }
        )

    # The ordering does not depend on the model backend's output ordering.
    fruits.sort(
        key=lambda item: (
            -item["confidence"],
            item["class_name"],
            item["bbox_xyxy_px"],
        )
    )
    confidence = max((item["confidence"] for item in fruits), default=None)
    return FruitInterpretation(
        stream="vision.fruits",
        observation_type="vision.prompted_fruits.v1",
        confidence=confidence,
        payload={
            "payload_schema_version": "1",
            "model_dataset": "open_vocabulary",
            "image": {"width_px": image_width, "height_px": image_height},
            "filter": {
                "classes": sorted(normalized_allowed),
                "min_confidence": min_confidence,
            },
            "count": len(fruits),
            "detections": fruits,
        },
    )
