"""Bounded color evidence for camera-only fruit identity debugging."""

from __future__ import annotations

import math
from typing import Any


def classify_bbox_color(
    source: Any,
    bbox_xyxy: tuple[float, float, float, float] | None,
) -> dict[str, object]:
    """Classify saturated red/orange pixels inside one detector box.

    This is descriptive evidence only. It does not authorize motion or replace
    the detector's label.
    """
    if bbox_xyxy is None:
        return _unknown()
    shape = getattr(source, "shape", None)
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        return _unknown()
    height, width = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = bbox_xyxy
    x1_i = max(0, min(width - 1, int(x1)))
    y1_i = max(0, min(height - 1, int(y1)))
    x2_i = max(x1_i + 1, min(width, math.ceil(x2)))
    y2_i = max(y1_i + 1, min(height, math.ceil(y2)))

    # Ignore the outer 15% where loose boxes most often include floor/background.
    inset_x = int((x2_i - x1_i) * 0.15)
    inset_y = int((y2_i - y1_i) * 0.15)
    x1_i, x2_i = x1_i + inset_x, x2_i - inset_x
    y1_i, y2_i = y1_i + inset_y, y2_i - inset_y
    if x2_i <= x1_i or y2_i <= y1_i:
        return _unknown()
    stride = max(1, int(math.sqrt(((x2_i - x1_i) * (y2_i - y1_i)) / 4096)))
    try:
        pixels = source[y1_i:y2_i:stride, x1_i:x2_i:stride].reshape(-1, 3).tolist()
    except (AttributeError, IndexError, TypeError, ValueError):
        return _unknown()
    return classify_bgr_pixels(pixels)


def classify_bgr_pixels(pixels: object) -> dict[str, object]:
    if not isinstance(pixels, list) or not pixels:
        return _unknown()
    red = 0
    orange = 0
    valid = 0
    for pixel in pixels:
        if not isinstance(pixel, (list, tuple)) or len(pixel) < 3:
            continue
        try:
            blue, green, red_channel = (float(value) for value in pixel[:3])
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (blue, green, red_channel)):
            continue
        maximum = max(blue, green, red_channel)
        minimum = min(blue, green, red_channel)
        if maximum < 40.0 or maximum - minimum < max(30.0, maximum * 0.25):
            continue
        valid += 1
        if red_channel < max(green, blue):
            continue
        delta = maximum - minimum
        hue_degrees = (60.0 * ((green - blue) / delta)) % 360.0
        # A dim orange shifts toward red, but unlike the red apple it does not
        # wrap around the zero-degree hue boundary. Keep the red band narrow
        # enough to preserve that exposure-stable distinction.
        if hue_degrees <= 8.0 or hue_degrees >= 345.0:
            red += 1
        elif hue_degrees <= 80.0:
            orange += 1

    classified = red + orange
    coverage = classified / valid if valid else 0.0
    dominance = max(red, orange) / classified if classified else 0.0
    identity = "unknown"
    if coverage >= 0.20 and dominance >= 0.65:
        identity = "red_apple" if red > orange else "orange"
    return {
        "identity": identity,
        "confidence": dominance * min(1.0, coverage / 0.50),
        "sample_count": valid,
        "classified_coverage": coverage,
        "red_fraction": red / valid if valid else 0.0,
        "orange_fraction": orange / valid if valid else 0.0,
    }


def classify_mango_color_evidence(color: object) -> dict[str, object]:
    """Recognize the staged mango's red/orange blend, or remain unknown.

    This evidence is only meaningful when paired with the COCO ``sports ball``
    geometry proposal. A single red or orange hue band is intentionally not
    enough to distinguish Mango from the red apple or Orange props.
    """
    unknown = {"identity": "unknown", "confidence": 0.0}
    if not isinstance(color, dict):
        return unknown
    try:
        sample_count = int(color.get("sample_count", 0))
        coverage = float(color.get("classified_coverage", 0.0))
        red_fraction = float(color.get("red_fraction", 0.0))
        orange_fraction = float(color.get("orange_fraction", 0.0))
    except (TypeError, ValueError):
        return unknown
    values = (coverage, red_fraction, orange_fraction)
    if not all(math.isfinite(value) for value in values):
        return unknown
    classified = red_fraction + orange_fraction
    if sample_count < 25 or coverage < 0.20 or classified <= 0.0:
        return unknown
    red_share = red_fraction / classified
    orange_share = orange_fraction / classified
    if min(red_share, orange_share) < 0.20:
        return unknown
    balance = min(red_share, orange_share) / 0.50
    return {
        "identity": "mango",
        "confidence": balance * min(1.0, coverage / 0.50),
    }


def classify_replacement_candidate(
    model_label: str, color: object
) -> dict[str, object]:
    """Combine one raw COCO proposal label with bounded color evidence."""
    normalized_label = model_label.casefold().strip()
    if normalized_label == "sports ball":
        return classify_mango_color_evidence(color)
    if normalized_label == "bowl" and isinstance(color, dict):
        try:
            confidence = float(color.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if color.get("identity") == "orange" and math.isfinite(confidence):
            return {"identity": "orange", "confidence": confidence}
    return {"identity": "unknown", "confidence": 0.0}


def _unknown() -> dict[str, object]:
    return {
        "identity": "unknown",
        "confidence": 0.0,
        "sample_count": 0,
        "classified_coverage": 0.0,
        "red_fraction": 0.0,
        "orange_fraction": 0.0,
    }
