"""Bounded color evidence for camera-only fruit identity debugging."""

from __future__ import annotations

import math
from typing import Any

from media.fruit_color_max import accelerated_band_counts


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
        region = source[y1_i:y2_i:stride, x1_i:x2_i:stride].reshape(-1, 3)
    except (AttributeError, IndexError, TypeError, ValueError):
        return _unknown()
    # The accelerated backends consume the array region directly, which also
    # skips the .tolist() marshalling the reference loop needs.
    counts = accelerated_band_counts(region)
    if counts is not None:
        return summarize_band_counts(*counts)
    try:
        pixels = region.tolist()
    except (AttributeError, IndexError, TypeError, ValueError):
        return _unknown()
    return classify_bgr_pixels(pixels)


def classify_bgr_pixels(pixels: object) -> dict[str, object]:
    if not isinstance(pixels, list) or not pixels:
        return _unknown()
    counts = accelerated_band_counts(pixels)
    if counts is None:
        counts = count_band_pixels(pixels)
    return summarize_band_counts(*counts)


def count_band_pixels(pixels: list) -> tuple[int, int, int]:
    """Reference hue-band count: returns (valid, red, orange).

    This is the trusted implementation. Accelerated backends in
    `media.fruit_color_max` must reproduce these three integers exactly.
    """
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
    return valid, red, orange


def summarize_band_counts(valid: int, red: int, orange: int) -> dict[str, object]:
    """Turn hue-band counts into the colour evidence dictionary.

    Shared by the reference loop and every accelerated backend, so identical
    counts always produce an identical result.
    """
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


def classify_replacement_candidate(
    model_label: str,
    color: object,
    bbox_xyxy: object,
    *,
    source_width: int | None,
    source_height: int | None,
) -> dict[str, object]:
    """Confirm the staged warm/yellow Mango proposal, or remain unknown."""
    unknown = {"identity": "unknown", "confidence": 0.0}
    if model_label.casefold().strip() not in {"bowl", "sports ball"} or not isinstance(
        color, dict
    ):
        return unknown
    if not isinstance(bbox_xyxy, (list, tuple)) or len(bbox_xyxy) != 4:
        return unknown
    if not source_width or not source_height:
        return unknown
    try:
        confidence = float(color.get("confidence", 0.0))
        sample_count = int(color.get("sample_count", 0))
        coverage = float(color.get("classified_coverage", 0.0))
        x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    except (TypeError, ValueError):
        return unknown
    values = (confidence, coverage, x1, y1, x2, y2)
    if not all(math.isfinite(value) for value in values):
        return unknown
    if (
        color.get("identity") != "orange"
        or not 0.65 <= confidence <= 1.0
        or sample_count < 25
        or coverage < 0.20
        or x1 < 0.0
        or y1 < 0.0
        or x2 <= x1
        or y2 <= y1
    ):
        return unknown
    width_ratio = (x2 - x1) / source_width
    height_ratio = (y2 - y1) / source_height
    center_y_ratio = ((y1 + y2) / 2.0) / source_height
    bottom_ratio = y2 / source_height
    # The proposal begins as a small lower-frame box and grows during the
    # existing approach controller. Keep a bounded lower-frame identity gate,
    # but do not impose the test scene's initial size as an Arrival ceiling.
    if not (
        0.005 <= width_ratio <= 0.50
        and 0.005 <= height_ratio <= 0.50
        and 0.50 <= center_y_ratio <= 1.0
        and bottom_ratio <= 1.0
    ):
        return unknown
    return {"identity": "mango", "confidence": confidence}


def _unknown() -> dict[str, object]:
    return {
        "identity": "unknown",
        "confidence": 0.0,
        "sample_count": 0,
        "classified_coverage": 0.0,
        "red_fraction": 0.0,
        "orange_fraction": 0.0,
    }
