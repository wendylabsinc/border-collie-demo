"""Pure image preparation and YOLO decoding for the MAX camera gate."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image


def prepare_jpeg(
    jpeg: bytes, *, input_size: int, dtype: np.dtype[Any] | type[np.generic]
) -> tuple[np.ndarray, dict[str, float | int]]:
    with Image.open(io.BytesIO(jpeg)) as encoded:
        image = encoded.convert("RGB")
    source_width, source_height = image.size
    scale = min(input_size / source_width, input_size / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = image.resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )
    pad_x = (input_size - resized_width) // 2
    pad_y = (input_size - resized_height) // 2
    canvas = Image.new("RGB", (input_size, input_size), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))
    tensor = np.asarray(canvas, dtype=dtype)
    tensor = np.ascontiguousarray(tensor[None] / np.asarray(255.0, dtype=dtype))
    return tensor, {
        "source_width": source_width,
        "source_height": source_height,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }


def _box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def decode_detections(
    output: np.ndarray,
    geometry: dict[str, float | int],
    *,
    confidence_floor: float,
    nms_iou: float = 0.45,
) -> list[dict[str, Any]]:
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 7:
        raise RuntimeError(f"unexpected YOLO output shape {output.shape}")
    channels = output[0].astype(np.float32, copy=False)
    class_scores = channels[4:7]
    class_ids = np.argmax(class_scores, axis=0)
    confidences = np.max(class_scores, axis=0)
    proposals: list[tuple[float, int, list[float]]] = []
    for index in np.flatnonzero(confidences >= confidence_floor):
        center_x, center_y, width, height = (
            float(channels[channel, index]) for channel in range(4)
        )
        proposals.append(
            (
                float(confidences[index]),
                int(class_ids[index]),
                [
                    center_x - width / 2.0,
                    center_y - height / 2.0,
                    center_x + width / 2.0,
                    center_y + height / 2.0,
                ],
            )
        )
    kept: list[tuple[float, int, list[float]]] = []
    for proposal in sorted(proposals, reverse=True):
        if all(
            proposal[1] != previous[1]
            or _box_iou(proposal[2], previous[2]) <= nms_iou
            for previous in kept
        ):
            kept.append(proposal)
    scale = float(geometry["scale"])
    pad_x = float(geometry["pad_x"])
    pad_y = float(geometry["pad_y"])
    source_width = int(geometry["source_width"])
    source_height = int(geometry["source_height"])
    class_names = ("pear", "apple", "banana")
    detections: list[dict[str, Any]] = []
    for confidence, class_id, box in kept:
        source_box = [
            max(0.0, min((box[0] - pad_x) / scale, source_width)),
            max(0.0, min((box[1] - pad_y) / scale, source_height)),
            max(0.0, min((box[2] - pad_x) / scale, source_width)),
            max(0.0, min((box[3] - pad_y) / scale, source_height)),
        ]
        if source_box[0] < source_box[2] and source_box[1] < source_box[3]:
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "confidence": confidence,
                    "box_xyxy": source_box,
                }
            )
    return detections
