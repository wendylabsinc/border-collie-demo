from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROBE_DIR = Path(__file__).parents[1] / "lab" / "max-yolo11s-candidate"
sys.path.insert(0, str(PROBE_DIR))
from snapshot_inference import decode_detections, prepare_jpeg


def test_prepare_jpeg_emits_native_nhwc_fp16_tensor() -> None:
    image = Image.new("RGB", (100, 50), (255, 0, 0))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG")

    tensor, geometry = prepare_jpeg(
        encoded.getvalue(), input_size=416, dtype=np.float16
    )

    assert tensor.shape == (1, 416, 416, 3)
    assert tensor.dtype == np.float16
    assert geometry["source_width"] == 100
    assert geometry["source_height"] == 50
    assert geometry["pad_y"] > 0


def test_decode_detections_filters_by_confidence_and_class_aware_nms() -> None:
    output = np.zeros((1, 7, 3549), dtype=np.float16)
    output[0, :4, 0] = [208, 208, 100, 100]
    output[0, 4, 0] = 0.9
    output[0, :4, 1] = [210, 210, 100, 100]
    output[0, 4, 1] = 0.8
    output[0, :4, 2] = [208, 208, 100, 100]
    output[0, 6, 2] = 0.7
    output[0, :4, 3] = [50, 50, 20, 20]
    output[0, 5, 3] = 0.1

    detections = decode_detections(
        output,
        {
            "source_width": 416,
            "source_height": 416,
            "scale": 1.0,
            "pad_x": 0,
            "pad_y": 0,
        },
        confidence_floor=0.25,
    )

    assert [item["class_name"] for item in detections] == ["pear", "banana"]
    assert detections[0]["confidence"] > detections[1]["confidence"]
    assert detections[0]["box_xyxy"] == [158.0, 158.0, 258.0, 258.0]
