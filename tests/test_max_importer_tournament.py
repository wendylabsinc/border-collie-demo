from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.max_tournament.tournament import (
    CorpusItem,
    ExpectedDetection,
    _grade_fixture,
    _raw_parity,
    decode_yolo,
    load_corpus,
    prepare_image,
)


def test_corpus_fails_closed_when_fixture_digest_changes(tmp_path: Path) -> None:
    image = tmp_path / "pear.jpg"
    Image.new("RGB", (8, 8), (0, 255, 0)).save(image)
    manifest = tmp_path / "corpus.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "classes": ["pear", "apple", "banana"],
                "fixtures": [
                    {
                        "id": "pear",
                        "path": "pear.jpg",
                        "sha256": "0" * 64,
                        "expected_classes": ["pear"],
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        load_corpus(tmp_path, manifest)


def test_prepare_image_is_deterministic_rgb_nchw(tmp_path: Path) -> None:
    image = tmp_path / "red.png"
    Image.new("RGB", (10, 5), (255, 0, 0)).save(image)

    first, geometry = prepare_image(image, 16)
    second, _ = prepare_image(image, 16)

    assert first.shape == (1, 3, 16, 16)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert geometry["pad_y"] > 0
    assert first[0, 0, 8, 8] == pytest.approx(1.0)
    assert first[0, 2, 8, 8] == pytest.approx(0.0)


def test_decoder_uses_fixed_pear_apple_banana_class_order() -> None:
    raw = np.zeros((1, 7, 4), dtype=np.float32)
    raw[0, :4, 0] = [8, 8, 4, 4]
    raw[0, 5, 0] = 0.9

    detections = decode_yolo(
        raw,
        {
            "source_width": 16,
            "source_height": 16,
            "scale": 1.0,
            "pad_x": 0.0,
            "pad_y": 0.0,
        },
        ("pear", "apple", "banana"),
        confidence_floor=0.25,
    )

    assert detections[0]["class_name"] == "apple"
    assert detections[0]["box_xyxy"] == [6.0, 6.0, 10.0, 10.0]


def test_wrong_pear_class_is_rejected_even_when_box_is_correct() -> None:
    item = CorpusItem(
        fixture_id="go2-pear",
        path=Path("unused"),
        sha256="unused",
        expected_classes=("pear",),
        expected=(ExpectedDetection("pear", (10.0, 10.0, 20.0, 20.0)),),
        tags=("stage_critical",),
    )
    apple = {
        "class_id": 1,
        "class_name": "apple",
        "confidence": 0.637,
        "box_xyxy": [10.0, 10.0, 20.0, 20.0],
    }

    result = _grade_fixture(item, [apple], [apple], 0.5)

    assert result["diagnostic_top_class"] == "apple"
    assert result["expected_class_hits"] == 0
    assert result["box_hits_iou50"] == 0
    assert result["passed"] is False


def test_raw_parity_distinguishes_runtime_parity_from_accuracy() -> None:
    reference = np.zeros((1, 7, 10), dtype=np.float32)
    matching_candidate = reference.copy()
    wrong_candidate = reference.copy()
    wrong_candidate[0, 4, 0] = 1.0

    assert _raw_parity(reference, matching_candidate)["passed"] is True
    assert _raw_parity(reference, wrong_candidate)["passed"] is False
