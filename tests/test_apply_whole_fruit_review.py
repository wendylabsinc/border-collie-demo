from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "training" / "apply_whole_fruit_review.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "apply_whole_fruit_review", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_review_withholds_partially_reviewed_images(tmp_path: Path) -> None:
    module = load_module()
    image = tmp_path / "source.jpg"
    image.write_bytes(b"jpeg")
    manifest_path = tmp_path / "manifest.json"
    ranking_path = tmp_path / "ranking.json"
    review_path = tmp_path / "review.json"
    manifest = {
        "records": [
            {
                "image_id": "image-1",
                "local_image": "source.jpg",
                "license_url": "https://creativecommons.org/licenses/by/2.0/",
                "annotations": [
                    {
                        "class_id": 1,
                        "class_name": "banana",
                        "bbox_xyxy_normalized": [0.1, 0.2, 0.3, 0.4],
                    },
                    {
                        "class_id": 1,
                        "class_name": "banana",
                        "bbox_xyxy_normalized": [0.5, 0.5, 0.8, 0.8],
                    },
                ],
            }
        ]
    }
    manifest_path.write_text(json.dumps(manifest))
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    ranking_path.write_text(
        json.dumps(
            {
                "source_manifest_sha256": manifest_hash,
                "candidates": [
                    {"candidate_id": "image-1:0", "class_name": "banana"},
                    {"candidate_id": "image-1:1", "class_name": "banana"},
                ],
            }
        )
    )
    review_path.write_text(
        json.dumps(
            {
                "source_manifest_sha256": manifest_hash,
                "review_class": "banana",
                "decisions": {
                    "image-1:0": {"decision": "accept_whole"},
                    "image-1:1": {"decision": "skip"},
                },
            }
        )
    )

    result = module.apply_review(
        manifest_path,
        ranking_path,
        review_path,
        tmp_path / "output",
    )

    assert result["counts"] == {
        "images": 0,
        "positive_images": 0,
        "hard_negative_images": 0,
        "accepted_boxes": 0,
        "hard_negative_boxes_by_reason": {},
        "withheld_images_by_reason": {"incomplete_image_review": 1},
    }
