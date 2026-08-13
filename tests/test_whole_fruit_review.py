from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SERVER_PATH = (
    Path(__file__).parents[1] / "training" / "whole-fruit-review" / "server.py"
)


def load_server_module():
    spec = importlib.util.spec_from_file_location("whole_fruit_review", SERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_review_filters_class_and_limits_candidates(tmp_path: Path) -> None:
    server = load_server_module()
    image = tmp_path / "banana.jpg"
    image.write_bytes(b"jpeg")
    manifest = tmp_path / "manifest.json"
    ranking = tmp_path / "ranking.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "image_id": "banana-1",
                        "local_image": "banana.jpg",
                        "license_url": "https://creativecommons.org/licenses/by/2.0/",
                        "author": "Author",
                    }
                ]
            }
        )
    )
    ranking.write_text(
        json.dumps(
            {
                "source_manifest_sha256": "sha",
                "candidates": [
                    {
                        "candidate_id": "banana-1:0",
                        "image_id": "banana-1",
                        "class_name": "banana",
                        "whole_fruit_score": 0.9,
                        "bbox_xyxy_normalized": [0.1, 0.2, 0.3, 0.4],
                    },
                    {
                        "candidate_id": "apple-1:0",
                        "image_id": "apple-1",
                        "class_name": "apple",
                    },
                ],
            }
        )
    )

    review, images = server.load_review(manifest, ranking, "banana", 1)

    assert review["candidate_count"] == 1
    assert review["candidates"][0]["license_url"].endswith("/by/2.0/")
    assert images == {"banana-1": image.resolve()}
