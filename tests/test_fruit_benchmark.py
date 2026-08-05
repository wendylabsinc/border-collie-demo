from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

BENCHMARK_PATH = (
    Path(__file__).parents[1] / "lab" / "fruit-recognition" / "benchmark.py"
)


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("fruit_benchmark", BENCHMARK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_reports_acquisition_separately_from_localization(
    tmp_path: Path,
) -> None:
    benchmark = load_benchmark_module()
    archive_path = tmp_path / "evidence.zip"
    labels_path = tmp_path / "labels.json"
    frames = []
    records = []
    for index, confidence in enumerate((0.8, 0.4), start=1):
        filename = f"frames/{index:06d}.jpg"
        frames.append(
            {
                "filename": filename,
                "detection": {
                    "bbox_xyxy": [10, 10, 30, 30],
                    "confidence": confidence,
                    "inference_s": 0.1,
                },
            }
        )
        records.append(
            {
                "filename": filename,
                "reviewed": True,
                "class_name": "banana",
                "bbox_xyxy": [10, 10, 30, 30],
            }
        )
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "manifest.json", json.dumps({"generation": "camera-1", "frames": frames})
        )
    labels_path.write_text(
        json.dumps(
            {
                "source_generation": "camera-1",
                "class_names": ["banana"],
                "records": records,
            }
        )
    )

    result = benchmark.benchmark_archive(
        archive_path,
        labels_path,
        confidence_threshold=0.5,
    )

    assert result["metrics"]["localization_recall_at_iou"] == 1.0
    assert result["metrics"]["acquisition_recall_at_gate"] == 0.5
    assert result["counts"]["exact_model_proposals_accepted"] == 2
    assert len(result["limitations"]) == 2
