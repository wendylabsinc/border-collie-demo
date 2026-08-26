"""Build and qualify a TensorRT FP16 engine for apple-pear-mango.pt.

Runs on Woof, because a TensorRT engine is specific to the GPU and the
TensorRT version that built it. It never touches the camera, the robot, or any
motion client; it reads one committed reference frame and writes one artifact.

Three phases:

1. Export `apple-pear-mango.pt` to `apple-pear-mango.engine` with `half=True`.
2. Qualify the engine against the `.pt` on the committed reference frame,
   loading the engine as `YOLO(path, task="segment")`. Without that task a
   serialized engine loses its task metadata and ultralytics decodes the 39
   output channels as 4 box + 35 classes instead of 4 box + 3 classes + 32 mask
   coefficients, which surfaces as `KeyError: 23`.
3. Serve the engine, the JSON report and the annotated frame over HTTP so the
   operator can pull them down and version the engine beside the checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WORK_DIR = Path(os.environ.get("EXPORT_WORK_DIR", "/export"))
CHECKPOINT = WORK_DIR / "apple-pear-mango.pt"
ENGINE = WORK_DIR / "apple-pear-mango.engine"
REFERENCE = WORK_DIR / "apple-pear-mango-reference.jpg"
REPORT = WORK_DIR / "tensorrt-export-result.json"

IMAGE_SIZE = int(os.environ.get("EXPORT_IMAGE_SIZE", "640"))
DEVICE = os.environ.get("EXPORT_DEVICE", "0")
FORCE = os.environ.get("EXPORT_FORCE", "0") == "1"
TIMING_PASSES = int(os.environ.get("EXPORT_TIMING_PASSES", "30"))
SERVE_PORT = int(os.environ.get("EXPORT_SERVE_PORT", "8124"))

# The .pt yields these on the committed reference frame. The engine has to
# reproduce every strong detection; the weak mango is reported but not gated.
EXPECTED_STRONG = {"pear": 0.960, "apple": 0.957, "mango": 0.901}
STRONG_FLOOR = 0.50
CONFIDENCE_TOLERANCE = float(os.environ.get("EXPORT_CONFIDENCE_TOLERANCE", "0.05"))
IOU_FLOOR = float(os.environ.get("EXPORT_IOU_FLOOR", "0.90"))
DETECTION_FLOOR = 0.05


def _iou(first, second) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _detections(model, image_path: Path) -> list[dict]:
    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=DETECTION_FLOOR,
        device=DEVICE,
        verbose=False,
    )
    names = model.names
    labels = names if isinstance(names, dict) else dict(enumerate(names))
    rows = []
    for result in results:
        if result.boxes is None:
            continue
        for class_id, confidence, bbox in zip(
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.xyxy.cpu().tolist(),
            strict=True,
        ):
            rows.append(
                {
                    "label": str(labels.get(int(class_id), int(class_id))),
                    "confidence": round(float(confidence), 4),
                    "bbox_xyxy": [round(float(value), 2) for value in bbox],
                }
            )
    rows.sort(key=lambda row: -row["confidence"])
    return rows


def _timing(model, image_path: Path) -> dict:
    timings = []
    for index in range(TIMING_PASSES):
        started = time.perf_counter()
        model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=DETECTION_FLOOR,
            device=DEVICE,
            verbose=False,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        if index >= 3:  # discard warmup passes
            timings.append(elapsed)
    timings.sort()
    return {
        "passes": len(timings),
        "p50_ms": round(timings[len(timings) // 2], 2),
        "p95_ms": round(timings[int(0.95 * (len(timings) - 1))], 2),
        "minimum_ms": round(timings[0], 2),
    }


def _class_names(model) -> dict:
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return {int(class_id): str(label).casefold().strip() for class_id, label in items}


def _compare(checkpoint_rows: list[dict], engine_rows: list[dict]) -> dict:
    """Pair each checkpoint detection with the best same-label engine detection."""
    remaining = list(engine_rows)
    pairs = []
    for row in checkpoint_rows:
        candidates = [
            (index, _iou(row["bbox_xyxy"], other["bbox_xyxy"]))
            for index, other in enumerate(remaining)
            if other["label"] == row["label"]
        ]
        if not candidates:
            pairs.append({"label": row["label"], "checkpoint": row, "engine": None})
            continue
        index, overlap = max(candidates, key=lambda item: item[1])
        matched = remaining.pop(index)
        pairs.append(
            {
                "label": row["label"],
                "checkpoint": row,
                "engine": matched,
                "iou": round(overlap, 4),
                "confidence_delta": round(
                    matched["confidence"] - row["confidence"], 4
                ),
            }
        )
    failures = []
    for pair in pairs:
        if pair["checkpoint"]["confidence"] < STRONG_FLOOR:
            continue  # weak proposals are reported, not gated
        label = pair["label"]
        if pair["engine"] is None:
            failures.append(f"{label}: engine produced no matching detection")
            continue
        if abs(pair["confidence_delta"]) > CONFIDENCE_TOLERANCE:
            failures.append(
                f"{label}: confidence moved {pair['confidence_delta']:+.4f}"
                f" (tolerance {CONFIDENCE_TOLERANCE})"
            )
        if pair["iou"] < IOU_FLOOR:
            failures.append(f"{label}: box IoU {pair['iou']} below {IOU_FLOOR}")
    strong_labels = {
        pair["label"]
        for pair in pairs
        if pair["checkpoint"]["confidence"] >= STRONG_FLOOR
    }
    for label in EXPECTED_STRONG:
        if label not in strong_labels:
            failures.append(
                f"{label}: checkpoint itself did not produce a strong detection;"
                " the reference frame or the checkpoint changed"
            )
    return {
        "pairs": pairs,
        "unmatched_engine_detections": remaining,
        "failures": failures,
        "passed": not failures,
    }


def _export() -> None:
    from ultralytics import YOLO

    if ENGINE.exists() and not FORCE:
        print(f"engine already present at {ENGINE}; set EXPORT_FORCE=1 to rebuild")
        return
    print(f"exporting {CHECKPOINT} -> TensorRT FP16 at imgsz={IMAGE_SIZE}")
    print("this takes many minutes on the Orin and is memory hungry")
    # The .pt carries its own task metadata, so no task= is needed here. It is
    # required on every subsequent load of the serialized engine.
    model = YOLO(str(CHECKPOINT))
    produced = Path(
        model.export(
            format="engine",
            half=True,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            batch=1,
            dynamic=False,
            verbose=False,
        )
    )
    if produced.resolve() != ENGINE.resolve():
        produced.replace(ENGINE)
    print(f"wrote {ENGINE} ({ENGINE.stat().st_size} bytes)")


def _qualify() -> dict:
    from ultralytics import YOLO

    checkpoint_model = YOLO(str(CHECKPOINT))
    # task="segment" is mandatory: a serialized engine has no task metadata and
    # ultralytics would otherwise decode 39 channels as 35 classes.
    engine_model = YOLO(str(ENGINE), task="segment")

    checkpoint_names = _class_names(checkpoint_model)
    engine_names = _class_names(engine_model)
    name_failures = []
    if sorted(engine_names.values()) != sorted(checkpoint_names.values()):
        name_failures.append(
            f"engine classes {engine_names} do not match checkpoint {checkpoint_names}"
        )
    if len(engine_names) != 3:
        name_failures.append(
            f"engine reports {len(engine_names)} classes, expected 3;"
            " this is the symptom of a missing task='segment'"
        )

    checkpoint_rows = _detections(checkpoint_model, REFERENCE)
    engine_rows = _detections(engine_model, REFERENCE)
    comparison = _compare(checkpoint_rows, engine_rows)
    comparison["failures"] = name_failures + comparison["failures"]
    comparison["passed"] = not comparison["failures"]

    return {
        "schema_version": 1,
        "reference_frame": REFERENCE.name,
        "image_size": IMAGE_SIZE,
        "half_precision": True,
        "checkpoint": {
            "path": str(CHECKPOINT),
            "sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
            "classes": checkpoint_names,
            "detections": checkpoint_rows,
            "timing": _timing(checkpoint_model, REFERENCE),
        },
        "engine": {
            "path": str(ENGINE),
            "bytes": ENGINE.stat().st_size,
            "sha256": hashlib.sha256(ENGINE.read_bytes()).hexdigest(),
            "classes": engine_names,
            "detections": engine_rows,
            "timing": _timing(engine_model, REFERENCE),
        },
        "expected_checkpoint_confidences": EXPECTED_STRONG,
        "thresholds": {
            "strong_detection_floor": STRONG_FLOOR,
            "confidence_tolerance": CONFIDENCE_TOLERANCE,
            "iou_floor": IOU_FLOOR,
        },
        "comparison": comparison,
        "control_authority": "none_measurement_only",
        "limitations": [
            "One committed reference frame is not a substitute for a live run.",
            "The engine is valid only for this GPU and this TensorRT version.",
        ],
    }


def _serve() -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(WORK_DIR))
    server = ThreadingHTTPServer(("0.0.0.0", SERVE_PORT), handler)
    print(f"serving {WORK_DIR} on port {SERVE_PORT}; stop the app when finished")
    server.serve_forever()


def main() -> None:
    for path in (CHECKPOINT, REFERENCE):
        if not path.is_file():
            raise SystemExit(f"missing required input: {path}")
    _export()
    report = _qualify()
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    verdict = "PASS" if report["comparison"]["passed"] else "FAIL"
    print(f"TENSORRT_EXPORT_RESULT={verdict}")
    for failure in report["comparison"]["failures"]:
        print(f"  failure: {failure}")
    _serve()


if __name__ == "__main__":
    main()
