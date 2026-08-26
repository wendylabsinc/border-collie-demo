from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


def _summary(values: list[float]) -> dict[str, float | None]:
    ordered = sorted(values)
    return {
        "minimum": min(ordered) if ordered else None,
        "median": statistics.median(ordered) if ordered else None,
        "mean": statistics.fmean(ordered) if ordered else None,
        "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else None,
        "maximum": max(ordered) if ordered else None,
    }


def _iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _ground_truth(label_path: Path, width: int, height: int) -> list[float] | None:
    content = label_path.read_text().strip()
    if not content:
        return None
    _, center_x, center_y, box_width, box_height = map(
        float, content.splitlines()[0].split()
    )
    return [
        (center_x - box_width / 2) * width,
        (center_y - box_height / 2) * height,
        (center_x + box_width / 2) * width,
        (center_y + box_height / 2) * height,
    ]


def _load_detector(model_path: Path | str, task: str | None) -> YOLO:
    """Load a checkpoint or a serialized artifact.

    A `.pt` carries its own task metadata. A serialized artifact (`.engine`,
    `.onnx`) does not, and ultralytics then decodes a segmentation head's 39
    output channels as 4 box + 35 classes instead of 4 box + 3 classes + 32
    mask coefficients, which surfaces as `KeyError: 23`. So a non-`.pt`
    artifact must always be told its task.
    """
    path = Path(model_path)
    if path.suffix.casefold() == ".pt":
        return YOLO(str(path)) if task is None else YOLO(str(path), task=task)
    if task is None:
        raise ValueError(
            f"--task is required for a non-.pt artifact: {path.name}"
            " (use segment for apple-pear-mango, detect for a box-only model)"
        )
    return YOLO(str(path), task=task)


def _maximum_streak(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def benchmark(
    model_path: Path | str,
    images_dir: Path,
    labels_dir: Path,
    *,
    target_class: str = "banana",
    confidence_threshold: float = 0.2,
    iou_threshold: float = 0.5,
    image_size: int = 640,
    device: str = "mps",
    task: str | None = None,
) -> dict[str, object]:
    model = _load_detector(model_path, task)
    names = model.names
    target_ids = {
        int(class_id)
        for class_id, name in (
            names.items() if isinstance(names, dict) else enumerate(names)
        )
        if str(name).casefold() == target_class.casefold()
    }
    if not target_ids:
        raise ValueError(f"model does not contain target class: {target_class}")
    image_paths = sorted(images_dir.glob("*.jpg"))
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=image_size,
        conf=0.001,
        iou=0.7,
        device=device,
        verbose=False,
        stream=False,
    )
    qualifying: list[bool] = []
    localized_confidences: list[float] = []
    best_ious: list[float] = []
    inference_ms: list[float] = []
    localized = acquired = ground_truth_positives = 0

    for image_path, result in zip(image_paths, results, strict=True):
        width, height = Image.open(image_path).size
        truth = _ground_truth(labels_dir / f"{image_path.stem}.txt", width, height)
        if truth is not None:
            ground_truth_positives += 1
        predictions = []
        if result.boxes is not None:
            for class_id, confidence, bbox in zip(
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.xyxy.cpu().tolist(),
                strict=True,
            ):
                if int(class_id) in target_ids:
                    predictions.append(
                        (float(confidence), [float(value) for value in bbox])
                    )
        best_overlap = 0.0
        best_confidence = None
        if truth is not None and predictions:
            scored = [
                (_iou(truth, bbox), confidence) for confidence, bbox in predictions
            ]
            best_overlap, best_confidence = max(scored)
        if best_overlap >= iou_threshold:
            localized += 1
            best_ious.append(best_overlap)
            assert best_confidence is not None
            localized_confidences.append(best_confidence)
        passed = bool(
            best_overlap >= iou_threshold
            and best_confidence is not None
            and best_confidence >= confidence_threshold
        )
        acquired += int(passed)
        qualifying.append(passed)
        inference_ms.append(float(result.speed["inference"]))

    model_file = Path(model_path)
    return {
        "schema_version": 1,
        "model": str(model_path),
        "model_sha256": (
            hashlib.sha256(model_file.read_bytes()).hexdigest()
            if model_file.is_file()
            else None
        ),
        "target_class": target_class,
        "thresholds": {
            "confidence": confidence_threshold,
            "iou": iou_threshold,
            "image_size": image_size,
        },
        "counts": {
            "frames": len(image_paths),
            "ground_truth_positive": ground_truth_positives,
            "localized": localized,
            "acquired_at_gate": acquired,
        },
        "metrics": {
            "localization_recall": localized / ground_truth_positives,
            "acquisition_recall": acquired / ground_truth_positives,
            "maximum_qualifying_streak": _maximum_streak(qualifying),
            "localized_confidence": _summary(localized_confidences),
            "localized_iou": _summary(best_ious),
            "local_inference_ms": _summary(inference_ms),
        },
        "limitations": [
            "The evaluation contains no negative frames, so precision is not measured.",
            "Local Mac inference timing is not a substitute for Woof device timing.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a detector on frozen fruit frames"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-class", default="banana")
    parser.add_argument("--confidence-threshold", default=0.2, type=float)
    parser.add_argument("--iou-threshold", default=0.5, type=float)
    parser.add_argument("--image-size", default=640, type=int)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--task",
        default=None,
        choices=("detect", "segment", "pose", "obb", "classify"),
        help="required for .engine/.onnx artifacts, which carry no task metadata",
    )
    args = parser.parse_args()
    result = benchmark(
        args.model,
        args.images,
        args.labels,
        target_class=args.target_class,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        image_size=args.image_size,
        device=args.device,
        task=args.task,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
