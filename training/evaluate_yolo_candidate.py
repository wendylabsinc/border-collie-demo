"""Evaluate an Ultralytics fruit detector with the MAX candidate match contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image, ImageOps
from ultralytics import YOLO

from training.max_native.evaluate import _metric_record, match_detections


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yolo_targets(label_path: Path, image_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load normalized YOLO labels as source-image xyxy boxes."""

    with Image.open(image_path) as source:
        width, height = ImageOps.exif_transpose(source).size
    boxes: list[list[float]] = []
    classes: list[int] = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id, center_x, center_y, box_width, box_height = line.split()[:5]
            center_x = float(center_x) * width
            center_y = float(center_y) * height
            box_width = float(box_width) * width
            box_height = float(box_height) * height
            boxes.append(
                [
                    center_x - box_width / 2.0,
                    center_y - box_height / 2.0,
                    center_x + box_width / 2.0,
                    center_y + box_height / 2.0,
                ]
            )
            classes.append(int(class_id))
    return (
        torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        torch.tensor(classes, dtype=torch.long),
    )


def _dataset_contract(data_path: Path) -> tuple[list[Path], Path, tuple[str, ...]]:
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    validation_images = Path(data["val"])
    if not validation_images.is_absolute():
        validation_images = root / validation_images
    validation_labels = validation_images.parent.parent / "labels" / validation_images.name
    names_value = data["names"]
    if isinstance(names_value, dict):
        classes = tuple(str(names_value[index]) for index in sorted(names_value))
    else:
        classes = tuple(str(name) for name in names_value)
    image_paths = sorted(
        path
        for path in validation_images.iterdir()
        if path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    return image_paths, validation_labels, classes


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    image_paths, labels_dir, classes = _dataset_contract(args.data)
    thresholds = tuple(sorted(set(args.confidence_thresholds)))
    minimum_threshold = min(thresholds)
    aggregate = {threshold: Counter() for threshold in thresholds}
    by_class = {
        threshold: {name: Counter() for name in classes} for threshold in thresholds
    }
    model = YOLO(args.checkpoint)
    started = time.perf_counter()
    results = model.predict(
        source=str(image_paths[0].parent),
        imgsz=args.input_size,
        conf=minimum_threshold,
        iou=args.nms_iou_threshold,
        device=args.device,
        batch=args.batch_size,
        verbose=False,
    )
    inference_seconds = time.perf_counter() - started
    validation_boxes = 0
    image_by_name = {path.name: path for path in image_paths}
    result_names = {Path(result.path).name for result in results}
    if result_names != set(image_by_name):
        raise RuntimeError("Ultralytics results do not match the validation image set")
    for result in results:
        image_path = image_by_name[Path(result.path).name]
        target_boxes, target_classes = _load_yolo_targets(
            labels_dir / f"{image_path.stem}.txt", image_path
        )
        validation_boxes += len(target_boxes)
        predicted_boxes = result.boxes.xyxy.cpu()
        predicted_scores = result.boxes.conf.cpu()
        predicted_classes = result.boxes.cls.to(dtype=torch.long, device="cpu")
        for threshold in thresholds:
            keep = predicted_scores >= threshold
            boxes = predicted_boxes[keep]
            class_ids = predicted_classes[keep]
            counts = match_detections(
                boxes,
                class_ids,
                target_boxes,
                target_classes,
                iou_threshold=args.match_iou_threshold,
            )
            aggregate[threshold].update(counts)
            for class_id, class_name in enumerate(classes):
                predicted = class_ids == class_id
                expected = target_classes == class_id
                class_counts = match_detections(
                    boxes[predicted],
                    class_ids[predicted],
                    target_boxes[expected],
                    target_classes[expected],
                    iou_threshold=args.match_iou_threshold,
                )
                by_class[threshold][class_name].update(class_counts)
    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "classes": list(classes),
        "device": args.device,
        "input_size": args.input_size,
        "validation_images": len(image_paths),
        "validation_boxes": validation_boxes,
        "wall_seconds": inference_seconds,
        "wall_images_per_second": len(image_paths) / max(inference_seconds, 1e-9),
        "match_iou_threshold": args.match_iou_threshold,
        "nms_iou_threshold": args.nms_iou_threshold,
        "thresholds": {
            str(threshold): {
                **_metric_record(aggregate[threshold]),
                "classes": {
                    name: _metric_record(counts)
                    for name, counts in by_class[threshold].items()
                },
            }
            for threshold in thresholds
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="+",
        default=(0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5),
    )
    parser.add_argument("--input-size", type=int, default=416)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.45)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
