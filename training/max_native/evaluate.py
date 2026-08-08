"""Evaluate a trained MAX-native fruit detector on its held-out image split."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .data import WholeFruitDataset, collate_batch, select_whole_fruit_records
from .detection import Detections, _pairwise_iou, decode_predictions
from .model import ARCHITECTURE_SHA256, DEFAULT_SPEC, TrainableFruitDetector


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def match_detections(
    predicted_boxes: Tensor,
    predicted_classes: Tensor,
    target_boxes: Tensor,
    target_classes: Tensor,
    *,
    iou_threshold: float,
) -> dict[str, int]:
    """Greedily match score-ordered predictions to same-class ground truth."""

    if len(predicted_boxes) != len(predicted_classes):
        raise ValueError("predicted boxes and classes must have equal lengths")
    if len(target_boxes) != len(target_classes):
        raise ValueError("target boxes and classes must have equal lengths")
    unmatched = torch.ones(len(target_boxes), dtype=torch.bool)
    true_positives = 0
    false_positives = 0
    for box, class_id in zip(predicted_boxes, predicted_classes, strict=True):
        candidates = (target_classes == class_id) & unmatched
        positions = candidates.nonzero(as_tuple=False).flatten()
        if not len(positions):
            false_positives += 1
            continue
        overlaps = _pairwise_iou(box, target_boxes[positions])
        best_overlap, local_index = overlaps.max(dim=0)
        if float(best_overlap) < iou_threshold:
            false_positives += 1
            continue
        unmatched[positions[int(local_index)]] = False
        true_positives += 1
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": int(unmatched.sum()),
    }


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _metric_record(counts: Counter[str]) -> dict[str, float | int]:
    true_positives = counts["true_positives"]
    false_positives = counts["false_positives"]
    false_negatives = counts["false_negatives"]
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["architecture_sha256"] != ARCHITECTURE_SHA256:
        raise RuntimeError("checkpoint does not match the current MAX architecture")
    if tuple(checkpoint["classes"]) != DEFAULT_SPEC.classes:
        raise RuntimeError("checkpoint class map does not match the deployment contract")
    input_size = int(checkpoint["input_size"])
    records = select_whole_fruit_records(
        args.manifest,
        args.ranking,
        minimum_whole_score=args.minimum_whole_score,
        split="val",
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    dataset = WholeFruitDataset(records, input_size=input_size, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_batch,
        persistent_workers=args.workers > 0,
    )
    model = TrainableFruitDetector(DEFAULT_SPEC)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    thresholds = tuple(sorted(set(args.confidence_thresholds)))
    aggregate = {threshold: Counter() for threshold in thresholds}
    by_class = {
        threshold: {name: Counter() for name in DEFAULT_SPEC.classes}
        for threshold in thresholds
    }
    inference_seconds = 0.0
    with torch.inference_mode():
        for images, target_boxes, target_classes in loader:
            started = time.perf_counter()
            outputs = tuple(output.cpu() for output in model(images.to(device)))
            inference_seconds += time.perf_counter() - started
            for threshold in thresholds:
                detections = decode_predictions(
                    outputs,
                    input_size=input_size,
                    confidence_threshold=threshold,
                    iou_threshold=args.nms_iou_threshold,
                )
                for detection, boxes, classes in zip(
                    detections, target_boxes, target_classes, strict=True
                ):
                    counts = match_detections(
                        detection.boxes,
                        detection.class_ids,
                        boxes,
                        classes,
                        iou_threshold=args.match_iou_threshold,
                    )
                    aggregate[threshold].update(counts)
                    for class_id, class_name in enumerate(DEFAULT_SPEC.classes):
                        predicted = detection.class_ids == class_id
                        expected = classes == class_id
                        class_counts = match_detections(
                            detection.boxes[predicted],
                            detection.class_ids[predicted],
                            boxes[expected],
                            classes[expected],
                            iou_threshold=args.match_iou_threshold,
                        )
                        by_class[threshold][class_name].update(class_counts)
    report = {
        "schema_version": 1,
        "architecture_sha256": ARCHITECTURE_SHA256,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "classes": list(DEFAULT_SPEC.classes),
        "device": str(device),
        "input_size": input_size,
        "validation_images": len(records),
        "validation_boxes": sum(len(record.boxes) for record in records),
        "inference_seconds": inference_seconds,
        "inference_images_per_second": len(records) / max(inference_seconds, 1e-9),
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
        "experimental_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ranking", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="+",
        default=(0.1, 0.15, 0.2, 0.25, 0.3),
    )
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.45)
    parser.add_argument("--minimum-whole-score", type=float, default=0.75)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
