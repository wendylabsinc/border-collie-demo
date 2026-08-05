from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from zipfile import ZipFile


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float | None]:
    return {
        "minimum": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p95": _percentile(values, 0.95),
        "maximum": max(values) if values else None,
    }


def _iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _maximum_streak(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _first_streak_completion(values: list[bool], length: int) -> int | None:
    current = 0
    for index, value in enumerate(values):
        current = current + 1 if value else 0
        if current >= length:
            return index + 1
    return None


def benchmark_archive(
    archive_path: Path,
    labels_path: Path,
    *,
    confidence_threshold: float,
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    with ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    labels = json.loads(labels_path.read_text())

    if labels.get("source_generation") != manifest.get("generation"):
        raise ValueError("label generation does not match camera archive")
    records = labels.get("records")
    frames = manifest.get("frames")
    if not isinstance(records, list) or not isinstance(frames, list):
        raise TypeError("labels or archive frames are missing")
    if len(records) != len(frames):
        raise ValueError("label and archive frame counts differ")
    if not records or any(record.get("reviewed") is not True for record in records):
        raise ValueError("every frame must have a human review")

    confidences: list[float] = []
    inference_times: list[float] = []
    paired_ious: list[float] = []
    qualifying: list[bool] = []
    predictions_at_gate = true_positives = 0
    ground_truth_positives = ground_truth_negatives = 0
    localization_matches = exact_proposal_acceptances = 0

    for record, frame in zip(records, frames, strict=True):
        if record.get("filename") != frame.get("filename"):
            raise ValueError("label and archive frame ordering differs")
        ground_truth = record.get("bbox_xyxy")
        detection = frame.get("detection") or {}
        prediction = detection.get("bbox_xyxy")
        confidence = detection.get("confidence")
        inference_s = detection.get("inference_s")
        if isinstance(confidence, (int, float)):
            confidences.append(float(confidence))
        if isinstance(inference_s, (int, float)):
            inference_times.append(float(inference_s))

        if ground_truth is None:
            ground_truth_negatives += 1
        else:
            ground_truth_positives += 1

        overlap = None
        if ground_truth is not None and prediction is not None:
            overlap = _iou(ground_truth, prediction)
            paired_ious.append(overlap)
            if ground_truth == prediction:
                exact_proposal_acceptances += 1
            if overlap >= iou_threshold:
                localization_matches += 1

        at_gate = (
            prediction is not None
            and isinstance(confidence, (int, float))
            and float(confidence) >= confidence_threshold
        )
        if at_gate:
            predictions_at_gate += 1
        is_true_positive = bool(
            at_gate
            and ground_truth is not None
            and overlap is not None
            and overlap >= iou_threshold
        )
        if is_true_positive:
            true_positives += 1
        qualifying.append(is_true_positive)

    proposal_acceptance_ratio = exact_proposal_acceptances / len(records)
    limitations = []
    if ground_truth_negatives == 0:
        limitations.append(
            "No negative frames are present, so scene-level false-positive behavior is not measured."
        )
    if proposal_acceptance_ratio >= 0.5:
        limitations.append(
            "Most human boxes exactly equal current-model proposals; localization metrics are biased toward the current model."
        )

    return {
        "schema_version": 1,
        "camera_generation": manifest.get("generation"),
        "target_class": (labels.get("class_names") or [None])[0],
        "thresholds": {
            "confidence": confidence_threshold,
            "iou": iou_threshold,
            "stable_detection_count": 5,
        },
        "counts": {
            "frames": len(frames),
            "human_reviewed": len(records),
            "ground_truth_positive": ground_truth_positives,
            "ground_truth_negative": ground_truth_negatives,
            "predictions_at_confidence_gate": predictions_at_gate,
            "localization_matches": localization_matches,
            "true_positives_at_gate": true_positives,
            "exact_model_proposals_accepted": exact_proposal_acceptances,
        },
        "metrics": {
            "localization_recall_at_iou": (
                localization_matches / ground_truth_positives
                if ground_truth_positives
                else None
            ),
            "acquisition_recall_at_gate": (
                true_positives / ground_truth_positives
                if ground_truth_positives
                else None
            ),
            "precision_at_gate": (
                true_positives / predictions_at_gate if predictions_at_gate else None
            ),
            "maximum_qualifying_streak": _maximum_streak(qualifying),
            "first_five_detection_completion_frame": _first_streak_completion(
                qualifying, 5
            ),
            "proposal_acceptance_ratio": proposal_acceptance_ratio,
            "iou": _summary(paired_ious),
            "confidence": _summary(confidences),
            "inference_s": _summary(inference_times),
        },
        "limitations": limitations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a labeled Go2 fruit capture"
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--confidence-threshold", required=True, type=float)
    parser.add_argument("--iou-threshold", default=0.5, type=float)
    args = parser.parse_args()
    result = benchmark_archive(
        args.archive,
        args.labels,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
