"""Run the production fruit pipeline against a fixed remote image corpus.

This command is intentionally camera-free and motion-free.  It loads the same
TensorRT general model, banana specialist, routing, crop-confirmation, and
acquisition thresholds as the production media sidecar, then emits one JSON
result for offline grading.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from media.model_router import FruitModelRouter
from media.perception_sidecar import (
    FRUIT_ACQUISITION_CONFIDENCE,
    SUPPORTED_FRUITS,
    PerceptionRuntime,
)


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        return response.read()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, round(0.95 * (len(ordered) - 1))))
    return {
        "median": float(statistics.median(ordered)),
        "p95": float(ordered[p95_index]),
        "maximum": float(ordered[-1]),
    }


def _iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _class_ids(model: Any) -> dict[str, int]:
    names = getattr(model, "names", {})
    items = names.items() if isinstance(names, dict) else enumerate(names)
    available = {
        str(label).casefold().strip(): int(class_id) for class_id, label in items
    }
    missing = sorted(set(SUPPORTED_FRUITS) - set(available))
    if missing:
        raise RuntimeError("general model is missing: " + ", ".join(missing))
    return {fruit: available[fruit] for fruit in SUPPORTED_FRUITS}


def _specialist_class_id(model: Any) -> int:
    names = getattr(model, "names", {})
    items = names.items() if isinstance(names, dict) else enumerate(names)
    available = {
        str(label).casefold().strip(): int(class_id) for class_id, label in items
    }
    try:
        return available["banana"]
    except KeyError as exc:
        raise RuntimeError("banana specialist is missing banana") from exc


class _Frame:
    def __init__(self, bgr: Any) -> None:
        self._bgr = bgr

    def to_ndarray(self, *, format: str) -> Any:
        if format != "bgr24":
            raise ValueError(f"unsupported frame format {format!r}")
        return self._bgr.copy()


def _prepare_runtime() -> PerceptionRuntime:
    general_path = os.environ.get("PEAR_MODEL_PATH", "/media/model.engine")
    specialist_path = os.environ.get(
        "BANANA_SPECIALIST_MODEL_PATH", "/media/banana-specialist.pt"
    )
    general = YOLO(general_path, task="segment")
    specialist = YOLO(specialist_path, task="detect")
    fruit_class_ids = _class_ids(general)
    runtime = PerceptionRuntime()
    runtime._model = general
    runtime._banana_specialist_model = specialist
    runtime._fruit_class_ids = fruit_class_ids
    runtime._model_router = FruitModelRouter(
        general_model=general,
        general_class_ids=fruit_class_ids,
        banana_specialist_model=specialist,
        banana_specialist_class_id=_specialist_class_id(specialist),
        banana_minimum_confidence=runtime.banana_specialist_minimum_confidence,
        banana_minimum_agreement_iou=(
            runtime.banana_specialist_minimum_agreement_iou
        ),
    )
    return runtime


def _run_once(
    runtime: PerceptionRuntime,
    *,
    bgr: Any,
    target: str,
    pts: int,
) -> tuple[dict[str, Any] | None, float, str | None]:
    runtime.select_target(target)
    received = time.monotonic()
    runtime.evidence.note_source(
        pts=pts,
        time_base="1/90000",
        received_monotonic_s=received,
        width=int(bgr.shape[1]),
        height=int(bgr.shape[0]),
    )
    started = time.perf_counter()
    runtime._process_frame(_Frame(bgr), received, pts, "1/90000")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    status = runtime.evidence.status()
    detection = status.get("detection")
    return (
        dict(detection) if isinstance(detection, dict) and detection else None,
        elapsed_ms,
        str(status["error"]) if status.get("error") else None,
    )


def _targets(record: dict[str, Any]) -> list[str]:
    expected = [str(value) for value in record.get("expected_classes", [])]
    if not expected:
        expected = sorted(
            {str(value["class_name"]) for value in record.get("expected", [])}
        )
    return expected or list(SUPPORTED_FRUITS)


def _fixture_url(manifest_url: str, relative_path: str) -> str:
    root_url = os.environ.get("CORPUS_ROOT_URL", "").strip()
    if root_url:
        return urllib.parse.urljoin(root_url.rstrip("/") + "/", relative_path)
    marker = "/lab/max-importer-tournament/"
    if marker not in manifest_url:
        raise ValueError("set CORPUS_ROOT_URL when the manifest URL is not under repo lab")
    return manifest_url.split(marker, 1)[0].rstrip("/") + "/" + relative_path


def run() -> dict[str, Any]:
    manifest_url = os.environ["CORPUS_MANIFEST_URL"]
    manifest = json.loads(_fetch(manifest_url))
    warmups = int(os.environ.get("CONTROL_WARMUPS", "3"))
    repetitions = int(os.environ.get("CONTROL_REPETITIONS", "3"))
    runtime = _prepare_runtime()
    started_wall = time.time()
    fixture_results: list[dict[str, Any]] = []
    all_latencies: list[float] = []

    with tempfile.TemporaryDirectory(prefix="fruit-control-") as temporary:
        temporary_root = Path(temporary)
        fixtures: list[tuple[dict[str, Any], Any]] = []
        for index, record in enumerate(manifest["fixtures"]):
            payload = _fetch(_fixture_url(manifest_url, str(record["path"])))
            digest = _sha256(payload)
            if digest != record["sha256"]:
                raise RuntimeError(
                    f"fixture {record['id']} digest mismatch: {digest}"
                )
            destination = temporary_root / f"{index:03d}.jpg"
            destination.write_bytes(payload)
            bgr = cv2.imread(str(destination), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"fixture {record['id']} is not a readable image")
            fixtures.append((record, bgr))

        zero = fixtures[-1][1] * 0
        for index in range(warmups):
            _run_once(runtime, bgr=zero, target="pear", pts=index + 1)

        pts = warmups + 1
        for record, bgr in fixtures:
            target_results: list[dict[str, Any]] = []
            expected_boxes = [
                value
                for value in record.get("expected", [])
                if value.get("box_xyxy") is not None
            ]
            for target in _targets(record):
                durations: list[float] = []
                detection: dict[str, Any] | None = None
                error: str | None = None
                for _ in range(repetitions):
                    detection, elapsed_ms, error = _run_once(
                        runtime,
                        bgr=bgr,
                        target=target,
                        pts=pts,
                    )
                    pts += 1
                    durations.append(elapsed_ms)
                all_latencies.extend(durations)
                confidence = (
                    None if detection is None else float(detection["confidence"])
                )
                accepted = bool(
                    confidence is not None
                    and confidence >= FRUIT_ACQUISITION_CONFIDENCE[target]
                )
                matching_boxes = [
                    value for value in expected_boxes if value["class_name"] == target
                ]
                best_iou = None
                if detection is not None and matching_boxes:
                    predicted_box = [float(value) for value in detection["bbox_xyxy"]]
                    best_iou = max(
                        _iou(predicted_box, list(value["box_xyxy"]))
                        for value in matching_boxes
                    )
                target_results.append(
                    {
                        "target": target,
                        "acquisition_threshold": FRUIT_ACQUISITION_CONFIDENCE[target],
                        "accepted_for_demo": accepted,
                        "detection": detection,
                        "best_expected_box_iou": best_iou,
                        "latency_ms": _summarize(durations),
                        "error": error,
                    }
                )
            expected_classes = set(_targets(record)) if _targets(record) != list(SUPPORTED_FRUITS) else set()
            is_negative = not record.get("expected") and not record.get(
                "expected_classes"
            )
            successful_targets = {
                value["target"]
                for value in target_results
                if value["accepted_for_demo"]
            }
            fixture_results.append(
                {
                    "fixture_id": record["id"],
                    "tags": record.get("tags", []),
                    "expected_classes": sorted(expected_classes),
                    "successful_targets": sorted(successful_targets),
                    "passed": (
                        not successful_targets if is_negative else expected_classes <= successful_targets
                    ),
                    "false_positive": is_negative and bool(successful_targets),
                    "targets": target_results,
                }
            )

    critical = [
        value for value in fixture_results if "stage_critical" in value["tags"]
    ]
    positives = [value for value in fixture_results if value["expected_classes"]]
    expected_count = sum(len(value["expected_classes"]) for value in positives)
    expected_hits = sum(
        len(set(value["expected_classes"]) & set(value["successful_targets"]))
        for value in positives
    )
    return {
        "schema_version": 1,
        "runtime": "tensorrt-production-control",
        "model": {
            "general_path": os.environ.get("PEAR_MODEL_PATH", "/media/model.engine"),
            "general_sha256": _sha256(
                Path(os.environ.get("PEAR_MODEL_PATH", "/media/model.engine")).read_bytes()
            ),
            "banana_specialist_path": os.environ.get(
                "BANANA_SPECIALIST_MODEL_PATH", "/media/banana-specialist.pt"
            ),
            "crop_confirm": asdict(runtime._crop_confirm),
        },
        "protocol": {
            "manifest_url": manifest_url,
            "manifest_name": manifest.get("name"),
            "fixture_count": len(fixture_results),
            "warmups": warmups,
            "timed_repetitions": repetitions,
            "motion_enabled": False,
            "camera_opened": False,
        },
        "accuracy": {
            "fixture_passes": sum(value["passed"] for value in fixture_results),
            "fixture_count": len(fixture_results),
            "expected_class_recall": expected_hits / max(1, expected_count),
            "negative_false_positives": sum(
                value["false_positive"] for value in fixture_results
            ),
            "stage_critical_passes": sum(value["passed"] for value in critical),
            "stage_critical_count": len(critical),
        },
        "latency_ms": _summarize(all_latencies),
        "fixtures": fixture_results,
        "started_unix_s": started_wall,
        "completed_unix_s": time.time(),
    }


def main() -> None:
    result = run()
    print("TENSORRT_CONTROL_RESULT=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
