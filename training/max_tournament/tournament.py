"""Run fixed-corpus accuracy and raw-output parity checks before Woof testing."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ExpectedDetection:
    class_name: str
    box_xyxy: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class CorpusItem:
    fixture_id: str
    path: Path
    sha256: str
    expected_classes: tuple[str, ...]
    expected: tuple[ExpectedDetection, ...]
    tags: tuple[str, ...]


class Runtime(Protocol):
    runtime_id: str

    def infer(self, tensor: np.ndarray) -> np.ndarray: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus(root: Path, manifest_path: Path) -> tuple[dict[str, Any], list[CorpusItem]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported corpus schema_version")
    classes = tuple(str(value) for value in manifest["classes"])
    if len(set(classes)) != len(classes):
        raise ValueError("corpus classes must be unique")

    items: list[CorpusItem] = []
    fixture_ids: set[str] = set()
    for record in manifest["fixtures"]:
        fixture_id = str(record["id"])
        if fixture_id in fixture_ids:
            raise ValueError(f"duplicate fixture id {fixture_id!r}")
        fixture_ids.add(fixture_id)
        path = root / str(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"fixture {fixture_id!r} is missing: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != record["sha256"]:
            raise ValueError(
                f"fixture {fixture_id!r} digest mismatch: "
                f"expected {record['sha256']}, got {actual_sha256}"
            )
        expected = tuple(
            ExpectedDetection(
                class_name=str(value["class_name"]),
                box_xyxy=(
                    tuple(float(component) for component in value["box_xyxy"])
                    if value.get("box_xyxy") is not None
                    else None
                ),
            )
            for value in record.get("expected", [])
        )
        expected_classes = tuple(
            str(value)
            for value in record.get(
                "expected_classes",
                sorted({value.class_name for value in expected}),
            )
        )
        unknown = (set(expected_classes) | {value.class_name for value in expected}) - set(
            classes
        )
        if unknown:
            raise ValueError(
                f"fixture {fixture_id!r} uses unknown classes: {sorted(unknown)}"
            )
        items.append(
            CorpusItem(
                fixture_id=fixture_id,
                path=path,
                sha256=actual_sha256,
                expected_classes=expected_classes,
                expected=expected,
                tags=tuple(str(value) for value in record.get("tags", [])),
            )
        )
    if not items:
        raise ValueError("corpus must contain at least one fixture")
    return manifest, items


def prepare_image(path: Path, input_size: int) -> tuple[np.ndarray, dict[str, float]]:
    with Image.open(path) as encoded:
        image = encoded.convert("RGB")
    source_width, source_height = image.size
    scale = min(input_size / source_width, input_size / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = image.resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )
    pad_x = (input_size - resized_width) // 2
    pad_y = (input_size - resized_height) // 2
    canvas = Image.new("RGB", (input_size, input_size), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))
    nhwc = np.asarray(canvas, dtype=np.float32)[None] / np.float32(255.0)
    tensor = np.ascontiguousarray(nhwc.transpose(0, 3, 1, 2))
    return tensor, {
        "source_width": float(source_width),
        "source_height": float(source_height),
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
    }


def _box_iou(first: list[float] | tuple[float, ...], second: list[float] | tuple[float, ...]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def decode_yolo(
    raw: np.ndarray,
    geometry: dict[str, float],
    classes: tuple[str, ...],
    confidence_floor: float,
    iou_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 4 + len(classes):
        raise ValueError(
            f"unexpected YOLO output shape {raw.shape}; expected [1,{4 + len(classes)},N]"
        )
    predictions = raw[0]
    boxes = predictions[:4].T.astype(np.float32, copy=False)
    scores = predictions[4:].T.astype(np.float32, copy=False)
    class_ids = np.argmax(scores, axis=1)
    confidences = scores[np.arange(scores.shape[0]), class_ids]
    candidates: list[dict[str, Any]] = []
    scale = geometry["scale"]
    pad_x = geometry["pad_x"]
    pad_y = geometry["pad_y"]
    source_width = geometry["source_width"]
    source_height = geometry["source_height"]
    for index in np.flatnonzero(confidences >= confidence_floor):
        center_x, center_y, width, height = (float(value) for value in boxes[index])
        box = [
            max(0.0, (center_x - width / 2.0 - pad_x) / scale),
            max(0.0, (center_y - height / 2.0 - pad_y) / scale),
            min(source_width, (center_x + width / 2.0 - pad_x) / scale),
            min(source_height, (center_y + height / 2.0 - pad_y) / scale),
        ]
        candidates.append(
            {
                "class_id": int(class_ids[index]),
                "class_name": classes[int(class_ids[index])],
                "confidence": float(confidences[index]),
                "box_xyxy": box,
            }
        )
    candidates.sort(key=lambda value: value["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(
            candidate["class_id"] == previous["class_id"]
            and _box_iou(candidate["box_xyxy"], previous["box_xyxy"])
            > iou_threshold
            for previous in kept
        ):
            continue
        kept.append(candidate)
    return kept


class PyTorchRuntime:
    def __init__(self, runtime_id: str, model_path: Path) -> None:
        import torch
        from ultralytics import YOLO

        self.runtime_id = runtime_id
        self._torch = torch
        self._model = YOLO(str(model_path)).model.eval().to("cpu")

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        with self._torch.inference_mode():
            output = self._model(self._torch.from_numpy(tensor))
        if isinstance(output, (list, tuple)):
            output = output[0]
        return np.ascontiguousarray(output.detach().cpu().numpy())


class OnnxRuntime:
    def __init__(self, runtime_id: str, model_path: Path) -> None:
        import onnxruntime

        self.runtime_id = runtime_id
        self._session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            self._session.run(None, {self._input_name: tensor})[0]
        )


def _runtime(root: Path, record: dict[str, Any]) -> Runtime:
    runtime_id = str(record["id"])
    model_path = root / str(record["model"])
    if not model_path.is_file():
        raise FileNotFoundError(f"runtime {runtime_id!r} model is missing: {model_path}")
    kind = record["kind"]
    if kind == "pytorch":
        return PyTorchRuntime(runtime_id, model_path)
    if kind == "onnxruntime":
        return OnnxRuntime(runtime_id, model_path)
    raise ValueError(f"unknown runtime kind {kind!r}")


def _summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, round(0.95 * (len(ordered) - 1))))
    return {
        "median": float(statistics.median(ordered)),
        "p95": float(ordered[p95_index]),
        "maximum": float(ordered[-1]),
    }


def _raw_parity(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "passed": False,
            "reason": f"shape mismatch {reference.shape} != {candidate.shape}",
        }
    difference = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    mean = float(np.mean(difference))
    p99 = float(np.quantile(difference, 0.99))
    maximum = float(np.max(difference))
    return {
        "passed": mean <= 0.001 and p99 <= 0.01 and maximum <= 0.1,
        "mean_absolute_difference": mean,
        "p99_absolute_difference": p99,
        "maximum_absolute_difference": maximum,
    }


def _grade_fixture(
    item: CorpusItem,
    accepted: list[dict[str, Any]],
    diagnostic: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, Any]:
    accepted_classes = {str(value["class_name"]) for value in accepted}
    expected_class_hits = sum(
        class_name in accepted_classes for class_name in item.expected_classes
    )
    unmatched_predictions = set(range(len(accepted)))
    box_hits = 0
    for expected in item.expected:
        best: tuple[float, int] | None = None
        for prediction_index in unmatched_predictions:
            prediction = accepted[prediction_index]
            if prediction["class_name"] != expected.class_name:
                continue
            if expected.box_xyxy is None:
                best = (1.0, prediction_index)
                break
            iou = _box_iou(expected.box_xyxy, prediction["box_xyxy"])
            if best is None or iou > best[0]:
                best = (iou, prediction_index)
        if best is not None and (
            expected.box_xyxy is None or best[0] >= iou_threshold
        ):
            unmatched_predictions.remove(best[1])
            box_hits += 1
    is_negative = not item.expected_classes and not item.expected
    false_positive = is_negative and bool(accepted)
    return {
        "fixture_id": item.fixture_id,
        "expected_classes": list(item.expected_classes),
        "accepted_classes": sorted(accepted_classes),
        "diagnostic_top_class": diagnostic[0]["class_name"] if diagnostic else None,
        "diagnostic_top_confidence": (
            diagnostic[0]["confidence"] if diagnostic else None
        ),
        "expected_class_hits": expected_class_hits,
        "expected_class_count": len(item.expected_classes),
        "box_hits_iou50": box_hits,
        "expected_box_count": sum(value.box_xyxy is not None for value in item.expected),
        "false_positive": false_positive,
        "accepted_detections": accepted,
        "passed": (
            not false_positive
            and expected_class_hits == len(item.expected_classes)
            and box_hits == sum(value.box_xyxy is not None for value in item.expected)
        ),
    }


def run_tournament(
    root: Path,
    corpus_path: Path,
    variants_path: Path,
) -> dict[str, Any]:
    corpus, items = load_corpus(root, corpus_path)
    variants = json.loads(variants_path.read_text(encoding="utf-8"))
    if variants.get("schema_version") != 1:
        raise ValueError("unsupported variants schema_version")
    records = variants["variants"]
    reference_id = str(variants["reference_variant"])
    input_size = int(variants["protocol"].get("input_size", corpus["input_size"]))
    classes = tuple(str(value) for value in corpus["classes"])
    warmups = int(variants["protocol"]["warmups"])
    repetitions = int(variants["protocol"]["timed_repetitions"])
    acceptance_floor = float(corpus["thresholds"]["acceptance_confidence"])
    diagnostic_floor = float(corpus["thresholds"]["diagnostic_confidence"])
    iou_threshold = float(corpus["thresholds"]["match_iou"])

    tensors: dict[str, tuple[np.ndarray, dict[str, float]]] = {
        item.fixture_id: prepare_image(item.path, input_size) for item in items
    }
    zero = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
    raw_outputs: dict[str, dict[str, np.ndarray]] = {}
    variant_results: list[dict[str, Any]] = []

    for record in records:
        runtime = _runtime(root, record)
        for _ in range(warmups):
            runtime.infer(zero)
        fixture_results: list[dict[str, Any]] = []
        latencies: list[float] = []
        raw_outputs[runtime.runtime_id] = {}
        for item in items:
            tensor, geometry = tensors[item.fixture_id]
            durations: list[float] = []
            raw: np.ndarray | None = None
            for _ in range(repetitions):
                started = time.perf_counter()
                raw = runtime.infer(tensor)
                durations.append((time.perf_counter() - started) * 1000.0)
            assert raw is not None
            raw_outputs[runtime.runtime_id][item.fixture_id] = raw
            latencies.extend(durations)
            accepted = decode_yolo(
                raw, geometry, classes, acceptance_floor
            )
            diagnostic = decode_yolo(
                raw, geometry, classes, diagnostic_floor
            )
            fixture = _grade_fixture(item, accepted, diagnostic, iou_threshold)
            fixture["latency_ms"] = _summarize(durations)
            fixture["raw_output_sha256"] = hashlib.sha256(raw.tobytes()).hexdigest()
            fixture_results.append(fixture)
        acceptance_pairs = [
            (item, value)
            for item, value in zip(items, fixture_results, strict=True)
            if "diagnostic_only" not in item.tags
        ]
        expected_classes = sum(
            value["expected_class_count"] for _, value in acceptance_pairs
        )
        expected_boxes = sum(
            value["expected_box_count"] for _, value in acceptance_pairs
        )
        class_hits = sum(
            value["expected_class_hits"] for _, value in acceptance_pairs
        )
        box_hits = sum(value["box_hits_iou50"] for _, value in acceptance_pairs)
        critical = [
            value
            for item, value in acceptance_pairs
            if "stage_critical" in item.tags
        ]
        accuracy = {
            "fixture_passes": sum(
                value["passed"] for _, value in acceptance_pairs
            ),
            "fixture_count": len(acceptance_pairs),
            "diagnostic_fixture_count": len(fixture_results) - len(acceptance_pairs),
            "expected_class_recall": class_hits / max(1, expected_classes),
            "box_recall_iou50": box_hits / max(1, expected_boxes),
            "negative_false_positives": sum(
                value["false_positive"] for _, value in acceptance_pairs
            ),
            "stage_critical_passes": sum(value["passed"] for value in critical),
            "stage_critical_count": len(critical),
        }
        variant_results.append(
            {
                "variant": runtime.runtime_id,
                "kind": record["kind"],
                "phase": "local_accuracy",
                "precision": record["precision"],
                "device": "cpu",
                "latency_ms_advisory_only": _summarize(latencies),
                "accuracy": accuracy,
                "fixtures": fixture_results,
            }
        )

    if reference_id not in raw_outputs:
        raise ValueError(f"reference variant {reference_id!r} did not run")
    gates = variants["gates"]
    records_by_id = {str(record["id"]): record for record in records}
    for result in variant_results:
        variant_id = result["variant"]
        parity_reference_id = records_by_id[variant_id].get("parity_reference")
        if parity_reference_id is None and variant_id == reference_id:
            parity_reference_id = reference_id
        if parity_reference_id is not None:
            parity_reference_id = str(parity_reference_id)
            try:
                parity_reference_outputs = raw_outputs[parity_reference_id]
            except KeyError as exc:
                raise ValueError(
                    f"variant {variant_id!r} references missing parity source "
                    f"{parity_reference_id!r}"
                ) from exc
            parity_by_fixture = {
                item.fixture_id: _raw_parity(
                    parity_reference_outputs[item.fixture_id],
                    raw_outputs[variant_id][item.fixture_id],
                )
                for item in items
            }
            parity_passed: bool | None = all(
                value["passed"] for value in parity_by_fixture.values()
            )
        else:
            parity_by_fixture = {}
            parity_passed = None
        accuracy = result["accuracy"]
        reasons: list[str] = []
        if parity_passed is False:
            reasons.append(
                f"raw output diverges from parity source {parity_reference_id}"
            )
        if accuracy["expected_class_recall"] < float(gates["minimum_class_recall"]):
            reasons.append("ground-truth class recall is below the local gate")
        if accuracy["box_recall_iou50"] < float(gates["minimum_box_recall_iou50"]):
            reasons.append("ground-truth box recall is below the local gate")
        if accuracy["negative_false_positives"]:
            reasons.append("negative fixtures produced accepted detections")
        if accuracy["stage_critical_passes"] != accuracy["stage_critical_count"]:
            reasons.append("one or more stage-critical Go2 fixtures failed")
        result["raw_parity_to_reference"] = {
            "passed": parity_passed,
            "reference_variant": parity_reference_id,
            "fixtures": parity_by_fixture,
        }
        result["verdict"] = "accepted" if not reasons else "rejected"
        result["rejection_reasons"] = reasons

    controls = variants.get("controls", [])
    leaderboard = [
        {
            "variant": result["variant"],
            "phase": result["phase"],
            "median_ms": result["latency_ms_advisory_only"]["median"],
            "p95_ms": result["latency_ms_advisory_only"]["p95"],
            "class_recall": result["accuracy"]["expected_class_recall"],
            "box_recall_iou50": result["accuracy"]["box_recall_iou50"],
            "raw_parity": result["raw_parity_to_reference"]["passed"],
            "verdict": result["verdict"],
        }
        for result in variant_results
    ]
    leaderboard.extend(controls)
    return {
        "schema_version": 1,
        "protocol": {
            "input_size": input_size,
            "classes": list(classes),
            "warmups": warmups,
            "timed_repetitions": repetitions,
            "local_precision": "fp32",
            "note": (
                "Local CPU latency is advisory. Woof candidates must repeat with FP16, "
                "native sm_87 execution, TensorRT corpus outputs, and safety telemetry."
            ),
        },
        "corpus": {
            "manifest": str(corpus_path),
            "fixture_count": len(items),
            "fixture_ids": [item.fixture_id for item in items],
        },
        "reference_variant": reference_id,
        "variants": variant_results,
        "leaderboard": leaderboard,
        "next_gate": (
            "Capture production TensorRT raw and decoded outputs for this exact corpus, "
            "then require each MAX importer variant to match them before Woof latency tests."
        ),
    }
