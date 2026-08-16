#!/usr/bin/env python3
"""Record sidecar frame/detection behavior during a supervised camera jiggle.

This tool is intentionally read-only. It polls the media sidecar status endpoint,
writes one JSON object per poll, and emits a compact summary beside the JSONL.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _status(url: str, timeout_s: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read())


def _finite(value: object) -> float | None:
    if not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _sample(
    payload: dict[str, Any],
    *,
    elapsed_s: float,
    latency_ms: float,
    previous_source_pts: object,
    previous_detection_pts: object,
) -> dict[str, Any]:
    source = payload.get("source") or {}
    detection = payload.get("detection") or {}
    bbox = detection.get("bbox_xyxy")
    width = _finite(source.get("width"))
    height = _finite(source.get("height"))
    geometry: dict[str, float] | None = None
    if (
        isinstance(bbox, list)
        and len(bbox) == 4
        and width
        and height
        and width > 0
        and height > 0
    ):
        left, top, right, bottom = (float(value) for value in bbox)
        geometry = {
            "center_x_ratio": ((left + right) / 2.0) / width,
            "center_y_ratio": ((top + bottom) / 2.0) / height,
            "bottom_ratio": bottom / height,
            "width_ratio": (right - left) / width,
            "height_ratio": (bottom - top) / height,
            "area_ratio": ((right - left) * (bottom - top)) / (width * height),
        }

    source_pts = source.get("pts")
    detection_pts = detection.get("source_pts")
    route = detection.get("model_route") or {}
    full_frame = route.get("full_frame") or {}
    return {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_s": round(elapsed_s, 4),
        "request_latency_ms": round(latency_ms, 3),
        "generation": payload.get("generation"),
        "source": {
            "pts": source_pts,
            "time_base": source.get("time_base"),
            "received_monotonic_s": source.get("received_monotonic_s"),
            "consecutive_frames": source.get("consecutive_frames"),
            "width": source.get("width"),
            "height": source.get("height"),
            "advanced": previous_source_pts is None or source_pts != previous_source_pts,
        },
        "detection": {
            "present": bool(detection),
            "source_pts": detection_pts,
            "advanced": (
                previous_detection_pts is None or detection_pts != previous_detection_pts
            ),
            "label": detection.get("label"),
            "confidence": detection.get("confidence"),
            "consecutive_detections": detection.get("consecutive_detections"),
            "inference_s": detection.get("inference_s"),
            "bbox_xyxy": bbox,
            "geometry": geometry,
            "inference_passes": detection.get("inference_passes"),
            "full_frame": {
                "mode": full_frame.get("mode"),
                "triggered": full_frame.get("triggered"),
                "confirmed": full_frame.get("confirmed"),
                "general_confidence": full_frame.get("general_confidence"),
                "specialist_confidence": full_frame.get("specialist_confidence"),
                "agreement_iou": full_frame.get("agreement_iou"),
            },
            "search_crop": route.get("search_crop"),
            "crop_confirmation": route.get("crop_confirmation"),
        },
    }


def _stats(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "samples": len(values),
        "minimum": min(values),
        "average": sum(values) / len(values),
        "maximum": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="woof.local")
    parser.add_argument("--port", type=int, default=8111)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--fruit", default="banana")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration <= 0 or args.hz <= 0 or args.timeout <= 0:
        parser.error("duration, hz, and timeout must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    status_url = f"http://{args.host}:{args.port}/status"
    interval_s = 1.0 / args.hz
    started = time.monotonic()
    deadline = started + args.duration
    previous_source_pts: object = None
    previous_detection_pts: object = None
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    print(f"READY recording {status_url} -> {args.output}", flush=True)
    try:
        with args.output.open("w", encoding="utf-8") as stream:
            while time.monotonic() < deadline:
                cycle_started = time.monotonic()
                try:
                    payload = _status(status_url, args.timeout)
                    latency_ms = (time.monotonic() - cycle_started) * 1000.0
                    record = _sample(
                        payload,
                        elapsed_s=cycle_started - started,
                        latency_ms=latency_ms,
                        previous_source_pts=previous_source_pts,
                        previous_detection_pts=previous_detection_pts,
                    )
                    previous_source_pts = record["source"]["pts"]
                    previous_detection_pts = record["detection"]["source_pts"]
                except Exception as exc:  # noqa: BLE001 - transport faults are evidence
                    error = f"{type(exc).__name__}: {exc}"
                    errors.append(error)
                    record = {
                        "recorded_at_utc": datetime.now(UTC).isoformat(),
                        "elapsed_s": round(cycle_started - started, 4),
                        "request_error": error,
                    }
                records.append(record)
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                stream.flush()
                time.sleep(max(0.0, interval_s - (time.monotonic() - cycle_started)))
    except KeyboardInterrupt:
        pass

    detections = [record.get("detection") or {} for record in records]
    banana = [
        detection
        for detection in detections
        if detection.get("label") == args.fruit
    ]
    confidences = [
        confidence
        for detection in banana
        if (confidence := _finite(detection.get("confidence"))) is not None
    ]
    bottoms = [
        bottom
        for detection in banana
        if isinstance(detection.get("geometry"), dict)
        and (bottom := _finite(detection["geometry"].get("bottom_ratio"))) is not None
    ]
    summary = {
        "schema_version": 1,
        "status_url": status_url,
        "target_fruit": args.fruit,
        "duration_s": time.monotonic() - started,
        "poll_hz_requested": args.hz,
        "samples": len(records),
        "request_errors": len(errors),
        "source_advanced_samples": sum(
            record.get("source", {}).get("advanced") is True for record in records
        ),
        "source_duplicate_samples": sum(
            record.get("source", {}).get("advanced") is False for record in records
        ),
        "detection_present_samples": sum(
            detection.get("present") is True for detection in detections
        ),
        "target_detection_samples": len(banana),
        "target_missing_samples": len(records) - len(banana) - len(errors),
        "confidence": _stats(confidences),
        "bottom_ratio": _stats(bottoms),
        "errors": errors,
        "jsonl": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"STOPPED summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
