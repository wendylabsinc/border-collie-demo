"""Stationary forward-range calibration for metric pear Arrival."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from urllib import request

KNOWN_CLEARANCES_M = (0.50, 0.30, 0.20, 0.15)


@dataclass(frozen=True)
class CalibrationSample:
    known_clearance_m: float
    ranges_m: tuple[float, ...]
    pose_age_s: float
    pear_visible: bool
    pear_center_error_ratio: float | None


def analyze(samples: list[CalibrationSample]) -> dict[str, object]:
    if not samples:
        raise ValueError("calibration requires samples")
    distances = sorted({sample.known_clearance_m for sample in samples})
    if distances != sorted(KNOWN_CLEARANCES_M):
        raise ValueError("calibration requires 50, 30, 20, and 15 cm samples")
    width = len(samples[0].ranges_m)
    if width == 0 or any(len(sample.ranges_m) != width for sample in samples):
        raise ValueError("range samples must have one stable non-empty shape")

    candidates: list[dict[str, object]] = []
    for index in range(width):
        medians = []
        usable = True
        for distance in distances:
            values = [
                sample.ranges_m[index]
                for sample in samples
                if sample.known_clearance_m == distance
                and math.isfinite(sample.ranges_m[index])
                and sample.ranges_m[index] > 0.0
            ]
            if not values:
                usable = False
                break
            medians.append(statistics.median(values))
        if not usable:
            continue
        slope = (medians[-1] - medians[0]) / (distances[-1] - distances[0])
        offsets = [
            sample.ranges_m[index] - sample.known_clearance_m
            for sample in samples
            if math.isfinite(sample.ranges_m[index])
            and sample.ranges_m[index] > 0.0
        ]
        offset = statistics.median(offsets)
        residuals = [
            value - (distance + offset)
            for distance, value in zip(distances, medians, strict=True)
        ]
        rmse = math.sqrt(sum(value * value for value in residuals) / len(residuals))
        monotonic = all(
            current > previous
            for previous, current in pairwise(medians)
        )
        candidates.append(
            {
                "index": index,
                "slope": slope,
                "offset_m": offset,
                "rmse_m": rmse,
                "monotonic": monotonic,
                "medians_m": medians,
                "score": abs(slope - 1.0) + 4.0 * rmse,
            }
        )
    qualified_candidates = [
        candidate
        for candidate in candidates
        if candidate["monotonic"] and 0.50 <= candidate["slope"] <= 1.50
    ]
    if not qualified_candidates:
        raise ValueError("no directional range follows the known forward clearances")
    selected = min(
        qualified_candidates,
        key=lambda candidate: float(candidate["score"]),
    )
    index = int(selected["index"])
    offset = float(selected["offset_m"])
    residuals = [
        sample.ranges_m[index] - sample.known_clearance_m - offset
        for sample in samples
        if sample.ranges_m[index] > 0.0
    ]
    noise_m = _percentile([abs(value) for value in residuals], 0.95)
    latency_s = _percentile([sample.pose_age_s for sample in samples], 0.95)
    pear_visible = all(sample.pear_visible for sample in samples)
    blockers = []
    if noise_m > 0.025:
        blockers.append("range noise exceeds half of the 5 cm Arrival tolerance")
    if latency_s > 0.25:
        blockers.append("range evidence age exceeds 250 ms")
    if not pear_visible:
        blockers.append("the floor-level pear was not centered and visible in every sample")
    return {
        "qualified": not blockers,
        "blockers": blockers,
        "forward_index": index,
        "sensor_to_front_envelope_m": offset,
        "sensor_latency_s_p95": latency_s,
        "noise_m_p95": noise_m,
        "candidate_channels": candidates,
        "braking_distance_m": None,
        "braking_qualification_required": True,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _get_json(url: str, timeout_s: float) -> Mapping[str, object]:
    with request.urlopen(url, timeout=timeout_s) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise TypeError(f"{url} did not return a JSON object")
    return payload


def _capture(
    app_url: str,
    media_url: str,
    known_clearance_m: float,
    *,
    count: int,
    interval_s: float,
) -> list[CalibrationSample]:
    captured = []
    for _ in range(count):
        app = _get_json(f"{app_url}/api/status", timeout_s=2.0)
        media = _get_json(f"{media_url}/status", timeout_s=2.0)
        hardware = app.get("hardware")
        pose = hardware.get("pose") if isinstance(hardware, dict) else None
        motion = pose.get("motion") if isinstance(pose, dict) else None
        ranges = (
            motion.get("obstacle_ranges_m") if isinstance(motion, dict) else None
        )
        age_s = pose.get("age_s") if isinstance(pose, dict) else None
        detection = media.get("detection")
        center_x = (
            detection.get("center_x_ratio")
            if isinstance(detection, dict)
            else None
        )
        label = (
            str(detection.get("label") or "").casefold()
            if isinstance(detection, dict)
            else ""
        )
        if not isinstance(ranges, list) or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in ranges
        ):
            raise RuntimeError("Go2 obstacle range telemetry is unavailable")
        if not isinstance(age_s, (int, float)) or isinstance(age_s, bool):
            raise TypeError("Go2 range age is unavailable")
        center_error = (
            float(center_x) - 0.5
            if isinstance(center_x, (int, float)) and not isinstance(center_x, bool)
            else None
        )
        captured.append(
            CalibrationSample(
                known_clearance_m=known_clearance_m,
                ranges_m=tuple(float(value) for value in ranges),
                pose_age_s=float(age_s),
                pear_visible=bool(
                    media.get("camera_healthy") is True
                    and label == "pear"
                    and center_error is not None
                    and abs(center_error) <= 0.15
                ),
                pear_center_error_ratio=center_error,
            )
        )
        time.sleep(interval_s)
    return captured


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate Go2 forward range at known front-envelope clearances"
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--interval-s", type=float, default=0.10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/forward-range-calibration.json"),
    )
    args = parser.parse_args()
    if args.samples < 5 or args.interval_s <= 0.0:
        parser.error("use at least five samples and a positive interval")
    host = args.host.strip()
    app_url = f"http://{host}:8110"
    media_url = f"http://{host}:8111"
    samples: list[CalibrationSample] = []
    for clearance_m in KNOWN_CLEARANCES_M:
        input(
            f"Place the pear {clearance_m * 100:.0f} cm from Woof's front "
            "body/paw envelope, center it in view, then press Enter: "
        )
        samples.extend(
            _capture(
                app_url,
                media_url,
                clearance_m,
                count=args.samples,
                interval_s=args.interval_s,
            )
        )
    result = analyze(samples)
    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "known_clearances_m": list(KNOWN_CLEARANCES_M),
        "analysis": result,
        "samples": [sample.__dict__ for sample in samples],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved {args.output}")
    return 0 if result["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
