"""Benchmark and parity-check the vectorised colour band backend.

Run in any environment that has numpy (the media container, or the training
venv on a laptop):

    python3 -m media.fruit_color_benchmark --iterations 200

The reference timing includes the `.tolist()` marshalling that the pure-Python
loop needs, because that is the real per-frame cost `classify_bbox_color` pays
when the backend is off. The numpy backend is timed from the same ndarray
region, which is what `classify_bbox_color` hands it. Exits non-zero if any
backend's counts differ from the reference by even one pixel.
"""

from __future__ import annotations

import argparse
import json
import time
from statistics import median

from media.fruit_color import count_band_pixels, summarize_band_counts
from media.fruit_color_backend import NumpyBandCounter


def _sample_region(pixel_count: int, seed: int):
    """A mango-like box: mostly warm orange, some background and specular pixels."""
    import numpy as np

    rng = np.random.default_rng(seed)
    orange = rng.normal((40.0, 150.0, 235.0), 18.0, size=(pixel_count, 3))
    background = rng.normal((110.0, 108.0, 112.0), 25.0, size=(pixel_count, 3))
    is_fruit = rng.random(pixel_count) < 0.62
    region = np.where(is_fruit[:, None], orange, background)
    return np.ascontiguousarray(np.clip(region, 0.0, 255.0))


def _timed(label, function, region, iterations):
    timings = []
    result = None
    for _ in range(iterations):
        started = time.perf_counter()
        result = function(region)
        timings.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(timings)
    return {
        "backend": label,
        "counts": list(result),
        "p50_ms": round(median(timings), 4),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 4),
        "mean_ms": round(sum(timings) / len(timings), 4),
        "minimum_ms": round(ordered[0], 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Colour band backend benchmark")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--pixels",
        type=int,
        default=4096,
        help="sampled pixels per box; classify_bbox_color strides to about 4096",
    )
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()
    if arguments.iterations < 1 or arguments.pixels < 1:
        raise SystemExit("--iterations and --pixels must be positive")

    region = _sample_region(arguments.pixels, arguments.seed)
    rows = [
        _timed(
            "python_reference",
            lambda array: count_band_pixels(array.tolist()),
            region,
            arguments.iterations,
        )
    ]
    warnings = []

    try:
        numpy_backend = NumpyBandCounter()
        rows.append(_timed("numpy", numpy_backend.count, region, arguments.iterations))
    except Exception as error:  # noqa: BLE001  # pragma: no cover
        warnings.append(f"numpy backend unavailable: {error}")

    baseline = rows[0]
    reference_counts = tuple(baseline["counts"])
    for row in rows:
        row["speedup_versus_reference"] = round(baseline["p50_ms"] / row["p50_ms"], 3)
        row["counts_match_reference"] = tuple(row["counts"]) == reference_counts
        row["summary_matches_reference"] = summarize_band_counts(
            *row["counts"]
        ) == summarize_band_counts(*reference_counts)

    report = {
        "iterations": arguments.iterations,
        "pixels_per_call": arguments.pixels,
        "results": rows,
        "warnings": warnings,
        "control_authority": "none_measurement_only",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if any(not row["counts_match_reference"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
