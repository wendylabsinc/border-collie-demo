"""Minimal MAX Conv2D filter-layout A/B for Jetson Orin.

The two graphs differ only in filter representation and the FilterLayout
metadata passed to MAX.  The FCRS weights are transposed to RSCF for the
control graph so both convolutions are mathematically identical.
"""

from __future__ import annotations

import json
import os
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


PORT = int(os.environ.get("MAX_CONV_AB_PORT", "8128"))
ARTIFACT_DIR = Path(os.environ.get("MAX_CONV_AB_ARTIFACT_DIR", "/artifacts"))
WARMUPS = int(os.environ.get("MAX_CONV_AB_WARMUPS", "5"))
SAMPLES = int(os.environ.get("MAX_CONV_AB_SAMPLES", "20"))
THERMAL_STATUS_URL = os.environ.get(
    "MAX_CONV_AB_THERMAL_STATUS_URL", "http://127.0.0.1:8102/api/status"
)
THERMAL_LIMIT_C = float(os.environ.get("MAX_CONV_AB_THERMAL_LIMIT_C", "82"))

# Representative YOLO feature-map convolution: enough work to make kernel
# selection obvious without compiling the whole detector.
INPUT_SHAPE = (1, 104, 104, 16)
OUTPUT_CHANNELS = 32
KERNEL_SIZE = 3
PADDING = 1
MIN_EXPECTED_SPEEDUP = 5.0


def make_weight_variants(seed: int = 20260807) -> tuple[np.ndarray, np.ndarray]:
    """Return identical convolution weights in FCRS and RSCF layouts."""

    rng = np.random.default_rng(seed)
    fcrs = rng.normal(
        0.0,
        0.02,
        size=(OUTPUT_CHANNELS, INPUT_SHAPE[-1], KERNEL_SIZE, KERNEL_SIZE),
    ).astype(np.float16)
    rscf = np.transpose(fcrs, (2, 3, 1, 0))
    return np.ascontiguousarray(fcrs), np.ascontiguousarray(rscf)


def evaluate_result(
    *, rscf_median_ms: float, fcrs_median_ms: float, maximum_absolute_error: float
) -> dict[str, Any]:
    speedup = rscf_median_ms / fcrs_median_ms
    parity_passed = maximum_absolute_error <= 0.05
    return {
        "speedup_rscf_over_fcrs": speedup,
        "parity_passed": parity_passed,
        "fcrs_is_faster": fcrs_median_ms < rscf_median_ms,
        "confirms_dispatch_hypothesis": (
            parity_passed
            and fcrs_median_ms < rscf_median_ms
            and speedup >= MIN_EXPECTED_SPEEDUP
        ),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _thermal_preflight() -> dict[str, Any]:
    with urllib.request.urlopen(THERMAL_STATUS_URL, timeout=2.0) as response:
        status = json.load(response)
    readings = {"jetson": float(status.get("hottest_jetson_c", 0.0))}
    go2 = status.get("go2") or {}
    if go2.get("imu_c") is not None:
        readings["go2.imu"] = float(go2["imu_c"])
    sensor, hottest_c = max(readings.items(), key=lambda item: item[1])
    if hottest_c >= THERMAL_LIMIT_C:
        raise RuntimeError(
            f"thermal gate closed: {sensor} is {hottest_c:.1f} C "
            f"(limit {THERMAL_LIMIT_C:.1f} C)"
        )
    return {"sensor": sensor, "hottest_c": hottest_c, "limit_c": THERMAL_LIMIT_C}


def _build_graph(
    layout_name: str, device: Any, fcrs: np.ndarray, rscf: np.ndarray
) -> tuple[Any, dict[str, np.ndarray]]:
    from max.dtype import DType
    from max.graph import DeviceRef, Graph, TensorType, Weight, ops
    from max.graph.type import FilterLayout

    device_ref = DeviceRef.from_device(device)
    input_type = TensorType(DType.float16, list(INPUT_SHAPE), device=device_ref)
    if layout_name == "fcrs":
        values = fcrs
        filter_layout = FilterLayout.FCRS
    elif layout_name == "rscf":
        values = rscf
        filter_layout = FilterLayout.RSCF
    else:
        raise ValueError(f"unknown filter layout {layout_name!r}")

    weight_name = f"conv.{layout_name}.weight"
    weight = Weight(weight_name, DType.float16, values.shape, device=device_ref)
    with Graph(f"conv_layout_ab_{layout_name}", input_types=[input_type]) as graph:
        output = ops.conv2d(
            graph.inputs[0],
            weight,
            stride=(1, 1),
            dilation=(1, 1),
            padding=(PADDING, PADDING, PADDING, PADDING),
            groups=1,
            filter_layout=filter_layout,
        )
        graph.output(output)
    return graph, {weight_name: values}


def run_probe() -> dict[str, Any]:
    from max import driver, engine

    if os.environ.get("CUDA_DISABLE_PTX_JIT") != "1":
        raise RuntimeError("probe requires CUDA_DISABLE_PTX_JIT=1")
    if int(driver.accelerator_count()) < 1:
        raise RuntimeError("MAX reports no accelerator")

    thermal = _thermal_preflight()
    device = driver.Accelerator(0)
    fcrs, rscf = make_weight_variants()
    session = engine.InferenceSession(devices=[device])

    models: dict[str, Any] = {}
    compile_ms: dict[str, float] = {}
    for name in ("rscf", "fcrs"):
        graph, registry = _build_graph(name, device, fcrs, rscf)
        started = time.perf_counter()
        compiled = session.compile(graph)
        models[name] = session.init(compiled, weights_registry=registry)
        compile_ms[name] = (time.perf_counter() - started) * 1000.0

    rng = np.random.default_rng(7)
    input_array = rng.normal(0.0, 1.0, size=INPUT_SHAPE).astype(np.float16)
    input_buffer = driver.Buffer.from_numpy(input_array).to(device)

    # Alternate the measured execution order to limit clock/thermal bias.
    sample_order: list[str] = []
    for index in range(SAMPLES):
        sample_order.extend(("rscf", "fcrs") if index % 2 == 0 else ("fcrs", "rscf"))

    # Warm both before collecting either set. Collection uses the same fixed
    # order list and persistent device input for both models.
    for model in models.values():
        for _ in range(WARMUPS):
            model.execute(input_buffer)[0].to_numpy()

    latencies: dict[str, list[float]] = {"rscf": [], "fcrs": []}
    final_outputs: dict[str, np.ndarray] = {}
    for name in sample_order:
        started = time.perf_counter()
        final_outputs[name] = models[name].execute(input_buffer)[0].to_numpy()
        latencies[name].append((time.perf_counter() - started) * 1000.0)

    summaries = {
        name: {
            "median_ms": statistics.median(values),
            "p95_ms": _percentile(values, 0.95),
            "minimum_ms": min(values),
            "maximum_ms": max(values),
            "samples": len(values),
        }
        for name, values in latencies.items()
    }
    maximum_absolute_error = float(
        np.max(
            np.abs(
                final_outputs["rscf"].astype(np.float32)
                - final_outputs["fcrs"].astype(np.float32)
            )
        )
    )
    verdict = evaluate_result(
        rscf_median_ms=summaries["rscf"]["median_ms"],
        fcrs_median_ms=summaries["fcrs"]["median_ms"],
        maximum_absolute_error=maximum_absolute_error,
    )
    result = {
        "schema_version": 1,
        "device": str(device),
        "input_shape": list(INPUT_SHAPE),
        "output_channels": OUTPUT_CHANNELS,
        "kernel_size": KERNEL_SIZE,
        "dtype": "float16",
        "warmups_per_model": WARMUPS,
        "thermal_preflight": thermal,
        "compile_ms": compile_ms,
        "layouts": summaries,
        "maximum_absolute_error": maximum_absolute_error,
        **verdict,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "max-conv-layout-ab-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("MAX_CONV_LAYOUT_AB_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI(title="MAX Conv2D filter-layout A/B")
    state: dict[str, Any]
    try:
        state = {"ready": True, "error": None, "result": run_probe()}
    except Exception as error:
        state = {"ready": False, "error": f"{type(error).__name__}: {error}", "result": None}
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / "max-conv-layout-ab-failure.json").write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("MAX_CONV_LAYOUT_AB_FAILURE=" + json.dumps(state, sort_keys=True), flush=True)

    @app.get("/result")
    def result() -> dict[str, Any]:
        return state

    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
