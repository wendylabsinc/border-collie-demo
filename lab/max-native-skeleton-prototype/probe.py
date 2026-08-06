"""PROTOTYPE: benchmark a trainable MAX-native detector skeleton on Woof."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI

PORT = int(os.environ.get("MAX_SKELETON_PORT", "8125"))
ARTIFACT_DIR = Path(os.environ.get("MAX_SKELETON_ARTIFACT_DIR", "/artifacts"))
RESOLUTIONS = tuple(
    int(item)
    for item in os.environ.get("MAX_SKELETON_RESOLUTIONS", "320,416,640").split(",")
)
WARMUPS = int(os.environ.get("MAX_SKELETON_WARMUPS", "5"))
SAMPLES = int(os.environ.get("MAX_SKELETON_SAMPLES", "30"))
THERMAL_STATUS_URL = os.environ.get(
    "MAX_SKELETON_THERMAL_STATUS_URL", "http://127.0.0.1:8102/api/status"
)
THERMAL_LIMIT_C = float(os.environ.get("MAX_SKELETON_THERMAL_LIMIT_C", "82"))
MIN_START_BATTERY_PERCENT = float(
    os.environ.get("MAX_SKELETON_MIN_START_BATTERY_PERCENT", "60")
)
BATTERY_ABORT_PERCENT = float(
    os.environ.get("MAX_SKELETON_BATTERY_ABORT_PERCENT", "25")
)
REQUIRE_BATTERY_STATUS = os.environ.get(
    "MAX_SKELETON_REQUIRE_BATTERY_STATUS", "1"
) not in {"0", "false", "False"}
MIN_AVAILABLE_BYTES = int(
    float(os.environ.get("MAX_SKELETON_MIN_AVAILABLE_GIB", "1")) * 1024**3
)
MAX_SWAP_USED_BYTES = int(
    float(os.environ.get("MAX_SKELETON_MAX_SWAP_USED_GIB", "2")) * 1024**3
)
RESOLUTION_TIMEOUT_S = float(
    os.environ.get("MAX_SKELETON_RESOLUTION_TIMEOUT_S", "1200")
)
CPU_AFFINITY_COUNT = int(
    os.environ.get("MAX_SKELETON_CPU_AFFINITY_COUNT", "8")
)
OOM_SCORE_ADJ = int(os.environ.get("MAX_SKELETON_OOM_SCORE_ADJ", "500"))
ARCHITECTURE_VERSION = 1
MEDIAN_GATE_MS = 150.0
P95_GATE_MS = 200.0

# MobileNetV3-like inverted residuals: input, output, expanded, stride.
BLOCKS = (
    (16, 16, 16, 1),
    (16, 24, 64, 2),
    (24, 24, 72, 1),
    (24, 40, 72, 2),
    (40, 40, 120, 1),
    (40, 40, 120, 1),
    (40, 80, 240, 2),
    (80, 80, 200, 1),
    (80, 80, 184, 1),
    (80, 80, 184, 1),
    (80, 112, 480, 1),
    (112, 112, 672, 1),
    (112, 160, 672, 2),
    (160, 160, 960, 1),
    (160, 160, 960, 1),
)
ARCHITECTURE = {
    "version": ARCHITECTURE_VERSION,
    "layout": "NHWC",
    "dtype": "float16",
    "classes": ["pear", "apple", "banana"],
    "stem_channels": 16,
    "blocks": BLOCKS,
    "head_strides": [8, 16, 32],
    "head_channels": 7,
    "postprocessing_in_graph": False,
}
ARCHITECTURE_SHA256 = hashlib.sha256(
    json.dumps(ARCHITECTURE, sort_keys=True).encode("utf-8")
).hexdigest()

_lock = Lock()
_worker_lock = Lock()
_worker_process: subprocess.Popen[str] | None = None
_state: dict[str, Any] = {
    "ready": False,
    "phase": "starting",
    "backend": "max",
    "prototype": True,
    "architecture": ARCHITECTURE,
    "architecture_sha256": ARCHITECTURE_SHA256,
    "resolutions": list(RESOLUTIONS),
    "error": None,
    "results": [],
}


def _set_state(**updates: Any) -> None:
    with _lock:
        _state.update(updates)


def _snapshot() -> dict[str, Any]:
    with _lock:
        return json.loads(json.dumps(_state))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _limit_process() -> None:
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, allowed[: min(CPU_AFFINITY_COUNT, len(allowed))])
    Path("/proc/self/oom_score_adj").write_text(
        f"{OOM_SCORE_ADJ}\n", encoding="utf-8"
    )


def _read_thermal_status() -> dict[str, Any]:
    with urllib.request.urlopen(THERMAL_STATUS_URL, timeout=1.0) as response:
        return json.load(response)


def _guard_values(status: dict[str, Any]) -> tuple[str, float, float | None]:
    readings = {"jetson": float(status.get("hottest_jetson_c", 0.0))}
    go2 = status.get("go2") or {}
    if go2.get("imu_c") is not None:
        readings["go2.imu"] = float(go2["imu_c"])
    sensor, temperature_c = max(readings.items(), key=lambda item: item[1])
    battery = go2.get("battery_soc_percent")
    return sensor, temperature_c, None if battery is None else float(battery)


def _memory_status() -> dict[str, int]:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open() as handle:
        for line in handle:
            name, raw = line.split(":", 1)
            if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                values[name] = int(raw.split()[0]) * 1024
    return {
        "total_bytes": values["MemTotal"],
        "available_bytes": values["MemAvailable"],
        "swap_total_bytes": values["SwapTotal"],
        "swap_used_bytes": values["SwapTotal"] - values["SwapFree"],
    }


def _preflight() -> None:
    status = _read_thermal_status()
    sensor, temperature_c, battery = _guard_values(status)
    memory = _memory_status()
    preflight = {
        "thermal_sensor": sensor,
        "temperature_c": temperature_c,
        "thermal_limit_c": THERMAL_LIMIT_C,
        "battery_percent": battery,
        "minimum_start_battery_percent": MIN_START_BATTERY_PERCENT,
        "available_memory_bytes": memory["available_bytes"],
    }
    _set_state(phase="preflight", preflight=preflight)
    if temperature_c >= THERMAL_LIMIT_C:
        raise RuntimeError(
            f"thermal gate closed: {sensor} is {temperature_c:.1f} C"
        )
    if REQUIRE_BATTERY_STATUS and battery is None:
        raise RuntimeError("battery telemetry is unavailable")
    if battery is not None and battery < MIN_START_BATTERY_PERCENT:
        raise RuntimeError(
            f"battery gate closed: {battery:.0f}% is below "
            f"{MIN_START_BATTERY_PERCENT:.0f}%"
        )
    if memory["available_bytes"] < MIN_AVAILABLE_BYTES:
        raise RuntimeError("available memory is below the start floor")


def _weight(
    name: str,
    shape: tuple[int, ...],
    *,
    device: Any,
    dtype: Any,
    rng: np.random.Generator,
    registry: dict[str, np.ndarray],
    zero: bool = False,
) -> Any:
    from max.graph import Weight

    values = (
        np.zeros(shape, dtype=np.float16)
        if zero
        else rng.normal(0.0, 0.02, size=shape).astype(np.float16)
    )
    registry[name] = np.ascontiguousarray(values)
    return Weight(name, dtype, shape, device=device)


def _build_graph(resolution: int) -> tuple[Any, dict[str, np.ndarray], dict[str, Any]]:
    from max.dtype import DType
    from max.graph import DeviceRef, Graph, TensorType, ops

    device = DeviceRef.GPU()
    dtype = DType.float16
    registry: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(20260806)
    layer_count = 0
    parameter_count = 0
    macs = 0

    input_type = TensorType(
        dtype, [1, resolution, resolution, 3], device=device
    )
    with Graph(
        f"fruit_detector_native_{resolution}", input_types=[input_type]
    ) as graph:
        x = graph.inputs[0]
        spatial = resolution

        def conv(
            value: Any,
            input_channels: int,
            output_channels: int,
            kernel: int,
            stride: int,
            groups: int,
            name: str,
            activate: bool,
            input_spatial: int,
        ) -> tuple[Any, int]:
            nonlocal layer_count, parameter_count, macs
            padding = kernel // 2
            output_spatial = (input_spatial + 2 * padding - kernel) // stride + 1
            filter_shape = (
                kernel,
                kernel,
                input_channels // groups,
                output_channels,
            )
            weight = _weight(
                f"{name}.weight",
                filter_shape,
                device=device,
                dtype=dtype,
                rng=rng,
                registry=registry,
            )
            bias = _weight(
                f"{name}.bias",
                (output_channels,),
                device=device,
                dtype=dtype,
                rng=rng,
                registry=registry,
                zero=True,
            )
            result = ops.conv2d(
                value,
                weight,
                stride=(stride, stride),
                padding=(padding, padding, padding, padding),
                groups=groups,
                bias=bias,
            )
            if activate:
                result = ops.relu(result)
            layer_count += 1
            parameters = int(np.prod(filter_shape)) + output_channels
            parameter_count += parameters
            macs += (
                output_spatial
                * output_spatial
                * output_channels
                * kernel
                * kernel
                * (input_channels // groups)
            )
            return result, output_spatial

        x, spatial = conv(x, 3, 16, 3, 2, 1, "stem", True, spatial)
        total_stride = 2
        features: dict[int, tuple[Any, int, int]] = {}
        for index, (input_channels, output_channels, expanded, stride) in enumerate(
            BLOCKS
        ):
            residual = x
            residual_spatial = spatial
            if expanded != input_channels:
                x, spatial = conv(
                    x,
                    input_channels,
                    expanded,
                    1,
                    1,
                    1,
                    f"blocks.{index}.expand",
                    True,
                    spatial,
                )
            x, spatial = conv(
                x,
                expanded,
                expanded,
                3,
                stride,
                expanded,
                f"blocks.{index}.depthwise",
                True,
                spatial,
            )
            x, spatial = conv(
                x,
                expanded,
                output_channels,
                1,
                1,
                1,
                f"blocks.{index}.project",
                False,
                spatial,
            )
            if stride == 1 and input_channels == output_channels:
                x = ops.add(x, residual)
                if spatial != residual_spatial:
                    raise RuntimeError("residual spatial shape changed")
            else:
                x = ops.relu(x)
            total_stride *= stride
            if total_stride in {8, 16, 32}:
                features[total_stride] = (x, output_channels, spatial)

        outputs = []
        output_shapes = []
        for stride in (8, 16, 32):
            feature, channels, feature_spatial = features[stride]
            head, _ = conv(
                feature,
                channels,
                7,
                1,
                1,
                1,
                f"heads.stride_{stride}",
                False,
                feature_spatial,
            )
            outputs.append(head)
            output_shapes.append([1, feature_spatial, feature_spatial, 7])
        graph.output(*outputs)

    return graph, registry, {
        "layer_count": layer_count,
        "parameter_count": parameter_count,
        "estimated_macs": macs,
        "output_shapes": output_shapes,
    }


def _worker_paths(resolution: int) -> tuple[Path, Path, Path, Path]:
    stem = f"native-skeleton-v{ARCHITECTURE_VERSION}-{resolution}"
    return (
        ARTIFACT_DIR / f"{stem}.mef",
        ARTIFACT_DIR / f"{stem}.weights.npz",
        ARTIFACT_DIR / f"{stem}.manifest.json",
        ARTIFACT_DIR / f"{stem}.worker-result.json",
    )


def _worker(resolution: int) -> None:
    from max import driver, engine

    _limit_process()
    mef_path, weights_path, manifest_path, result_path = _worker_paths(resolution)
    result_path.unlink(missing_ok=True)
    started = time.perf_counter()
    graph, weights, model_stats = _build_graph(resolution)
    device = driver.Accelerator(0)
    session = engine.InferenceSession(devices=[device])
    compatible = False
    if mef_path.is_file() and weights_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        compatible = (
            manifest.get("architecture_sha256") == ARCHITECTURE_SHA256
            and manifest.get("resolution") == resolution
        )

    compile_seconds = 0.0
    init_started = time.perf_counter()
    if compatible:
        with np.load(weights_path) as archive:
            weights = {
                name: np.ascontiguousarray(archive[name]) for name in archive.files
            }
        model = session.load(mef_path, weights_registry=weights)
    else:
        compile_started = time.perf_counter()
        compiled = session.compile(graph)
        compile_seconds = time.perf_counter() - compile_started
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        compiled.export_mef(mef_path)
        np.savez(weights_path, **weights)
        _write_json(
            manifest_path,
            {
                "architecture_sha256": ARCHITECTURE_SHA256,
                "resolution": resolution,
                "mef_sha256": _sha256(mef_path),
                "weights_sha256": _sha256(weights_path),
            },
        )
        model = session.init(compiled, weights_registry=weights)
    init_seconds = time.perf_counter() - init_started

    input_array = np.zeros(
        (1, resolution, resolution, 3), dtype=np.float16
    )
    for _ in range(WARMUPS):
        outputs = model.execute(driver.Buffer.from_numpy(input_array).to(device))
        for output in outputs:
            output.to_numpy()

    latencies_ms: list[float] = []
    final_outputs: list[np.ndarray] = []
    cpu_started = time.process_time()
    benchmark_started = time.perf_counter()
    for _ in range(SAMPLES):
        inference_started = time.perf_counter()
        outputs = model.execute(driver.Buffer.from_numpy(input_array).to(device))
        final_outputs = [output.to_numpy() for output in outputs]
        latencies_ms.append((time.perf_counter() - inference_started) * 1000.0)
    benchmark_seconds = time.perf_counter() - benchmark_started
    cpu_seconds = time.process_time() - cpu_started

    output_digest = hashlib.sha256()
    for output in final_outputs:
        output_digest.update(np.ascontiguousarray(output).tobytes())
    median_ms = statistics.median(latencies_ms)
    p95_ms = _percentile(latencies_ms, 0.95)
    result = {
        "resolution": resolution,
        "artifact_reused": compatible,
        "compile_seconds": compile_seconds,
        "initialization_seconds": init_seconds,
        "total_seconds": time.perf_counter() - started,
        "warmups": WARMUPS,
        "samples": SAMPLES,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "maximum_ms": max(latencies_ms),
        "fps_from_median": 1000.0 / median_ms,
        "passes_median_gate": median_ms <= MEDIAN_GATE_MS,
        "passes_p95_gate": p95_ms <= P95_GATE_MS,
        "process_cpu_percent_of_one_core": cpu_seconds / benchmark_seconds * 100.0,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "mef_bytes": mef_path.stat().st_size,
        "weights_bytes": weights_path.stat().st_size,
        "mef_sha256": _sha256(mef_path),
        "weights_sha256": _sha256(weights_path),
        "output_shapes": [list(output.shape) for output in final_outputs],
        "output_dtypes": [str(output.dtype) for output in final_outputs],
        "output_digest_sha256": output_digest.hexdigest(),
        **model_stats,
    }
    _write_json(result_path, result)


def _run_worker_process(resolution: int) -> dict[str, Any]:
    global _worker_process
    _set_state(
        phase=f"benchmarking_{resolution}",
        active_resolution=resolution,
        active_elapsed_seconds=0.0,
    )
    _worker_paths(resolution)[3].unlink(missing_ok=True)
    started = time.perf_counter()
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(resolution)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    with _worker_lock:
        _worker_process = process
    while process.poll() is None:
        elapsed = time.perf_counter() - started
        _set_state(active_elapsed_seconds=elapsed)
        if elapsed > RESOLUTION_TIMEOUT_S:
            process.terminate()
            process.wait(timeout=10)
            raise RuntimeError(
                f"{resolution}px worker exceeded {RESOLUTION_TIMEOUT_S:.0f}s"
            )
        time.sleep(1.0)
    stdout, stderr = process.communicate()
    with _worker_lock:
        _worker_process = None
    if process.returncode != 0:
        raise RuntimeError(
            f"{resolution}px worker exited {process.returncode}: "
            f"{(stderr or stdout)[-2000:]}"
        )
    result_path = _worker_paths(resolution)[3]
    if not result_path.is_file():
        raise RuntimeError(f"{resolution}px worker did not write a result")
    return json.loads(result_path.read_text(encoding="utf-8"))


async def _abort(reason: str) -> None:
    with _worker_lock:
        process = _worker_process
    if process is not None and process.poll() is None:
        process.terminate()
    _set_state(ready=False, phase="guard_abort", error=reason)
    _write_json(ARTIFACT_DIR / "native-skeleton.guard-abort.json", _snapshot())
    print(f"MAX native skeleton aborted: {reason}", file=sys.stderr, flush=True)
    await asyncio.sleep(0.25)
    os._exit(86)


async def _watchdog() -> None:
    missing_battery_samples = 0
    while True:
        await asyncio.sleep(2.0)
        try:
            status = _read_thermal_status()
            sensor, temperature_c, battery = _guard_values(status)
            memory = _memory_status()
        except Exception as exc:
            await _abort(f"safety telemetry failed: {exc}")
            return
        if battery is None:
            missing_battery_samples += 1
        else:
            missing_battery_samples = 0
        _set_state(
            guard={
                "thermal_sensor": sensor,
                "temperature_c": temperature_c,
                "thermal_limit_c": THERMAL_LIMIT_C,
                "battery_percent": battery,
                "battery_abort_percent": BATTERY_ABORT_PERCENT,
                "memory": memory,
            }
        )
        if temperature_c >= THERMAL_LIMIT_C:
            await _abort(f"{sensor} reached {temperature_c:.1f} C")
        if battery is not None and battery <= BATTERY_ABORT_PERCENT:
            await _abort(f"battery reached {battery:.0f}%")
        if REQUIRE_BATTERY_STATUS and missing_battery_samples >= 3:
            await _abort("battery telemetry was unavailable for three samples")
        if memory["available_bytes"] < MIN_AVAILABLE_BYTES:
            await _abort(
                "available memory fell below "
                f"{MIN_AVAILABLE_BYTES / 1024**3:.1f} GiB"
            )
        if memory["swap_used_bytes"] > MAX_SWAP_USED_BYTES:
            await _abort(
                f"swap use exceeded {MAX_SWAP_USED_BYTES / 1024**3:.1f} GiB"
            )


async def _orchestrate() -> None:
    try:
        _limit_process()
        _preflight()
        results: list[dict[str, Any]] = []
        for resolution in RESOLUTIONS:
            result = await asyncio.to_thread(_run_worker_process, resolution)
            results.append(result)
            _set_state(results=results, completed_resolutions=len(results))
            await asyncio.sleep(2.0)
        verdict = all(
            result["passes_median_gate"] and result["passes_p95_gate"]
            for result in results
        )
        _set_state(
            ready=True,
            phase="ready",
            active_resolution=None,
            error=None,
            results=results,
            verdict={
                "train_candidate": verdict,
                "median_gate_ms": MEDIAN_GATE_MS,
                "p95_gate_ms": P95_GATE_MS,
            },
        )
        _write_json(ARTIFACT_DIR / "native-skeleton.result.json", _snapshot())
    except Exception as exc:
        _set_state(ready=False, phase="failed", error=str(exc))
        _write_json(ARTIFACT_DIR / "native-skeleton.failure.json", _snapshot())
        print(f"MAX native skeleton failed: {exc}", file=sys.stderr, flush=True)


@asynccontextmanager
async def lifespan(_application: FastAPI):
    worker = asyncio.create_task(_orchestrate())
    watchdog = asyncio.create_task(_watchdog())
    yield
    worker.cancel()
    watchdog.cancel()


app = FastAPI(title="MAX-native detector skeleton prototype", lifespan=lifespan)


@app.get("/status")
async def status() -> dict[str, Any]:
    return _snapshot()


@app.get("/healthz")
async def healthz() -> dict[str, bool]:
    return {"ok": True}


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker(int(sys.argv[2]))
    else:
        uvicorn.run(app, host="0.0.0.0", port=PORT)
