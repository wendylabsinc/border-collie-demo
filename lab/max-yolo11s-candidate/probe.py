"""Compile and benchmark the exact fruit graph with MAX on Woof."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import resource
import statistics
import sys
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from max_onnx import (
    build_max_graph,
    native_max_input_shape,
    supported_operator_types,
    unsupported_operator_types,
)
from snapshot_inference import decode_detections, prepare_jpeg

INPUT_SIZE = int(os.environ.get("MAX_PROBE_INPUT_SIZE", "640"))
if INPUT_SIZE <= 0:
    raise ValueError("MAX_PROBE_INPUT_SIZE must be positive")
INPUT_SHAPE = (1, 3, INPUT_SIZE, INPUT_SIZE)
SNAPSHOT_URL = os.environ.get(
    "MAX_PROBE_SNAPSHOT_URL", "http://127.0.0.1:8122/snapshot.jpg"
)
PRECISION = os.environ.get("MAX_PROBE_PRECISION", "fp16").strip().lower()
LAYOUT = os.environ.get("MAX_PROBE_LAYOUT", "native_nhwc").strip().lower()
NUMPY_INPUT_DTYPES = {
    "fp16": np.float16,
    "fp32": np.float32,
}
if PRECISION not in NUMPY_INPUT_DTYPES:
    raise ValueError("MAX_PROBE_PRECISION must be fp16 or fp32")
MODEL_PATH = Path(
    os.environ.get("MAX_PROBE_MODEL_PATH", "/models/fruit-detection.onnx")
)
ARTIFACT_DIR = Path(os.environ.get("MAX_PROBE_ARTIFACT_DIR", "/artifacts"))
ARTIFACT_STEM = os.environ.get(
    "MAX_PROBE_ARTIFACT_STEM", f"fruit-detection-{PRECISION}-{LAYOUT}"
)
if not ARTIFACT_STEM or any(
    character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    for character in ARTIFACT_STEM
):
    raise ValueError(
        "MAX_PROBE_ARTIFACT_STEM must contain only letters, digits, - or _"
    )
MEF_PATH = ARTIFACT_DIR / f"{ARTIFACT_STEM}.cuda-sm87.mef"
WEIGHTS_PATH = ARTIFACT_DIR / f"{ARTIFACT_STEM}.cuda-sm87.weights.npz"
ARTIFACT_MANIFEST_PATH = ARTIFACT_DIR / f"{ARTIFACT_STEM}.artifact-manifest.json"
RESULT_PATH = ARTIFACT_DIR / f"{ARTIFACT_STEM}.result.json"
FAILURE_PATH = ARTIFACT_DIR / f"{ARTIFACT_STEM}.failure.json"
GUARD_ABORT_PATH = ARTIFACT_DIR / f"{ARTIFACT_STEM}.guard-abort.json"
GRAPH_COMPATIBILITY_VERSION = 6
THERMAL_STATUS_URL = os.environ.get(
    "MAX_PROBE_THERMAL_STATUS_URL", "http://127.0.0.1:8102/api/status"
)
THERMAL_LIMIT_C = float(os.environ.get("MAX_PROBE_THERMAL_LIMIT_C", "82"))
MEMORY_LIMIT_BYTES = int(
    float(os.environ.get("MAX_PROBE_MEMORY_LIMIT_GIB", "9")) * 1024**3
)
MIN_RUNTIME_AVAILABLE_BYTES = int(
    float(os.environ.get("MAX_PROBE_MIN_RUNTIME_AVAILABLE_GIB", "0.5")) * 1024**3
)
MAX_RUNTIME_SWAP_USED_BYTES = int(
    float(os.environ.get("MAX_PROBE_MAX_RUNTIME_SWAP_USED_GIB", "2")) * 1024**3
)
COMPILE_TIMEOUT_S = float(os.environ.get("MAX_PROBE_COMPILE_TIMEOUT_S", "480"))
RUNTIME_INIT_TIMEOUT_S = float(
    os.environ.get("MAX_PROBE_RUNTIME_INIT_TIMEOUT_S", "300")
)
START_DELAY_S = float(os.environ.get("MAX_PROBE_START_DELAY_S", "3"))
MIN_START_BATTERY_PERCENT = float(
    os.environ.get("MAX_PROBE_MIN_START_BATTERY_PERCENT", "70")
)
BATTERY_ABORT_PERCENT = float(os.environ.get("MAX_PROBE_BATTERY_ABORT_PERCENT", "25"))
REQUIRE_BATTERY_STATUS = os.environ.get(
    "MAX_PROBE_REQUIRE_BATTERY_STATUS", "1"
) not in {"0", "false", "False"}
CPU_AFFINITY_COUNT = int(os.environ.get("MAX_PROBE_CPU_AFFINITY_COUNT", "8"))
OOM_SCORE_ADJ = int(os.environ.get("MAX_PROBE_OOM_SCORE_ADJ", "500"))
if not -1000 <= OOM_SCORE_ADJ <= 1000:
    raise ValueError("MAX_PROBE_OOM_SCORE_ADJ must be between -1000 and 1000")

_state_lock = Lock()
_inference_lock = Lock()
_loaded_model: Any | None = None
_loaded_device: Any | None = None
_state: dict[str, Any] = {
    "ready": False,
    "backend": "max",
    "precision": PRECISION,
    "layout": LAYOUT,
    "phase": "starting",
    "error": None,
}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _limit_process_cpu_affinity() -> list[int]:
    """Restrict MAX's inherited CPU mask before it creates worker threads."""

    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return []
    allowed = sorted(os.sched_getaffinity(0))
    selected = allowed[: min(CPU_AFFINITY_COUNT, len(allowed))]
    if selected:
        os.sched_setaffinity(0, selected)
    return selected


def _set_oom_score_adj() -> int:
    """Make this disposable probe a better OOM victim than host services."""

    path = Path("/proc/self/oom_score_adj")
    path.write_text(f"{OOM_SCORE_ADJ}\n", encoding="utf-8")
    applied = int(path.read_text(encoding="utf-8").strip())
    if applied != OOM_SCORE_ADJ:
        raise RuntimeError(
            f"failed to apply OOM score adjustment {OOM_SCORE_ADJ}; got {applied}"
        )
    return applied


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_state(**updates: Any) -> None:
    with _state_lock:
        _state.update(updates)


def _write_json(name: str, payload: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _infer_jpeg(jpeg: bytes, confidence_floor: float) -> dict[str, Any]:
    tensor, geometry = prepare_jpeg(
        jpeg, input_size=INPUT_SIZE, dtype=NUMPY_INPUT_DTYPES[PRECISION]
    )
    with _inference_lock:
        if _loaded_model is None or _loaded_device is None:
            raise RuntimeError("MAX model is not loaded")
        from max import driver

        started = time.perf_counter()
        outputs = _loaded_model.execute(driver.Buffer.from_numpy(tensor).to(_loaded_device))
        arrays = [output.to_numpy() for output in outputs]
        latency_ms = (time.perf_counter() - started) * 1000.0
    if len(arrays) != 1:
        raise RuntimeError(f"expected one YOLO output, received {len(arrays)}")
    return {
        "backend": "max",
        "precision": PRECISION,
        "input_size": INPUT_SIZE,
        "latency_ms": latency_ms,
        "source": {
            "width": geometry["source_width"],
            "height": geometry["source_height"],
        },
        "detections": decode_detections(
            arrays[0], geometry, confidence_floor=confidence_floor
        ),
    }


def _cached_artifact_is_compatible(model_sha256: str) -> bool:
    if not (
        MEF_PATH.is_file()
        and WEIGHTS_PATH.is_file()
        and ARTIFACT_MANIFEST_PATH.is_file()
    ):
        return False
    try:
        manifest = json.loads(ARTIFACT_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        manifest.get("model_sha256") == model_sha256
        and manifest.get("precision") == PRECISION
        and manifest.get("layout") == LAYOUT
        and manifest.get("graph_compatibility_version") == GRAPH_COMPATIBILITY_VERSION
    )


def _read_thermal_status() -> dict[str, Any]:
    try:
        with urllib.request.urlopen(THERMAL_STATUS_URL, timeout=1.0) as response:
            return json.load(response)
    except Exception:  # fall back to host sensors
        temperatures: dict[str, float] = {}
        for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
            try:
                name = (zone / "type").read_text(encoding="utf-8").strip()
                temperature_c = (
                    float((zone / "temp").read_text(encoding="utf-8").strip()) / 1000.0
                )
                temperatures[name or zone.name] = temperature_c
            except (OSError, ValueError):
                continue
        if not temperatures:
            raise
        hottest_name, hottest_c = max(temperatures.items(), key=lambda item: item[1])
        return {
            "captured_at": None,
            "hottest_jetson_zone": hottest_name,
            "hottest_jetson_c": hottest_c,
            "jetson_c": temperatures,
            "go2": {},
            "source": "sysfs",
        }


def _hottest_guarded_temperature(status: dict[str, Any]) -> tuple[str, float]:
    readings = {"jetson": float(status.get("hottest_jetson_c", 0.0))}
    go2 = status.get("go2") or {}
    if go2.get("imu_c") is not None:
        readings["go2.imu"] = float(go2["imu_c"])
    return max(readings.items(), key=lambda item: item[1])


def _battery_percent(status: dict[str, Any]) -> float | None:
    go2 = status.get("go2") or {}
    value = go2.get("battery_soc_percent")
    return None if value is None else float(value)


def _preflight() -> None:
    thermal = _read_thermal_status()
    sensor, temperature_c = _hottest_guarded_temperature(thermal)
    battery_percent = _battery_percent(thermal)
    _set_state(
        phase="preflight",
        preflight={
            "thermal_sensor": sensor,
            "temperature_c": temperature_c,
            "thermal_limit_c": THERMAL_LIMIT_C,
            "battery_percent": battery_percent,
            "minimum_start_battery_percent": MIN_START_BATTERY_PERCENT,
            "battery_status_required": REQUIRE_BATTERY_STATUS,
        },
    )
    if temperature_c >= THERMAL_LIMIT_C:
        raise RuntimeError(
            f"preflight thermal gate closed: {sensor} is {temperature_c:.1f} C "
            f"(limit {THERMAL_LIMIT_C:.1f} C)"
        )
    if REQUIRE_BATTERY_STATUS and battery_percent is None:
        raise RuntimeError("preflight battery status is unavailable")
    if battery_percent is not None and battery_percent < MIN_START_BATTERY_PERCENT:
        raise RuntimeError(
            f"preflight battery gate closed: {battery_percent:.0f}% "
            f"(minimum {MIN_START_BATTERY_PERCENT:.0f}%)"
        )


def _resident_memory_bytes() -> int:
    with Path("/proc/self/status").open() as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS is missing from /proc/self/status")


def _system_memory_status() -> dict[str, int]:
    """Return the small host-memory subset needed to explain a guarded stop."""

    wanted = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open() as meminfo:
        for line in meminfo:
            name, raw_value = line.split(":", 1)
            if name in wanted:
                values[name] = int(raw_value.split()[0]) * 1024
    missing = wanted - values.keys()
    if missing:
        raise RuntimeError(
            "missing /proc/meminfo fields: " + ", ".join(sorted(missing))
        )
    return {
        "total_bytes": values["MemTotal"],
        "available_bytes": values["MemAvailable"],
        "swap_total_bytes": values["SwapTotal"],
        "swap_used_bytes": values["SwapTotal"] - values["SwapFree"],
    }


def _cpu_status() -> dict[str, Any]:
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    )
    load_parts = Path("/proc/loadavg").read_text(encoding="utf-8").split()
    frequencies: list[int] = []
    for path in Path("/sys/devices/system/cpu").glob(
        "cpu[0-9]*/cpufreq/scaling_cur_freq"
    ):
        try:
            frequencies.append(int(path.read_text(encoding="utf-8").strip()))
        except (OSError, ValueError):
            continue
    return {
        "host_logical_cpus": os.cpu_count(),
        "process_affinity": affinity,
        "load_1m": float(load_parts[0]),
        "load_5m": float(load_parts[1]),
        "load_15m": float(load_parts[2]),
        "minimum_frequency_khz": min(frequencies) if frequencies else None,
        "maximum_frequency_khz": max(frequencies) if frequencies else None,
    }


def _update_resource_peaks(
    *, rss_bytes: int, system_memory: dict[str, int], cpu: dict[str, Any]
) -> None:
    with _state_lock:
        previous = _state.get("resource_peaks") or {}
        _state["resource_peaks"] = {
            "maximum_rss_bytes": max(previous.get("maximum_rss_bytes", 0), rss_bytes),
            "minimum_available_bytes": min(
                previous.get(
                    "minimum_available_bytes", system_memory["available_bytes"]
                ),
                system_memory["available_bytes"],
            ),
            "maximum_swap_used_bytes": max(
                previous.get("maximum_swap_used_bytes", 0),
                system_memory["swap_used_bytes"],
            ),
            "maximum_load_1m": max(
                previous.get("maximum_load_1m", 0.0), cpu["load_1m"]
            ),
        }


async def _abort_probe(message: str) -> None:
    failure = {"ready": False, "phase": "guard_abort", "error": message}
    _set_state(**failure)
    with _state_lock:
        snapshot = json.loads(json.dumps(_state))
    _write_json(GUARD_ABORT_PATH.name, snapshot)
    print(message, file=sys.stderr, flush=True)
    await asyncio.sleep(0.25)
    os._exit(86)


def compile_and_benchmark() -> None:
    global _loaded_device, _loaded_model
    from max import driver, engine

    started = time.perf_counter()
    total_cpu_started = time.process_time()
    device = driver.Accelerator(0)
    session = engine.InferenceSession(devices=[device])
    model_sha256 = _sha256(MODEL_PATH)
    artifact_reused = _cached_artifact_is_compatible(model_sha256)

    if artifact_reused:
        _set_state(
            phase="loading_cached_artifact",
            artifact_reused=True,
            runtime_init_started_at=time.time(),
        )
        with np.load(WEIGHTS_PATH) as archive:
            weights = {
                name: np.ascontiguousarray(archive[name]) for name in archive.files
            }
        # MAX weight placeholders are host-side registry inputs even when the
        # graph and inference tensors are GPU-resident. MAX stages them onto
        # the target device while initializing the executable model.
        model = session.load(MEF_PATH, weights_registry=weights)
        compile_seconds = 0.0
        _set_state(phase="artifact_loaded")
    else:
        _set_state(phase="building_graph", artifact_reused=False)
        unsupported = unsupported_operator_types(MODEL_PATH)
        if unsupported:
            raise RuntimeError(
                "unsupported ONNX operators: " + ", ".join(sorted(unsupported))
            )
        graph, weights = build_max_graph(
            MODEL_PATH,
            INPUT_SHAPE,
            precision=PRECISION,
            layout=LAYOUT,
        )

        _set_state(
            phase="compiling_graph",
            compile_started_at=time.time(),
            operator_types=sorted(supported_operator_types(MODEL_PATH)),
        )
        compile_started = time.perf_counter()
        compiled = session.compile(graph)
        compile_seconds = time.perf_counter() - compile_started

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        compiled.export_mef(MEF_PATH)
        np.savez(WEIGHTS_PATH, **weights)
        _write_json(
            ARTIFACT_MANIFEST_PATH.name,
            {
                "model_sha256": model_sha256,
                "precision": PRECISION,
                "layout": LAYOUT,
                "graph_compatibility_version": GRAPH_COMPATIBILITY_VERSION,
                "mef_sha256": _sha256(MEF_PATH),
                "weights_sha256": _sha256(WEIGHTS_PATH),
                "compile_seconds": compile_seconds,
            },
        )

        _set_state(
            phase="initializing_model",
            compile_seconds=compile_seconds,
            runtime_init_started_at=time.time(),
        )
        model = session.init(compiled, weights_registry=weights)
        _set_state(phase="artifact_loaded")
    input_array = np.zeros(
        native_max_input_shape(INPUT_SHAPE),
        dtype=NUMPY_INPUT_DTYPES[PRECISION],
    )

    _set_state(phase="warming_model", warmup_completed=0)
    for warmup_index in range(5):
        model.execute(driver.Buffer.from_numpy(input_array).to(device))
        _set_state(warmup_completed=warmup_index + 1)

    latencies_ms: list[float] = []
    output_arrays: list[np.ndarray] = []
    _set_state(phase="benchmarking_model")
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    for _ in range(30):
        inference_started = time.perf_counter()
        outputs = model.execute(driver.Buffer.from_numpy(input_array).to(device))
        output_arrays = [output.to_numpy() for output in outputs]
        latencies_ms.append((time.perf_counter() - inference_started) * 1000.0)
    wall_seconds = time.perf_counter() - wall_started
    cpu_seconds = time.process_time() - cpu_started

    output_digest = hashlib.sha256()
    for output in output_arrays:
        output_digest.update(np.ascontiguousarray(output).tobytes())

    with _inference_lock:
        _loaded_device = device
        _loaded_model = model

    _set_state(
        ready=True,
        phase="ready",
        error=None,
        model_sha256=model_sha256,
        precision=PRECISION,
        layout=LAYOUT,
        logical_input_shape=list(INPUT_SHAPE),
        physical_input_shape=list(input_array.shape),
        mef_sha256=_sha256(MEF_PATH),
        weights_sha256=_sha256(WEIGHTS_PATH),
        compile_seconds=compile_seconds,
        artifact_reused=artifact_reused,
        total_startup_seconds=time.perf_counter() - started,
        total_cpu_seconds=time.process_time() - total_cpu_started,
        warm_inference={
            "samples": len(latencies_ms),
            "median_ms": statistics.median(latencies_ms),
            "p95_ms": _percentile(latencies_ms, 0.95),
            "maximum_ms": max(latencies_ms),
            "fps_from_median": 1000.0 / statistics.median(latencies_ms),
        },
        process_cpu_percent_of_one_core=(cpu_seconds / wall_seconds) * 100.0,
        process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        output_shapes=[list(output.shape) for output in output_arrays],
        output_dtypes=[str(output.dtype) for output in output_arrays],
        output_digest_sha256=output_digest.hexdigest(),
        artifact_sizes={
            "mef_bytes": MEF_PATH.stat().st_size,
            "weights_bytes": WEIGHTS_PATH.stat().st_size,
        },
    )
    with _state_lock:
        result = json.loads(json.dumps(_state))
    _write_json(RESULT_PATH.name, result)


async def _run_compile() -> None:
    await asyncio.sleep(START_DELAY_S)
    try:
        await asyncio.to_thread(_preflight)
        await asyncio.to_thread(compile_and_benchmark)
    except Exception as exc:  # noqa: BLE001 - diagnostics must preserve failure
        failure = {
            "ready": False,
            "phase": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _set_state(**failure)
        with _state_lock:
            snapshot = json.loads(json.dumps(_state))
        _write_json(FAILURE_PATH.name, snapshot)


async def _thermal_watchdog() -> None:
    missing_battery_samples = 0
    while True:
        try:
            thermal = await asyncio.to_thread(_read_thermal_status)
            sensor, temperature_c = _hottest_guarded_temperature(thermal)
            _set_state(
                thermal_guard={
                    "limit_c": THERMAL_LIMIT_C,
                    "hottest_sensor": sensor,
                    "hottest_c": temperature_c,
                    "captured_at": thermal.get("captured_at"),
                    "battery_percent": _battery_percent(thermal),
                    "battery_abort_percent": BATTERY_ABORT_PERCENT,
                }
            )
            battery_percent = _battery_percent(thermal)
            if battery_percent is None:
                missing_battery_samples += 1
            else:
                missing_battery_samples = 0
            rss_bytes = _resident_memory_bytes()
            system_memory = _system_memory_status()
            cpu = _cpu_status()
            with _state_lock:
                compile_started_at = _state.get("compile_started_at")
                runtime_init_started_at = _state.get("runtime_init_started_at")
                phase = _state.get("phase")
            compile_elapsed_s = (
                time.time() - compile_started_at if compile_started_at else 0.0
            )
            _set_state(
                resource_guard={
                    "rss_bytes": rss_bytes,
                    "memory_limit_bytes": MEMORY_LIMIT_BYTES,
                    "minimum_runtime_available_bytes": MIN_RUNTIME_AVAILABLE_BYTES,
                    "maximum_runtime_swap_used_bytes": MAX_RUNTIME_SWAP_USED_BYTES,
                    "compile_elapsed_s": compile_elapsed_s,
                    "compile_timeout_s": COMPILE_TIMEOUT_S,
                    "runtime_init_elapsed_s": (
                        time.time() - runtime_init_started_at
                        if runtime_init_started_at
                        else 0.0
                    ),
                    "runtime_init_timeout_s": RUNTIME_INIT_TIMEOUT_S,
                    "system_memory": system_memory,
                    "cpu": cpu,
                }
            )
            _update_resource_peaks(
                rss_bytes=rss_bytes, system_memory=system_memory, cpu=cpu
            )
            if temperature_c >= THERMAL_LIMIT_C:
                await _abort_probe(
                    f"MAX probe aborted: {sensor} reached {temperature_c:.1f} C "
                    f"(limit {THERMAL_LIMIT_C:.1f} C)"
                )
            if REQUIRE_BATTERY_STATUS and missing_battery_samples >= 3:
                await _abort_probe(
                    "MAX probe aborted: battery status was unavailable for "
                    f"{missing_battery_samples} consecutive samples"
                )
            if battery_percent is not None and battery_percent <= BATTERY_ABORT_PERCENT:
                await _abort_probe(
                    f"MAX probe aborted: battery reached {battery_percent:.0f}% "
                    f"(floor {BATTERY_ABORT_PERCENT:.0f}%)"
                )
            if rss_bytes >= MEMORY_LIMIT_BYTES:
                await _abort_probe(
                    f"MAX probe aborted: RSS reached {rss_bytes / 1024**3:.2f} GiB "
                    f"(limit {MEMORY_LIMIT_BYTES / 1024**3:.2f} GiB)"
                )
            if phase in {
                "loading_cached_artifact",
                "initializing_model",
                "artifact_loaded",
                "warming_model",
            }:
                available_bytes = system_memory["available_bytes"]
                swap_used_bytes = system_memory["swap_used_bytes"]
                if available_bytes <= MIN_RUNTIME_AVAILABLE_BYTES:
                    await _abort_probe(
                        "MAX probe aborted: runtime initialization left "
                        f"{available_bytes / 1024**3:.2f} GiB available "
                        f"(floor {MIN_RUNTIME_AVAILABLE_BYTES / 1024**3:.2f} GiB)"
                    )
                if swap_used_bytes >= MAX_RUNTIME_SWAP_USED_BYTES:
                    await _abort_probe(
                        "MAX probe aborted: runtime initialization used "
                        f"{swap_used_bytes / 1024**3:.2f} GiB swap "
                        f"(limit {MAX_RUNTIME_SWAP_USED_BYTES / 1024**3:.2f} GiB)"
                    )
                if (
                    runtime_init_started_at
                    and time.time() - runtime_init_started_at
                    >= RUNTIME_INIT_TIMEOUT_S
                ):
                    await _abort_probe(
                        "MAX probe aborted: runtime initialization reached "
                        f"{time.time() - runtime_init_started_at:.1f}s "
                        f"(limit {RUNTIME_INIT_TIMEOUT_S:.1f}s, phase {phase})"
                    )
            if phase == "compiling_graph" and compile_elapsed_s >= COMPILE_TIMEOUT_S:
                await _abort_probe(
                    f"MAX probe aborted: compilation reached {compile_elapsed_s:.1f}s "
                    f"(limit {COMPILE_TIMEOUT_S:.1f}s)"
                )
        except Exception as exc:  # noqa: BLE001 - watchdog reports but remains active
            _set_state(thermal_guard_error=f"{type(exc).__name__}: {exc}")
        await asyncio.sleep(2.0)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    compile_task = asyncio.create_task(_run_compile())
    watchdog_task = asyncio.create_task(_thermal_watchdog())
    try:
        yield
    finally:
        compile_task.cancel()
        watchdog_task.cancel()


app = FastAPI(title="Border Collie MAX Runtime Probe", lifespan=lifespan)


@app.get("/status")
async def status() -> dict[str, Any]:
    with _state_lock:
        return json.loads(json.dumps(_state))


@app.get("/artifacts/{name}")
async def artifact(name: str) -> FileResponse:
    paths = {
        MEF_PATH.name: MEF_PATH,
        WEIGHTS_PATH.name: WEIGHTS_PATH,
        RESULT_PATH.name: RESULT_PATH,
        FAILURE_PATH.name: FAILURE_PATH,
        GUARD_ABORT_PATH.name: GUARD_ABORT_PATH,
        ARTIFACT_MANIFEST_PATH.name: ARTIFACT_MANIFEST_PATH,
    }
    path = paths.get(name)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="artifact is not ready")
    return FileResponse(path)


@app.post("/infer-snapshot")
async def infer_snapshot(confidence: float = 0.25) -> dict[str, Any]:
    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(status_code=400, detail="confidence must be in [0, 1]")
    with _state_lock:
        ready = bool(_state.get("ready"))
    if not ready:
        raise HTTPException(status_code=503, detail="MAX model is not ready")
    try:
        jpeg = await asyncio.to_thread(
            lambda: urllib.request.urlopen(SNAPSHOT_URL, timeout=5.0).read()
        )
        result = await asyncio.to_thread(_infer_jpeg, jpeg, confidence)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"snapshot inference failed: {type(exc).__name__}: {exc}",
        ) from exc
    result["snapshot_url"] = SNAPSHOT_URL
    return result


if __name__ == "__main__":
    _set_state(
        cpu_affinity=_limit_process_cpu_affinity(),
        oom_score_adj=_set_oom_score_adj(),
    )
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("MAX_PROBE_PORT", "8123")))
