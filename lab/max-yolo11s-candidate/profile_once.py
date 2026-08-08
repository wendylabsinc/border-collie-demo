"""Capture exactly one warm MAX inference under an external GPU profiler."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import resource
import time
import urllib.request
from pathlib import Path

import numpy as np


ARTIFACT_DIR = Path(os.environ.get("MAX_PROBE_ARTIFACT_DIR", "/artifacts"))
ARTIFACT_STEM = os.environ.get(
    "MAX_PROBE_ARTIFACT_STEM", "fruit-yolo11n-whole-v1-416-fp16-native-nhwc"
)
MEF_PATH = ARTIFACT_DIR / f"{ARTIFACT_STEM}.cuda-sm87.mef"
WEIGHTS_PATH = ARTIFACT_DIR / f"{ARTIFACT_STEM}.cuda-sm87.weights.npz"
RESULT_PATH = ARTIFACT_DIR / "max-yolo11n-one-warm-profile.json"
THERMAL_STATUS_URL = os.environ.get(
    "MAX_PROBE_THERMAL_STATUS_URL", "http://127.0.0.1:8102/api/status"
)
INPUT_SIZE = int(os.environ.get("MAX_PROBE_INPUT_SIZE", "416"))
WARMUP_COUNT = int(os.environ.get("MAX_PROFILE_WARMUPS", "5"))
MIN_BATTERY_PERCENT = float(os.environ.get("MAX_PROFILE_MIN_BATTERY_PERCENT", "50"))
THERMAL_LIMIT_C = float(os.environ.get("MAX_PROBE_THERMAL_LIMIT_C", "82"))


def _thermal_status() -> dict[str, object]:
    with urllib.request.urlopen(THERMAL_STATUS_URL, timeout=3.0) as response:
        return json.load(response)


def _validate_preflight(status: dict[str, object]) -> None:
    if not status.get("ok"):
        raise RuntimeError("thermal monitor did not report ok")
    hottest = float(status.get("hottest_monitored_c", 999.0))
    if hottest >= THERMAL_LIMIT_C:
        raise RuntimeError(
            f"guarded temperature {hottest:.1f} C is at or above {THERMAL_LIMIT_C:.1f} C"
        )
    go2 = status.get("go2")
    battery = go2.get("battery_soc_percent") if isinstance(go2, dict) else None
    # A zero floor explicitly disables the battery gate. This is useful when the
    # Go2 telemetry has not repopulated after a reboot; thermal protection stays
    # active independently.
    if MIN_BATTERY_PERCENT > 0 and (
        battery is None or float(battery) < MIN_BATTERY_PERCENT
    ):
        raise RuntimeError(
            f"battery {battery!r} is below the {MIN_BATTERY_PERCENT:.0f}% profiling floor"
        )


def _cuda_profiler_api() -> ctypes.CDLL:
    for library_name in ("libcudart.so.12", "libcudart.so"):
        try:
            library = ctypes.CDLL(library_name)
            break
        except OSError:
            continue
    else:
        raise RuntimeError("CUDA runtime library was not found")
    library.cudaProfilerStart.restype = ctypes.c_int
    library.cudaProfilerStop.restype = ctypes.c_int
    return library


def _check_cuda(code: int, operation: str) -> None:
    if code != 0:
        raise RuntimeError(f"{operation} failed with CUDA error code {code}")


def main() -> None:
    if not MEF_PATH.is_file() or not WEIGHTS_PATH.is_file():
        raise RuntimeError(
            "cached MEF/weights are missing; refusing to compile during the profiling run"
        )
    preflight = _thermal_status()
    _validate_preflight(preflight)
    try:
        Path("/proc/self/oom_score_adj").write_text("500\n", encoding="ascii")
    except OSError:
        pass

    from max import driver, engine

    started = time.perf_counter()
    device = driver.Accelerator(0)
    session = engine.InferenceSession(devices=[device])
    session.gpu_profiling("detailed")
    with np.load(WEIGHTS_PATH) as archive:
        weights = {
            name: np.ascontiguousarray(archive[name]) for name in archive.files
        }
    model = session.load(MEF_PATH, weights_registry=weights)
    input_array = np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float16)

    for _ in range(WARMUP_COUNT):
        warm_outputs = model.execute(driver.Buffer.from_numpy(input_array).to(device))
        for output in warm_outputs:
            output.to_numpy()

    cuda = _cuda_profiler_api()
    _check_cuda(cuda.cudaProfilerStart(), "cudaProfilerStart")
    inference_started = time.perf_counter()
    outputs = model.execute(driver.Buffer.from_numpy(input_array).to(device))
    output_arrays = [output.to_numpy() for output in outputs]
    inference_ms = (time.perf_counter() - inference_started) * 1000.0
    _check_cuda(cuda.cudaProfilerStop(), "cudaProfilerStop")

    digest = hashlib.sha256()
    for output in output_arrays:
        digest.update(np.ascontiguousarray(output).tobytes())
    result = {
        "schema_version": 1,
        "profile_scope": "one warm inference",
        "max_profiling": "detailed",
        "warmups_outside_capture": WARMUP_COUNT,
        "profiled_inferences": 1,
        "inference_ms": inference_ms,
        "startup_and_warmup_seconds": time.perf_counter() - started,
        "input_shape": list(input_array.shape),
        "output_shapes": [list(output.shape) for output in output_arrays],
        "output_dtypes": [str(output.dtype) for output in output_arrays],
        "output_digest_sha256": digest.hexdigest(),
        "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "preflight": preflight,
        "postflight": _thermal_status(),
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"MAX_PROFILE_RESULT={json.dumps(result, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
