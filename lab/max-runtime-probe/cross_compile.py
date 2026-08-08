"""Cross-compile the complete fruit model for Woof without a physical GPU."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
from max import driver, engine

MODEL_PATH = Path(os.environ.get("MAX_PROBE_MODEL_PATH", "/models/fruit.onnx"))
ARTIFACT_DIR = Path(os.environ.get("MAX_PROBE_ARTIFACT_DIR", "/artifacts"))
INPUT_SHAPE = (1, 3, 640, 640)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    driver.set_virtual_device_api("cuda")
    driver.set_virtual_device_target_arch("sm_87")
    if hasattr(driver, "set_virtual_cpu_target"):
        driver.set_virtual_cpu_target("generic")
    driver.set_virtual_device_count(1)

    # Import graph code only after selecting virtual compilation targets.
    from max_onnx import build_max_graph, unsupported_operator_types

    started = time.perf_counter()
    print("phase=building_graph target=linux-arm64+cuda-sm_87", flush=True)
    unsupported = unsupported_operator_types(MODEL_PATH)
    if unsupported:
        raise RuntimeError(
            "unsupported ONNX operators: " + ", ".join(sorted(unsupported))
        )
    graph, weights = build_max_graph(MODEL_PATH, INPUT_SHAPE)

    print("phase=compiling_graph", flush=True)
    device = driver.Accelerator(0)
    session = engine.InferenceSession(devices=[device])
    compile_started = time.perf_counter()
    compiled = session.compile(graph)
    compile_seconds = time.perf_counter() - compile_started

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    mef_path = ARTIFACT_DIR / "fruit.linux-arm64.cuda-sm87.mef"
    weights_path = ARTIFACT_DIR / "fruit.weights.npz"
    mef_temp = mef_path.with_suffix(".mef.partial")
    weights_temp = weights_path.with_suffix(".npz.partial")
    print("phase=exporting", flush=True)
    compiled.export_mef(mef_temp)
    with weights_temp.open("wb") as handle:
        np.savez(handle, **weights)
    mef_temp.replace(mef_path)
    weights_temp.replace(weights_path)

    result = {
        "ready": True,
        "target": "linux-arm64+cuda-sm_87",
        "model_sha256": _sha256(MODEL_PATH),
        "mef_sha256": _sha256(mef_path),
        "weights_sha256": _sha256(weights_path),
        "compile_seconds": compile_seconds,
        "total_seconds": time.perf_counter() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "artifact_sizes": {
            "mef_bytes": mef_path.stat().st_size,
            "weights_bytes": weights_path.stat().st_size,
        },
    }
    (ARTIFACT_DIR / "compile-result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
