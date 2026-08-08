"""Prove that pinned MAX emits and executes native sm_87 code on Woof."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import hashlib
import json
import os
import re
import resource
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

ARTIFACT_DIR = Path(os.environ.get("MAX_SM87_ARTIFACT_DIR", "/artifacts"))
PORT = int(os.environ.get("MAX_SM87_PROBE_PORT", "8124"))
THERMAL_LIMIT_C = float(os.environ.get("MAX_SM87_THERMAL_LIMIT_C", "82"))
PTXAS_PATH = Path(
    os.environ.get("MODULAR_NVPTX_COMPILER_PATH", "/usr/local/cuda/bin/ptxas")
)
CUOBJDUMP_PATH = Path(os.environ.get("MAX_SM87_CUOBJDUMP", "/usr/local/cuda/bin/cuobjdump"))
VECTOR_LENGTH = 1024

_lock = Lock()
_state: dict[str, Any] = {
    "ready": False,
    "phase": "starting",
    "requested_device": "accelerator",
    "resolved_device": None,
    "gpu_arch": None,
    "result_correct": False,
    "error": None,
}


def _set_state(**updates: Any) -> None:
    with _lock:
        _state.update(updates)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _cgroup_cpu_limit() -> dict[str, Any]:
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.is_file():
        raw = cpu_max.read_text(encoding="utf-8").strip()
        quota, period = raw.split()
        cores = None if quota == "max" else float(quota) / float(period)
        return {"source": str(cpu_max), "cores": cores, "raw": raw}

    for root in (Path("/sys/fs/cgroup/cpu"), Path("/sys/fs/cgroup/cpu,cpuacct")):
        quota_path = root / "cpu.cfs_quota_us"
        period_path = root / "cpu.cfs_period_us"
        if not quota_path.is_file() or not period_path.is_file():
            continue
        quota = int(quota_path.read_text(encoding="utf-8").strip())
        period = int(period_path.read_text(encoding="utf-8").strip())
        cores = None if quota < 0 else float(quota) / float(period)
        return {
            "source": f"{quota_path},{period_path}",
            "cores": cores,
            "raw": f"{quota} {period}",
        }
    return {"source": None, "cores": None, "raw": None}


def _command_version(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    if completed.returncode != 0:
        raise RuntimeError(f"version command failed ({completed.returncode}): {' '.join(command)}: {output}")
    return output


def _build_graph(device: Any) -> Any:
    from max.dtype import DType
    from max.graph import DeviceRef, Graph, TensorType, ops

    device_ref = DeviceRef.from_device(device)
    vector_type = TensorType(DType.float32, [VECTOR_LENGTH], device=device_ref)
    with Graph("sm87_vector_add", input_types=[vector_type, vector_type]) as graph:
        graph.output(ops.add(graph.inputs[0], graph.inputs[1]))
    return graph


def _compile_target(target_arch: str, output: Path) -> dict[str, Any]:
    from max import driver, engine

    child_started = time.perf_counter()
    child_cpu_started = time.process_time()
    driver.set_virtual_device_api("cuda")
    driver.set_virtual_device_target_arch(target_arch)
    if hasattr(driver, "set_virtual_cpu_target"):
        driver.set_virtual_cpu_target("generic")
    driver.set_virtual_device_count(1)
    device = driver.Accelerator(0)
    graph = _build_graph(device)
    session = engine.InferenceSession(devices=[device])
    started = time.perf_counter()
    compiled = session.compile(graph)
    compile_seconds = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    compiled.export_mef(output)
    return {
        "target_arch": target_arch,
        "artifact": output.name,
        "artifact_sha256": _sha256(output),
        "artifact_bytes": output.stat().st_size,
        "compile_seconds": compile_seconds,
        "child_wall_seconds": time.perf_counter() - child_started,
        "child_cpu_seconds": time.process_time() - child_cpu_started,
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _execute_artifact(path: Path) -> dict[str, Any]:
    from max import driver, engine

    if os.environ.get("CUDA_DISABLE_PTX_JIT") != "1":
        raise RuntimeError("execution proof requires CUDA_DISABLE_PTX_JIT=1")
    if int(driver.accelerator_count()) < 1:
        raise RuntimeError("MAX reports no accelerator")
    device = driver.Accelerator(0)
    session = engine.InferenceSession(devices=[device])
    model = session.load(str(path))
    left = np.arange(VECTOR_LENGTH, dtype=np.float32)
    right = np.full(VECTOR_LENGTH, 2.5, dtype=np.float32)
    expected = left + right
    started = time.perf_counter()
    output = model.execute(
        driver.Buffer.from_numpy(left).to(device),
        driver.Buffer.from_numpy(right).to(device),
    )[0]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    actual = output.to(driver.CPU()).to_numpy()
    correct = bool(np.array_equal(actual, expected))
    result = {
        "requested_device": "accelerator",
        "resolved_device": "gpu",
        "device_description": str(device),
        "cuda_disable_ptx_jit": os.environ["CUDA_DISABLE_PTX_JIT"],
        "result_correct": correct,
        "maximum_absolute_error": float(np.max(np.abs(actual - expected))),
        "execution_ms": elapsed_ms,
    }
    if not correct:
        raise RuntimeError(f"vector-add result mismatch: {result}")
    return result


def _extract_if_archive(path: Path, destination: Path) -> list[Path]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            archive.extractall(destination)
        return [item for item in destination.rglob("*") if item.is_file()]
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            root = destination.resolve()
            for member in members:
                candidate = (destination / member.name).resolve()
                if candidate != root and root not in candidate.parents:
                    raise RuntimeError(
                        "MEF archive member escapes inspection directory: "
                        + member.name
                    )
            archive.extractall(destination, members=members)
        return [item for item in destination.rglob("*") if item.is_file()]
    return []


def _embedded_elf_size(payload: bytes, offset: int) -> int:
    elf = memoryview(payload)[offset:]
    if len(elf) < 52 or elf[:4] != b"\x7fELF":
        raise RuntimeError(f"invalid embedded ELF at offset {offset}")
    byte_order = "<" if elf[5] == 1 else ">"
    elf_class = elf[4]
    if elf_class == 2:
        program_offset = struct.unpack_from(byte_order + "Q", elf, 32)[0]
        section_offset = struct.unpack_from(byte_order + "Q", elf, 40)[0]
        header_size, program_entry_size, program_count, section_entry_size, section_count = (
            struct.unpack_from(byte_order + "HHHHH", elf, 52)
        )
        program_file_offset_at = 8
        program_file_size_at = 32
        section_type_at = 4
        section_file_offset_at = 24
        section_file_size_at = 32
        offset_format = "Q"
    elif elf_class == 1:
        program_offset = struct.unpack_from(byte_order + "I", elf, 28)[0]
        section_offset = struct.unpack_from(byte_order + "I", elf, 32)[0]
        header_size, program_entry_size, program_count, section_entry_size, section_count = (
            struct.unpack_from(byte_order + "HHHHH", elf, 40)
        )
        program_file_offset_at = 4
        program_file_size_at = 16
        section_type_at = 4
        section_file_offset_at = 16
        section_file_size_at = 20
        offset_format = "I"
    else:
        raise RuntimeError(f"unknown embedded ELF class {elf_class}")

    end = max(
        header_size,
        program_offset + program_entry_size * program_count,
        section_offset + section_entry_size * section_count,
    )
    for index in range(program_count):
        entry = program_offset + index * program_entry_size
        file_offset = struct.unpack_from(
            byte_order + offset_format, elf, entry + program_file_offset_at
        )[0]
        file_size = struct.unpack_from(
            byte_order + offset_format, elf, entry + program_file_size_at
        )[0]
        end = max(end, file_offset + file_size)
    for index in range(section_count):
        entry = section_offset + index * section_entry_size
        section_type = struct.unpack_from(
            byte_order + "I", elf, entry + section_type_at
        )[0]
        file_offset = struct.unpack_from(
            byte_order + offset_format, elf, entry + section_file_offset_at
        )[0]
        file_size = struct.unpack_from(
            byte_order + offset_format, elf, entry + section_file_size_at
        )[0]
        if section_type != 8:  # SHT_NOBITS has no bytes in the file.
            end = max(end, file_offset + file_size)
    if end > len(elf):
        raise RuntimeError(
            f"embedded ELF at offset {offset} exceeds MEF by {end - len(elf)} bytes"
        )
    return end


def _extract_embedded_elfs(path: Path, destination: Path) -> list[Path]:
    """Expose ELF payloads from MAX's MEF container for CUDA inspection."""
    payload = path.read_bytes()
    magic = b"\x7fELF"
    offsets: list[int] = []
    cursor = 0
    while True:
        offset = payload.find(magic, cursor)
        if offset < 0:
            break
        offsets.append(offset)
        cursor = offset + len(magic)

    extracted: list[Path] = []
    for index, offset in enumerate(offsets):
        candidate = destination / f"{path.name}.embedded-{index}.elf"
        size = _embedded_elf_size(payload, offset)
        candidate.write_bytes(payload[offset : offset + size])
        extracted.append(candidate)
    return extracted


def _inspect_cuda_images(path: Path) -> dict[str, Any]:
    candidates = [path]
    with tempfile.TemporaryDirectory(prefix="max-sm87-inspect-") as temporary:
        inspection_dir = Path(temporary)
        candidates.extend(_extract_if_archive(path, inspection_dir))
        candidates.extend(_extract_embedded_elfs(path, inspection_dir))
        inspections: list[dict[str, Any]] = []
        combined_output: list[str] = []
        for candidate in candidates:
            completed = subprocess.run(
                [str(CUOBJDUMP_PATH), "--list-elf", str(candidate)],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = "\n".join(
                part.strip()
                for part in (completed.stdout, completed.stderr)
                if part.strip()
            )
            if output:
                inspections.append(
                    {
                        "path": candidate.name,
                        "sha256": _sha256(candidate),
                        "bytes": candidate.stat().st_size,
                        "returncode": completed.returncode,
                        "output": output,
                    }
                )
                combined_output.append(output)
        text = "\n".join(combined_output)
        architectures = sorted(set(re.findall(r"sm_[0-9]+", text)))
        return {
            "tool": str(CUOBJDUMP_PATH),
            "architectures": architectures,
            "contains_sm_87": "sm_87" in text,
            "inspections": inspections,
        }


def _native_cuda_elf(path: Path) -> tuple[bytes, str]:
    with tempfile.TemporaryDirectory(prefix="max-sm87-native-") as temporary:
        candidates = _extract_embedded_elfs(path, Path(temporary))
        for candidate in candidates:
            completed = subprocess.run(
                [str(CUOBJDUMP_PATH), "--list-elf", str(candidate)],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = "\n".join(
                part.strip()
                for part in (completed.stdout, completed.stderr)
                if part.strip()
            )
            if completed.returncode == 0 and ".cubin" in output:
                architectures = sorted(set(re.findall(r"sm_[0-9]+", output)))
                if len(architectures) != 1:
                    raise RuntimeError(
                        f"native CUDA ELF has ambiguous architectures: {output}"
                    )
                return candidate.read_bytes(), architectures[0]
    raise RuntimeError(f"no native CUDA ELF found in {path.name}")


def _cuda_error_name(driver: Any, code: int) -> str:
    name = ctypes.c_char_p()
    if driver.cuGetErrorName(code, ctypes.byref(name)) == 0 and name.value:
        return name.value.decode("ascii", errors="replace")
    return f"CUDA_ERROR_{code}"


def _load_native_cubin(path: Path) -> dict[str, Any]:
    if os.environ.get("CUDA_DISABLE_PTX_JIT") != "1":
        raise RuntimeError("native CUBIN proof requires CUDA_DISABLE_PTX_JIT=1")
    payload, architecture = _native_cuda_elf(path)
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuCtxCreate_v2.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
        ctypes.c_int,
    ]
    cuda.cuCtxCreate_v2.restype = ctypes.c_int
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuCtxDestroy_v2.restype = ctypes.c_int
    cuda.cuModuleLoadData.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    cuda.cuModuleLoadData.restype = ctypes.c_int
    cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
    cuda.cuModuleUnload.restype = ctypes.c_int
    cuda.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    cuda.cuGetErrorName.restype = ctypes.c_int

    initialized = cuda.cuInit(0)
    if initialized != 0:
        raise RuntimeError(_cuda_error_name(cuda, initialized))
    device = ctypes.c_int()
    device_result = cuda.cuDeviceGet(ctypes.byref(device), 0)
    if device_result != 0:
        raise RuntimeError(_cuda_error_name(cuda, device_result))
    context = ctypes.c_void_p()
    context_result = cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device.value)
    if context_result != 0:
        raise RuntimeError(_cuda_error_name(cuda, context_result))

    module = ctypes.c_void_p()
    image = ctypes.create_string_buffer(payload)
    try:
        load_result = cuda.cuModuleLoadData(
            ctypes.byref(module), ctypes.cast(image, ctypes.c_void_p)
        )
        load_error = _cuda_error_name(cuda, load_result)
        result = {
            "gpu_arch": architecture,
            "cuda_disable_ptx_jit": os.environ["CUDA_DISABLE_PTX_JIT"],
            "native_cubin_bytes": len(payload),
            "native_cubin_sha256": hashlib.sha256(payload).hexdigest(),
            "cuda_module_load_code": load_result,
            "cuda_module_load_error": load_error,
            "module_loaded": load_result == 0,
        }
        return result
    finally:
        if module.value:
            cuda.cuModuleUnload(module)
        cuda.cuCtxDestroy_v2(context)


def _run_child(arguments: list[str], *, jit_disabled: bool = False) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["MODULAR_NVPTX_COMPILER_PATH"] = str(PTXAS_PATH)
    if jit_disabled:
        environment["CUDA_DISABLE_PTX_JIT"] = "1"
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *arguments],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )


def _child_json(completed: subprocess.CompletedProcess[str], label: str) -> dict[str, Any]:
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed ({completed.returncode}): stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{label} produced no JSON output")
    return json.loads(lines[-1])


def _hottest_jetson_temperature() -> tuple[str, float]:
    readings: list[tuple[str, float]] = []
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            name = (zone / "type").read_text(encoding="utf-8").strip()
            temperature = float((zone / "temp").read_text(encoding="utf-8").strip()) / 1000.0
        except (OSError, ValueError):
            continue
        readings.append((name or zone.name, temperature))
    if not readings:
        raise RuntimeError("no Jetson thermal zones are readable")
    return max(readings, key=lambda item: item[1])


def _write_json(name: str, payload: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_proof() -> None:
    proof_started = time.perf_counter()
    proof_cpu_started = _cpu_seconds()
    sensor, temperature = _hottest_jetson_temperature()
    if temperature >= THERMAL_LIMIT_C:
        raise RuntimeError(
            f"thermal gate closed: {sensor} is {temperature:.1f} C (limit {THERMAL_LIMIT_C:.1f} C)"
        )
    _set_state(phase="recording_versions", thermal_before_c=temperature, thermal_sensor=sensor)
    versions = {
        "max": _command_version(["max", "--version"]),
        "mojo": _command_version(["mojo", "--version"]),
        "ptxas": _command_version([str(PTXAS_PATH), "--version"]),
        "cuobjdump": _command_version([str(CUOBJDUMP_PATH), "--version"]),
    }

    positive_path = ARTIFACT_DIR / "vector-add.cuda-sm87.mef"
    negative_path = ARTIFACT_DIR / "vector-add.cuda-sm80.mef"
    _set_state(phase="compiling_sm87", versions=versions)
    positive_compile = _child_json(
        _run_child(["--compile-target", "sm_87", "--output", str(positive_path)]),
        "sm_87 compile",
    )
    positive_inspection = _inspect_cuda_images(positive_path)
    _write_json("cuobjdump-sm87.json", positive_inspection)
    if not positive_inspection["contains_sm_87"]:
        raise RuntimeError("cuobjdump did not find a native sm_87 CUDA image")

    _set_state(phase="loading_native_sm87_without_ptx_jit")
    positive_native_load = _child_json(
        _run_child(["--load-native-cubin", str(positive_path)], jit_disabled=True),
        "sm_87 native CUBIN load",
    )
    if not positive_native_load["module_loaded"]:
        raise RuntimeError(
            "native sm_87 CUBIN did not load: "
            + json.dumps(positive_native_load, sort_keys=True)
        )

    _set_state(phase="executing_sm87_without_ptx_jit")
    positive_execution = _child_json(
        _run_child(["--execute", str(positive_path)], jit_disabled=True),
        "sm_87 JIT-disabled execution",
    )

    _set_state(phase="compiling_negative_sm80")
    negative_compile = _child_json(
        _run_child(["--compile-target", "sm_80", "--output", str(negative_path)]),
        "sm_80 compile",
    )
    negative_inspection = _inspect_cuda_images(negative_path)
    _write_json("cuobjdump-sm80.json", negative_inspection)
    negative_native_load = _child_json(
        _run_child(
        ["--load-native-cubin", str(negative_path)], jit_disabled=True
        ),
        "sm_80 native CUBIN negative control",
    )
    negative_error = negative_native_load["cuda_module_load_error"]
    negative_control_expected_error = negative_error == "CUDA_ERROR_NO_BINARY_FOR_GPU"
    negative_control_passed = bool(
        not negative_native_load["module_loaded"]
        and negative_error
        in {"CUDA_ERROR_NO_BINARY_FOR_GPU", "CUDA_ERROR_INVALID_SOURCE"}
    )
    if not negative_control_passed:
        raise RuntimeError(
            "sm_80 negative control was not rejected as an incompatible native image: "
            + json.dumps(negative_native_load, sort_keys=True)
        )

    negative_max_execution = _run_child(
        ["--execute", str(negative_path)], jit_disabled=True
    )
    negative_max_text = (
        f"{negative_max_execution.stdout}\n{negative_max_execution.stderr}".strip()
    )
    if negative_max_execution.returncode == 0:
        raise RuntimeError("MAX unexpectedly executed the sm_80 negative control")

    sensor_after, temperature_after = _hottest_jetson_temperature()
    proof_wall_seconds = time.perf_counter() - proof_started
    proof_cpu_seconds = _cpu_seconds() - proof_cpu_started
    result = {
        "ready": True,
        "requested_device": "accelerator",
        "resolved_device": positive_execution["resolved_device"],
        "gpu_arch": "sm_87",
        "result_correct": positive_execution["result_correct"],
        "cuda_disable_ptx_jit": "1",
        "native_sm87_found": positive_inspection["contains_sm_87"],
        "negative_control_passed": negative_control_passed,
        "negative_control_expected_error": negative_control_expected_error,
        "negative_control_error": negative_error,
        "negative_control_note": (
            "Jetson/Tegra returned CUDA_ERROR_INVALID_SOURCE instead of "
            "CUDA_ERROR_NO_BINARY_FOR_GPU for the exact native sm_80 CUBIN."
            if not negative_control_expected_error
            else None
        ),
        "versions": versions,
        "positive_compile": positive_compile,
        "positive_execution": positive_execution,
        "positive_native_cubin_load": positive_native_load,
        "positive_cuobjdump": positive_inspection,
        "negative_compile": negative_compile,
        "negative_cuobjdump": negative_inspection,
        "negative_native_cubin_load": negative_native_load,
        "negative_max_rejected": True,
        "negative_max_error": negative_max_text,
        "thermal": {
            "limit_c": THERMAL_LIMIT_C,
            "before": {"sensor": sensor, "temperature_c": temperature},
            "after": {"sensor": sensor_after, "temperature_c": temperature_after},
        },
        "timing": {
            "proof_wall_seconds": proof_wall_seconds,
            "proof_cpu_seconds": proof_cpu_seconds,
            "average_cpu_cores_used": proof_cpu_seconds / proof_wall_seconds,
            "configured_cpu_limit": _cgroup_cpu_limit(),
            "host_logical_cpu_count": os.cpu_count(),
        },
    }
    _write_json("result.json", result)
    _set_state(phase="ready", error=None, **result)


async def _run_proof() -> None:
    try:
        await asyncio.to_thread(run_proof)
    except Exception as exc:  # noqa: BLE001 - status preserves probe failure
        error = f"{type(exc).__name__}: {exc}"
        _write_json("failure.json", {"ready": False, "error": error})
        _set_state(ready=False, phase="failed", error=error)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(_run_proof())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="MAX native sm_87 proof", lifespan=lifespan)


@app.get("/status")
async def status() -> dict[str, Any]:
    with _lock:
        return json.loads(json.dumps(_state))


@app.get("/artifacts/{name}")
async def artifact(name: str) -> FileResponse:
    allowed = {
        "result.json",
        "failure.json",
        "cuobjdump-sm87.json",
        "cuobjdump-sm80.json",
        "vector-add.cuda-sm87.mef",
        "vector-add.cuda-sm80.mef",
    }
    path = ARTIFACT_DIR / name
    if name not in allowed or not path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-target", choices=("sm_80", "sm_87"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", type=Path)
    parser.add_argument("--load-native-cubin", type=Path)
    arguments = parser.parse_args()
    if arguments.compile_target:
        if arguments.output is None:
            parser.error("--output is required with --compile-target")
        print(json.dumps(_compile_target(arguments.compile_target, arguments.output), sort_keys=True))
        return
    if arguments.execute:
        print(json.dumps(_execute_artifact(arguments.execute), sort_keys=True))
        return
    if arguments.load_native_cubin:
        print(json.dumps(_load_native_cubin(arguments.load_native_cubin), sort_keys=True))
        return
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
