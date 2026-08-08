# Fruit inference runtime benchmark

The TensorRT control is recorded in
[`benchmarks/results/2026-08-05-tensorrt-perception-baseline.json`](../benchmarks/results/2026-08-05-tensorrt-perception-baseline.json).
It is a read-only observation of the deployed media service; it sent no motion
command and changed no target selection.

## Control result

At the lower-overhead status polling rate, `/status` responded in 15.94 ms
median, 20.64 ms p95, and 27.36 ms maximum. Observed inference took 60.22 ms
median, 71.70 ms p95, and 75.96 ms maximum. A separate 20-second high-rate
probe observed 168 unique inference results at 61.57 ms median, 78.03 ms p95,
and 98.37 ms maximum. Every observed result used one inference pass.

The source was 1280×720. Its median observed source interval in the high-rate
probe was 71.18 ms, equivalent to 14.05 fps. That probe requested status at
approximately 25 Hz and materially perturbed status tail latency, so its
133.17 ms p95 and 281.66 ms maximum must not be compared to an unstressed MAX
candidate. The approximately 5 Hz probe is the control for endpoint latency.

The media process used approximately 54.1% of one CPU core over a 10-second
sample and reported 1,428,248 KiB RSS. The combined Wendy app group reported
1.935 GB median memory. Direct `tegrastats` samples observed GR3D at 99%, 99%,
and 41%; the coarser Wendy metric reported zero in its snapshots, demonstrating
that it can miss short GPU work. Three direct samples are not enough to claim a
stable GPU percentile.

The live target was pear, but the scene did not have labeled ground truth.
Confidence and bounding-box aggregates therefore describe model behavior only;
they do not measure accuracy. No crop-and-confirm or banana-specialist pass
occurred in the window.

## MAX candidate boundary

`media/inference_runtime.py` keeps TensorRT as the validated default and adds a
MAX candidate behind the model interface already consumed by the fruit router.
The candidate expects a fixed-shape compiled `.mef`. Its default contract
matches the exact exported fruit checkpoint: a `(1, 39, 8400)` YOLOE detection
and mask-coefficient tensor plus `(1, 32, 160, 160)` mask prototypes. The demo
does not need masks, so the adapter decodes boxes and the three fruit class
scores from the first output and retains BGR-to-RGB letterboxing,
normalization, target-class filtering, NMS, and source-coordinate restoration
outside the compiled graph. A separate decoded boxes/scores/class-IDs contract
remains available for a future graph with postprocessing included.

Selection is explicit:

- `FRUIT_INFERENCE_BACKEND=tensorrt` keeps the current path and is the default.
- `FRUIT_INFERENCE_BACKEND=max` requests MAX and fails startup when the MAX
  candidate cannot initialize.
- `MAX_ALLOW_TENSORRT_FALLBACK=1` permits an explicit fallback and reports the
  requested backend, active backend, and reason in `/status`.
- `MAX_MODEL_PATH`, `MAX_FRUIT_CLASS_IDS_JSON`, `MAX_INPUT_SIZE`, and
  `MAX_DEVICE_INDEX` identify the compiled candidate contract.
- `MAX_WEIGHTS_PATH` identifies the NumPy weight registry emitted alongside
  the MEF by the repo-local compiler.
- `MAX_OUTPUT_CONTRACT` defaults to `yoloe_segment_raw` for the current fruit
  checkpoint; `decoded` selects the three-output alternative.

Automatic fallback is deliberately disabled. A stage operator must never
believe MAX is under test while TensorRT is silently serving detections.

## Current blockers

The clean repository has the validated TensorRT `.engine` and banana PyTorch
checkpoint but no MAX graph/module or compiled fruit `.mef`. The legacy demo
does contain the exact general source checkpoint:
`collie-fruit-yoloe11m.pt`, 59,997,395 bytes, SHA-256
`7c75fcc5d449a8b00785dfd0c955cbf11bd6bde6a5ede1ea8d34c097413bc53e`.
A bounded local proof exported it successfully through Ultralytics 8.4.102 to
a fixed 640×640 ONNX graph with the two output tensors described above.

That ONNX graph uses 18 operator types. The existing workspace
ONNX-to-MAX importer supports all of them except one `ConvTranspose` node. MAX
26.4 documents `max.graph.ops.conv2d_transpose`, so this is a bounded importer
extension rather than an unknown architecture port. It still needs shape,
filter-layout, numerical-parity, and compile verification in the pinned MAX
environment; operator-name coverage alone is not execution proof.

Woof also has NVIDIA driver 540.4, below MAX's documented driver 580 minimum.
The documented older-driver path requires a system `ptxas` and
`MODULAR_NVPTX_COMPILER_PATH`; the current host probe found no `ptxas`. An
existing workspace MAX-on-Orin project demonstrates the intended shape with a
pinned MAX 26.4 package, a CUDA 12.6 `ptxas`, an ONNX-to-MAX importer, and an
`sm_87` `.mef`. The exact next step is to extend its importer for the fruit
graph's single `ConvTranspose`, cross-compile the fruit `.mef` for `sm_87`, and
load it with the imported weight registry. That compile must happen in the
Jetson/MAX candidate environment; this Mac cannot prove the CUDA artifact.

The detailed API and platform evidence is in
[`docs/max-runtime-research.md`](max-runtime-research.md).

## Bounded compiler commands

Export the exact source checkpoint at fixed shape without embedded NMS:

```bash
yolo export \
  model=/path/to/collie-fruit-yoloe11m.pt \
  format=onnx imgsz=640 dynamic=false simplify=false nms=false opset=17
```

The check-only command requires ONNX and NumPy, but not MAX or a GPU:

```bash
PYTHONPATH=. python -m media.compile_max_fruit \
  --onnx /path/to/collie-fruit-yoloe11m.onnx \
  --check-only
```

The exact exported checkpoint reports no unsupported operators with the
repo-local importer. That is operator coverage, not compile or numerical
parity.

Compilation must run inside the pinned MAX 26.4 Jetson candidate environment
that provides CUDA 12.6 `ptxas`:

```bash
export MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda-12.6/bin/ptxas
PYTHONPATH=. python -m media.compile_max_fruit \
  --onnx /path/to/collie-fruit-yoloe11m.onnx \
  --mef /path/to/fruit.cuda-sm87.mef \
  --weights /path/to/fruit.cuda-sm87.weights.npz
```

The compiler translates ONNX to a native MAX graph, exports the compiled MEF,
and saves the external weight registry. A successful command still does not
qualify the model: load, warm inference, frozen-corpus parity, and performance
must be proven on Woof before selecting `FRUIT_INFERENCE_BACKEND=max`.

## Required comparison

Do not call MAX faster or equivalent until both runtimes process the same
frozen frame corpus with the same target, thresholds, crop decisions, and
specialist routes. Record cold startup/compile time, warm end-to-end inference
median/p95/max, source throughput, status latency, per-process CPU/RSS, direct
GPU utilization/memory, power, output parity, and all crop/specialist pass
counts. The MAX candidate must preserve classes, confidences, boxes, freshness,
camera failure behavior, and the mission's fail-closed safety deadlines before
TensorRT can be removed.
