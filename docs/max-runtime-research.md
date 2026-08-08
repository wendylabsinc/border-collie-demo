# Modular MAX runtime feasibility for the fruit detector

Research date: 2026-08-05. This note uses current Modular documentation and
first-party package metadata. It distinguishes current MAX 26.x APIs from old
24.x behavior.

## Conclusion

MAX can run on the class of hardware used by Woof, but replacing the current
TensorRT detector is **not a backend flag change**. Modular lists Jetson Orin /
Orin Nano as “known compatible for development,” and publishes Linux ARM64
packages, but does not classify Jetson as “tested for serving.” More
importantly, the current MAX 26.x `InferenceSession` API documents MAX
`Graph`/`Module` and compiled `.mef` inputs, not direct ONNX or YOLO import.
The practical migration is therefore a native MAX graph/module port of the
YOLO forward pass, followed by output-parity and performance qualification on
Woof.

Do not remove the TensorRT adapter until the MAX candidate produces equivalent
classes, boxes, confidence behavior, and safety-facing freshness data on the
same frozen corpus.

## Installation and Jetson support

- Modular’s current package requirements are Linux with glibc 2.34 or later,
  Python 3.10–3.14, a C linker, and either x86-64-v3 or ARM64 Neoverse N1 or
  newer. Ubuntu 22.04 or later is the continuously tested Linux distribution.
  The current package guide recommends `max[all]` through `uv` or `max-all`
  through `pixi`; smaller `max` packages are available when serving and
  benchmarking extras are unnecessary. [MAX packages and system
  requirements](https://docs.modular.com/packages/)
- The official nightly package index contains `manylinux_2_34_aarch64` wheels,
  so an ARM64 distribution channel exists. [Official MAX wheel
  index](https://whl.modular.com/nightly/simple/max/)
- MAX 26.4 is the current stable release; nightlies are explicitly less mature.
  Pin the experiment rather than silently following nightly. [MAX release
  index](https://docs.modular.com/releases/)
- Jetson Orin / Orin Nano appears under “Known compatible for development,”
  not “Tested for serving.” The documented NVIDIA requirement is driver 580 or
  newer. With an older driver, Modular requires
  `MODULAR_NVPTX_COMPILER_PATH` to point to a system `ptxas` binary. [MAX GPU
  compatibility](https://docs.modular.com/packages/#gpu-compatibility)

A read-only Woof probe on 2026-08-05 reported ARM64, Ubuntu 22.04, glibc 2.35,
Python 3.10.12, JetPack 6.2.1, Orin `sm_87`, 16 GB RAM, and NVIDIA driver
540.4.0. The base OS and Python meet the published minimums, but the driver is
below 580 and no host `ptxas` was found. The documented older-driver workaround
must therefore be added to and proven inside the candidate container before GPU
inference is viable. Package installation alone is not proof that the
accelerator can compile or execute a model.

## Current Python execution API

Device selection is explicit:

```python
from max.driver import Accelerator, CPU, accelerator_count
from max.engine import InferenceSession

device = Accelerator(0) if accelerator_count() else CPU()
session = InferenceSession(devices=[device])
compiled = session.compile(graph)
model = session.init(compiled, weights_registry=weights)
outputs = model(input_buffer)
```

`Accelerator()` selects a GPU and reports an NVIDIA device through the CUDA
API; `CPU()` selects the host. [MAX device API](https://docs.modular.com/api/python/generated/max.driver.Device/)
The preferred lifecycle is `compile()` followed by `init()` so compiled work
can be reused. `load()` still combines both operations, but is marked for
eventual deprecation. `Model` is callable and returns MAX buffers; input and
output metadata expose names, shapes, and dtypes. [InferenceSession
API](https://docs.modular.com/api/python/generated/max.engine.InferenceSession/)
[Model API](https://docs.modular.com/max/api/python/generated/max.engine.Model/)

NumPy input can enter through `Buffer.from_numpy()` and then move to the GPU
with `Buffer.to(device)`. A GPU result converted with `to_numpy()` incurs a host
copy. [Buffer API](https://docs.modular.com/api/python/generated/max.driver.Buffer/)
That transfer must be included in end-to-end detector timing, not hidden behind
kernel-only measurements.

MAX can add NVTX/CUDA profiling instrumentation through
`session.gpu_profiling("on" | "detailed")`, but Modular warns that it adds
runtime overhead. Record ordinary steady-state measurements separately from
profiled runs. [MAX GPU profiling
API](https://docs.modular.com/api/python/generated/max.engine.InferenceSession/#gpu_profiling)

## ONNX and YOLO status

Current 26.x documentation for `InferenceSession.compile()` accepts a MAX
`Graph`, MAX `Module`, or a saved compiled model such as `.mef`; `load()`
documents a `Graph` or saved model. It does not document `.onnx` as a supported
input. The current model catalog and model-development path are centered on
MAX-native graph architectures and do not provide a first-class YOLO object
detector pipeline. [InferenceSession API](https://docs.modular.com/api/python/generated/max.engine.InferenceSession/)
[MAX model support](https://docs.modular.com/max/model-formats)

Older MAX 24.x material did advertise ONNX execution. That is historical
evidence, not a supported basis for a new 26.x integration when the current API
surface no longer documents that input path. [Historical MAX 24.1 release
notes](https://docs.modular.com/max/changelog/v0.24.1/)

Consequences for this repository:

1. The deployed `.engine` file cannot be loaded by MAX. The exact source
   checkpoint was subsequently located in the legacy demo, so the remaining
   work starts from that checkpoint rather than attempting to reconstruct the
   TensorRT artifact.
2. Exporting another ONNX file does not, by itself, create a current supported
   MAX integration.
3. The YOLO backbone, neck, and detection head must be represented with MAX
   graph/module operations and their trained weights mapped into the graph, or
   Modular must provide a newly documented importer in a later release.
4. Preprocessing, YOLO output decode, confidence filtering, and NMS remain the
   application’s responsibility unless deliberately moved into the MAX graph.

## Lowest-risk migration shape

Preserve the runtime-neutral perception adapter and port only the neural
forward pass first:

1. Freeze a batch-1, 640×640 input and a labeled frame corpus.
2. Keep the current OpenCV/NumPy letterbox, BGR-to-RGB conversion, and scaling
   exactly unchanged. Feed the native MAX candidate as NHWC
   `[1, 640, 640, 3]`; the importer preserves logical ONNX NCHW axes and
   restores the documented ONNX output layout at the graph boundary.
3. Rebuild the YOLO forward graph with MAX `Graph`/`Module`, bind weights, and
   execute on `Accelerator(0)`.
4. Copy raw head output to the host and reuse the current Python decode and NMS.
5. Compare raw tensors first, then final boxes/classes/confidences against the
   TensorRT control. Only after parity should resize, normalization, decode, or
   NMS move into MAX/Mojo.

MAX’s model-development documentation identifies `max.graph` and `max.nn` as
the stable production APIs for custom architectures; its eager APIs are still
described as experimental. [MAX model development
overview](https://docs.modular.com/max/develop/)

## Qualification gates

The MAX candidate is not a replacement until all of these pass on Woof:

- the pinned package installs in an ARM64 container;
- `Accelerator(0)` initializes with the 540 driver through a verified system
  `ptxas`, or the platform is upgraded to a supported driver;
- the native graph compiles for `sm_87` and executes repeatedly;
- output parity passes on the same pear, apple, banana, distractor, small-fruit,
  and negative frames used for TensorRT;
- cold compile/startup, warm inference latency, end-to-end frame latency, FPS,
  CPU utilization, GPU utilization/memory, RAM, and power are recorded for both
  runtimes under equivalent conditions;
- camera freshness, status responsiveness, conditional crop-and-confirm, and
  specialist routing remain behaviorally identical;
- failures remain fail-closed and TensorRT remains an immediate rollback.

## Exact fruit-checkpoint feasibility proof

After the API research, the exact general-model checkpoint was located at the
legacy demo's `models/collie/collie-fruit-yoloe11m.pt`. It is 59,997,395 bytes
with SHA-256
`7c75fcc5d449a8b00785dfd0c955cbf11bd6bde6a5ede1ea8d34c097413bc53e`.
Ultralytics 8.4.102 exported that checkpoint successfully with fixed input
`(1, 3, 640, 640)`. The ONNX outputs are `(1, 39, 8400)` predictions and
`(1, 32, 160, 160)` mask prototypes.

The workspace's existing pinned MAX 26.4 ONNX importer supports every operator
type in that exact graph except its single `ConvTranspose` node. The graph has
125 `Conv` nodes and one `ConvTranspose`; the other operator types are Add,
Concat, Constant, Div, Gather, MatMul, MaxPool, Mul, Reshape, Resize, Shape,
Sigmoid, Slice, Softmax, Split, Sub, and Transpose. MAX documents
`max.graph.ops.conv2d_transpose`, including its NHWC input and RSCF filter
layout, so the next implementation step is bounded: extend the importer,
cross-compile an `sm_87` `.mef`, and compare its raw tensors to the exported
ONNX control. [MAX `conv2d_transpose` API](https://docs.modular.com/max/api/python/graph/ops/#conv2d_transpose)

This operator-coverage result is not a successful MAX compile. Filter layout,
padding, output shape, imported weight binding, the driver-540 `ptxas` path,
and numerical parity still require proof in the pinned Jetson/MAX environment.

The repo-local `media/max_onnx.py` preserves the provenance of the proven G1
importer and adds the exact mapping from ONNX NCHW/CFRS transpose convolution
to MAX NHWC/RSCF `ops.conv2d_transpose`. The focused contract test verifies
filter-axis mapping, stride, dilation, padding, output padding, contiguous
layout, and fail-closed rejection of grouped transpose convolution. The
check-only compiler entry point verified that the exact exported fruit graph
now has zero unsupported operator types. Compilation remains deliberately
unclaimed until that entry point runs in the Jetson/MAX environment.

The blocker is therefore a bounded importer extension plus on-device
compatibility and parity, not the perception service interface or missing
weights.
