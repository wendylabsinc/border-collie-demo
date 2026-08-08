# MAX full fruit-model attempts on Woof

Date: 2026-08-06

## Outcome

The pruned 640 x 640 FP16 detection graph now **compiles and executes with MAX
on Woof's GPU**. A clean cached load completed with PTX JIT disabled and
produced the expected FP16 `[1, 7, 8400]` tensor. It is not suitable for the
live demo: initialization took 418.6 seconds and median inference latency was
11.85 seconds (0.084 FPS). The full FP32 segmentation and detection graphs
remain unusable because of runtime memory pressure.

The strongest current explanation is a **MAX 26.4 whole-graph runtime-memory
plan**, not an incompatible GPU or oversized artifact. The compile peaked at
13.38 GiB RSS and completed by using zram. Loading the resulting 16.7 MB MEF and
89.6 MB weight archive in a clean process consumed roughly 12.8 GiB of unified
host/GPU memory outside ordinary process RSS, leaving too little space for
cuDNN's transpose-convolution workspace.

## Test configuration

- Device: `woof.local`, ARM64, eight online CPU cores in bounded 25 W mode
- GPU: NVIDIA Orin, `sm_87`
- Physical memory: 15.29 GiB
- Model: full fruit segmentation ONNX
- Model SHA-256:
  `f5b24d8ea80dc25189f01eb83b0e58c0cdd29292a5e944c730c6955ad1718056`
- Input: `1 x 3 x 640 x 640`, FP32
- Imported weights: 251 weights, 89,518,464 bytes
- MAX: 26.4.0
- Mojo: 1.0.0b2
- CUDA/PTXAS: 12.6
- Native-code gate: `CUDA_DISABLE_PTX_JIT=1`
- CPU quota and affinity: 8.0 cores, CPUs 0-7
- Final software RSS ceiling: 13.75 GiB
- Hard container ceiling: 14 GiB
- Thermal cutoff: 82 C
- Battery emergency cutoff: 25%

## Attempts

| Attempt | Change under test | Result | Timing and resource evidence |
| --- | --- | --- | --- |
| 1. Original full graph | Compile the exact full segmentation graph with a 2-core quota | **Graph compilation completed**, then model initialization failed because the probe copied the weights to CUDA while the compiled weight registry expected host buffers | Compile: 1,793.03 seconds (29m 53s). Peak RSS observed around 13.2 GiB. Jetson stayed near 50 C. |
| 2. Cached artifact with host weights | Reuse attempt 1's MEF and pass NumPy/host weights to MAX | **Initialization succeeded**; first execution reached the GPU kernel and failed because MAX 26.4 does not support this softmax on a non-innermost axis | Cached load reached execution in about 20 seconds and stayed below roughly 0.55 GiB RSS during loading. This isolated the next failure from compilation. |
| 3. Softmax compatibility lowering, 13 GiB guard | Transpose the requested softmax axis to the innermost position, run softmax, then restore the layout | **Guarded stop**, exit 86, during graph compilation | Roughly 18 minutes into compilation. RSS crossed the conservative 13 GiB software ceiling. No thermal or battery failure. |
| 4. Softmax compatibility lowering, 13.5 GiB guard | Allow another 0.5 GiB while keeping the 14 GiB hard container cap | **Guarded stop**, exit 86 | Roughly 19 minutes into compilation. RSS reached 13.55 GiB. Jetson remained near 50 C. |
| 5. Channel-last softmax/Conv fusion | Compose redundant transposes and feed the channel-last softmax result directly into MAX's NHWC convolution path | **Guarded stop**, exit 86 | Roughly 19 minutes into compilation. RSS still reached 13.55 GiB. This indicates the peak is dominated by whole-graph compiler state rather than those layout ops alone. |
| 6. `InferenceSession(num_threads=2)` | Ask MAX directly to use two inference-session threads | **Immediate MAX/LLVM trap**, exit 133 | Log: `MLRT::getOrCreateCPUDevice called requesting different options to those used to create the existing CPUDevice.` No model compilation occurred. |
| 7. Two-core process affinity | Remove the conflicting `num_threads` option and restrict the process to CPUs `[0, 1]` before MAX creates worker threads | **Guarded stop**, exit 86 | Compile ran for about 18m 25s before RSS reached 13.55 GiB. Battery went from 63% to 57%; Jetson remained below 50 C in the final run. CPU affinity did not materially reduce peak compiler memory. |
| 8. Eight-core 25 W compile with only thermal diagnostics | Stop every other Wendy app, enable CPUs 0-7, retain the 14 GiB container ceiling, and allow a 13.75 GiB software guard | **Compilation succeeded; initialization failed** | Compile: 730.94 seconds. Peak RSS: 13.38 GiB. Minimum available host RAM: 122,081,280 bytes. Peak zram use: 2,654,994,432 bytes. Battery fell from 95% to 89%; Jetson peaked below 58 C and Go2 IMU stayed at 79-80 C. The exported MEF and weights were preserved before cuDNN returned `CUDNN_STATUS_ALLOC_FAILED` for the segmentation `ConvTranspose`. |
| 9. Clean cached-artifact load | Stop the compiler process, recover host memory, then restart and load only the exported MEF and weights with PTX JIT disabled | **Runtime initialization failed** | Process RSS stayed below about 0.55 GiB, but host available memory fell from about 13.17 GiB to 1.25 GiB. The same cuDNN allocation failed at `ConvTranspose`, proving the new blocker is the compiled runtime memory plan rather than retained compiler state. |
| 10. Detection-only pruned graph | Remove the unused mask coefficients, mask prototype output, and `ConvTranspose`; compile the resulting `[1, 7, 8400]` class/box graph on all eight cores | **Compilation succeeded; host OOM during initialization** | Compile: 665.25 seconds. The MEF and weights were preserved. Peak observed compile RSS was about 13.12 GiB. During initialization, available RAM fell to about 0.21 GiB and zram rose from about 0.52 GiB to 6.82 GiB. The kernel killed the MAX process and several system services. No `ConvTranspose` or cuDNN workspace error occurred. The probe had no restart policy, critical NVIDIA/network services were restored, and host memory recovered. |
| 11. FP16 detection-only graph | Keep the same 640 x 640 detector and output contract, but lower floating inputs, weights, constants, activations, and outputs to FP16 while preserving integer shape/index tensors | **Compilation succeeded; guarded stop during initialization** | Compile: 372.40 seconds, 44.0% faster than FP32 detection. Peak RSS: 9.53 GiB, 27.3% lower. MEF: 7,005,075 bytes, 56.4% smaller. Weights: 40,208,094 bytes, 50.0% smaller. Runtime initialization still reduced available memory to 462,974,976 bytes (0.43 GiB), so the new 0.5 GiB headroom guard stopped the process with exit 86. Swap was only 124,301,312 bytes and no host OOM occurred. Battery fell from 71% to 67%; guarded temperature stayed at or below 80 C. |
| 12. Clean cached FP16 load, user-authorized unbounded run | Start a fresh process with only the saved FP16 MEF/weights, disable the RAM and swap aborts, retain thermal/battery/no-restart controls, and set `oom_score_adj=500` so MAX is killed before host services if OOM occurs | **Runtime and 30 timed inferences succeeded** | Artifact load plus warm/timed execution took 418.59 seconds. Median latency: 11,851.05 ms; p95: 11,908.43 ms; maximum: 23,695.16 ms; throughput: 0.084 FPS. Output: FP16 `[1, 7, 8400]`. Peak process RSS: 618,971,136 bytes. Minimum available host memory: 2,944,606,208 bytes (2.74 GiB). Peak swap: 182,235,136 bytes. Battery fell from 62% to 58%; guarded temperature stayed at or below 80 C. No host OOM occurred. |

## What the attempts prove

### Proven

1. MAX can produce and execute native GPU code on Woof's `sm_87` GPU. The
   isolated probe passes with PTX JIT disabled.
2. The full segmentation graph is translatable by the current ONNX-to-MAX
   importer. It contains 251 imported weights totaling 89.5 MB.
3. The uncorrected full graph can finish compilation in about 30 minutes.
4. Weight initialization must use host registry buffers. Moving the registry
   weights to CUDA in the probe was incorrect and is now fixed.
5. The full model reaches a real MAX GPU-kernel limitation: MAX 26.4's softmax
   implementation rejects the model's non-innermost reduction axis.
6. The compatibility-corrected graph compiles as one graph with eight online
   cores, zram, a 13.75 GiB software guard, and every other application stopped.
7. The corrected MEF is 16,739,805 bytes and its weights archive is 89,584,750
   bytes; artifact size does not explain the runtime allocation.
8. A clean cached-artifact load still consumes roughly 12.8 GiB of unified
   memory before cuDNN requests the `ConvTranspose` workspace.
9. The failures were not caused by overheating, battery exhaustion, or an
   incompatible GPU architecture. The guards stopped the process deliberately.
10. The pruned detector contains 439 nodes, one `[1, 7, 8400]` output, no mask
    output, no `ConvTranspose`, and no operators unsupported by the importer.
11. Removing segmentation saves about 66 seconds of compilation and about 9 MB
    of serialized weights, but does not make MAX 26.4's FP32 runtime fit safely
    in Woof's 15.29 GiB unified memory.
12. FP16 materially improves compiler memory, compile time, and artifact size.
    Compiling and initializing in the same process still consumes nearly all
    unified memory, so compilation and runtime must remain separate operations.
13. The runtime headroom guard prevents the known host-OOM path: the FP16 run
    exited once at 0.43 GiB available with no new kernel OOM event and no system
    service recovery required.

14. A clean cached process is materially different from compiling and loading
    in the same process. The clean FP16 load retained 2.74 GiB host headroom and
    completed successfully, whereas the compile process still held compiler
    allocations and crossed the conservative headroom guard.
15. The full 640 x 640 FP16 detection model executes end to end on MAX with PTX
    JIT disabled. This proves native runtime compatibility, but only on the
    probe's zero input; perception parity is not yet proven.

### Not yet proven

1. The complete segmentation model executes end to end with PTX JIT disabled.
2. MAX outputs match TensorRT for fruit classes, confidence, and boxes on fixed
   real images.
3. MAX is fast enough for live camera perception. The current measured rate is
   only 0.084 FPS, which is already a failing result for the demo.

## My diagnosis

### Primary issue: runtime working memory

The compile-memory blocker is resolved under a stripped-down eight-core setup.
The remaining blocker is MAX 26.4's runtime memory plan for both tested FP32
graphs. The clearest evidence is that:

- the corrected graph compiled and exported successfully;
- the MEF and weight files total only about 106 MB;
- a clean load used little process RSS but removed roughly 12.8 GiB from host
  available memory; and
- cuDNN then failed to allocate workspace for the segmentation
  transpose-convolution; and
- removing that complete branch changed the failure from a cuDNN error to a
  host OOM during runtime initialization.

This means the runtime cannot safely coexist with WendyOS and the camera/demo
stack for the FP32 graphs even if their container limit were raised. The clean
FP16 detector does retain 2.74 GiB headroom, but camera/demo coexistence has not
been tested and its throughput already fails the live-demo requirement.

### Secondary issue: MAX 26.4 GPU softmax support

The model's DFL head contains a softmax whose reduction axis is not the
innermost tensor dimension. MAX 26.4 rejects that GPU kernel form. Our importer
now preserves the model's semantics by moving the axis to the innermost
position before softmax and composing/removing redundant layout conversions
where possible. This fix is what must be present in the final compiled
artifact.

### Resolved probe issue: weight placement

The first post-compile failure was caused by the probe, not the model or MAX
code generation. The probe copied weight-registry tensors to CUDA even though
the compiled model expected host registry tensors. Passing contiguous NumPy
arrays fixes initialization and allows MAX to stage the weights correctly.

### MAX thread-setting defect or limitation

MAX documents `InferenceSession(num_threads=...)`, but setting it after MAX had
already created its internal CPU device triggered a fatal option-mismatch trap
on this version. Process affinity avoided that trap, but it did not solve peak
compile memory.

## Recommended next step

Do not use either FP32 artifact in the demo. The 640 x 640 FP16 artifact now
passes native execution and memory headroom, but its 11.85-second median
latency makes it unusable for live tracking. The next investigation should
profile why MAX 26.4 holds the GPU at 99% for minutes during load and takes
roughly 12 seconds per inference before changing resolution or model behavior.
The probe now aborts runtime initialization when available memory falls to
0.5 GiB or total swap reaches 2 GiB in its checked-in configuration, preventing
another known OOM path. Attempt 12 explicitly overrode those limits for one
user-authorized run; the currently running process retains that override, while
future deployments return to the guarded configuration.

If segmentation masks become required later, partition the backbone and heads.
Raising Woof's memory limit is not safe because the current runtime already
starved host services before the camera or motion stack was running.

The alternative is to compile the corrected graph on a higher-memory ARM64/MAX
host while explicitly targeting `sm_87`, then transfer the MEF and repeat the
JIT-disabled execution and parity tests on Woof.

### 2026-08-07 candidate update

The next compile target is now the trained 416px YOLO11n from
`MAX-NATIVE-006`, not another optimization of the 9.41-million-parameter
YOLO11s graph. The nano candidate has 2.58 million parameters and 6.3 GFLOPs,
reaches mAP50 0.775 and mAP50-95 0.620 on the fixed held-out split, exports to
`[1, 7, 3549]`, and uses only operators already supported by the importer.
PyTorch-to-ONNX parity passed locally.

This changes the hypothesis being tested: the earlier 2.822-second 416px MAX
result measured the larger YOLO11s graph, while the nano graph has 73% fewer
parameters and roughly 70% fewer GFLOPs. It does not prove a proportional
speedup. Compile it under the existing battery, thermal, memory, no-restart,
native-`sm_87`, and PTX-JIT-disabled gates, then run fixed-corpus and live-camera
parity with motion disabled. TensorRT remains the rollback and production
default until those gates pass.

## Native NHWC importer candidate

The next importer revision keeps rank-four activations in MAX's native NHWC
layout instead of converting NCHW to NHWC before every convolution and back to
NCHW afterward. ONNX axes remain logical NCHW inside the importer; a small
layout value maps each logical axis to its current physical MAX axis. Concat,
Split, Slice, Resize, pooling, elementwise operations, and convolution use that
mapping. Layout is materialized only at order-sensitive boundaries such as
Reshape, Transpose, MatMul, and the final ONNX output.

A shape-aware dry traversal of the exact 439-node detection graph produced:

- physical MAX input: `[1, 640, 640, 3]` (NHWC);
- unchanged logical output: `[1, 7, 8400]`;
- unchanged external weight count: 225; and
- 16 runtime tensor permutations across the complete graph.

The old importer necessarily emitted at least 226 runtime permutations around
its 113 convolutions alone, plus pooling and other layout boundaries. The new
count is structural evidence that the repeated conversion pattern is gone; it
is not yet performance evidence. A new artifact stem,
`fruit-detection-fp16-native-nhwc`, and graph compatibility version 6 preserve
the known-good FP16 artifact for rollback.

The candidate has passed importer unit tests and exact-graph shape traversal.
It has not yet been compiled or benchmarked on Woof. At the time this candidate
was prepared, Woof's battery was 46%, below the checked-in 60% compile-start
gate, so no device build was started.

## Docker and deployment measurements

These timings describe packaging and deployment, not MAX graph compilation:

- Initial full-model image build: 194.373 seconds
- Initial Wendy deployment: 416.117 seconds
- Latest cached image build: 3.968 seconds
- Latest cached Wendy deployment: 50.083 seconds
- Latest image size: 2,118,916,041 bytes across nine layers
- Latest build reused seven of nine image layers
- Eight-core diagnostic deployment: 429.867 seconds total; 388.905 seconds
  build, 38.523 seconds unpack, 1.585 seconds readiness, and 0.020 seconds
  transfer. This cold rebuild reused none of its ten reported build units.

The cold and cached runs are not a paired optimization experiment, so they do
not prove that DLO itself produced a speedup.

## Evidence files

- `benchmarks/results/2026-08-06-max-sm87-native-probe.json`
- `benchmarks/results/2026-08-06-max-full-fruit-seg-cpu2-failure.json`
- `benchmarks/results/2026-08-06-max-full-fruit-seg-woof.json`
- `benchmarks/results/2026-08-06-max-fruit-detection-cpu8-runtime-oom.json`
- `benchmarks/results/2026-08-06-max-fruit-detection-fp16-guarded.json`
- `benchmarks/results/2026-08-06-max-fruit-detection-fp16-runtime-success.json`
