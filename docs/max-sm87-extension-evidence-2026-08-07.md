# MAX 26.4 `sm_87` extension: evidence and pre-commit experiment plan

**Date:** 2026-08-07

**Status:** Research plan only; no runtime fix has been selected or implemented.

**Hardware rule:** Every new hardware measurement in this plan runs on **G1**. Woof results are historical clues only, not acceptance evidence.

## Executive conclusion

The public MAX 26.4 source confirms a large architecture-specific convolution dispatch gap:

- MAX's normal Python `conv2d` path uses NHWC activations and RSCF filters by default.
- On an `sm_100` Blackwell target, MAX 26.4 first attempts dedicated structured convolution, then an `im2col + matmul` path.
- On a non-`sm_100` NVIDIA target such as Jetson Orin `sm_87`, an RSCF filter falls through to a scalar, thread-per-output-pixel naive kernel. An FCRS filter is instead sent to cuDNN.
- The 2-D cuDNN wrapper in this tag uses deprecated legacy APIs, has known C-ABI mismatches in its generated Mojo bindings, accepts the first heuristic algorithm, allocates that algorithm's workspace per invocation, and has neither a workspace cap nor an allocation-failure retry.

This is enough to explain why MAX 26.4 can work very well for one vision network yet be very slow for YOLO on Orin: YOLO is convolution-heavy, and the normal RSCF route reaches a kernel that does scalar loads and nested loops over every output channel, filter element, and input channel. It does **not** prove which replacement is best. The fastest safe next step is a controlled G1 tournament that first separates a cuDNN/MAX-wrapper failure from a genuine `sm_87` kernel gap, then compares four extension methods under identical YOLO shapes.

The plan therefore tests, without committing to one:

1. a minimal legacy-cuDNN wrapper repair,
2. a cuDNN Graph API custom op,
3. an `sm_87` adaptation of MAX's open `im2col + matmul` route, and
4. native Mojo kernels specialized for the most important YOLO shape families.

ONNX Runtime CUDA is the G1 production control. A same-model TensorRT build is
an optional tactic/performance oracle, not a candidate implementation inside
MAX and not the production baseline.

## Source scope and version pin

The source observations below are pinned to Modular's public `max/v26.4.0` tag at commit [`cb643611fa4ae40b799fbc195b9f243220db5d97`](https://github.com/modular/modular/commit/cb643611fa4ae40b799fbc195b9f243220db5d97). Current web documentation is used only for documented platform support and APIs; code-path claims use immutable commit links.

Modular announced Jetson Orin `sm_87` programming support in MAX 25.3 and, in the same release, published the MAX AI kernels. That establishes that `sm_87` is a supported compilation target, but it does not claim equal kernel coverage or performance across architectures. See the official [MAX 25.3 release notes](https://docs.modular.com/releases/v25.3/). Modular's current requirements distinguish B200 `sm_100` as continuously tested and Jetson Orin `sm_87` as known compatible; they also document the older-driver compiler-path requirement for Jetson. See [Mojo hardware and software requirements](https://mojolang.org/docs/requirements/).

## What `sm_87` and `sm_100` mean here

`sm_87` and `sm_100` are NVIDIA GPU code-generation targets, not MAX versions:

- `sm_87` is the embedded Ampere target used by Jetson Orin. MAX's open GPU metadata gives it a 32-thread warp, 1,536 resident threads per SM, and 164 KiB shared memory per SM ([Ampere embedded family](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/mojo/stdlib/std/gpu/host/info.mojo#L60-L90)). The exposed Orin target emits NVPTX for `sm_87` with PTX 8.1 features ([Orin target metadata](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/mojo/stdlib/std/gpu/host/info.mojo#L681-L714)).
- `sm_100a` is the Blackwell target used by B100/B200 in this source tree, emitted with PTX 8.8 features ([Blackwell target metadata](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/mojo/stdlib/std/gpu/host/info.mojo#L929-L976)). It has different matrix-multiply and memory-movement facilities, so an optimized Blackwell kernel cannot simply be relabeled as an Ampere kernel.

One item to verify rather than assume: the public MAX target model is named `OrinNano` and records eight SMs. G1's actual Orin SKU, SM count, power mode, and memory capacity must come from G1 inventory. A generic or mismatched target model could affect tuning decisions even though both report compute capability 8.7.

## Confirmed MAX 26.4 convolution path

### 1. The normal graph path favors RSCF

The high-level `Conv2D` module permutes NCHW input to NHWC when `permute=True`, selects FCRS in that case, and otherwise selects RSCF ([`max.nn.Conv2D.__call__`](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/python/max/nn/conv.py#L269-L304)). The Graph API's documented default is NHWC input with RSCF filters ([MAX Graph `conv2d` API](https://docs.modular.com/stable/max/api/python/graph.ops/#max.graph.ops.conv2d)).

The registered convolution lowering accepts only NHWC input, computes `filter_is_fcrs` directly from the graph layout, and passes that compile-time flag to `conv_gpu` ([registered `mo.conv` lowering](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/kernels/src/graph_compiler/builtin_kernels/conv.mojo#L309-L467)). This is not merely a weight-storage convention; the layout chooses a materially different GPU implementation.

### 2. `sm_100` has two optimized paths before fallback

For rank-4 convolution, `conv_gpu` checks `_is_sm10x_gpu`. On Blackwell BF16 it attempts a dedicated structured convolution whose source comment reports 4–7× over cuDNN, subject to stride, dilation, group, and alignment constraints ([structured `sm_100` dispatch](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/kernels/src/nn/conv/conv.mojo#L4510-L4606)). If that does not apply, an `sm_100`-gated `im2col + matmul` route handles other channel alignments ([Blackwell `im2col + matmul` dispatch](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/kernels/src/nn/conv/conv.mojo#L4608-L4629)).

Those gates explain why “MAX has optimized 2-D convolution” and “YOLO is slow on Orin” can both be true. The relevant optimized implementations are architecture- and dtype-specific.

### 3. `sm_87` reaches cuDNN only with FCRS; RSCF reaches a naive kernel

After the architecture-specific paths, non-`sm_100` FCRS convolution constructs cuDNN-compatible tensors and invokes cuDNN; the epilogue form also creates a temporary convolution output before running the fused epilogue ([FCRS fallback](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/kernels/src/nn/conv/conv.mojo#L4912-L5021)). RSCF falls through to the naive GPU launch ([RSCF fallback](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/kernels/src/nn/conv/conv.mojo#L5022-L5036)).

The naive kernel assigns a thread to one output spatial position, then loops serially over all output channels, filter rows, filter columns, and input channels. Its loads are scalar-width (`width=1`) ([naive NHWC/RSCF implementation](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/kernels/src/nn/conv/conv.mojo#L3259-L3335)). That is fundamentally different from mapping the convolution to Tensor Core matrix operations.

This dispatch asymmetry is the strongest confirmed cause of the observed YOLO gap.

## The FCRS/cuDNN path is promising but not yet trusted

MAX 26.4's 2-D cuDNN wrapper:

- describes activations/output as NHWC and filters as FCRS;
- enables tensor-op math conversion for FP16/BF16;
- calls `cudnnGetConvolutionForwardAlgorithm_v7` for one heuristic result;
- asks cuDNN for that algorithm's workspace;
- allocates a runtime buffer of that size on every invocation; and
- calls `cudnnConvolutionForward` ([2-D cuDNN wrapper](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/kernels/src/nn/conv/conv.mojo#L3565-L3783)).

There are two high-value diagnostic concerns.

### Binding ABI concern

The source itself says its generated performance-result structure has an incorrect C layout because Mojo represents some cuDNN enums as one-byte values. It works around the result structure with a raw 48-byte buffer, but the binding still declares `requestedAlgoCount` and `returnedAlgoCount` as `Int16` ([MAX binding declaration](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/kernels/src/_cudnn/cnn_infer.mojo#L597-L628)); the enum wrapper itself stores an `Int8` ([forward algorithm enum](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/kernels/src/_cudnn/infer.mojo#L1775-L1789)). NVIDIA declares both counts as C `int`/`int *` in the official [`cudnnGetConvolutionForwardAlgorithm_v7` signature](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.3.0/api/cudnn-cnn-library.html#cudnngetconvolutionforwardalgorithm-v7).

That mismatch is a credible source of corrupted counts or adjacent stack data. It is not proven to be the cause of the allocation failure. The same MAX file's 3-D implementation explicitly documents the count as a four-byte C `int`, uses an `Int32` backing value, and bitcasts around the binding ([3-D ABI handling](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/kernels/src/nn/conv/conv.mojo#L5838-L5873)). The experiment must therefore compare an ABI-correct control before changing the production route.

### Workspace and algorithm-selection concern

The 2-D path takes the first heuristic result without a workspace budget, creates the requested workspace per call, and immediately errors if allocation or cuDNN execution fails. NVIDIA documents the v7 call as a heuristic list ordered by expected performance and documents the separate workspace-size query; both legacy calls are deprecated in cuDNN 9 ([cuDNN legacy convolution API](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.3.0/api/cudnn-cnn-library.html)). NVIDIA also documents `CUDNN_STATUS_ALLOC_FAILED` for failed internal or explicit allocation, so the status alone does not identify which allocation failed.

MAX's 3-D convolution code already provides an informative design precedent: it uses a 256 MiB search cap, chooses a successful algorithm within that cap, and retries allocation failures with zero-workspace `IMPLICIT_GEMM` ([3-D selection and fallback](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/kernels/src/nn/conv/conv.mojo#L5840-L5940), [3-D allocation retry](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/kernels/src/nn/conv/conv.mojo#L5954-L6037)). This is evidence that bounded selection and retry are relevant patterns, not evidence that copying the 3-D policy is the right 2-D fix.

cuDNN 9 deprecates the legacy API in favor of the Graph API ([cuDNN API overview](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.3.0/api/overview.html)). The official Graph API support table includes compute capability 8.7 and FP16 forward convolution, including grouped convolution, in supported engine surfaces ([cuDNN Graph API support](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.3.0/developer/graph-api.html)). JetPack 6.2.1 ships CUDA 12.6, TensorRT 10.3, and cuDNN 9.3 on Orin ([JetPack 6.2.1 components](https://developer.nvidia.com/embedded/jetpack-sdk-621)). Therefore “cuDNN cannot accelerate convolution on `sm_87`” is not a sound assumption; the MAX integration must be isolated from cuDNN itself.

## Runtime memory: visible pieces and missing pieces

The open runtime includes a buffer planner that accepts static sizes, runtime sizes, alignment requirements, and a compiler-provided `can_share` lifetime matrix. It returns allocation offsets and a pool high-water mark ([`mgp.buffer.plan`](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/kernels/src/graph_compiler/builtin_primitives/primitives.mojo#L548-L619)). Its implementation greedily reuses compatible blocks and can log allocation count, reuse count, requested bytes, pool size, and savings ([planner implementation](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/kernels/src/graph_compiler/builtin_primitives/buffer_plan.mojo#L61-L105), [greedy reuse](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/kernels/src/graph_compiler/builtin_primitives/buffer_plan.mojo#L126-L278)).

The public tree does not expose the native implementation that constructs the liveness matrix, performs the full fusion/lowering search, or schedules all execution buffers. Generated Python stubs describe kernel selection and layout inference but not their implementation ([`MOToMOGG` and kernel selection](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/python/max/_core/dialects/mo/passes.pyi#L23-L31), [`InferLayouts`](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/python/max/_core/dialects/mo/passes.pyi#L236-L242)). We can observe compiled kernel summaries through the public model interface ([`Model.kernel_summaries`](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/python/max/_core/engine.pyi#L150-L158)), profiler traces, CUDA artifacts, and device memory. We cannot infer allocator ownership or fusion decisions from source alone.

## NMS is a separate, later problem

MAX's compiled NMS implementation performs a dry run to determine the dynamic output size, then uses a serial greedy selection loop with repeated sorting and pairwise IoU suppression ([NMS shape function and kernel](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/kernels/src/nn/nms.mojo#L194-L365)). The interpreter implementation is explicitly CPU-only and documents quadratic sorting/NMS behavior ([interpreter NMS](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/python/max/_interpreter_ops/nms_ops.mojo#L14-L24), [quadratic loop](https://github.com/modular/modular/blob/cb643611fa4ae40b799fbc195b9f243220db5d97/max/python/max/_interpreter_ops/nms_ops.mojo#L155-L199)).

That may become material after convolution is fixed, but it should not be mixed into the first convolution tournament. First record which NMS implementation the G1 model actually uses and its measured share of end-to-end latency.

## What is observable and what is not

| Question | Observable with public MAX 26.4/G1 | Not available from the public source |
|---|---|---|
| Which architecture was compiled? | Target metadata, compiler flags, MEF/cubin/PTX inspection, kernel names | Internal rationale for selecting that target |
| Which convolution path ran? | Nsight kernel symbols, cuDNN API log, isolated RSCF/FCRS A/B, `kernel_summaries` | Native compiler cost model and full kernel-selection implementation |
| What workspace did cuDNN request? | Direct API query, instrumented experimental wrapper, CUDA free-memory readings | cuDNN's proprietary heuristic and all undocumented internal allocations |
| Why buffers overlap or remain live? | Open greedy planner and its optional statistics; external memory timeline | Compiler construction of `can_share`, full schedule, and native ownership logic |
| Whether a custom op can target `sm_87` | Official custom-op API and explicit accelerator compilation target | A guarantee that a proposed kernel is fast without G1 measurements |
| Whether MAX NMS is material | G1 end-to-end trace and actual dispatch | Relevance inferred only from source or another machine |

MAX officially supports hardware-specific custom Mojo operators and graph integration while managing device placement ([custom operators](https://docs.modular.com/develop/custom-ops/), [building GPU custom ops](https://docs.modular.com/develop/build-custom-ops/)). Mojo can explicitly compile GPU assembly for an `sm_*` accelerator target ([Mojo compilation targets](https://mojolang.org/docs/tools/compilation/)). These are the public extension seams available if a built-in fix is not immediately upstreamable.

## G1 experiment design

### Non-negotiable controls

Before testing a method, freeze these variables:

- **Machine:** G1 only, with exclusive access during samples.
- **Software manifest:** L4T/JetPack, kernel, driver, CUDA runtime/toolkit, cuDNN library and headers, TensorRT, MAX wheel/build, Mojo compiler, Python, container/image digest, and `MODULAR_NVPTX_COMPILER_PATH` if set.
- **Hardware state:** exact Orin SKU, compute capability, SM count, memory size, power mode, clock policy, temperature, throttling flags, and available memory before each run.
- **Model:** the immutable G1 YOLO11n ONNX digest, fixed 416-pixel input, batch 1,
  and its current FP32 semantics. FP16 is a separately labeled optimization
  variable tested only after the FP32 control is reproducible.
- **Data:** deterministic synthetic tensors for microbenchmarks plus a frozen real-image calibration corpus for model parity.
- **Timing:** compile/load, first inference, and warmed steady-state are separate metrics. Alternate candidate/control order, warm each path, collect at least 30 steady samples, and report median, p95, and dispersion.
- **Correctness:** compare each layer against a common FP32 reference and compare
  full-model decoded boxes/scores against the G1 ONNX Runtime CUDA control. A
  same-model TensorRT oracle may be added, but it does not replace the production
  control. Freeze numerical tolerances before looking at performance results.
- **Profiles:** collect an Nsight Systems trace first; use Nsight Compute only on the top kernels. Modular recommends this progression in its [GPU profiling guide](https://docs.modular.com/gpu-system-profiling/).

Each process should emit a machine-readable result containing all manifest fields, shape, layout, selected method/algorithm, requested workspace, peak memory, status, latency distribution, correctness result, kernel symbols, and artifact digests. A result without its manifest is not comparable.

### Phase 0 — establish the G1 baseline and shape corpus

**E0. G1 inventory and stability check**

1. Record the full manifest and actual GPU properties from CUDA, not from MAX's `OrinNano` constant.
2. Run a 10-minute fixed ONNX Runtime CUDA workload while recording `tegrastats`,
   temperature, clocks, power mode, and latency. If latency drifts or throttling
   occurs, fix the test environment before any MAX comparison.
3. Verify that the installed MAX package is exactly 26.4 and that its build can be associated with the inspected tag. If not, repeat source inspection against the installed build's exact source revision.

**Pass condition:** reproducible ONNX Runtime CUDA p95 within a predeclared
variance band across two fresh processes; no thermal/power-mode drift.

**E1. Compile-artifact audit**

1. Cross-compile the current minimal MAX convolution graph for `sm_87` off-device
   where possible, then load and execute the immutable artifact on G1. Do not
   repeat the prior memory-heavy full-model on-device JIT path.
2. Capture compiler command/target metadata and inspect generated PTX/cubin/MEF with NVIDIA binary tools.
3. Confirm the artifact contains `sm_87` code and record kernel symbols/resource usage. Check that no silent PTX JIT to a different target or generic fallback is occurring.
4. Print `Model.kernel_summaries` and align each graph convolution with its profiler kernel.

**Pass condition:** unambiguous mapping from graph op → compiled kernel symbol → `sm_87` artifact.

**E2. Build the YOLO convolution corpus**

Extract every unique convolution signature from the frozen YOLO graph:

`(N,H,W,Cin,Cout,R,S,stride,dilation,padding,groups,dtype,epilogue)`.

Weight each unique signature by call count and baseline GPU time. Ensure the corpus explicitly covers:

- 1×1 pointwise convolution,
- 3×3 stride 1,
- 3×3 stride 2,
- shallow high-spatial-resolution layers,
- deep high-channel layers,
- grouped/depthwise convolution if present, and
- no-epilogue versus bias/SiLU epilogue forms.

This corpus, rather than a single friendly shape, is the unit of comparison.

### Phase 1 — discriminate cuDNN capability from MAX integration

**E3. Independent cuDNN control**

Build a small standalone C++ harness on G1 using exactly MAX's NHWC activation/output descriptors, FCRS filter descriptors, FP16 data, FP32 accumulation, shapes, padding, stride, dilation, and group count. Test each shape in a fresh process under these modes:

1. Legacy `IMPLICIT_GEMM` with zero workspace.
2. Legacy `IMPLICIT_PRECOMP_GEMM` with queried workspace.
3. All successful algorithms returned by the v7 heuristic, not only result zero, with requested and actual workspace recorded.
4. cuDNN Graph API heuristics with explicit workspace caps.

Enable official cuDNN debug logging with `CUDNN_LOGLEVEL_DBG` and `CUDNN_LOGDEST_DBG`; NVIDIA documents that this records API arguments and diagnostic tracebacks ([cuDNN troubleshooting/logging](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.3.0/reference/troubleshooting.html)). Query CUDA free memory immediately before and after plan creation, workspace allocation, execution, and synchronization.

**Interpretation:**

- If all direct cuDNN methods fail on a shape, investigate the G1 software stack/descriptors before modifying MAX.
- If direct cuDNN succeeds but current MAX FCRS fails, the defect is in the binding, algorithm selection, memory interaction, or wrapper lifecycle.
- If direct cuDNN succeeds but is itself far behind ONNX Runtime CUDA, a native
  MAX kernel may be needed; an optional same-model TensorRT oracle can determine
  whether vendor tactic selection or fusion leaves additional headroom.

**E4. Current MAX layout A/B**

For every corpus shape, compile two separate one-op graphs with mathematically identical weights:

- NHWC + RSCF through the current built-in route;
- NHWC + FCRS through the current built-in route.

Use separate processes so cached metadata from one shape cannot contaminate another. Run no-epilogue and bias/SiLU variants separately. Capture cuDNN logs, MAX kernel summaries, Nsight kernel names, requested workspace if observable, peak memory, and exact failing API/status.

**Expected diagnostic, not a required result:** RSCF should show the named naive kernel; FCRS should show cuDNN or fail at a precisely identified allocation/API boundary.

### Phase 2 — method tournament

All changes in this phase live in disposable experimental branches or isolated custom-op packages. Do not alter the deployed demo or make a production architecture decision.

#### Method A — minimal legacy-cuDNN wrapper repair

Create the smallest experimental variant of the 2-D wrapper that:

- uses C-ABI-correct four-byte count storage and correctly sized enum/performance records;
- requests multiple candidates and rejects unsuccessful results;
- records selected algorithm, math type, and workspace;
- accepts an explicit workspace cap derived from G1's measured headroom;
- always includes zero-workspace `IMPLICIT_GEMM` as a control; and
- retries an allocation failure only after synchronization and workspace release.

Test four independent variables rather than landing them together:

| Variant | ABI | Selection | Workspace policy | Purpose |
|---|---|---|---|---|
| A0 | current | current first result | unbounded | reproduce current behavior |
| A1 | corrected | current first result | unbounded | isolate ABI effect |
| A2 | corrected | enumerate successful results | caps: 0, 16, 64, 128, 256 MiB and measured-safe maximum | isolate selection/workspace effect |
| A3 | corrected | best passing result | bounded + zero-workspace retry | test robust policy |

The 256 MiB point is included because MAX 26.4 uses it in the 3-D path; it is not the presumed answer. Reject any cap that leaves inadequate concurrent headroom for the real G1 camera/demo process.

#### Method B — cuDNN Graph API custom op

Implement a throwaway MAX custom op that calls cuDNN Graph API for the same corpus. Compare:

- convolution alone,
- convolution + bias,
- convolution + bias + SiLU when the graph supports the fusion,
- heuristic modes with the same workspace-cap sweep, and
- one cached execution plan per immutable shape versus plan creation per process.

Measure standalone cuDNN Graph performance first, then the identical plan through the MAX custom-op boundary. The delta isolates MAX scheduling/custom-op overhead. Record any layout conversion explicitly; offline FCRS weight packing is allowed, but per-inference weight transposition is not hidden in the timing.

#### Method C — `sm_87` `im2col + matmul`

Adapt the open route that is currently gated to `sm_100`, without assuming its Blackwell implementation is portable. Test two submethods:

- **C1, 1×1 direct GEMM:** reinterpret/reshape NHWC convolution as matrix multiplication without materializing `im2col` where mathematically valid.
- **C2, 3×3/stride convolution:** materialize tiled `im2col` into a bounded scratch buffer and call the best existing `sm_87` MAX matmul; sweep tile sizes and fusion of `im2col` production with GEMM consumption.

For each, capture scratch-buffer bytes, copy/materialization time, Tensor Core instructions, tensor-pipe utilization, occupancy, register use, shared memory, memory bandwidth, and epilogue fusion. A fast GEMM with an expensive `im2col` is not a winning convolution.

#### Method D — native Mojo `sm_87` kernels

Only after E2 ranks the shapes, prototype narrow kernels for the smallest set of families covering most baseline convolution time:

1. 1×1 stride-1, contiguous channels;
2. 3×3 stride-1;
3. 3×3 stride-2; and
4. grouped/depthwise only if the actual model makes it material.

Compile explicitly for `sm_87`. Test vectorized global loads, shared-memory tiles, Ampere Tensor Core MMA, double-buffering, channel-tail handling, and fused bias/SiLU as separate variables. Use NVIDIA's compiler resource report and Nsight Compute to reject variants limited by register spilling, occupancy, or redundant global traffic.

This method has the greatest maintenance surface. It advances only if it materially beats Method B/C on the weighted corpus or provides coverage they cannot.

### Phase 3 — integration ladder

A microkernel winner is not yet a MAX/YOLO fix. Promote each viable method through the same ladder:

1. **Single-op parity:** every corpus signature passes numerical and memory checks.
2. **Stage slice:** replace the highest-time contiguous YOLO stage; preserve all surrounding MAX operations.
3. **All-convolution forward:** replace all supported convolutions and use explicit fallback for unsupported signatures.
4. **Detector forward without NMS:** compare raw boxes/class logits against the frozen reference.
5. **Full detector with current post-processing:** measure real end-to-end G1 latency, transfers, and CPU/GPU overlap.
6. **Soak:** run at least 1,000 inferences plus a camera-duration soak under the intended G1 power/thermal mode, checking memory growth and error rate.

At every rung, compare against:

- unchanged MAX 26.4 RSCF,
- ONNX Runtime CUDA on the same G1/model/input,
- optional same-model TensorRT as a tactic/performance oracle,
- the independent direct-cuDNN control, and
- the previous rung's numerical output.

Compile latency, model-load latency, first inference, and warmed inference remain separate. A method may be acceptable for a fixed robot model even with expensive one-time compilation, but that trade must be visible.

### Phase 4 — NMS only if the new profile justifies it

After convolution is no longer dominant, profile the actual deployed post-processing path. If NMS is at least 5% of p95 end-to-end latency, compare:

1. existing host/vectorized post-processing,
2. MAX built-in NMS, and
3. a GPU custom op using sort plus bitmask suppression.

Use identical pre-NMS boxes and scores and include device-transfer/synchronization costs. If NMS is below the threshold, leave it alone; source-level inefficiency is not sufficient reason to expand this work.

## Provisional acceptance gates

These thresholds should be approved before performance data is viewed. They are proposed decision rules, not current results.

### Required for any candidate

- 100% of the frozen convolution corpus passes the predeclared FP16 numerical tolerance or has a documented, correct fallback.
- Full decoded detections preserve the agreed validation metric and per-image parity envelope against the frozen reference.
- No CUDA/cuDNN/MAX error in 1,000 consecutive inferences and no monotonic memory growth in the camera-duration soak.
- All workspace and temporary allocations remain inside a G1 memory budget derived from measured concurrent headroom, not merely from total device RAM.
- Artifacts demonstrably target `sm_87`; no accidental CPU fallback or hidden per-inference layout conversion.
- Every reported number is reproduced in two fresh processes with the full manifest attached.

### Proposed performance gates

- Weighted microbenchmark median no worse than 1.20× the fastest correct independent cuDNN method for the signatures the candidate claims to support.
- At least 5× faster than current naive RSCF on the weighted corpus; otherwise the integration/maintenance cost is unlikely to justify the change.
- In the full forward profile, the named naive RSCF kernels account for less than 10% of GPU time.
- Full detector p95 on G1 is no worse than 1.25× the ONNX Runtime CUDA control,
  unless the team explicitly accepts a different latency target for the benefit
  of one runtime.
- Peak device memory and p95 do not regress when the camera/demo process runs concurrently.

If those targets prove physically inconsistent with the direct-cuDNN control, revise them openly before choosing a method; do not relax them only for a favored implementation.

## Decision matrix before committing a fix

Score each method using measured evidence, with correctness and robustness as hard gates:

| Dimension | Weight | Evidence required |
|---|---:|---|
| Correctness/coverage | gate | corpus pass rate, unsupported-shape fallback, detector validation |
| Reliability | gate | repeated-process runs, 1,000-inference and camera soak, no memory growth |
| G1 p95 performance | 30% | weighted corpus, stage slice, full detector |
| Peak memory/workspace | 20% | explicit workspace plus observed runtime peak under concurrent demo load |
| Shape and epilogue coverage | 15% | actual YOLO signature coverage, not synthetic breadth |
| Maintainability | 15% | source size, architecture-specific assumptions, test burden, dependency stability |
| Upstreamability | 10% | compatibility with public MAX extension seams and likelihood of Modular acceptance |
| Build/startup cost | 5% | compile, model load, first inference |
| Observability/fallback | 5% | algorithm logging, bounded allocation, deterministic fallback behavior |

Do not choose a production fix until at least Methods A, B, and C have completed the weighted microbenchmark corpus. Method D can be stopped after a representative kernel if it cannot beat the vendor-library or matmul routes.

## Stop conditions that save time

- Stop MAX-wrapper work if the independent cuDNN control fails the same way; fix or change the G1 CUDA/cuDNN environment first.
- Stop a method after three representative high-weight shapes if it cannot beat naive RSCF by 2× or fails correctness without a clear local cause.
- Stop `im2col + matmul` if materialization plus GEMM is already slower than bounded cuDNN on the representative shapes.
- Stop native-kernel expansion if the first high-weight family is not competitive after profiling two materially different tilings.
- Stop NMS work if it remains below 5% of end-to-end p95 after convolution changes.
- Stop integration if a method relies on an unbounded workspace or leaves insufficient memory for concurrent G1 duties, even if isolated latency is excellent.

## Evidence package required for the decision

Keep experimental code separate from production and preserve:

- immutable source/model/container/package digests;
- G1 hardware/software/power/thermal manifest;
- JSON results for each shape and run;
- cuDNN debug logs with sensitive paths/environment values removed;
- Nsight Systems reports and selected Nsight Compute summaries;
- kernel summaries and `sm_87` binary/resource inspection;
- correctness outputs and detector validation results;
- workspace/free-memory timeline; and
- a one-page comparison generated from raw results, not hand-entered numbers.

Only after this evidence package satisfies the gates should the team decide
between an upstream MAX kernel change, a maintained Wendy custom op, or
continuing to use ONNX Runtime CUDA for YOLO.

## Historical local clues, explicitly not G1 acceptance evidence

The existing [Woof Nsight checkpoint](./max-yolo11n-nsys-profile-checkpoint-2026-08-07.md) reports a profile dominated by the naive RSCF convolution kernel, and the existing `lab/max-conv-layout-ab` artifacts report an FCRS/cuDNN allocation failure. Those results are consistent with the public 26.4 dispatch source and motivated E3/E4. They must not be used as proof of G1 behavior; all decisions above require fresh G1 measurements.
