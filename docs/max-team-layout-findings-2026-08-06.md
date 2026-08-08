# MAX 26.4 Orin fruit-detector findings

Date: 2026-08-06

## Executive summary

MAX 26.4 successfully compiled and executed the complete 640 x 640 FP16 fruit
detector on Woof's Jetson Orin GPU (`sm_87`). Execution succeeded with PTX JIT
disabled and produced the expected FP16 `[1, 7, 8400]` output. Native GPU
compatibility is therefore proven for this graph.

Performance is not yet viable for live perception. A clean cached-artifact run
measured 11,851.05 ms median inference latency, 11,908.43 ms p95, and 0.084 FPS
over 30 timed zero-input inferences.

The leading performance hypothesis is excessive layout traffic introduced by
our narrow ONNX-to-MAX importer. ONNX represents image activations logically as
NCHW, while MAX's optimized convolution and pooling paths consume NHWC. The
current importer converts NCHW to NHWC before each convolution and converts the
result back to NCHW afterward. The detector contains 113 convolutions, so the
graph expresses hundreds of large rank-four permutations.

This is an importer-level hypothesis, not yet a confirmed MAX runtime defect.
MAX may fuse or remove some permutations during compilation. A GPU profile and
an A/B graph benchmark are required to establish how much of the measured
latency is attributable to physical layout conversion.

## Layout terminology

- **NCHW**: Number (batch), Channels, Height, Width.
- **NHWC**: Number (batch), Height, Width, Channels.

For one 640 x 640 RGB frame:

- Logical ONNX shape: `[1, 3, 640, 640]` (NCHW)
- Native MAX convolution shape: `[1, 640, 640, 3]` (NHWC)

The proposed importer retains ONNX's logical axis semantics while tracking a
separate physical-axis order. Rank-four activations enter MAX as NHWC, remain
NHWC across convolution, activation, residual, concatenation, pooling, and
resize blocks, and convert only at genuine semantic boundaries such as
attention reshapes and detection-head flattening. The public output remains
the exact ONNX contract `[1, 7, 8400]`.

## Reproducible environment

| Item | Value |
| --- | --- |
| Device | `woof.local` |
| CPU | ARM64, 8 logical cores |
| GPU | NVIDIA Jetson Orin, `sm_87` |
| Unified physical memory | 16,416,137,216 bytes |
| MAX | 26.4.0 |
| Mojo | 1.0.0b2 |
| CUDA/PTXAS | 12.6 |
| Precision | FP16 |
| Input | `[1, 3, 640, 640]` logical NCHW |
| Output | `[1, 7, 8400]` FP16 |
| Native-code gate | `CUDA_DISABLE_PTX_JIT=1` |
| Model SHA-256 | `ecfb3c7bfa88661338cf08cb1d639c4cb570f7a308f3173eee2351571cf3dfea` |
| MEF SHA-256 | `b4f88f5d1b4e4a27b02c26006d9a5fab186c881decbc4a7bcb77eeab0fd09131` |

## Measured results

### Successful native FP16 run

| Metric | Result |
| --- | ---: |
| Compile time | 372.40 s |
| Cached artifact load plus benchmark startup | 418.59 s |
| Timed inference samples | 30 |
| Median latency | 11,851.05 ms |
| p95 latency | 11,908.43 ms |
| Maximum latency | 23,695.16 ms |
| Throughput from median | 0.084 FPS |
| MEF size | 7,005,075 bytes |
| Weights archive | 40,208,094 bytes |
| Minimum available host memory | 2,944,606,208 bytes |
| Peak process RSS | 618,971,136 bytes |

The successful run used a zero tensor. It proves execution and timing, but not
numerical or detection parity with TensorRT on real images.

### Graph characteristics relevant to layout

The pruned detection graph contains 439 nodes, including:

- 113 `Conv`
- 106 `Mul`
- 103 `Sigmoid`
- 26 `Concat`
- 22 `Add`
- 11 `Reshape`
- 10 `Split`
- 3 `MaxPool`
- 3 `Transpose`
- 2 `Resize`
- 2 `MatMul`
- 2 `Softmax`

Of the 113 convolution outputs, 102 feed `Sigmoid` and `Mul` activation paths.
Those long convolutional blocks are layout-preserving and should not require a
round trip through NCHW after every convolution.

## Importer behavior under investigation

The baseline importer lowers a typical convolution as:

```text
logical NCHW
  -> physical permute to NHWC
  -> MAX conv2d with RSCF weights
  -> physical permute back to NCHW
```

The proposed native lowering is:

```text
NHWC graph input
  -> MAX conv2d
  -> activation / residual / concat / pool / resize in NHWC
  -> convert only before an operation whose logical memory order requires it
  -> restore the declared ONNX output layout once
```

The new lowering must translate logical ONNX axes to physical MAX axes for
`Concat`, `Split`, `Slice`, `Softmax`, and `Resize`. `Reshape`, `Transpose`, and
`MatMul` are treated as explicit layout boundaries unless their equivalence can
be proven.

## A/B validation required

The native-NHWC graph should be accepted only if all of the following pass:

1. Compile the same 640 x 640 FP16 detector for `sm_87`.
2. Execute with `CUDA_DISABLE_PTX_JIT=1`.
3. Preserve the exact FP16 `[1, 7, 8400]` output contract.
4. Compare outputs with the baseline MAX graph and TensorRT on a fixed real
   image suite, including boxes, classes, confidence, and tolerances.
5. Warm each artifact and run at least 30 paired inferences.
6. Compare median, p95, GPU utilization, memory, temperature, startup time, and
   output digests.
7. Capture a GPU profile to determine whether permutation kernels or related
   memory copies remain after MAX compilation.

## Questions for the MAX team

1. Does MAX 26.4's compiler reliably eliminate adjacent NCHW/NHWC `permute`
   operations around `conv2d`, or are those materialized as GPU kernels/copies?
2. Is an NHWC graph input and persistent channel-last activation layout the
   recommended lowering for imported CNNs on Jetson Orin?
3. Is there a supported layout annotation or ONNX import path that communicates
   logical NCHW axes without emitting explicit runtime permutations?
4. Which profiler or compiler diagnostics expose post-lowering permutation
   kernels, fusion decisions, and tensor allocation sizes?
5. Is the roughly seven-minute cached artifact initialization expected for a
   7 MB MEF plus 40 MB weight archive on this target?
6. Are there known MAX 26.4 performance issues for this YOLO-style pattern on
   `sm_87`, especially DFL softmax and mixed reshape/convolution boundaries?

## Why this report is useful

The evidence isolates three different statements that are often conflated:

- Native `sm_87` execution works.
- The FP16 detector fits without host OOM in a clean runtime process.
- Current end-to-end inference performance is unusable.

It also gives the MAX team a fixed model hash, exact runtime versions, a
repeatable native-code gate, graph operator counts, and an A/B experiment that
can confirm or reject the layout hypothesis without changing model resolution
or output behavior.

## Local evidence

- `benchmarks/results/2026-08-06-max-sm87-native-probe.json`
- `benchmarks/results/2026-08-06-max-fruit-detection-fp16-runtime-success.json`
- `docs/max-full-model-attempts-2026-08-06.md`

