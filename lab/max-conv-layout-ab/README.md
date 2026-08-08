# MAX Conv2D filter-layout A/B

This is a motion-free, single-operation diagnostic for Woof's Jetson Orin.
It compiles two mathematically identical FP16 Conv2D graphs with MAX 26.4:

- native NHWC activation plus RSCF filter metadata (the current importer path);
- the same activation and weights represented as FCRS with
  `FilterLayout.FCRS` (the MAX cuDNN fallback path).

The probe uses a representative 104x104, 16-to-32-channel, 3x3 convolution,
warms both graphs, alternates 20 synchronized measurements per layout, checks
raw-output parity, and records the median/p95 latency and speedup
in `/artifacts/max-conv-layout-ab-result.json`. It imports no robot SDK and
cannot issue motion commands.

The hypothesis is considered confirmed when the outputs agree within FP16
tolerance and FCRS is at least 5x faster than RSCF. This deliberately avoids a
whole-YOLO compile so kernel dispatch can be tested in minutes rather than tens
of minutes.

## First Orin result

The 2026-08-07 run confirmed that FCRS reaches MAX's cuDNN branch, but that
branch failed with `CUDNN_STATUS_ALLOC_FAILED` for isolated 208x208, 104x104,
and even 8x8 inputs. The matching 104x104 RSCF control completed at 6.505 ms
median. This confirms the dispatch analysis but disproves the stronger claim
that changing the importer alone produces a usable speedup on MAX 26.4.

The captured result is in
`results/2026-08-07-sm87-fcrs-cudnn-check.json`.
