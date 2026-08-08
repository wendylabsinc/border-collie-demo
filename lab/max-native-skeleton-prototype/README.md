# PROTOTYPE — MAX-native detector skeleton

Question: can a small, detection-only network written directly with MAX Graph
run fast enough on Woof's `sm_87` GPU to justify training matching fruit-model
weights?

This is throwaway measurement code. It does not read the camera, publish fruit
detections, issue motion, or modify the TensorRT demo. It builds a
MobileNet-style backbone with three raw detection heads, then benchmarks fixed
NHWC inputs at 320, 416, and 640 pixels. Each resolution runs in a fresh child
process so compiler and GPU allocations cannot leak into the next result.

The gate is a 150 ms median forward pass and a 200 ms p95 forward pass. Passing
only means training is worth attempting; it does not prove detector accuracy.

## Training outcome

The 1.31-million-parameter custom skeleton passed the original runtime gate but
did not pass the accuracy gate. Three architecture/training rounds peaked at
held-out F1 0.227; the later feature-pyramid and LTRB-target variants did not
improve it. These negative results are recorded in `MAX-NATIVE-003` through
`MAX-NATIVE-005` under `training/results/`.

The active MAX candidate is therefore not this custom skeleton. `MAX-NATIVE-006`
uses a 2.58-million-parameter YOLO11n trained at 416px. It retains 92% of the
YOLO11s reference mAP50 with 73% fewer parameters, and its ONNX export passes
the existing importer operator gate. That candidate still requires a guarded
Woof-native MAX compile and camera-only benchmark before any motion use.

Run from this directory:

```bash
/Users/olivertaylor/Library/Python/3.14/bin/dlo deploy --root . --dockerfile Dockerfile --adapter wendy --target woof.local -- wendy run --device woof.local --yes --detach --builder docker --chunking off --no-restart
```

Status is available at `http://woof.local:8125/status`. The app requires fresh
battery telemetry, at least 60% charge, and stays below the existing 82 C
guard. It aborts at 25% battery, low available memory, excessive swap, or a
per-resolution timeout.
