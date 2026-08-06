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

Run from this directory:

```bash
/Users/olivertaylor/Library/Python/3.14/bin/dlo deploy --root . --dockerfile Dockerfile --adapter wendy --target woof.local -- wendy run --device woof.local --yes --detach --builder docker --chunking off --no-restart
```

Status is available at `http://woof.local:8125/status`. The app requires fresh
battery telemetry, at least 60% charge, and stays below the existing 82 C
guard. It aborts at 25% battery, low available memory, excessive swap, or a
per-resolution timeout.
