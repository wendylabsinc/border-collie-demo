# MAX inference variant

`demo/max` is the verified presentation checkpoint
`go2-demo-verified-2026-08-17` (`Go2-modcon-final`, `acce417`) plus one
substitution: the general apple/pear/mango detector is loaded through Modular
MAX instead of the presented Ultralytics `apple-pear-mango.pt` path.

The mission, UI, controller, voice, camera, crop-confirm, model-router,
threshold, and evidence implementations are inherited unchanged from the
presentation checkpoint. Banana still uses the same resident specialist, and
it remains unqualified for motion.

## Runtime seam

`media/inference_runtime.py` exposes the compiled MAX model through the same
`predict(...)` shape consumed by `FruitModelRouter`. The sidecar reports both
the requested and active runtime in `/status.inference_runtime`. The branch
sets:

- `FRUIT_INFERENCE_BACKEND=max`
- `MAX_ALLOW_ULTRALYTICS_FALLBACK=0`
- `MAX_FRUIT_CLASS_IDS_JSON={"apple": 0, "pear": 1, "mango": 2}`
- `MAX_OUTPUT_CONTRACT=yoloe_segment_raw`

Fallback is prohibited so a missing or unloadable MAX candidate cannot look
like a passing MAX run while actually serving the presentation model.

## Current qualification boundary

Source alignment does not qualify the MAX runtime. The exact compiled
`apple-pear-mango` MEF and weight registry are not versioned in this branch,
and this branch has not passed same-frame parity, on-device latency, or a
supervised physical run. The manifest therefore keeps
`BORDER_COLLIE_AUTONOMY_ENABLED=0` even though the presentation checkpoint had
autonomy enabled.

Before deploying, the media image must install the matching MAX runtime and
ship these exact artifacts:

- `/media/apple-pear-mango.cuda-sm87.mef`
- `/media/apple-pear-mango.cuda-sm87.weights.npz`

Before enabling autonomy, require all of the following on Woof:

1. MAX is the active runtime with fallback disabled.
2. The exact presentation checkpoint produces matching fruit decisions on the
   frozen reference/camera corpus, with approximately 0.90 or better box IoU.
3. Median inference is below 100 ms and p95 below 200 ms on the Orin `sm_87`.
4. Camera freshness, memory, and thermal gates remain stable in a camera-only
   soak.
5. A supervised run matches the presentation behavior and reaches a terminal,
   safely disarmed state.

Until those gates pass, this branch is a source-aligned MAX integration target,
not a deployable or physically validated replacement for the presented build.
