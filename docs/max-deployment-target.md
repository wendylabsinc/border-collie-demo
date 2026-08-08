# MAX deployment target branch

This branch (`demo/max`) is `demo/base` plus the Modular MAX inference path,
selected through `FRUIT_INFERENCE_BACKEND=max`. It exists so the MAX runtime has
a stable, polished demo to land in once the runtime clears its gates. It is
**not** cleared for autonomous motion today.

## Why autonomy is disabled here

`BORDER_COLLIE_AUTONOMY_ENABLED` is `0` in this branch's manifest, unlike
`demo/base`. Per
[`max-vs-tensorrt-retrospective-2026-08-07.md`](max-vs-tensorrt-retrospective-2026-08-07.md),
the best usable MAX candidate ran at 841.45 ms median (1.19 FPS) against a
100 ms motion gate — a 8.4x miss — and in its first clear live pear test ranked
the pear as apple. Perception at that latency cannot satisfy the freshness rules
the approach and final-push stages depend on, so motion stays off until the
runtime gates pass.

The camera, perception, evidence, and bark paths all still run. Deploying this
branch shows MAX inference behaving on real Woof camera frames without the robot
moving on stale detections.

`MAX_ALLOW_TENSORRT_FALLBACK` is `0` deliberately. This branch should fail
loudly if the MAX artifact is missing or unloadable rather than quietly serving
TensorRT results and looking like a passing MAX run. Set it to `1` only for a
deliberate side-by-side comparison.

## What is already wired

- `media/inference_runtime.py` — `InferenceRuntimeConfig`, `MaxCompiledModelRunner`
  (loads one precompiled MAX artifact and retains its accelerator session), and
  `MaxYoloModelAdapter`, which exposes a compiled MAX detector through the
  existing YOLO model seam. Target filtering, NMS, and source-coordinate
  restoration stay in the adapter. `max` is imported lazily inside the runner
  constructor, so the module stays importable without the MAX runtime installed.
- `media/perception_sidecar.py` — selects the backend via `load_general_model`
  and publishes an `inference_runtime` block in `/status` carrying
  `requested_backend`, `active_backend`, `fallback_used`, `fallback_reason`, and
  `candidate_validated`.
- `media/max_onnx.py`, `media/compile_max_fruit.py`,
  `media/prune_max_detection_onnx.py` — the ONNX import and artifact build path.

The banana specialist and the runtime-neutral camera/perception contract are
unchanged, so switching backends does not change the interface the app consumes.

## Unconfirmed settings

The compiled artifact does not exist yet, so these manifest values are
placeholders that must be confirmed against the real artifact when it is built.
They are the branch's open questions, not settled configuration:

| Setting | Current value | What must confirm it |
| --- | --- | --- |
| `MAX_MODEL_PATH` / `MAX_WEIGHTS_PATH` | `/media/fruit.cuda-sm87.*` | Paths the build actually emits and the media image actually ships |
| `MAX_INPUT_SIZE` | `640` | Matches the newest go2-domain head (`yolo11n-640-go2-head`). The retrospective's best-benchmarked candidate was 416 px — pick whichever head is actually compiled |
| `MAX_OUTPUT_CONTRACT` | `yoloe_segment_raw` | Whether the compiled graph emits raw YOLO channels or three decoded `boxes`/`scores`/`class_ids` outputs. Wrong choice silently produces garbage boxes |
| `MAX_FRUIT_CLASS_IDS_JSON` | `{"pear": 0, "apple": 1, "banana": 2}` | Taken from `training/prepare_go2_domain.py` `CLASS_IDS`. Must match the ordering the compiled model was trained with |

## Gates before this branch can enable motion

1. A compiled `sm_87` artifact exists and loads on Woof through
   `MaxCompiledModelRunner` without falling back.
2. Median inference meets the 100 ms motion gate on Woof.
3. Live fruit ranking is correct on a clear pear scene — the specific failure
   that rejected the previous candidate.
4. Status responsiveness and freshness behavior hold at that latency.
5. A supervised Woof run matches `demo/base` end-to-end behavior.

Only after all five should `BORDER_COLLIE_AUTONOMY_ENABLED` return to `1`.
