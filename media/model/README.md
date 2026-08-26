# Pear model runtime

## Active checkpoint: apple-pear-mango.pt

`PEAR_MODEL_PATH` points at `apple-pear-mango.pt`, a YOLOE checkpoint carrying
`apple`, `pear` and `mango`. It is not a fine-tune. YOLOE is open-vocabulary,
and the checkpoint is produced by baking one visual-prompt exemplar per class
into the base weights — no training data, no labels, no gradient steps.

Reproduce it from `collie-demo/tools/build_visual_prompt_weights.py` using the
reference frame versioned beside it as `apple-pear-mango-reference.jpg`, which
was captured from Woof's own camera so the exemplars carry this stage's
lighting, floor and fisheye:

```sh
python tools/build_visual_prompt_weights.py \
  --weights models/candidates/yoloe-11m-seg.pt \
  --reference apple-pear-mango-reference.jpg \
  --output apple-pear-mango.pt \
  --device mps \
  --prompt apple=966,504,995,531 \
  --prompt pear=204,476,237,517 \
  --prompt mango=699,492,746,521
```

Base weights: `yoloe-11m-seg.pt` from the Ultralytics v8.4.0 assets release,
sha256 `785199b7cc75a5f4041dec15782a4b8a4105e5a34d7ffb287a0a10781724c5bd`. The
baked checkpoint is 59,997,080 bytes, sha256
`8b3bdf67e3f0daa62058565bb85974c8133e8adf2e510740a6fad3a97ae69a0a`.

Measured on Woof: apple 0.952, pear 0.957, mango 0.899, one inference pass at
roughly 0.100 s. Over the frozen 40-frame `MANGO-EVAL-001` clip the mango holds
40/40 at IoU 0.50 with median confidence 0.848 and a worst true positive of
0.822, against a top false positive of 0.209.

The crop-and-confirm small-box rule is disabled for this checkpoint
(`PEAR_CROP_CONFIRM_SMALL_AREA_RATIO`), because it triggers on box size alone
and every floor-level fruit is under the default ratio. At these confidences
the second pass changed no outcome and doubled inference past the 0.25 s
detection freshness gate.

This checkpoint runs as PyTorch, not TensorRT, at roughly 100 ms per frame on
the Orin. It is fast enough that the export below is an optimisation rather
than a prerequisite.

## Pending optimisation: TensorRT FP16 for apple-pear-mango

Prior art on this exact Orin ran TensorRT 10.7 FP16 in production, so an FP16
engine for this checkpoint is expected to be substantially faster than the
100 ms PyTorch pass. Ultralytics loads an engine directly, so this is a path
change and not a rewrite.

Build it with the one-shot Wendy app in
[`lab/tensorrt-export`](../../lab/tensorrt-export/README.md). It **must** be
built on Woof: an engine is specific to the GPU and to the TensorRT version
that produced it. The app exports from `apple-pear-mango.pt` with `half=True`,
then qualifies the engine against the `.pt` on the committed reference frame
`apple-pear-mango-reference.jpg`, where the `.pt` yields pear 0.960, apple
0.957 and mango 0.901, plus a weak 0.212 mango on a yellow object. The three
strong detections must reappear within 0.05 confidence and 0.90 box IoU; the
weak mango is reported but not gated.

`PEAR_MODEL_PATH` stays on the `.pt` until the engine passes. Switching is
configuration only: add a `model/apple-pear-mango.engine` copy entry to
`media/build.stagefile.yaml` and point `PEAR_MODEL_PATH` at
`/media/apple-pear-mango.engine` in `wendy.json`.

`task="segment"` is mandatory on every load of a non-`.pt` artifact. A
serialized engine loses its task metadata and ultralytics decodes the 39 output
channels as 4 box + 35 classes instead of 4 box + 3 classes + 32 mask
coefficients, which surfaces as `KeyError: 23`. `media/perception_sidecar.py`
already passes it for `PEAR_MODEL_PATH`, so no runtime code change is needed.

## Previous adapter: TensorRT

The validated hardware-specific TensorRT engine is versioned as `model.engine`
and copied to `/media/model.engine` in the media container. Model binaries are
part of the reproducible demo checkpoint; do not exclude them from Git.

Long-distance detection currently uses crop-and-confirm before changing this
engine: one conditional second pass through the same engine enlarges a small or
uncertain full-frame pear proposal. The runtime records both confidences, the
crop, spatial agreement, promotion decision, pass count, and combined latency.
This is an adapter behavior, not a new model artifact.

## Banana specialist router

The general TensorRT model remains resident and handles the first pass for all
fruits. When its requested-class result contains a banana proposal, the media
sidecar routes the same image to the resident banana specialist at
`/media/banana-specialist.pt`. A banana detection is published only when the
specialist reaches `BANANA_SPECIALIST_MIN_CONFIDENCE` and its box overlaps the
general proposal by at least `BANANA_SPECIALIST_MIN_IOU`.

Apple and pear never spend a specialist pass. A missing general banana proposal
does not invoke the specialist, and a specialist rejection returns no detection
so the bounded search continues. Both adapters load once at process startup;
the frame loop never unloads or cold-swaps model files. Route, confidence,
agreement, inference-pass count, and combined latency are included in detection
evidence.

The specialist checkpoint is versioned as `banana-specialist.pt`. Banana
remains camera-only until on-device timing, negative-frame behavior, and a
guarded physical run are independently qualified.

## Planned adapter: Modular MAX and Mojo

TensorRT is temporary. The intended model runtime is Modular MAX, using the
MAX/Mojo stack rather than a TensorRT engine. Keep the camera/perception status
contract runtime-neutral so this migration can replace the media-side model
adapter without changing mission safety, freshness, or detection evidence.
