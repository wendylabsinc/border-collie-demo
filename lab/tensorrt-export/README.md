# TensorRT FP16 export for apple-pear-mango

A one-shot, read-only Wendy app. It builds a TensorRT FP16 engine from
`media/model/apple-pear-mango.pt`, qualifies it against the `.pt` on the
committed reference frame, and serves the result over HTTP. It imports no
motion client and exposes no robot-control endpoint.

**The engine must be built on Woof.** A TensorRT engine is specific to the GPU
and the TensorRT version that built it; an engine built anywhere else will not
deserialize in the media container. The Dockerfile here uses the same
`dustynv/pytorch:2.7-r36.4.0-cu128-24.04` base as `media/`, so the engine is
built against the runtime that will load it.

## Run it

Stage the two inputs into this build context (they are versioned under
`media/model/`, but that path is outside this Docker build context):

```sh
mkdir -p lab/tensorrt-export/model
cp media/model/apple-pear-mango.pt \
   media/model/apple-pear-mango-reference.jpg \
   lab/tensorrt-export/model/
```

Then, with Woof idle and cool — the export saturates the GPU for many minutes
and this device throttles:

```sh
wendy run --prefix lab/tensorrt-export --device woof.local -y
```

Watch for `TENSORRT_EXPORT_RESULT=PASS`. The export itself is the slow part;
expect tens of minutes and high memory pressure. On failure the report lists
every reason under `comparison.failures`.

## Collect the artifacts

```sh
curl --fail --output media/model/apple-pear-mango.engine \
  http://woof.local:8124/apple-pear-mango.engine
curl --fail --output lab/tensorrt-export/results/TENSORRT-EXPORT-001.json \
  http://woof.local:8124/tensorrt-export-result.json
wendy device apps stop border-collie-tensorrt-export --device woof.local
```

## What "equivalent" means here

The report runs both artifacts over `apple-pear-mango-reference.jpg` at
`conf=0.05`, pairs each `.pt` detection with the best same-label engine
detection by IoU, and gates on the three strong detections the checkpoint
produces:

| label | `.pt` confidence |
| ----- | ---------------- |
| pear  | 0.960            |
| apple | 0.957            |
| mango | 0.901            |

Each must reappear in the engine within `EXPORT_CONFIDENCE_TOLERANCE` (0.05)
and at `EXPORT_IOU_FLOOR` (0.90) box overlap. FP16 moves confidences by a few
thousandths; a swing larger than that means the export lost something.

The `.pt` also yields a weak 0.212 mango on a yellow object. That one is
reported under `comparison.pairs` but not gated, because a proposal that far
below the acquisition thresholds is not evidence either way. Read it as a
smoke signal: if it vanishes entirely, or climbs above the mango acquisition
confidence, look harder before shipping the engine.

The report additionally records `engine.classes`. It must be exactly three
entries — `apple`, `pear`, `mango`. If it shows 35, the engine was loaded
without `task="segment"`; see below.

## The task="segment" gotcha

A serialized engine loses its task metadata. Ultralytics then decodes the 39
output channels as 4 box + 35 classes instead of 4 box + 3 classes + 32 mask
coefficients, which surfaces as `KeyError: 23`. Every load of a non-`.pt`
artifact must pass the task explicitly:

```python
YOLO('apple-pear-mango.engine', task='segment')
```

`media/perception_sidecar.py` already does this for `PEAR_MODEL_PATH`, so no
runtime code change is needed to adopt the engine.

## Switching the runtime over

Leave the default on the `.pt` until the engine is qualified. When it is, the
switch is configuration only:

1. Add the engine to the media image in `media/build.stagefile.yaml`:

   ```yaml
   - from: local
     paths: [model/apple-pear-mango.engine]
     dest: /media/apple-pear-mango.engine
   ```

2. Point `PEAR_MODEL_PATH` at it in `wendy.json` (media service):

   ```json
   "PEAR_MODEL_PATH": "/media/apple-pear-mango.engine"
   ```

Reverting is the same two lines back. Nothing else in the runtime knows the
difference.
