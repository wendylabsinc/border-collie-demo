# Media service

This directory contains the single Go2 WebRTC owner for camera frames and bark
audio. It owns fruit inference on a dedicated worker thread and deliberately
imports no Unitree motion client.

The service contract must expose monotonically increasing frame IDs, source
timestamps, connection generations, and explicit freshness health.

## Crop-and-confirm

The existing TensorRT engine performs one full-frame pass. When that pass finds
a fruit candidate at or above 0.35 confidence and the box is small (at most 0.5%
of the frame) or uncertain (below 0.65), the same worker performs exactly one
additional pass over a bounded square crop around that candidate. The crop
defaults to at least 256×256 pixels with six box-widths/heights of context.

The crop result is promoted only when it reaches 0.55 confidence, improves on
the full-frame confidence, and overlaps the original box by at least 0.10 IoU.
Otherwise the original result is retained. Both passes remain serialized on the
dedicated inference worker, and `inference_s` measures their combined time
against the unchanged 0.200-second mission deadline.

The behavior is visible in `/status`, the annotated preview, and evidence
manifests. Its environment controls are:

- `PEAR_CROP_CONFIRM_ENABLED` (default `1`)
- `PEAR_CROP_CONFIRM_MIN_CANDIDATE_CONFIDENCE` (default `0.35`)
- `PEAR_CROP_CONFIRM_UNCERTAIN_BELOW_CONFIDENCE` (default `0.65`)
- `PEAR_CROP_CONFIRM_SMALL_AREA_RATIO` (default `0.005`)
- `PEAR_CROP_CONFIRM_MIN_SIDE_PX` (default `256`)
- `PEAR_CROP_CONFIRM_CONTEXT_SCALE` (default `6.0`)
- `PEAR_CROP_CONFIRM_MIN_CONFIDENCE` (default `0.55`)
- `PEAR_CROP_CONFIRM_MIN_IOU` (default `0.10`)

## Evidence and Fieldmark

- `/api/camera/frame.jpg` is the current annotated operator preview.
- `/api/camera/raw.jpg` is the current unannotated JPEG and can be used as
  Fieldmark's configurable Go2 frame endpoint.
- `/api/evidence/clip.zip` is a rolling evidence archive. Production retains 40
  raw frames at 0.5-second intervals, approximately the latest 20 seconds.

The archive contains `frames/*.jpg` for manual labeling, `manifest.json` with
source markers and recognition statistics, and `terminal/annotated.jpg` for
comparison with the current detector. JPEG and archive generation stay on the
dedicated inference worker or a background thread so the `/status` safety
deadline is not blocked.

`/status.pipeline` exposes the active `baseline` or `throughput-v1` scheduling
profile and its source, processed, dropped, inference-pass, route, odometry,
and preview evidence. The throughput profile publishes source-bound motion
evidence before latest-only visual-odometry and JPEG workers; those auxiliary
workers can drop pending obsolete frames but cannot reorder or refresh motion
evidence.

Fieldmark may capture from
`http://woof.local:8111/api/camera/raw.jpg`, or an operator may extract and
upload `frames/*.jpg` from a failed Run Result. Fieldmark is perception-only;
it is never given a motion API or lease.
