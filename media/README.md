# Media service

This directory contains the single Go2 WebRTC owner for camera frames and bark
audio. It owns pear inference on a dedicated worker thread and deliberately
imports no Unitree motion client.

The service contract must expose monotonically increasing frame IDs, source
timestamps, connection generations, and explicit freshness health.

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

Fieldmark may capture from
`http://woof.local:8111/api/camera/raw.jpg`, or an operator may extract and
upload `frames/*.jpg` from a failed Run Result. Fieldmark is perception-only;
it is never given a motion API or lease.
