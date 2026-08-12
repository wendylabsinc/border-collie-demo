# Perception throughput experiment

Release `stage-camera-v23-perception-throughput` isolates scheduling changes
from fruit-policy changes. It does not change confidence floors, camera
generation fencing, source freshness, the 200 ms detector limit, the 250 ms
detection-age limit, geometry requirements, or the raw weak-detection publish
contract.

## Baseline

Read-only measurements on Woof on 2026-08-11, before this branch was deployed:

- WebRTC source: 14.285 FPS at 1280 x 720.
- Completed detector frames: 5.665 FPS.
- Visual odometry: 5.665 FPS with zero rate-limit skips, because it received a
  frame only after serial detector completion.
- Visual-odometry processing: 20.98 ms average against a 30 ms budget.
- Recent non-null detection inference: 125 ms median and 136 ms p90.

The source delivered 116 frames while the detector completed 46 in the same
8.12-second interval. The capacity-one queue correctly discarded obsolete
frames, but Woof received new motion evidence only every 176 ms on average.

## Profiles

`PERCEPTION_PIPELINE_PROFILE=baseline` preserves the prior ordering:

```text
latest frame -> inference -> visual odometry -> publish -> preview/JPEG
```

`PERCEPTION_PIPELINE_PROFILE=throughput-v1` uses independent latest-only
auxiliary lanes:

```text
latest frame -> inference -> publish motion evidence
                         \-> latest-only visual odometry
                         \-> latest-only preview/evidence JPEG
```

For a missing apple or banana, `throughput-v1` schedules one full-frame pass
and then one lower-center crop pass across fresh advancing frames. A crop
candidate keeps the normalized crop route across subsequent fresh frames so
its consecutive acquisition evidence is not reset by interleaved full-frame
misses; losing it returns to full-frame search. Baseline continues to run both
passes against one frame. A small uncertain proposal
retains the existing bounded same-frame confirmation pass in this first
experiment; changing that safety behavior belongs in a separately replayed
iteration.

Every lane is capacity one. Busy workers replace pending obsolete work rather
than building latency. `/status.pipeline` reports source/processed/drop
counts, inference passes and timing, route counts, last published PTS, and
independent visual-odometry and preview completion/drop/error metrics.

## Acceptance and rollback

Compare both profiles under identical fruit placement, target sequence, camera
generation, and run policy. Qualify `throughput-v1` only if all of these hold:

- source delivery remains at least 14 FPS;
- motion evidence reaches at least 10 FPS for apple, banana, and pear search
  and approach routes;
- inference p95 is below 100 ms and no sample exceeds the existing 200 ms
  safety limit;
- source-to-published-evidence p95 is below 150 ms;
- confidence/geometry traces and terminal failure classifications do not lose
  recall or false-positive rejection;
- visual odometry remains fresh through the posture sequence; and
- the supervised fruit and Home behavior is no worse than baseline.

Rollback requires no source rewrite: select `baseline` in the media service
environment and redeploy with the normal Stagefile-aware `wendy run --detach`
path. The preceding v22 branch remains an additional Git rollback point.

## Banana TensorRT artifact

The general fruit model is already a TensorRT engine. The banana specialist
remains `banana-specialist.pt` in this checkpoint because TensorRT engines are
platform-specific and must be exported and benchmarked on the Jetson runtime.
The existing `BANANA_SPECIALIST_MODEL_PATH` accepts an Ultralytics-compatible
`.engine` path without a code change. Do not replace the pinned `.pt` artifact
until the engine has matching detection/IoU output and improves p95 latency on
Woof.
