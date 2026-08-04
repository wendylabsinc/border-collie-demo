# Fruit model benchmark and training gate

The current TensorRT model must be measured on a frozen, labeled validation
set before a replacement model is trained. Live confidence from one placement
is useful diagnostic evidence, but it is not an accuracy benchmark.

## Phase 0: deployed baseline

`BANANA-BASELINE-001` records the first no-motion baseline at the current
banana placement. It contains 159 distinct frames. Median confidence was 0.199,
79 frames reached the provisional 0.20 gate, and none reached the 0.35
crop-confirm trigger. Median inference was 86 ms and p95 was 135 ms.

## Phase 1: freeze evaluation data

Capture evaluation clips with the Go2 camera before collecting training data.
Keep every evaluation scene out of training and augmentation inputs. Group the
split by complete clip or physical setup, never by adjacent frames, so nearly
identical video frames cannot leak across train and validation.

The initial evaluation set should contain:

- banana, red apple, green apple, and pear;
- small, medium, and large image-area buckets;
- center, left, and right image positions;
- multiple orientations, backgrounds, and lighting conditions;
- partially occluded fruit; and
- hard negatives such as yellow packaging, balls, leaves, and empty floor.

Each retained frame needs a human-reviewed bounding box and class. Record the
camera resolution, clip/setup identifier, approximate distance when available,
and whether the frame is inside the intended stage Operating Envelope.

## Phase 2: benchmark the current model

Run the current engine over the frozen evaluation set and record:

- per-class precision, recall, AP50, and AP50:95;
- recall by image-area bucket and scene/setup;
- false positives per negative frame;
- time to five consecutive qualifying detections on clips;
- detection-gap duration while the fruit remains visible;
- median, p95, and maximum inference time;
- model artifact size and runtime memory; and
- crop pass frequency, promotion rate, and combined latency.

Thresholds must be selected from precision-recall evidence. Confidence values
from different models are not directly comparable and should not be optimized
in isolation.

## Phase 3: build training data

Combine a license-compatible public fruit dataset with new Go2-camera images.
Public images supply appearance diversity; Go2 images supply the low camera
angle, floor backgrounds, small-object scale, blur, lighting, and perspective
that determine stage performance. Deduplicate near-identical video frames and
include hard negatives.

Keep dataset manifests, source licenses, transformations, class mappings, and
content hashes with each training run. Raw third-party data and large model
artifacts should remain outside Git.

## Phase 4: train and compare

Start from pretrained weights and fine-tune a three- or four-class detector.
Training is runtime-neutral: retain an interchange format suitable for the
planned Modular MAX adapter, then export a temporary TensorRT candidate only
for an equivalent Woof comparison.

A candidate may replace the current model only when:

- it improves the frozen-set banana recall and clip acquisition reliability;
- it does not materially regress qualified pear or red-apple results;
- false positives remain inside the selected precision gate;
- p95 end-to-end inference remains below the 0.200-second mission deadline;
- camera freshness and event-loop responsiveness tests pass; and
- a supervised no-motion fruit test succeeds before any autonomous run.

After those gates, qualify one fruit at a time through the complete Demo Run.

## Recorded baselines

- [`BANANA-BASELINE-001`](../lab/fruit-recognition/results/BANANA-BASELINE-001.json)

The public-dataset survey and licensing decision are recorded separately in
`fruit-model-dataset-research.md`.
