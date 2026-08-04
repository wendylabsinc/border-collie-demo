# Fruit detector dataset research

Research date: 2026-08-04

## Decision

Do not train before measuring the current model on the frozen Go2 evaluation
set defined in [`fruit-model-benchmark-plan.md`](fruit-model-benchmark-plan.md).
The recommended first candidate is a fine-tune, not a model trained from random
initialization:

1. real-scene `Apple`, `Banana`, and `Pear` boxes from Open Images V7;
2. Wendy-owned, labeled Go2 frames from the actual demo environment; and
3. optionally, Fruits-360 crops for appearance diversity or synthetic
   compositing, but not as the primary detection data.

This is the shortest defensible path to a commercially usable detector. Public
data provides appearance variety, while Go2 data provides the camera geometry
and scene distribution that determine whether the demo works.

Licensing notes below are an engineering screen, not legal advice. Preserve a
manifest containing every source image ID, original URL, license URL, author,
annotation source, transformation, and content hash. Exclude any item whose
rights cannot be verified.

## Recommended sources

### 1. Open Images V7 — primary real-scene source

**Recommendation:** use as the main public detection source, subject to an
image-by-image license audit.

- **Provenance:** Google Open Images V7.
- **Coverage:** the official boxable-class file contains `Apple`
  (`/m/014j1m`), `Banana` (`/m/09qck`), and `Pear` (`/m/061_f`). The mapping is
  available through the official [class-description download](https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv).
- **Annotations:** rectangular object boxes. Open Images also contains masks
  for a subset of its vocabulary, but boxes are the verified common annotation
  needed here.
- **Approximate size:** across all 600 box classes, V7 has 14,610,229 training
  boxes on 1,743,042 training images, plus 303,980 validation boxes and 937,327
  test boxes. Exact fruit-subset counts should be computed from the official
  box CSVs before download rather than estimated from a third-party mirror.
  [Official V7 description](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#bounding-boxes)
- **Access:** the official download page provides annotation CSVs, image
  metadata, image URLs, and TFDS access. Filter the box annotations by the
  three MIDs first, then fetch only referenced images.
  [Official V7 download instructions](https://storage.googleapis.com/openimages/web/download_v7.html)
- **License/commercial constraint:** Google licenses annotations under
  CC BY 4.0 and lists images as CC BY 2.0. Google explicitly makes no warranty
  about each image's license and tells users to verify each image themselves.
  A production manifest therefore must retain and validate the per-image
  license and attribution fields; do not treat the dataset-level statement as
  blanket clearance.
  [Official license statement](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#licenses)

Why it fits: it supplies all three classes in varied, cluttered scenes and has
real small-object examples. It is still not camera-matched to Woof, and its
long-tailed class distribution means the three classes must be sampled and
audited explicitly.

### 2. Wendy-owned Go2 camera data — required domain source

**Recommendation:** make this a required part of training and the only source
for the final held-out evaluation set.

- **Provenance:** recordings produced by Wendy Labs with the deployed Go2
  camera and fruit props.
- **Coverage:** banana, red apple, green apple, pear, empty-floor scenes, and
  visually similar negatives such as yellow packaging, balls, leaves, and
  reflections.
- **Annotations:** human-reviewed boxes; masks are optional because the
  application consumes detection geometry.
- **Size:** collect by physical setup and clip, not by extracting thousands of
  nearly identical adjacent frames. Start with hundreds of diverse labeled
  frames per fruit and expand based on benchmark error clusters rather than a
  fixed image-count target.
- **Access/license:** internal capture and labeling pipeline. Record consent or
  ownership for any people, artwork, packaging, or third-party material visible
  in the scene.

Split by complete recording session or physical setup. Adjacent frames from
one clip must never appear on opposite sides of train/validation/test because
that would make the benchmark look much better than deployment.

### 3. Fruits-360 — commercial-friendly auxiliary appearance data

**Recommendation:** use only as auxiliary classification/crop data or for
carefully generated composites.

- **Provenance:** dataset authors Horea Muresan and Mihai Oltean; the original
  author repository documents how each fruit was filmed and extracted from a
  white background.
- **Coverage:** multiple red and green apple varieties, yellow/red/Lady Finger
  bananas, and multiple pear varieties.
- **Annotations:** class labels on single-object 100x100 crops. It does not
  provide authored scene-level boxes or masks. The 103 multi-fruit images are
  a small test set, not a substantial detection set.
- **Approximate size:** 90,483 images across 131 fruit/vegetable classes:
  67,692 train, 22,688 test, and 103 multi-fruit images.
- **Access:** clone or selectively fetch from the authors' repository.
- **License/commercial constraint:** the repository carries the MIT license,
  which permits use, modification, redistribution, sublicensing, and sale when
  its copyright and license notice are preserved.
  [Author repository, dataset description, and license](https://github.com/Horea94/Fruit-Images-Dataset)

Why it is not sufficient alone: nearly all images are centered, tightly
cropped, low-resolution objects on a white background. A detector trained
mainly on this distribution could classify clean crops while still missing the
56x36-pixel banana on Woof's floor-level camera.

## Conditional supplements

### GreenFruitDetector — useful pear data, unresolved commercial provenance

- **Coverage and annotation:** 1,201 Korla pear orchard images captured at
  multiple distances and angles and annotated with minimum bounding
  rectangles. The accompanying work also describes 2,431 annotated green-apple
  images.
- **Access:** the authors link the pear and green-apple archives from the
  official repository.
- **License issue:** the paper is CC BY and the code repository is MIT, but the
  green-apple set combines MinneApple data with web-scraped images. Those
  statements do not establish commercial rights for every underlying image.
  Do not add these archives to the production training corpus until the archive
  itself has source-level rights metadata. The authors' own pear captures are
  the more promising portion to clear.

Sources: [peer-reviewed PLOS ONE data description](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0312164#sec006),
[official repository and download links](https://github.com/1997shangyu/GreenFruitDetector),
and [repository license](https://github.com/1997shangyu/GreenFruitDetector/blob/main/LICENSE).

## Do not use in a commercial production model without additional permission

### MinneApple

MinneApple is high-quality apple research data: the University of Minnesota
describes more than 41,000 polygon-mask instances in 1,000 orchard images, and
its loader can derive boxes from those masks.
[Official project page](https://rsn.umn.edu/MinneApple)

However, the University's dataset record lists
**CC BY-NC-SA 3.0 US**, which restricts use to noncommercial purposes. The MIT
license shown in the companion GitHub repository applies to that repository's
software and is not a replacement for the dataset record's license. Exclude
MinneApple from a commercial training run unless Wendy obtains separate
permission from the rights holder.
[Official repository metadata API](https://conservancy.umn.edu/server/api/pid/find?id=11299%2F206575),
[license text](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).

### Repacked community datasets without image-level provenance

Several small Hugging Face, Kaggle, and Roboflow collections advertise all
three fruit labels and permissive dataset-level licenses. A repackager's label
does not prove that it owns or can relicense the underlying images. Do not use
one unless it supplies original-image provenance, an image-level license, and
the annotation author's rights for every retained item. This is especially
important for web-scraped collections.

## Why the public data cannot replace held-out Go2 frames

The deployed failure is domain-specific: the banana occupied only a small part
of a 1280x720 frame, the camera was near floor height, and confidence changed
with distance, pose, blur, lighting, video compression, and the robot's motion.
Neither orchard imagery nor centered product crops reproduce that joint
distribution.

A public-only test could therefore show strong AP while the robot still fails
to acquire a banana, loses the fruit during approach, or mistakes a yellow
background object for a target. Keep complete Go2 scenes out of training and
use them to compare the current and candidate models on:

- per-fruit precision, recall, AP50, and AP50:95;
- small-object recall and recall by approximate distance;
- false positives on demo-specific hard negatives;
- time to five consecutive qualifying detections;
- detection gaps while a fruit remains visible;
- confidence calibration, not just raw confidence magnitude; and
- median/p95 end-to-end latency on Woof, including crop-confirm behavior.

## Acquisition sequence

1. Freeze and label the Go2 evaluation clips before downloading training data.
2. Benchmark the current engine and save predictions, timings, and thresholds.
3. Produce an Open Images fruit-subset manifest and complete the per-image
   license/attribution audit.
4. Collect separate Go2 training sessions covering the baseline's error
   clusters; do not reuse frozen evaluation setups.
5. Fine-tune one small pretrained detector using Open Images plus Go2 data.
   Treat Fruits-360 as optional auxiliary input and measure whether it helps.
6. Compare on the untouched Go2 set, then export the same winning weights to
   the temporary TensorRT runtime and, later, the planned MAX runtime.

No dataset should be downloaded in bulk until steps 1 and 2 establish what the
current model actually fails on.
