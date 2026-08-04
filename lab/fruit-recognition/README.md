# Read-only fruit recognition

Use `/fruit-test` to evaluate additional fruit classes without exposing motion
controls. A test passes its provisional recognition gate when the box is on the
requested fruit and the configured confidence threshold is sustained for at
least five consecutive fresh detections.

Results and their annotated camera snapshots live in `results/`. Passing this
camera-only gate does not qualify a fruit for autonomous approach or the full
demo.

`benchmark.py` compares a human-reviewed evidence archive with the model
predictions embedded in its manifest. Acquisition recall and localization are
reported separately so a correctly located but under-confident fruit remains a
visible failure.

Frozen comparison scenes are recorded in `evaluation/`. Their raw archives
live under `artifacts/fruit-model/evaluation/` and stay outside Git. Every
capture marked `training_use_allowed: false` is evaluation-only: do not use its
frames for training, augmentation, threshold selection, or synthetic data.

To review a captured scene with the local labeler:

```bash
python3 lab/run-labeler/server.py \
  --archive artifacts/fruit-model/evaluation/BANANA-EVAL-001/evidence.zip \
  --class-name banana
```
