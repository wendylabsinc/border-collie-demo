# Fruit model training

Training data and model artifacts live under `artifacts/fruit-model/` and are
never committed. The repository keeps only reproducible preparation code,
source manifests, licenses, benchmark summaries, and small configuration files.

The first public source is a license-audited subset of Open Images V7 containing
Apple, Banana, and Pear boxes. Open Images records are a candidate pool, not
training data, until they pass `whole-fruit-policy.md`. The final evaluation set
consists of separate, frozen Go2 camera sessions.

Prepare the public source after downloading the official annotation and image
metadata CSV files described in `docs/fruit-model-dataset-research.md`:

```bash
python3 training/prepare_open_images.py \
  --annotations artifacts/fruit-model/sources/open-images-v7/validation-annotations-bbox.csv \
  --metadata artifacts/fruit-model/sources/open-images-v7/validation-images-with-rotation.csv \
  --output artifacts/fruit-model/candidates/open-images-v7
```

The output manifest retains attribution, license URLs, original URLs, boxes,
and content hashes. It explicitly marks the records as training-ineligible
until curation is complete. A future training run must combine the curated
public pool with new Go2 training sessions. It must not read anything under
`artifacts/fruit-model/evaluation/`.

Open Images `GroupOf` and `Depiction` boxes are excluded. They are useful for
the source dataset's evaluation semantics but are not tight physical-fruit
targets for this detector.

Rank candidate crops for whole-fruit review with:

```bash
.venv-training/bin/python training/rank_whole_fruit.py \
  --manifest artifacts/fruit-model/candidates/open-images-v7/manifest.json \
  --output artifacts/fruit-model/candidates/open-images-v7/whole-fruit-ranking.json
```

The vision-language score only orders the review queue. It never makes a crop
training-eligible by itself.

Apply exported review decisions with `apply_whole_fruit_review.py`. The script
withholds any image containing another target class or an unfinished fruit-box
decision, preventing unlabeled fruit from becoming accidental background.

## BANANA-001 local result

`BANANA-001` is the first banana-only experiment. Its selected local checkpoint
was trained from 72 whole-fruit positive images plus 21 reviewed cut, food, or
bunch hard negatives. The frozen 40-frame Go2 evaluation is excluded from
training.

The stable fine-tune acquired the banana in 40 of 40 frozen frames at the demo's
0.20 confidence and 0.50 IoU gate, with 0.784 mean confidence. The current
deployed detector acquired 28 of 40 frames with 0.215 mean confidence. This is
promising local evidence, not production approval: the evaluation has no
negative frames, the checkpoint is banana-only, and it has not been benchmarked
on Woof.

The reproducible experiment metadata is in `experiments/BANANA-001.json`; the
small aggregate comparison is in `results/BANANA-001.json`. Model weights,
images, and run output remain ignored under `artifacts/fruit-model/`.
