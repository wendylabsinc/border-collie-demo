# COCO Replacement Tester

Use `/fruit-test` to compare stage props with a stock YOLO11n detector trained
on the complete 80-class COCO label set. The tester is camera-only: its output
never enters Target Fruit evidence, guidance, Arrival, or motion authority.

## Procedure

1. Confirm no Demo Run is active and open `http://woof.local:8110/fruit-test`.
2. Put one candidate prop where Banana normally sits, with the normal lighting
   and viewing distance.
3. Press **Start / reset sample** and leave the prop still for at least ten
   processed samples.
4. Record the top class's all-frame score, detection rate, and maximum.
5. Stop the tester and repeat from a reset sample for each candidate prop.

Prefer the candidate with the highest repeatable **all-frame score**. That score
includes every processed frame and treats a miss as zero, so a prop detected at
90% once and missed nine times scores below one detected around 60% in every
frame. Maximum confidence alone is not a reliability measure.

## Runtime limits

- The tester is off by default and samples at most once every `0.5 s`.
- `COCO_TEST_INTERVAL_S` is measured in seconds, defaults to `0.5`, and must
  remain within `0.1..5.0`.
- The UI confidence floor defaults to `0.05` and accepts `0.01..0.95`.
- Starting or selecting a Demo Run disables the tester before mission evidence
  is evaluated.
- The stock model is downloaded at build time from the pinned Stagefile URL and
  verified by SHA-256; it is loaded only when the tester is enabled.

COCO's fruit labels are `apple`, `banana`, and `orange`. Other food/object
classes are still displayed because the purpose is to find the most stable
stage prop, not to restrict the experiment to botanical fruit.

## Stage candidate identities

The camera-only Mango selector uses COCO's raw `bowl` proposal for its box. It
confirms the staged Mango candidate only when that proposal is small, in the
bounded lower-frame stage band, and contains strong warm/yellow color evidence.
Fewer than 25 useful color samples, less than 20% classified coverage, weak
color evidence, another raw class (including the pear's `sports ball` false
positive), or out-of-band geometry remains `unknown`.

The status retains each proposal's original COCO label, confidence, and
bounding box next to the derived candidate evidence. This is a bounded test of
the current prop and stage placement, not a general botanical Mango-vs-Orange
classifier. Candidate identities remain diagnostic and never enter Demo Run
perception or motion authority.
