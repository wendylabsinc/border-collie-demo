# Hardware lab

Human-operated camera, fruit, motion, and return-home tests belong here. Lab
code must never be imported by `src/border_collie_demo` or included in the
production startup path. Each completed hardware test will emit a compact JSON
result and a visual snapshot.

- [`run-labeler`](run-labeler/README.md) opens a recorded run directly for
  long-distance fruit-box correction and training-label export.
- [`tensorrt-export`](tensorrt-export/README.md) builds the TensorRT FP16
  engine for `apple-pear-mango.pt` on Woof and qualifies it against the `.pt`
  on the committed reference frame. Camera-free and motion-free.
