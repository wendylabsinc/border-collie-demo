# MAX importer tournament

This lab rejects inaccurate model/runtime candidates before spending Woof time
on `sm_87` compilation or profiling. It is isolated from the production demo and
contains no motion code.

The first corpus is fixed at 416 px and contains nine held-out public images,
three human-validated Go2 fruit frames, and one visually reviewed negative Go2
frame. Every source file is SHA-256 checked before inference. The ignored public
dataset and banana evidence artifacts must already exist locally; the manifest
fails closed if an image is missing or changed.

Run the local source-versus-ONNX accuracy and raw-tensor parity gate:

```bash
.venv-training/bin/python -m training.max_tournament.run \
  --corpus lab/max-importer-tournament/corpus.json \
  --variants lab/max-importer-tournament/variants.json \
  --output lab/max-importer-tournament/results/local-accuracy-baseline.json
```

Add `--require-accepted` to make rejected non-reference candidates return a
non-zero exit status. That is the tight regression signal for the current
accuracy problem.

Local CPU timing is advisory only. A candidate that passes local accuracy must
next be run on the exact same corpus through production TensorRT and through a
MAX FP16 native-`sm_87` artifact. The Woof gate must add raw/decoded TensorRT
parity, CPU-fallback rejection, Nsight convolution classification, memory,
temperature, camera freshness, and `/status` responsiveness.

The current MAX/YOLO11n work is expected to be rejected even when ONNX matches
PyTorch numerically: the source nano checkpoint itself misses or misclassifies
stage-critical fruit frames. Runtime parity and real-world accuracy are separate
columns and both must pass.

Checkpoint selection and importer parity are also separate. A checkpoint
candidate may intentionally differ from the current source checkpoint. Once a
checkpoint is selected, every ONNX/MAX runtime must name that exact PyTorch
checkpoint as its `parity_reference` and match it.

## First local result

The first run tested four ranked explanations for the live pear failure:

1. The trained YOLO11n checkpoint is inaccurate on the stage scene. Prediction:
   PyTorch and ONNX will agree on the wrong class.
2. The ONNX export changes the model. Prediction: raw PyTorch/ONNX tensors will
   diverge before decoding.
3. Preprocessing or class order is inconsistent. Prediction: fixed RGB/NCHW
   preprocessing or the explicit `pear, apple, banana` mapping will change the
   winning class.
4. The confidence threshold alone hides a correct pear proposal. Prediction: a
   diagnostic 0.01 threshold will rank pear first even if it does not qualify.

The first explanation was confirmed. PyTorch and ONNX both produced 50% class
recall, 58.33% IoU-0.5 box recall, and 0/3 stage-critical Go2 fixture passes.
The Go2 pear frame ranked banana first at 0.166; a held-out pear ranked apple at
0.269. Lowering the threshold exposed the wrong class rather than a hidden
correct pear.

All 13 raw ONNX outputs passed parity against PyTorch. Across the corpus, the
largest mean absolute difference was 0.0000471, p99 difference was 0.000549,
and absolute difference was 0.00728. This falsifies the ONNX export as the
primary accuracy cause for this checkpoint.

| Variant | Class recall | Box recall @ 0.5 | Go2 stage fixtures | Raw parity | Verdict |
| --- | ---: | ---: | ---: | --- | --- |
| PyTorch YOLO11n source | 50.0% | 58.3% | 0/3 | reference | Rejected |
| ONNX MAX source graph | 50.0% | 58.3% | 0/3 | Pass | Rejected |
| TensorRT production | Pending frozen-corpus capture | Pending | Pending | Pending | Pending |

The generated per-frame evidence and leaderboard are in
`results/local-accuracy-baseline.json`. No MAX importer using these same weights
may advance to Woof merely because it achieves tensor parity; the source model
already fails the local accuracy gate.
