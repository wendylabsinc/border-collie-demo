# Pear model runtime

## Current adapter: TensorRT

Provision the validated `collie-fruit-yoloe11m.engine` as
`/media/model.engine` in the media container. The engine is a hardware-specific
deployment artifact and is intentionally not committed to Git.

The current local validated copy is under
`lab/pear-evidence/model/collie-fruit-yoloe11m.engine`. Container assembly must
copy that exact file only after DLO captures the pre-change build baseline.

Long-distance detection currently uses crop-and-confirm before changing this
engine: one conditional second pass through the same engine enlarges a small or
uncertain full-frame pear proposal. The runtime records both confidences, the
crop, spatial agreement, promotion decision, pass count, and combined latency.
This is an adapter behavior, not a new model artifact.

## Banana specialist router

The general TensorRT model remains resident and handles the first pass for all
fruits. When its requested-class result contains a banana proposal, the media
sidecar routes the same image to the resident banana specialist at
`/media/banana-specialist.pt`. A banana detection is published only when the
specialist reaches `BANANA_SPECIALIST_MIN_CONFIDENCE` and its box overlaps the
general proposal by at least `BANANA_SPECIALIST_MIN_IOU`.

Apple and pear never spend a specialist pass. A missing general banana proposal
does not invoke the specialist, and a specialist rejection returns no detection
so the bounded search continues. Both adapters load once at process startup;
the frame loop never unloads or cold-swaps model files. Route, confidence,
agreement, inference-pass count, and combined latency are included in detection
evidence.

The specialist checkpoint is an ignored experimental artifact. Banana remains
camera-only until on-device timing, negative-frame behavior, and a guarded
physical run are independently qualified.

## Candidate adapter: Modular MAX and Mojo

TensorRT remains the production default, but the MAX migration now has a
concrete candidate. `MAX-NATIVE-006` trained a 416px YOLO11n fruit detector
with 2.58 million parameters. On the same 107-image held-out split it reached
mAP50 0.775 and mAP50-95 0.620, compared with 0.843 and 0.657 for the 9.41
million-parameter YOLO11s reference. Its exported ONNX graph has no operators
unsupported by the checked-in MAX importer and matches the PyTorch output to a
mean absolute difference of 2.8e-5.

Those local results authorize only a guarded MAX compile and camera-only
benchmark on Woof. They do not authorize autonomous motion. Before this model
can replace TensorRT, the native `sm_87` artifact must pass PTX-JIT-disabled
execution, fixed-corpus output parity, live-camera latency, memory, thermal,
freshness, and false-positive gates. The full record is
[`training/results/MAX-NATIVE-006.json`](../../training/results/MAX-NATIVE-006.json).

Keep the camera/perception status contract runtime-neutral so the MAX/Mojo
adapter can replace the media-side inference implementation without changing
mission safety, freshness, or detection evidence.
