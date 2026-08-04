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

## Planned adapter: Modular MAX and Mojo

TensorRT is temporary. The intended model runtime is Modular MAX, using the
MAX/Mojo stack rather than a TensorRT engine. Keep the camera/perception status
contract runtime-neutral so this migration can replace the media-side model
adapter without changing mission safety, freshness, or detection evidence.
