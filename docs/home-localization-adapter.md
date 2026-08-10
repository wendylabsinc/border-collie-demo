# Fused Home localization

`HomeLocalizer` treats Go2 SportModeState as the metric-scale source and sparse
camera motion as independent relative-motion evidence. It does not use a Home
marker and it does not convert monocular image translation into metres.

The media process tracks sparse features on frames already decoded for fruit
inference. Processing is CPU-only, rate limited, resolution limited, and
adaptively slowed when its measured processing time exceeds budget. Old frames
are not queued and a media-generation change clears all visual history.

## Visual motion interface

The mission process reads one compact observation:

```python
class VisualOdometryAdapter(Protocol):
    def observe_motion(self) -> VisualOdometryObservation | None: ...
```

The observation includes the media generation, source frame and motion
sequences, capture timestamp, accumulated image-space translation and yaw,
feature/inlier counts, motion quality, and any geometrically verified natural
scene loop closure.

Image translation remains pixels by contract. The Go2 supplies metric distance;
vision currently contributes qualified yaw, motion continuity, and diagnostic
loop-closure evidence. This prevents scale-free monocular motion from becoming
a fabricated Home Distance.

## Trust behavior

- Fresh Go2 pose is always required for a metric Home estimate.
- Missing, initializing, stale, low-quality, or restarted visual odometry falls
  back to fresh Go2 metric pose and is reported explicitly.
- Fresh agreeing visual yaw reduces heading uncertainty and participates in the
  fused heading.
- One visual disagreement is evidence, not an immediate stop. Persistent
  qualified disagreement makes localization unavailable and stops motion.
- Natural-feature loop closure is recorded for trajectory qualification. It
  does not by itself prove the final 0.10 m Home gate.

The visual yaw sign, fusion weight, disagreement gate, feature thresholds, and
CPU budget require physical qualification on Woof before deployment.
