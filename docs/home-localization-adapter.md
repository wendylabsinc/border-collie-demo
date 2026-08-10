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

Image translation remains pixels by contract. An essential-matrix estimate may
provide scale-free body direction and yaw when enough geometry survives RANSAC.
The Go2 pose delta supplies metric distance; the visual direction cannot create
or enlarge that distance.

## Stateful planar fusion

`PlanarSensorFusion` maintains Home-relative position, heading, planar velocity,
and gyroscope yaw bias with explicit covariance. Its interface is only
`reset(observation)` and `update(observation)`.

Each advancing Go2 sample performs:

1. prediction from velocity and IMU yaw rate;
2. innovation-gated Go2 position and heading updates;
3. body-velocity conversion into the Home frame;
4. a zero-velocity and gyro-bias update when at least three qualified feet are
   loaded and reported velocity/yaw rate are stationary; and
5. qualified scale-free visual direction/yaw updates, using Go2 displacement
   magnitude as the translation scale.

Non-advancing timestamps with changed poses fail immediately. A bounded number
of metric innovations may be rejected; persistent rejection makes localization
unavailable. A long observation gap can re-seed only while a qualified
stationary stance is observed.

## Trust behavior

- Fresh Go2 pose is always required for a metric Home estimate.
- Missing, initializing, stale, low-quality, or restarted visual odometry falls
  back to fresh Go2 metric pose and is reported explicitly.
- Fresh agreeing scale-free visual direction and yaw reduce uncertainty and
  participate in the fused state.
- One visual disagreement is evidence, not an immediate stop. Persistent
  qualified disagreement makes localization unavailable and stops motion.
- Natural-feature loop closure is recorded for trajectory qualification. It
  does not by itself prove the final 0.10 m Home gate.
- The final metric gate uses the farther of raw Go2 distance and filtered
  distance. Filtering therefore cannot turn an outside-10-cm raw pose into a
  successful arrival.

Camera field of view/extrinsics, foot-force threshold, process/measurement
variances, innovation gates, and CPU budget require physical qualification on
Woof before deployment.
