# Absolute Home observation adapter

`HomeLocalizer` is the SDK-neutral authority for Home estimates. It combines
fresh short-term Go2 odometry with an optional absolute observation, rejects
stale or contradictory evidence, and returns either `trusted` or `unavailable`.
Motion and recovery code must not read a camera SDK or fiducial detector
directly.

## Adapter interface

A future AprilTag, ArUco, or equivalent producer implements one method:

```python
class AbsoluteHomeObservationAdapter(Protocol):
    def observe_home(self, home: Pose2D) -> AbsoluteHomeObservation | None: ...
```

The method must be non-blocking and return the newest observation available at
the time of the call. `None` means no absolute observation is currently
available. An exception means the adapter is unhealthy and makes the combined
Home estimate unavailable.

`AbsoluteHomeObservation.pose_from_home` is the robot pose in the coordinate
frame established when Home was captured:

- Home is `(0, 0, 0)`.
- Positive x follows the captured Home heading.
- Positive y points left from that heading.
- Yaw is relative to the captured Home heading.

The adapter owns camera calibration, camera-to-body extrinsics, fiducial pose
solving, and transformation into this frame. It must bind an observation to the
same physical marker and Home calibration used for the current Demo Run.
`reference_id` identifies that marker or calibrated reference.

`captured_monotonic_s` must use the application process's monotonic clock. It is
the time of image acquisition, not detector completion or publication. The
adapter must never refresh this timestamp when replaying an older detection.

## Trust behavior

- Without an adapter, fresh Go2 odometry preserves current behavior and the
  evidence reports `absolute.state = not_configured`.
- With an adapter but no current observation, fresh odometry remains trusted
  and reports `absolute.state = not_observed`.
- A fresh, agreeing observation corrects odometry with the configured absolute
  weight.
- A stale, future-dated, malformed, or contradictory observation makes Home
  unavailable. It is not allowed to silently replace odometry or fall back to
  an apparently healthy distance.
- Both raw measurements, their disagreement, thresholds, fusion weight, source,
  and reference ID are retained in Home-localization evidence.

The initial software thresholds are conservative defaults, not a physically
qualified Operating Envelope. Physical integration still requires marker
selection and placement, calibration, occlusion and lighting tests, measured
latency, disagreement-threshold qualification, and a new return-to-Home soak.
