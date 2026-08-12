# Tracking Confidence Average

Release `stage-camera-v32-confidence-average` keeps acquisition conservative:
the Target Fruit must still satisfy its existing current-frame confidence and
consecutive-frame requirements before the track is acquired. After acquisition,
confidence safety authority is the arithmetic mean of the latest fresh,
advancing, same-fruit, same-generation observations.

`BORDER_COLLIE_TRACKING_CONFIDENCE_WINDOW_FRAMES` configures the maximum number
of observations in that window.

- Unit: frames.
- Default: `10`.
- Valid range: `1` through `120`, inclusive.
- Reload behavior: read at app process startup; restart the app process after a
  change.
- Safety contract: the average is the only confidence floor after acquisition;
  there is no separate raw per-frame confidence floor. Missing or stale
  evidence, wrong labels or generations, unhealthy camera state, and invalid
  geometry remain independent immediate stops. Duplicate frames do not enter
  the average. A confidence-qualified frame still cannot advance Arrival unless
  the separate fresh geometry and Arrival contracts pass.

The physical v31 Pear trace ended on raw confidence `0.5253688693`, below the
Pear tracking floor `0.55`. Its ten-frame arithmetic mean was `0.6002081841`.
Replaying those exact fresh observations through v32 keeps the track in
`SLOW / close_range_steering` instead of producing
`STOP / target_lost_off_axis`.
