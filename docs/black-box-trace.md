# Demo Run black-box trace

Every terminal Demo Run retains one run-filtered `run-trace.ndjson` beside its
materialized `result.json`. The trace is derived from the fsynced rolling flight
recorder, keeps each event's original global sequence and SHA-256 fields, and is
available at:

```text
GET /api/results/{run_id}/trace
```

The trace is retained for both completed and failed runs. Failed-run Home
recovery refreshes the same file after the recovery attempt, so its motion tail
is not omitted. Trace capture failure cannot mask stop/disarm or change the Run
Result outcome; the result records `black_box_trace.available: false` and the
unavailable reason.

The Home diagnostics include two complementary 5 Hz/decision streams:

- `home_fusion_sample`: captured Home, raw `SportModeState` pose and motion,
  fused Home-relative pose and distance, covariance, velocities, yaw bias,
  stationary evidence, trust state, and rejection reason.
- `home_navigation_sample`: the exact raw and fused evidence consumed by the
  return planner, current and target poses, Home or breadcrumb target kind,
  chosen mode, distance, heading error, forward/yaw commands, pulse accounting,
  progress timer, and active arrival/heading/stall thresholds.

Other run-owned lifecycle, perception, authority, and motion-command events are
retained in timestamp order. This artifact is intended to answer a failure from
one download; the larger rolling recorder remains the cross-run/process source.
