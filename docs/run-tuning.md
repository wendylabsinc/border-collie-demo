# Per-run tuning contract

`RunTuning` is the single interface for values that may vary between supervised
Demo Runs. The server publishes its schema in `GET /api/status` under
`run_tuning`. The audience UI renders controls from that schema and submits one
target-bound object to `POST /api/run`.

The server resolves omitted values, rejects unknown, nonfinite, out-of-range,
contradictory, or target-mismatched values, and creates an immutable snapshot
before selecting a perception target, persisting a run, or permitting motion.
The exact effective snapshot is stored as `run_tuning` in the Run Result and in
the black-box `run_started` event. Changing UI values never mutates an active
run and does not mutate process environment or require a restart/deployment.

## Tunable groups

The authoritative defaults, units, ranges, step sizes, and safety notes are
machine-readable in `RunTuning.contract()` rather than repeated here:

- `search`: yaw rate, bounded sweep, timeout, and the rollbackable progressive
  focused-yaw profile;
- `recognition`: selected-fruit focus/lock/tracking confidence and required
  fresh centered frames;
- `centering`: lock band, moving corridor, moving yaw, and in-place recenter
  yaw;
- `approach`: forward speed, timeout, duplicate hold, source freshness, and
  detection freshness;
- `arrival`: visual near gates, confirmation/loss policy, and the one bounded
  final push;
- `home`: initial alignment, return translation/steering, progress, stall, and
  completion tolerances.

The focused-yaw profile is consumed directly by the mission-lifetime guidance
module. Disabling `progressive_focus_yaw_enabled` restores fixed
`focus_yaw_rps`; restoring `lock_tolerance_ratio` to `0.08` completes the prior
lock profile without rebuilding or reverting source.

## Non-tunable safety invariants

Runtime tuning cannot weaken single-owner motion leases, the command watchdog,
absolute hardware motion envelopes, exact-zero disarm, inter-run Home
clearance, or immediate stops for stale evidence, wrong target/generation, and
camera failure. Freshness settings may only tighten the deployed hard ceilings
of 0.350 seconds for source evidence and 0.250 seconds for detections.

## Cohort integration

Cohort policy is separate from mission tuning. A cohort request may carry a
`tuning` template, but the cohort controller must call
`RunTuning.from_payload(fruit, template)` separately for every scheduled fruit
and pass that new snapshot to `FruitMission(tuning=...)`. It must never store
mission tuning inside `CohortPolicy` or reuse a target-bound object for a
different randomized fruit.
