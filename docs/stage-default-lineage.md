# Stage-default lineage

`codex/stage-default-base-guardrails` is a replace-not-layer branch from the
physically proven `648469f` base checkpoint. That checkpoint completed three
supervised Pear runs and is the motion baseline for this branch.

The active stage path keeps the base's bounded camera search, reliable
factory-avoidance translation, Banana specialist, measured Home pose, terminal
evidence bundle, and both Stagefiles. It adds three deliberately narrow
modules:

- `StageDemo` owns activation, preflight, Home capture, terminal result, and
  exact stop/disarm behind one caller-neutral interface.
- `FruitGuidance` owns one Target Fruit identity from search through Arrival,
  with bounded duplicate hold, one center contract, a measured one-revolution
  search cap, two-sample lower-edge loss confirmation, and one final push.
- `PositionOnlyFailureEpilogue` owns the no-bark down/stand action and one
  capability-gated Home attempt after a non-operator failure.

The default path intentionally excludes LiDAR Arrival, persistent full-frame
tracking, breadcrumbs, multi-source fusion, randomized starting orientation,
and heading restoration as a success gate. Those experiments remain in their
existing worktrees and Git history; they are not hidden switches in the stage
controller.

Release identity: `stage-default-v2-muted-black-box`, Wendy application version
`1.1.1-stage-default`. This iteration adds the muted-except-bark system audio
policy and the append-only per-run black-box trace; physical speaker behavior
and timing remain unqualified until deployment and a supervised zero-motion
audio check.
