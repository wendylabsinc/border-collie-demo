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

Release identity at the branch point: `stage-default-v2-muted-black-box`, Wendy
application version `1.1.1-stage-default`. This derived candidate removes the
muted-except-bark startup dependency: direct bark is best effort and cannot
block readiness, posture cleanup, or Home. It also makes the final push a
frozen one-run UI tuning and separates Home alignment (regular Sports,
yaw-only) from Home translation (factory avoidance, forward plus bounded yaw).
The integrated candidate is `stage-default-v3-runtime-tuning-cohorts`, Wendy
application version `1.1.2-stage-default`. It adds the immutable per-run tuning
interface and independently gated cohort controller. None of these changes is
physically qualified until its first supervised run completes.

The derived software candidate is
`stage-default-v4-banana-reliability-replay`, Wendy application version
`1.1.3-stage-default`. It keeps the 250 ms motion-evidence ceiling while adding
a stopped 500 ms terminal grace for otherwise-valid slow inference, persistent
fine pre-lock focus, exact inference timing evidence, and post-disarm
position-only reconciliation of a near-Home heading-gate escape. Its three
minimized physical traces are deterministic software replays; this candidate
is not physically qualified or deployed by that replay result.

The next software-only candidate is `stage-default-v5-fruit-bearing-map`, Wendy
application version `1.1.4-stage-default`. The media general-model pass now
reports raw best-per-fruit observations while preserving selected `detection`
as the only motion-consumed interface. A process-local app map records only
observations inside the existing fine-focus corridor (within 12% of image
center) against the exact processed camera frame and measured body yaw; it may
select a shortest Sports yaw-only turn, but normal selected-target guidance
must still reacquire and alone can authorize translation or Arrival. Map reuse
requires unchanged camera generation and odometry epoch. Later iterations use
the new run's freshly captured Home heading to calculate the shortest signed
turn and retain Home position offset as diagnostic evidence rather than a map
invalidation. This changes the shared media/app status contract, so it requires
a whole-project deployment; the voice command path is otherwise unchanged.

The runtime/UI A/B switch is `BORDER_COLLIE_BEARING_ROUTING_ENABLED` and the
frozen per-run `search.bearing_routing_enabled`, both defaulting to false.
All-fruit mapping remains active when false; only the initial mapped turn is
disabled. This permits a baseline broad scan followed by a treatment run in the
same app process without rebuilding or restarting services.

The observability-only candidate is `stage-default-v9-operator-logs`, Wendy
application version `1.1.8-stage-default`. It projects selected durable
black-box events into compact INFO JSON for operators and suppresses routine
app/media HTTP access lines. It does not change mission policy, perception,
motion commands, Home gates, or stored evidence.

The passive-diagnostics successor is `stage-default-v10-pose-drift-recorder`,
Wendy application version `1.1.9-stage-default`. It retains every fresh DDS
pose from Home capture through the terminal result and emits one
`DRIFT DETECTED` warning when a yaw-only episode translates by the configured
distance. The recorder has no command writer and the alert never changes motion.

The rollbackable focus-lock experiment is
`stage-default-v11-progressive-lock-yaw`, Wendy application version
`1.1.10-stage-default`. It narrows the lock corridor from `+/-8%` to `+/-5%`
and reduces agreeing fresh focused-yaw passes from `0.40` to `0.30` to
`0.20 rad/s`. Per-run tuning can disable the progressive profile and restore
the prior `0.08` corridor without reverting source.
