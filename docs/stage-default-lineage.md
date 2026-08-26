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

The diagnostic-only derivative is `stage-default-v3-coco-replacement-tester`,
Wendy application version `1.2.0-stage-default`. It preserves the v3 mission
controller while adding an opt-in, rate-limited stock COCO detector to the
camera-only `/fruit-test` page. COCO observations never enter mission evidence
or motion authority.
