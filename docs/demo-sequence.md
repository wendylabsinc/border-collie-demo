# Demo sequence and safety states

The production sequence is:

```text
IDLE -> PREFLIGHT -> CAPTURE_HOME -> WAIT_FOR_COMMAND
     -> [TURN_TO_FRUIT -> FIND_FRUIT] -> APPROACH_FRUIT -> ARRIVED
     -> SIT_AND_BARK -> STAND -> TURN_TOWARD_HOME -> RETURN_HOME
     -> RESTORE_HEADING -> COMPLETE
```

Square brackets mark the **conditional initial search**. `TURN_TO_FRUIT` and
`FIND_FRUIT` remain in every Run Result for auditability, but when the selected
Target Fruit is already freshly qualified they are recorded as skipped and no
search rotation is sent. They execute normally only when that qualification is
not already present.

`STOPPED`, `FAILED`, and `REMOTE_TAKEOVER` are off-ramps from active work.

For the qualified-fruit milestone, facing the person is an operator setup
condition established before activation. Autonomous person detection is not
required and the application must not silently adjust the captured Home pose.

Preflight records explicit checks for durable Run Result storage, connected
hardware, the autonomous-motion gate, fresh pose, disarmed application motion,
and an advancing healthy camera/perception source. Bark status is observable
but is not a motion-readiness gate. Target Fruit
evidence is intentionally not an activation gate. The initial search is
conditional: at the start of both search stages, fresh qualified evidence for
the selected Target Fruit records
`target_already_visible` and skips search without arming motion. Otherwise,
`TURN_TO_FRUIT` rotates through at most one measured revolution until fresh
qualified fruit evidence stops the turn, and `FIND_FRUIT` confirms or briefly
reacquires that evidence before approach. Completing the bounded search with healthy,
advancing frames but no qualified Target Fruit records `TARGET_RECOGNITION_FAILURE`
with the strongest candidate statistics and a downloadable raw-frame evidence
archive for Fieldmark labeling. Any failed or errored preflight check seals the Demo Run as
`FAILED` with reason `PREFLIGHT_FAILURE`; only an all-ready report may advance
to `CAPTURE_HOME`. Home capture then reads a new fresh, disarmed
`rt/sportmodestate` pose, persists its position and heading, and advances to
`WAIT_FOR_COMMAND`. If pose freshness is lost before capture completes, the run
fails closed in `capture_home` without arming motion.

A frozen, repeated, disconnected, or otherwise stale camera frame must stop
motion immediately and terminate the current run as `FAILED`. The run result
must record the phase where freshness was lost and the reason
`CAMERA_FAILURE`. Autonomous work may not resume within that run; a new run is
allowed only after camera freshness passes preflight again.

From activation through `TURN_TO_FRUIT` and `FIND_FRUIT`, every velocity command
has zero forward input. During `APPROACH_FRUIT`, translation remains zero while
a fixed 0.50 rad/s correction first turns toward the Target Fruit. Once its
center enters 0.08 of the horizontal frame center, yaw becomes zero and must
remain centered for three fresh samples. After that initial gate, approach may
combine forward input with bounded yaw to steer toward the fruit. Confirmed
near-fruit geometry arms the lower-edge Arrival gate; it does not command a
zero-motion hold. While the fresh Target Fruit remains visible, forward
approach continues. Every forward heartbeat is counted, including the single
bounded final push after qualified lower-edge disappearance. Its one-run
defaults are `0.60 m/s` by `1.0 s`; the activation may safely select
`0.50..1.0 m/s` and either zero duration (disabled) or `0.10..1.50 s` without a
rebuild or redeploy.

An acquired red-apple track may survive its observed close-range confidence
collapse only while its box remains low and spatially continuous with the last
accepted box. This continuation cannot acquire a fruit, and discontinuous or
missing evidence commands zero motion.
Arrival releases motion before `SIT_AND_BARK`; Woof stops, settles, lies down,
and attempts the direct bark. Bark is audience feedback, not a success gate:
failure records bounded detail and `bark_played: false`, but the five-second
down hold still completes before `STAND` begins.

`TURN_TOWARD_HOME` is the sole yaw-only Home alignment phase. It uses the
regular SportClient path and must finish on a new fresh pose whose measured
bearing is inside the configured course gate. Only then may `RETURN_HOME` arm
factory obstacle avoidance and replay forward heartbeats with bounded yaw
steering. Moving yaw is exactly zero inside the configured deadband (5 degrees
by default). Outside that deadband, each non-zero correction is at least the
physically verified 0.50 rad/s signal and no greater than the configured
maximum. Both values are frozen in the per-run tuning snapshot and exposed in
the UI. Once translation starts it never falls back to yaw-only correction; a
heading escape outside the qualified forward-steering gate (20 degrees by
default) stops and disarms first. The controller then takes one fresh
authoritative Home-position measurement. If Woof is already inside the frozen
Home Distance gate (0.10 m by default), the stage records positional success
with `heading_gate_escape_reconciled: true` and performs no further turn. An
outside, stale, or unavailable post-disarm measurement retains
`RETURN_HOME_FAILURE`.

The bounded failed-run epilogue uses the same two-phase Home contract after its
down/hold/stand posture sequence: it first proves a fresh bearing within 5
degrees through regular SportClient yaw, confirms exact-zero disarm, and only
then permits the factory-avoidance position return. A failed or unverifiable
alignment records its own evidence and never authorizes forward recovery.

Return-to-Home must not use open-ended recovery. Loss of trustworthy pose or
failure to make bounded progress must stop and disarm Woof, terminate the run
as `FAILED`, and record `RETURN_HOME_FAILURE`. The Run Result must include the
latest trustworthy Home Distance; if pose freshness was lost, it must record
Home Distance as unavailable instead of estimating it.

The detailed state and evidence rules are defined in the draft
[`return-home-contract.md`](return-home-contract.md). Its marked Operating
Envelope and motion thresholds are not qualified until WDY-2281 and WDY-2283
are completed.

`REMOTE_TAKEOVER` is latched for the lifetime of the process. Detection of any
valid physical remote input must stop application command output, release the
motion owner, invalidate the run, record the interrupted phase, and require an
application restart. It cannot be cleared through the web UI or API.

Every accepted activation and terminal path must satisfy the durable evidence
requirements in [`run-result-contract.md`](run-result-contract.md).
