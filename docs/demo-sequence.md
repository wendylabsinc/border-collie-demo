# Demo sequence and safety states

The production sequence is:

```text
IDLE -> PREFLIGHT -> CAPTURE_HOME -> WAIT_FOR_COMMAND
     -> TURN_TO_FRUIT -> FIND_FRUIT -> APPROACH_FRUIT -> ARRIVED
     -> SIT_AND_BARK -> STAND -> TURN_TOWARD_HOME -> RETURN_HOME
     -> RESTORE_HEADING -> COMPLETE
```

`STOPPED`, `FAILED`, and `REMOTE_TAKEOVER` are off-ramps from active work.

For the pear-qualified milestone, facing the person is an operator setup
condition established before activation. Autonomous person detection is not
required and the application must not silently adjust the captured Home pose.

Preflight records explicit checks for durable Run Result storage, connected
hardware, the autonomous-motion gate, fresh pose, disarmed application motion,
an advancing healthy camera/perception source, and bark-media readiness. Pear
evidence is intentionally not an activation gate: `TURN_TO_FRUIT` happens
first and rotates through at most one measured revolution until fresh qualified
pear evidence stops the turn. `FIND_FRUIT` then confirms or briefly reacquires
that evidence before approach. Completing the bounded search with healthy,
advancing frames but no qualified pear records `TARGET_RECOGNITION_FAILURE`
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

During `APPROACH_FRUIT`, translation remains zero while a fixed 0.50 rad/s
correction turns toward the pear. Once its center enters 0.08 of the horizontal
frame center, yaw becomes zero and must remain centered for three fresh
samples. Every later forward heartbeat is counted, including the single
bounded 0.3 m/s by 1.0-second push after qualified lower-edge disappearance.
Arrival releases motion before `SIT_AND_BARK`; Woof barks while down and holds
that posture for 5 seconds before `STAND` may begin. After the measured turn
toward Home,
`RETURN_HOME` replays the recorded number of forward heartbeats at the same
1.0 m/s signal. Heading-only course corrections do not consume a forward
heartbeat.

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
