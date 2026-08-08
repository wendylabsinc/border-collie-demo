# Demo sequence and safety states

The production sequence is:

```text
IDLE -> PREFLIGHT -> CAPTURE_HOME -> WAIT_FOR_COMMAND
     -> [TURN_TO_FRUIT -> FIND_FRUIT] -> APPROACH_FRUIT -> ARRIVED
     -> SIT_AND_BARK -> STAND -> STEP_BACK -> TURN_TOWARD_HOME
     -> RETURN_HOME -> RESTORE_HEADING -> COMPLETE
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
an advancing healthy camera/perception source, and bark-media readiness. Target Fruit
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
combine forward input with bounded yaw to steer toward the fruit. While the
track's lower edge remains above the 0.70 close-range boundary the center
band stays at 0.08; once the lower edge crosses that boundary the band
tightens to 0.04 so the same fixed 0.30 rad/s correction engages earlier and
the heading is close to the fruit axis before the fruit leaves the frame.
An offset beyond the wide 0.08 band corrects immediately as it always has;
the tighter band engages only after two consecutive off-band samples so
single-frame detection jitter near 0.04 cannot toggle the correction.
Forward translation continues through every steering sample, so the
forward-heartbeat count is unaffected by the tighter band. Confirmed
near-fruit geometry arms the lower-edge Arrival gate; it does not command a
zero-motion hold. While the fresh Target Fruit remains visible, forward
approach continues. For arrival purposes "visible" requires the detection to
meet the fruit's close-range tracking confidence and to remain spatially
continuous with the last accepted track box: a matching label below that
floor, or one discontinuous with the fruit being approached, counts as
disappearance. (Supervised run `45a1e796` on 2026-08-08 showed a static
0.010-0.016-confidence phantom holding the arrival gate open for ~12 seconds
after the pear had genuinely dropped below the camera at arrival distance,
which suppressed the final push until the deadline. The continuity condition
keeps that scenario closed even with the lowered pear close-range floor.)
Every forward heartbeat is counted, including the single
bounded 0.3 m/s by 0.6-second push after qualified lower-edge disappearance.
The push trigger is unchanged; only its window was shortened from 1.0 s after
the 2026-08-08 supervised baseline finished too close to the fruit. Every
terminal approach path — Arrival, identity change, and timeout — seals the
same approach evidence dict (near gate state, pulse counts, centering
counters, and sub-floor visibility samples) so a failed approach is never
blind in the Run Result.

An acquired track may survive its observed close-range confidence collapse
only while its box remains low and spatially continuous with the last
accepted box; the per-fruit floors live in `fruits.py` (apple 0.10, banana
0.20, pear 0.20 — the pear value was lowered from 0.55 after supervised run
`d740a5f2` proved a real pear collapses to ~0.27 confidence when it fills
the frame at arrival distance). Close-range-continued frames count toward
near-fruit Arrival confirmation exactly like fully qualified frames. This
continuation cannot acquire a fruit, and discontinuous or missing evidence
commands zero motion.
Arrival releases motion before `SIT_AND_BARK`; Woof barks while down and holds
that posture for 5 seconds before `STAND` may begin.

`STEP_BACK` then reverses for one bounded window (1.0 m/s for 0.4 seconds)
so the following Home turn rotates with clearance from the fruit. The
reverse pulse travels through the direct SportClient, not factory avoidance:
the avoidance controller's perception is forward-facing, cannot validate
space behind the robot, and silently refuses reverse translation (supervised
run on 2026-08-08 measured -0.001 m over five accepted avoidance reverse
commands). Bypassing avoidance is acceptable only for this bounded step
because Woof reverses into space it traversed seconds earlier during its own
approach, the pulse is short and speed-bounded under the same command
watchdog and StopMove release, displacement is verified by fresh odometry,
and the demo is operator-supervised. The reverse window timer starts only
after the motion arm, including its remote-API settle, is confirmed
complete. The stage reads a fresh pose before and after the window and fails
closed with `ACTION_FAILURE` if odometry does not confirm at least
0.05 meters of backward movement.
`STEP_BACK` takes no credit against the return replay: the outbound
forward-heartbeat count recorded during approach is passed to `RETURN_HOME`
unchanged, and the return controller keeps measuring the real pose from
wherever Woof actually stands. (The 2026-08-07 attempt that credited
step-back pulses against the replay desynchronized return-home and is
quarantined on its own branch.)

After the measured turn
toward Home,
`RETURN_HOME` replays the recorded number of forward heartbeats at the same
1.0 m/s signal. Heading-only corrections do not consume a forward heartbeat;
bounded course correction may accompany forward replay after approach.

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
