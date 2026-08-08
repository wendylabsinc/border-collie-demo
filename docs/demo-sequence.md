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
Approach translation runs at the qualified constant 1.0 m/s on every
visible-track frame, right up until sight loss. A close-range duty-cycle
slowdown (drive one frame in three) was tried on 2026-08-08 (r7) and
retired the same day: the hold pattern validated by the search stage is a
YAW pattern — rotation resumes instantly from rest, but forward gait does
not, and a single 0.1-second 1.0 m/s burst through the avoidance module
plants no step before the next zero-hold stops it. r7 measured roughly
10 cm of twitching advance in 15.5 seconds with the pear rock-solid in
view, a guaranteed stall under the sight-lost contract. The slowdown's
original purpose (winning the retired near gate's frame race) no longer
exists, and its secondary purpose (a gentler stop) is served by the
contract's immediate zero velocity on sight loss. If arrival distance
proves too close in practice, the tunable is the arrival geometry (the
0.70 bottom threshold), not stop-start speed control. Every visible-track
frame consumes one forward heartbeat at the same 1.0 m/s signal, so the
return-home replay budget maps one-to-one exactly as it always has.

Arrival is the operator-ruled sight-lost contract (2026-08-08): while the
fresh Target Fruit remains visible, forward approach continues; when the
qualified track is lost and stays lost through the 0.75-second grace
window, Woof declares Arrival and stops where it stands **only if** the
last accepted track geometry was already close (lower edge at or above the
0.70 boundary) and roughly centered (horizontal center within 0.15 of frame
center — a guard against a sideways frame exit). There is no blind final
push and no near-confirmation counting: the push existed to end
nose-at-fruit, the operator explicitly prefers stopping farther, and the
push's protective near gate caused three of the four recorded arrival
failures. A distant or off-center sight loss never declares Arrival — the
stage holds zero motion, waits out the deadline, and fails closed, because
Woof must never sit down in the middle of the room after a tracking
dropout. For loss purposes "visible" requires the detection to meet the
fruit's close-range tracking confidence and to remain spatially continuous
with the last accepted track box: a matching label below that floor, or one
discontinuous with the fruit being approached, counts as disappearance.
(Supervised run `45a1e796` showed a static 0.010-0.016-confidence phantom
keeping the fruit counted as visible for ~12 seconds; the floor and the
continuity condition keep that scenario closed, and a phantom can never
supply the arrival geometry because it is never accepted as the track.)
Every terminal approach path — Arrival, identity change, and timeout —
seals the same approach evidence dict (arrival mode, last track geometry,
slowdown engagement, pulse counts, centering counters, and sub-floor
visibility samples) so a failed approach is never blind in the Run Result.

An acquired track may survive its observed close-range confidence collapse
only while its box remains low and spatially continuous with the last
accepted box; the per-fruit floors live in `fruits.py` (apple 0.10, banana
0.20, pear 0.20 — the pear value was lowered from 0.55 after supervised run
`d740a5f2` proved a real pear collapses to ~0.27 confidence when it fills
the frame at arrival distance). Close-range-continued frames update the
accepted track geometry exactly like fully qualified frames, which is what
lets the sight-lost Arrival read the true loss point. This continuation
cannot acquire a fruit, and discontinuous or missing evidence commands zero
motion.
Arrival releases motion before `SIT_AND_BARK`; Woof barks while down and holds
that posture for 5 seconds before `STAND` may begin.

`STEP_BACK` then reverses for one bounded window (a single 0.5 m/s reverse
setpoint held for 0.6 seconds, about 0.30 m commanded) so the following
Home turn rotates with clearance from the fruit. Two hardware facts shape
this stage. First, the Go2 obstacle-avoidance module is a robot-global
switch, not a per-client path: while engaged it owns velocity control and
vetoes reverse translation from every client — two supervised runs on
2026-08-08 measured -0.001 m over five accepted reverse commands, first
through the avoidance client (r2) and then through the direct SportClient
with the module still engaged (r4). Second, `sport.Move` is a velocity
setpoint: re-sending it on a 0.1 s cadence restarts gait initiation each
time, so a reverse step never gets planted — r5 sent six direct-sport
setpoints inside a correctly suspended avoidance window and still measured
only 0.007 m. The proven reverse shape from `go2-local-web-remote` (which
has physically reversed this robot) is ONE `Move(-vx)`, a silent hold, then
`StopMove()`. The stage therefore records the module's prior state,
switches it off with a 0.45 s settle, sends exactly one direct-sport
reverse setpoint and holds it for the bounded window, issues `StopMove`,
and always re-engages and confirms the module afterwards. During the silent
hold the command watchdog is renewed on a timer without emitting a new
setpoint: the watchdog's guarantee survives (a wedged process is still
stopped within 0.35 s) while the setpoint stays untouched — re-sending it
is exactly the failure r5 measured. A failed module restore is a hard
fault: the motion adapter latches the fault, stops, disarms, and the run
seals `FAILED` — the demo never continues with avoidance silently off. This
roughly one-second avoidance-off window is acceptable only for this bounded
step because Woof reverses into space it traversed seconds earlier during
its own approach, the hold is short and speed-bounded, displacement is
verified by fresh odometry, and the demo is operator-supervised. The stage
reads a fresh pose before and after the window and fails closed with
`ACTION_FAILURE` if odometry does not confirm at least 0.05 meters of
backward movement; the switch states, settle and off-window durations, the
setpoint timestamp, the stop origin, poses, and measured displacement are
sealed in the Run Result on success and on every failure path.
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
