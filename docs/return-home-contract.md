# Return-to-Home control contract (draft)

This contract defines the bounded behavior that returns Woof to the Home pose
captured at the start of a Demo Run. Its state model is ready for implementation,
but its marked numeric gates are **provisional** until WDY-2281 qualifies the
Operating Envelope and WDY-2283 qualifies the required motion primitives.
WDY-2286 therefore remains blocked and must not be treated as complete.

The executable, hardware-free design probe lives in
[`../lab/return-home-contract-prototype/`](../lab/return-home-contract-prototype/).

## Home and pose authority

- Home is a position and heading captured at each operator-initiated start and
  held as the Stage Home for that start. It is the physical spot the stage is
  set to, not a reading each back-to-back run takes for itself.
- Capture requires a stable window of advancing local-pose samples. The sample
  count and permitted position and yaw spread are qualification parameters.
- Every activation reads a fresh, disarmed pose before `capture_home`
  completes, so the disarm and pose-freshness gates are unchanged. On an
  operator start that reading becomes Home. On a back-to-back run inside a
  cohort it is recorded instead as `home_provenance.activation_offset_m`, the
  measured drift from the cohort's Home at the start of that run.
- An individual Demo Run start and a cohort start both capture Home fresh, so
  the spot Woof is standing on becomes Home. The remaining runs of that cohort
  reuse it, which is what keeps Home error from compounding across the cohort.
  A failed run inside a cohort does not reset it.
- Home is not carried across an application restart. A restarted process holds
  no Home and captures one on the next start.
- `POST /api/home/recapture` retargets the Home a cohort already underway
  returns to. It is refused while a Demo Run or cohort owns activation, and
  while Remote Takeover is latched.
- The recorded outbound forward-heartbeat count bounds the return translation:
  after turning toward Home, the return controller replays at most that many
  forward heartbeats at the same 1.0 m/s signal. Heading-only corrections do
  not consume the count.
- Fresh measured local pose remains the authority for course, progress, early
  stop, the 0.10 m Home gate, and the reported Home Distance. Pulse count and
  requested velocity never substitute for measured Home Distance or prove
  arrival.
- A pose is usable only when its age is at most **0.50 seconds**. This is the
  current clean-app freshness boundary and must be rechecked during hardware
  qualification.
- A stale, missing, non-finite, discontinuous, or out-of-envelope pose stops
  return motion and terminates the run with `RETURN_HOME_FAILURE`.
- When the pose is untrustworthy, Home Distance and heading error are recorded
  as unavailable rather than copied from the last estimate.

## Bounded return sequence

1. Stop and settle after standing from the fruit action.
2. Read a fresh pose and recompute bearing and distance to Home.
3. In `TURN_TOWARD_HOME`, turn in place through regular SportClient until a new
   fresh pose proves the measured Home bearing is inside the qualified course
   gate. This is the only Home phase that may issue yaw-only commands.
4. In `RETURN_HOME`, arm factory obstacle avoidance and replay the recorded
   outbound forward-heartbeat count through a
   collision-aware motion owner while continuously measuring Home Distance,
   pose age, route state, and progress. Every non-zero command combines forward
   translation with bounded yaw steering. Yaw is exactly zero inside the
   per-run moving-yaw deadband. Outside it, a correction uses at least the
   configured minimum and no more than the configured maximum. The production
   defaults are a 5-degree deadband and 0.50 rad/s for both minimum and maximum;
   the minimum cannot be configured below the physically verified 0.50 rad/s
   factory-avoidance turning signal.
5. Stop translation inside the position gate.
6. If the bearing escapes the qualified forward-steering gate after translation
   starts, command exact zero, disarm, and fail. Do not re-enter an in-place
   turn from `RETURN_HOME`.
7. Request stop, release the motion owner, and confirm disarm.
8. Report success only when fresh pose confirms the position gate and the final
   safety state is `DISARMED_CONFIRMED`. Captured heading remains evidence, not
   a completion gate.

The initial acceptance target is **0.10 meters** Home Distance. The
`TURN_TOWARD_HOME` course-entry gate is **5 degrees**. They are proposed gates,
not qualified claims. The earlier
prototype failed its 0.10-meter gate with 0.207 meters remaining, so the clean
implementation must earn these values in a new acceptance run.

## Inter-run stage clearance

Mission completion keeps the strict 0.10 m position target above. A repeated
stage soak has a separate operator margin, configured by
`BORDER_COLLIE_STAGE_HOME_MARGIN_M`: meters, default `0.50`, valid range
`0.10..1.0`. This margin never changes whether an individual Demo Run passed.

Inside a cohort, a back-to-back run does not start until fresh current pose is
within this margin of the cohort's Stage Home and the prior run ended
`DISARMED_CONFIRMED`. A failed return therefore aborts the cohort instead of
capturing the fruit-side position as a new Home.

The margin is measured against the cohort's one Stage Home rather than against
each run's own reading. A cohort therefore cannot random-walk away from the
stage by staying inside the margin on every individual hop.

`/api/status.activation.ready` does not fall to false on this distance, and
`activation.inter_run` is published as evidence rather than as a blocker. An
operator start captures Home fresh, so it cannot be refused for standing too
far from a previous run's Home.

## Route, progress, and recovery

- Translation is forward-only. The return controller does not reverse toward
  an unseen route. Bounded course correction may accompany forward replay.
- All yaw-only Home commands precede the first forward command. After that
  first forward command, an excessive course error is a terminal safety event,
  not authorization for a stationary correction.
- Translation must retain factory obstacle avoidance or use another
  independently qualified collision-aware planner. Direct unprotected body
  translation cannot implement production return-to-Home.
- Progress means a measured reduction in Home Distance over a bounded time
  window. Distance traveled in some other direction is not progress.
- Minimum progress, progress-window duration, course gate, maximum Home
  Distance, total return timeout, speed, and pulse duration are configuration
  owned by WDY-2281 and WDY-2283. Prototype defaults are illustrative only.
- A blocked route first stops and disarms. At most one stationary,
  collision-aware replan may be accepted. A second blockage, unavailable route,
  exhausted replan, or failure to make qualified progress terminates the run.
- Recovery is never open-ended and never continues with stale evidence.

## Failure and stop behavior

Pose failure, stalled progress, route blockage without a bounded replan,
motion-owner loss, watchdog failure, timeout, tolerance failure, operator stop,
or Remote Takeover immediately exits autonomous return. The application requests
zero motion, stops/releases every motion client it owns, and records whether
disarm was actually confirmed.

An unsuccessful stop is not hidden. The run remains `FAILED` with reason
`RETURN_HOME_FAILURE` and final safety state
`STOP_REQUESTED_UNCONFIRMED` or `UNKNOWN` as appropriate. Remote Takeover uses
the separate latched `REMOTE_TAKEOVER` terminal path and `REMOTE_OWNED` safety
state.

## Run Result evidence

The return phase adds these fields or events to the Run Result:

- captured Home pose, source, capture window, and freshness;
- every accepted return pose's age, Home Distance, bearing/course error, and
  Home-heading error;
- qualified threshold-set identifier and the exact values used;
- motion-owner, avoidance/planner, watchdog, command, and stop evidence;
- progress-window start and end measurements;
- requested outbound forward-heartbeat count and the number actually replayed;
- obstacle, replan, timeout, and failure decisions;
- latest trustworthy terminal Home Distance and heading error, or explicit
  unavailability reasons; and
- the first confirmed zero/disarmed state.

Return success must be reconstructable from this evidence without relying on
console output or visual observation alone.
