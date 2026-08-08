# Return-to-Home control contract (draft)

This contract defines the bounded behavior that returns Woof to the Home pose
captured at the start of a Demo Run. Its state model is ready for implementation,
but its marked numeric gates are **provisional** until WDY-2281 qualifies the
Operating Envelope and WDY-2283 qualifies the required motion primitives.
WDY-2286 therefore remains blocked and must not be treated as complete.

The executable, hardware-free design probe lives in
[`../lab/return-home-contract-prototype/`](../lab/return-home-contract-prototype/).

## Home and pose authority

- Home is a position and heading captured immediately before the run begins.
- Capture requires a stable window of advancing local-pose samples. The sample
  count and permitted position and yaw spread are qualification parameters.
- The recorded outbound forward-heartbeat count bounds the return translation:
  after turning toward Home, the return controller replays at most that many
  forward heartbeats at the same 1.0 m/s signal. Heading-only corrections do
  not consume the count.
- The bounded `STEP_BACK` clearance stage that precedes the Home turn never
  reduces or credits this count. Its backward movement is verified by fresh
  odometry and simply leaves the robot nearer Home; the return controller
  continues to work from the measured pose, and the unchanged replay count
  remains an upper bound, not a distance claim. Crediting commanded (rather
  than measured) step-back motion against the replay is what desynchronized
  the 2026-08-07 return-home attempt (0.187-0.375 m misses) and is
  prohibited.
- `STEP_BACK` is the single sanctioned exception to the avoidance-owned
  translation rule, and it does not weaken this contract's return rules:
  return translation itself remains forward-only through factory avoidance.
  The obstacle-avoidance module is robot-global and vetoes reverse from any
  client while engaged (2026-08-08, twice: five accepted reverse commands
  measured -0.001 m through the avoidance client, then again through the
  direct SportClient with the module engaged), so the step suspends the
  module for one bounded under-a-second window: prior state recorded,
  SwitchSet off with a short settle, direct-sport reverse, StopMove, then a
  mandatory SwitchSet-on with confirmation. A failed restore latches a hard
  motion fault and seals the run `FAILED` — autonomous work never continues
  with avoidance silently off. The window is bounded and evidenced —
  reverse-only with no yaw or lateral mixing, the same forward-speed limit
  and command watchdog, fresh-pose displacement verification with a
  fail-closed gate, switch states and off-window duration sealed in the Run
  Result, a path the robot itself cleared seconds earlier during approach,
  and an operator supervising the run.
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
3. Turn in place until the measured Home bearing enters the qualified course
   gate.
4. Replay the recorded outbound forward-heartbeat count through a
   collision-aware motion owner while continuously measuring Home Distance,
   pose age, route state, and progress.
5. Stop translation inside the position gate.
6. Restore the captured Home heading in place.
7. Re-read position after the heading turn. If the position gate was lost,
   repeat the bounded turn-and-translate sequence within the same limits.
8. Request stop, release the motion owner, and confirm disarm.
9. Report success only when fresh pose confirms both position and heading gates
   and the final safety state is `DISARMED_CONFIRMED`.

The initial acceptance targets are **0.10 meters** Home Distance and **5 degrees**
heading error. They are proposed gates, not qualified claims. The earlier
prototype failed its 0.10-meter gate with 0.207 meters remaining, so the clean
implementation must earn these values in a new acceptance run.

## Route, progress, and recovery

- Translation is forward-only. The return controller does not reverse toward
  an unseen route. Bounded course correction may accompany forward replay.
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
