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
- The recorded outbound forward-heartbeat count bounds return translation.
  It is a safety budget, not a route estimate. Heading-only corrections do not
  consume the count.
- Fresh fused local pose remains the authority for course and progress. The
  0.10 m gate conservatively uses the farther of fresh raw Go2 distance and
  filtered distance. Pulse count and requested velocity never substitute for
  measured Home Distance or prove arrival.
- A pose is usable only when its age is at most **0.50 seconds**. This is the
  current clean-app freshness boundary and must be rechecked during hardware
  qualification.
- A stale, missing, non-finite, discontinuous, or out-of-envelope pose stops
  return motion and terminates the run with `RETURN_HOME_FAILURE`.
- When the pose is untrustworthy, Home Distance and heading error are recorded
  as unavailable rather than copied from the last estimate.

The implementation fuses Go2 position deltas, body velocity, IMU yaw rate,
loaded-foot zero-velocity updates, and bounded CPU-only sparse visual motion
through
[`home-localization-adapter.md`](home-localization-adapter.md). Vision supplies
scale-free direction/yaw, continuity, and natural-scene revisit evidence. It
does not claim metric monocular translation or replace the fresh Go2 pose
requirement.

## Bounded return sequence

1. Stop and settle after standing from the fruit action.
2. Read a fresh pose and recompute bearing and distance to Home.
3. Turn in place until the measured Home bearing enters the qualified course
   gate.
4. Follow the recorded outbound pose breadcrumbs in reverse through a
   collision-aware motion owner while continuously measuring Home Distance,
   pose age, route state, and progress. The heartbeat count remains the maximum
   translation budget.
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
  an unseen route. Bounded course correction accompanies reverse-order
  breadcrumb traversal.
- Translation must retain factory obstacle avoidance or use another
  independently qualified collision-aware planner. Direct unprotected body
  translation cannot implement production return-to-Home.
- Progress means a measured reduction in distance to the active breadcrumb or
  Home over a bounded time window. Reaching a breadcrumb advances the route.
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

## Recovery after a failed run

A failed run with `DISARMED_CONFIRMED` and a persisted Home pose may accept an
explicit recovery attempt. Translation additionally requires at least one
recorded forward approach heartbeat; a run already inside the position gate is
sealed as recovered without arming motion.
The recovery is separate from the terminal mission: the original run remains
`FAILED`, no new Home is captured, and the failed mission stages are not
resumed.

`TARGET_LOST_OFF_AXIS` starts this recovery automatically after terminal
evidence and confirmed disarm. The recovery record is persisted before the
failure action. Woof lies down without barking, holds for five seconds, stands,
and requires the existing continuous-fusion readiness gate before turning
toward Home. Pose, motion, posture, or stop failures remain fail-closed and do
not attempt translation. Other failed-run reasons retain the explicit recovery
endpoint.

The API persists the recovery record before motion, returns `202 Accepted`, and
runs recovery asynchronously so a client disconnect cannot cause an ambiguous
second activation. Recovery turns toward the saved Home and uses 1.5 times the
original approach heartbeat count as its maximum forward-return budget. The
reserve accounts for factory obstacle avoidance reducing actual return speed;
pose-based arrival stops playback early. Fresh pose,
factory obstacle avoidance, progress, course, timeout, watchdog, stop, and
0.10-meter position gates remain authoritative.

Failed-run recovery is position-only. It records the final heading but does not
restore it, because the in-place heading primitive can move the feet back
outside the position gate. Recovery reports `HOME_POSITION_RECOVERED` only when
a fresh terminal pose is inside 0.10 meters and stop/disarm is confirmed. Any
failure or operator stop seals the recovery attempt with its own outcome and
full evidence without altering the failed run's original reason.
One correction attempt is allowed only when the first recovery ended `FAILED`
with `DISARMED_CONFIRMED`. No third attempt is accepted, and a completed,
stopped, active, interrupted, or unconfirmed recovery is not retryable.

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
