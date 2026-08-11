# Known hardware facts to revalidate

These observations came from the earlier prototype and are design inputs, not
proof that this clean implementation works:

- Direct SportClient movement produced physical steps at 0.25 and 0.50 m/s.
- Factory obstacle-avoidance commands below 0.55 m/s can load the legs and
  produce a lean without translation; 0.55 m/s is the minimum permitted signal.
- Corrective factory-avoidance yaw values around 0.24–0.30 rad/s can change
  posture without producing a useful turn. The clean centering candidate uses
  the separately observed working 0.50 rad/s turn signal outside its center
  band and zero yaw inside the band.
- Direct SportClient search can use a 1.0 rad/s broad sweep, but the August 10
  live run crossed a strong pear candidate before the fifth qualifying frame.
  The current candidate holds fresh detections at or above 0.50 confidence for
  0.75 seconds, then resumes at no more than 0.50 rad/s while evidence persists.
  It keeps the existing five-frame gate and returns to the broad rate only
  after 0.50 seconds without a plausible candidate. This policy still needs a
  supervised physical qualification run.
- The same factory-avoidance calibration produced visible physical movement
  at 1.0 m/s. Production fruit approach therefore uses 1.0 m/s rather than
  operating below the 0.55 m/s movement floor; camera steering,
  near-fruit geometry, slowed close approach, timeout, and automatic release
  remain mandatory.
- Short travel should use a reliable velocity with bounded pulse duration,
  rather than reducing velocity below the movement deadband.
- The previous final approach used one direct 1.0 m/s, 0.4-second push only
  after confirmed near-fruit evidence and lower-camera disappearance.
- The first clean combined Arrival used that 0.4-second value but stopped too
  far from the pear. A later 1.0 m/s by 1.0-second final movement was too fast.
  The current production candidate removes off-screen movement entirely and
  uses continuous close-track evidence plus zero-motion sight-lost Arrival; it
  requires supervised qualification.
- `SportClient.StandDown()` returns before the visible posture completes. The
  old application held the down posture for 5 seconds; the clean production
  sequence now restores that hold before stand-up.
- Camera safety requires frame identity, source time, connection generation,
  and a strict stale-data boundary.
- **Observed on Woof (2026-08-11, run `acfdb493`):** while a centered pear was
  clipped at the bottom of the 1280x720 camera image, its detected area shrank
  54.6% between fresh advancing frames even though horizontal center changed
  only from 0.560 to 0.571 and the lower edge remained at 1.0. A generic 35%
  area-retreat continuity guard therefore produced a false discontinuity.
  Treat this geometry as stopped loss evidence only inside Final Approach; do
  not relax the global continuity threshold or permit off-axis/non-bottom
  shrinkage to authorize a final push.
- **Observed on Woof (2026-08-11, run `5e2dc136`):** search qualified an apple
  after five detections at 0.7106289 confidence, then approach created a new
  tracker and observed 165 fresh same-label apple samples whose confidence
  peaked at 0.6477978. The unchanged 0.70 acquisition gate therefore produced
  zero qualified approach samples, zero forward pulses, and an Arrival timeout.
  Preserve the 0.70 ordinary apple gate; transfer the completed search identity
  through the bounded search-to-approach handoff and require current tracking-
  floor geometry plus fresh centering before translation.
- Return success must include measured position, restored heading, and a final
  disarmed state.

Every value must receive a new acceptance result in this repository before it
is treated as qualified.

## First reuse decision

The clean foundation reuses the factory-avoidance connection and motion
boundary because `MOTION-DEADBAND-001` has explicit human observation and a
verified disarmed final state. Direct SportClient translation is deliberately
not exposed yet. Although 0.25 and 0.50 m/s produced direct-path steps, that
path disables factory avoidance and belongs behind the future return-home
collision-planning contract.
