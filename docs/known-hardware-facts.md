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
