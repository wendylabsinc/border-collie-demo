# Known hardware facts to revalidate

These observations came from the earlier prototype and are design inputs, not
proof that this clean implementation works:

- Direct SportClient movement produced physical steps at 0.25 and 0.50 m/s.
- Factory obstacle-avoidance movement required approximately 0.50 m/s.
- Corrective factory-avoidance yaw values around 0.24–0.30 rad/s can change
  posture without producing a useful turn. The clean centering candidate uses
  the separately observed working 0.50 rad/s turn signal outside its center
  band and zero yaw inside the band.
- The same factory-avoidance calibration produced visible physical movement
  at 1.0 m/s. Production fruit approach therefore uses 1.0 m/s rather than
  operating exactly at the observed 0.50 m/s deadband edge; camera steering,
  near-fruit geometry, the final-push gate, timeout, and automatic release
  remain mandatory.
- Short travel should use a reliable velocity with bounded pulse duration,
  rather than reducing velocity below the movement deadband.
- The factory obstacle-avoidance MODULE is a robot-global switch
  (`SwitchSet`/`SwitchGet`), not a per-client path. While engaged it owns
  velocity control and silently vetoes reverse translation from every
  client: reverse `Move` RPCs are accepted with no physical motion both
  through the avoidance client (2026-08-08 r2: five commands, -0.001 m by
  odometry) and through the direct SportClient with the module still
  engaged (2026-08-08 r4: identical signature). Any reverse pulse must
  switch the module off for the bounded window, verify displacement by
  odometry, and re-engage and confirm the module afterwards. This also
  means the 2026-08-07 avoidance-path step-back, which had no odometry
  check, most likely never physically executed.
- The previous final approach used one direct 1.0 m/s, 0.4-second push only
  after confirmed near-fruit evidence and lower-camera disappearance.
- The first clean combined Arrival used that 0.4-second value but stopped too
  far from the pear. A later 1.0 m/s by 1.0-second final movement was too fast.
  The current production candidate keeps the one-second bound but reduces only
  this off-screen movement to 0.3 m/s; it requires supervised qualification.
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
