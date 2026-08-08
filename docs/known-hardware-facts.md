# Known hardware facts to revalidate

These observations came from the earlier prototype and are design inputs, not
proof that this clean implementation works:

- Direct SportClient movement produced physical steps at 0.25 and 0.50 m/s.
- Factory obstacle-avoidance movement required approximately 0.50 m/s.
- Corrective factory-avoidance yaw values around 0.24–0.30 rad/s can change
  posture without producing a useful turn. Rotation-only operations now use
  the old implementation's direct SportClient yaw lease instead, with the
  separately observed 0.50 rad/s turn signal outside the center band and zero
  yaw inside it. This reassignment is automated-test validated but requires a
  new supervised physical qualification.
- The same factory-avoidance calibration produced visible physical movement
  at 1.0 m/s. Production fruit approach therefore uses 1.0 m/s rather than
  operating exactly at the observed 0.50 m/s deadband edge; camera steering,
  near-fruit geometry, the final-push gate, timeout, and automatic release
  remain mandatory.
- Short travel should use a reliable velocity with bounded pulse duration,
  rather than reducing velocity below the movement deadband.
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

The clean foundation reuses factory avoidance for every forward or
forward-plus-yaw command because `MOTION-DEADBAND-001` has explicit human
observation and a verified disarmed final state. Rotation-only work uses the
old implementation's separately owned direct SportClient yaw path so obstacle
avoidance cannot reshape a requested in-place turn. Direct SportClient
translation remains deliberately unavailable; it disables factory avoidance
and belongs behind the future return-home collision-planning contract.
