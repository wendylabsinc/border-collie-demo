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

## Banana reliability observations (2026-08-13, Woof)

- **Measured:** Run `af45a566-d0c8-4c97-9640-28abe5ee1512` retained a Banana
  detection until its age reached 267.8 ms, 17.8 ms beyond the motion evidence
  ceiling. The camera source still advanced. Practical consequence: remove
  motion authority at 250 ms, but distinguish a bounded stopped inference wait
  from terminal camera failure.
- **Measured:** Run `689e9005-5ae9-4579-ad4c-3e25022c1779` repeatedly saw an
  off-center Banana, resumed broad yaw through intervening misses, and later
  saw it on the opposite side. Practical consequence: a pre-lock focus state
  must retain its last corrective direction briefly and must not count missing
  or weak frames toward lock.
- **Measured:** Run `1d5129e4-9f18-4cb3-8f74-59e81b95dbc1` crossed the moving
  Home heading gate but a fresh measurement after stopping placed Woof about
  0.065 m from Home. Practical consequence: stop/disarm before measuring, then
  accept positional Home only inside the frozen 0.10 m gate; do not issue
  another turn.
- **Implemented, software-only:** `stage-default-v4-banana-reliability-replay`
  encodes those three contracts and a deterministic non-motion replay. The
  `0.20 rad/s` focused Sports yaw and physical end-to-end behavior remain
  unqualified until a supervised comparison run.

## First reuse decision

The clean foundation reuses the factory-avoidance connection and motion
boundary because `MOTION-DEADBAND-001` has explicit human observation and a
verified disarmed final state. Direct SportClient translation is deliberately
not exposed yet. Although 0.25 and 0.50 m/s produced direct-path steps, that
path disables factory avoidance and belongs behind the future return-home
collision-planning contract.
