# Metric Arrival contract

Arrival is measured from Woof's front body/paw envelope, not from the LiDAR
origin. The initial qualified clearance is 0.15 m with a tolerance of 0.05 m.
Image geometry can enter Final Approach but cannot complete Arrival.

Production requires a stationary calibration before motion. Run:

```bash
python3 scripts/calibrate_forward_range.py --host <woof-address>
```

The operator places a centered floor-level pear at 50, 30, 20, and 15 cm from
the front envelope. The harness samples all four `SportModeState.range_obstacle`
channels and the camera detection. It selects a forward channel only when that
channel follows the known clearances, then measures the sensor-to-envelope
offset, p95 noise, and p95 evidence age. It does not assume Unitree's channel
ordering.

The stationary result does not qualify braking distance. A supervised motion
qualification must measure stopping distance at the production close speed of
0.55 m/s. Production configuration is complete only when all of these values
are set:

- `BORDER_COLLIE_FORWARD_RANGE_INDEX`
- `BORDER_COLLIE_RANGE_SENSOR_TO_FRONT_ENVELOPE_M`
- `BORDER_COLLIE_RANGE_SENSOR_LATENCY_S`
- `BORDER_COLLIE_RANGE_BRAKING_DISTANCE_M`
- `BORDER_COLLIE_RANGE_NOISE_M`

The controller begins braking early using sensor latency, braking distance, and
two noise bounds. Arrival is emitted only after the command is zero, measured
velocity and yaw are near zero, and a fresh associated range confirms the final
clearance. Missing, stale, unassociated, or uncalibrated range evidence stops
motion and fails the stage with `RANGE_UNAVAILABLE`; image geometry is never a
fallback.
