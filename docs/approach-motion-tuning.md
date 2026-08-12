# Approach Motion Tuning

Release `stage-camera-v33-base-motion-approach` restores the base demo's
factory-avoidance motion policy: qualified normal and close-range tracking both
command a sustained `1.0 m/s`. Arrival owns the stop, and the one bounded final
push remains `0.6 m/s` for one second.

The app reads these values at process startup:

| Variable | Unit | Default | Valid range and safety constraint |
| --- | --- | --- | --- |
| `BORDER_COLLIE_APPROACH_FORWARD_MPS` | m/s | `1.0` | At least `BORDER_COLLIE_MIN_FORWARD_MPS` and at most `BORDER_COLLIE_MAX_FORWARD_MPS`. |
| `BORDER_COLLIE_CLOSE_RANGE_MPS` | m/s | `1.0` | Same hardware envelope and no greater than the approach speed. |
| `BORDER_COLLIE_FINAL_PUSH_MPS` | m/s | `0.6` | Same hardware envelope; used only by the bounded closeout episode. |
| `BORDER_COLLIE_FINAL_PUSH_DURATION_S` | seconds | `1.0` | Finite and greater than zero. |

Changing a value requires restarting only the app service; it does not alter
the media image or perception contract. These settings cannot bypass motion
authority, watchdog, exact-zero disarm, stale-frame, wrong-label, generation,
camera-health, geometry, or Arrival gates.
