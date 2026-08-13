# Stage-default camera guidance

`FruitGuidance` is the single camera-guidance interface from Target Fruit search
through Arrival. A caller feeds each sidecar status sample to `observe(...)` and
executes only the returned command. The module privately retains the acquired
fruit identity, camera generation, fresh-frame counters, lower-edge latch, and
final-push state; there is no search-to-approach handoff token.

The defaults intentionally reproduce the physically useful base behavior:
search at `0.50 rad/s`, require three fresh centered samples, translate
continuously at `1.0 m/s`, steer while moving outside the center band, remove
forward authority outside the outer corridor, and permit one `0.60 m/s` by
`1.0 s` final push only after centered lower-edge evidence disappears.

## Runtime environment

All values are read when a `FruitGuidance` instance is created. Changing one
therefore requires restarting only the app service, not rebuilding the image or
restarting the media service. Validation is fail-fast and cannot widen the
camera freshness or hardware motion safety limits.

| Environment variable | Units | Default | Valid range | Safety constraint |
| --- | --- | ---: | ---: | --- |
| `BORDER_COLLIE_GUIDANCE_SEARCH_YAW_RPS` | rad/s | `0.50` | `0.50..0.80` | Values below the observed useful factory-avoidance yaw are rejected. |
| `BORDER_COLLIE_GUIDANCE_CENTER_TOLERANCE_RATIO` | frame-width ratio from center | `0.08` | greater than `0`, less than outer corridor | Three fresh samples must be inside this band before lock. |
| `BORDER_COLLIE_GUIDANCE_CENTER_CONFIRMATIONS` | fresh frames | `3` | integer `>=1` | Duplicate frames never advance this count. |
| `BORDER_COLLIE_GUIDANCE_APPROACH_FORWARD_MPS` | m/s | `1.0` | `0.50..1.0` | Preserves the factory-avoidance translation floor and configured maximum. |
| `BORDER_COLLIE_GUIDANCE_APPROACH_YAW_RPS` | rad/s | `0.30` | greater than `0`, at most `0.80` | Applied only while translating inside the outer corridor. |
| `BORDER_COLLIE_GUIDANCE_OUTER_CORRIDOR_RATIO` | frame-width ratio from center | `0.20` | greater than center band, less than `0.50` | Forward authority is removed outside this band. |
| `BORDER_COLLIE_GUIDANCE_RECENTER_YAW_RPS` | rad/s | `0.50` | `0.50..0.80` | In-place recentering never includes forward motion. |
| `BORDER_COLLIE_GUIDANCE_DUPLICATE_HOLD_S` | seconds | `0.250` | greater than `0`, at most `0.250` | A duplicate cannot extend the original fresh-evidence authority window. |
| `BORDER_COLLIE_GUIDANCE_SOURCE_MAXIMUM_AGE_S` | seconds | `0.350` | greater than `0`, at most `0.350` | Older source evidence stops terminally. |
| `BORDER_COLLIE_GUIDANCE_DETECTION_MAXIMUM_AGE_S` | seconds | `0.250` | greater than `0`, at most `0.250` | Older detection evidence stops terminally. |
| `BORDER_COLLIE_GUIDANCE_NEAR_BOTTOM_RATIO` | frame-height ratio | `0.86` | greater than `0`, at most `1` | Counts only while the fruit is also centered. |
| `BORDER_COLLIE_GUIDANCE_NEAR_CENTER_RATIO` | frame-height ratio | `0.72` | greater than `0`, at most `1` | Counts only on fresh matching evidence. |
| `BORDER_COLLIE_GUIDANCE_NEAR_CONFIRMATIONS` | fresh frames | `3` | integer `>=1` | Duplicate or weak observations never advance Arrival. |
| `BORDER_COLLIE_GUIDANCE_NEAR_LOSS_GRACE_S` | seconds | `0.75` | greater than `0` | Missing fruit can close out only while the centered near latch is recent. |
| `BORDER_COLLIE_GUIDANCE_FINAL_PUSH_MPS` | m/s | `0.60` | `0.50..1.0` | Exactly one final-push episode is allowed. |
| `BORDER_COLLIE_GUIDANCE_FINAL_PUSH_DURATION_S` | seconds | `1.0` | greater than `0`, at most `1.0` | The push is terminal and cannot be restarted. |

Fruit acquisition and tracking confidence remain owned by
`border_collie_demo.fruits.FRUIT_POLICIES`, including the Banana specialist's
sidecar route. Guidance does not replace or bypass those per-fruit thresholds.

## Safety behavior

- Camera/source staleness, generation changes, regressed frames, invalid
  geometry, and a changed identity produce a terminal zero command.
- A duplicate frame reuses the previous already-authorized command for at most
  `250 ms`; it never advances acquisition or Arrival counters.
- A fruit outside the outer corridor remains identified, but forward authority
  is zero until recentered.
- Lower-edge disappearance before the centered near latch stops. After the
  latch it starts one bounded final push and then returns an Arrival stop.
