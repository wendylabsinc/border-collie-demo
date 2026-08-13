# Stage-default camera guidance

`FruitGuidance` is the single camera-guidance interface from Target Fruit search
through Arrival. A caller feeds each sidecar status sample to `observe(...)` and
executes only the returned command. The module privately retains the acquired
fruit identity, camera generation, fresh-frame counters, lower-edge latch, and
final-push state; there is no search-to-approach handoff token.

The defaults retain the physically useful base behavior with the qualified
slower-search experiment: search at `0.40 rad/s`, require three fresh centered samples, translate
continuously at `1.0 m/s`, steer while moving outside the center band, remove
forward authority outside the outer corridor, and permit one `0.60 m/s` by
`1.0 s` final push only after centered lower-edge evidence disappears. An
activation may replace the final-push speed and duration for that one run;
those values are frozen with the mission and never mutate process-wide
guidance.

## Per-run tuning

The current interface is documented in [`run-tuning.md`](run-tuning.md). The
server publishes defaults, units, valid ranges, and safety constraints to the
UI; each activation persists one immutable `RunTuning` snapshot. Changing a
value for the next run does not mutate the environment, rebuild an image,
restart a service, or change an active run.

## Legacy runtime environment

The following variables remain deployment defaults for backward compatibility.
They are no longer the experiment interface. The media sidecar publishes raw
same-label temporal evidence and does not read motion-policy thresholds.
Validation is fail-fast and cannot widen the camera freshness or hardware
motion safety limits.

| Environment variable | Units | Default | Valid range | Safety constraint |
| --- | --- | ---: | ---: | --- |
| `BORDER_COLLIE_GUIDANCE_SEARCH_YAW_RPS` | rad/s | `0.40` | `0.40..0.80` | `0.50` is physically proven; `0.40` is the supervised slower-search experiment. Lower values remain rejected because `0.24..0.30` produced posture changes without a useful turn. |
| `BORDER_COLLIE_APPLE_FOCUS_CONFIDENCE` | confidence ratio | `0.50` | `0.50..0.70`, at least the Apple acquisition floor | The first fresh Apple observation at this floor stops the broad sweep and starts zero-motion focused confirmation. It never authorizes translation. |
| `BORDER_COLLIE_APPLE_ACQUISITION_CONFIDENCE` | confidence ratio | `0.40` | `0.40..0.70`, at most the Apple focus floor | After focus begins, three fresh centered Apple observations at or above this app-owned floor lock identity. A weaker or missing observation holds at zero and resets confirmation; stale or unhealthy evidence still fails closed. |
| `BORDER_COLLIE_GUIDANCE_SEARCH_SWEEP_RAD` | radians | `6.283185` | greater than `0`, at most one revolution | Fresh measured pose bounds the search; command duration is not treated as rotation proof. |
| `BORDER_COLLIE_GUIDANCE_CENTER_TOLERANCE_RATIO` | frame-width ratio from center | `0.08` | greater than `0`, less than outer corridor | Three fresh samples must be inside this band before lock. |
| `BORDER_COLLIE_GUIDANCE_CENTER_CONFIRMATIONS` | fresh frames | `3` | integer `>=1` | Duplicate frames never advance this count. |
| `BORDER_COLLIE_GUIDANCE_APPROACH_FORWARD_MPS` | m/s | `1.0` | `0.50..1.0` | Preserves the factory-avoidance translation floor and configured maximum. |
| `BORDER_COLLIE_GUIDANCE_APPROACH_YAW_RPS` | rad/s | `0.30` | greater than `0`, at most `0.80` | Applied only while translating inside the outer corridor. |
| `BORDER_COLLIE_GUIDANCE_OUTER_CORRIDOR_RATIO` | frame-width ratio from center | `0.20` | greater than center band, less than `0.50` | Forward authority is removed outside this band. |
| `BORDER_COLLIE_GUIDANCE_RECENTER_YAW_RPS` | rad/s | `0.50` | `0.50..0.80` | In-place recentering never includes forward motion. |
| `BORDER_COLLIE_GUIDANCE_DUPLICATE_HOLD_S` | seconds | `0.250` | greater than `0`, at most `0.250` | A duplicate cannot extend the original fresh-evidence authority window. |
| `BORDER_COLLIE_GUIDANCE_SOURCE_MAXIMUM_AGE_S` | seconds | `0.350` | greater than `0`, at most `0.350` | Older source evidence stops terminally. |
| `BORDER_COLLIE_GUIDANCE_DETECTION_MAXIMUM_AGE_S` | seconds | `0.250` | greater than `0`, at most `0.250` | Older detection evidence stops terminally. |
| `BORDER_COLLIE_GUIDANCE_NEAR_BOTTOM_RATIO` | frame-height ratio | `0.90` | greater than `0`, at most `1` | One fresh confidence-qualified observation at or below this image edge confirms Arrival. |
| `BORDER_COLLIE_GUIDANCE_NEAR_CENTER_RATIO` | frame-height ratio | `0.72` | greater than `0`, at most `1` | Counts only on fresh matching evidence. |
| `BORDER_COLLIE_GUIDANCE_NEAR_CONFIRMATIONS` | fresh frames | `3` | integer `>=1` | Duplicate or weak observations never advance Arrival. |
| `BORDER_COLLIE_GUIDANCE_NEAR_LOSS_CONFIRMATIONS` | fresh frames | `2` | integer `>=2` | One weak or missing observation stops but cannot start the final push. |
| `BORDER_COLLIE_GUIDANCE_NEAR_LOSS_GRACE_S` | seconds | `0.75` | greater than `0` | A disappearance can confirm Arrival only after a fresh confidence-qualified observation reached the configured lower-edge threshold. Otherwise loss commands exact zero and waits for reacquisition until timeout. |
| `BORDER_COLLIE_GUIDANCE_FINAL_PUSH_MPS` | m/s | `0.60` | `0.50..1.0` | Exactly one final-push episode is allowed. |
| `BORDER_COLLIE_GUIDANCE_FINAL_PUSH_DURATION_S` | seconds | `1.0` | `0`, or `0.10..1.50` | The push is terminal and cannot be restarted. Zero disables it and leaves Arrival stopped. |

Fruit acquisition and tracking confidence remain owned by
`border_collie_demo.fruits.FRUIT_POLICIES`, including the Banana specialist's
sidecar route. Guidance does not replace or bypass those per-fruit thresholds.

The audience UI may supply bounded focus and lock confidence for the selected
Target Fruit as part of one Demo Run activation. Those values are validated
against the per-fruit ranges in [`search-experiment.md`](search-experiment.md),
persisted with the selected fruit before preflight, and applied by replacing
the `FruitPolicy` passed to that run's `FruitGuidance`. The process-wide policy
map is never mutated. Activations without an override preserve the baseline
policy, including no separate focus phase for Pear and Banana.

The same activation may include `final_push.speed_mps` in `0.50..1.0` m/s and
`final_push.duration_s` equal to `0` or in `0.10..1.50` seconds. For example,
`0.60 m/s` for `0.40 s` is a valid one-run experiment. The server rejects
unknown, non-finite, or out-of-range fields before preflight, persists the exact
effective pair in the activation, Run Result, and black-box trace, and treats a
changed pair under the same activation ID as an idempotency conflict.

## Safety behavior

- Camera/source staleness, generation changes, regressed frames, invalid
  geometry, and a changed identity produce a terminal zero command.
- A duplicate frame reuses the previous already-authorized command for at most
  `250 ms`; it never advances acquisition or Arrival counters.
- A fruit outside the outer corridor remains identified, but forward authority
  is zero until recentered.
- Lower-edge disappearance before the centered near latch stops. After the
  latch it starts one bounded final push and then returns an Arrival stop.
