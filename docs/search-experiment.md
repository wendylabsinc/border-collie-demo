# Search yaw experiment

The audience UI exposes a bounded search experiment for the next Demo Run. The
values are part of the activation request and are persisted in the Run Result
before motion begins. They do not change process-wide environment variables and
do not rebuild or redeploy either service.

| Setting | Default | Allowed range | Unit |
| --- | ---: | ---: | --- |
| Initial search yaw | `0.40` | `0.40..0.80` | radians/second |
| Apple focus confidence | `0.50` | `0.50..0.70` | probability |
| Apple acquisition confidence | `0.40` | `0.40..0.70`, no greater than focus | probability |
| Centered confirmations | `3` | `2..5` | fresh frames |
| Center tolerance | `0.08` | `0.05..0.12` | fraction of image width from center |

Existing camera health, freshness, generation, frame-progress, geometry,
motion-authority, watchdog, and exact-zero stop requirements remain mandatory.
Changing a search setting cannot bypass those interlocks.

## Evidence

Every processed frame during the initial turn records:

- target fruit and source frame identity;
- raw confidence and horizontal box center;
- requested yaw rate, measured yaw angle, and cumulative search turn;
- controller action and reason;
- whether the frame advanced; and
- whether the target was locked.

Successful turn evidence is stored at
`stage_results.turn_to_fruit.search_trace`. A failed initial turn stores the
same evidence at `failure_details.recognition.search_trace`. Both include a
confidence summary with detected-frame count, minimum, maximum, mean, and lock
confidence.

`GET /api/experiments/search` groups persisted runs by fruit and selected yaw
rate and reports run count, lock count, terminal success count, lock rate, and
end-to-end success rate. The UI renders the latest trace and the grouped
scorecard so successive yaw settings can be compared without rebuilding.
