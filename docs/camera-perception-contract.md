# Camera and perception evidence contract

Motion may be authorized only by perception derived from a camera source that
proves source progress. Callback receipt, local counters, and changing HTTP
responses are not source-progress evidence.

## Required frame identity

Every accepted frame must carry:

- an opaque connection generation;
- a source-owned marker that strictly advances within that generation;
- local monotonic receipt time; and
- dimensions plus image data belonging to that same marker.

For the WebRTC adapter, decoded-frame PTS together with its time base is the
source marker. `CAMERA-SOURCE-001` and `CAMERA-RECONNECT-001` established that
both values are present and PTS advances strictly within each measured
connection generation. PTS remains an opaque identity marker; it is not a
wall-clock age or cadence measurement.

An adapter without a trustworthy generation and advancing source marker must
never authorize motion. A content fingerprint may help diagnostics, but it is
not sufficient evidence because a stationary live scene may contain unchanged
pixels.

## Detection binding

Every pear detection must retain the generation and source marker of the exact
frame used for inference. A callback that repeats or regresses a source marker:

- does not refresh camera age;
- does not advance detection stability;
- does not refresh an existing pear observation; and
- cannot extend a motion command lease.

## Production readiness seam

The mission process does not import WebRTC, GPU, TensorRT, or detector code. A
read-only perception sidecar owns those details and exposes one JSON status
document. The mission-side adapter independently validates this evidence rather
than trusting a sidecar-provided `ready` flag:

```json
{
  "generation": "opaque-generation",
  "source": {
    "pts": 12345,
    "time_base": "1/90000",
    "received_monotonic_s": 100.0,
    "consecutive_frames": 10
  },
  "detection": {
    "label": "pear",
    "confidence": 0.81,
    "consecutive_detections": 5,
    "inference_s": 0.08,
    "completed_monotonic_s": 100.05
  }
}
```

The sidecar may also expose its latest JPEG preview for operator visibility.
That preview is annotated with model state and pear evidence, is never used as
mission input, and grants the browser no camera or motion authority.

Both processes must run on the same Woof host so their monotonic timestamps use
the same kernel clock. A missing, future, malformed, stale, unreachable, or
cross-host timestamp fails readiness closed. The complete validated document is
persisted in the Demo Run preflight evidence.

## Qualified thresholds

These thresholds apply to the current pear-qualified Operating Envelope. They
may be relaxed only after new acceptance evidence is recorded.

### Source progress and preflight

- Maximum source-progress age is **0.350 seconds**, measured with the local
  monotonic clock from receipt of the last strictly advancing source marker.
- Camera preflight requires **10 consecutive accepted frames** from one opaque
  connection generation. Every frame must have present, strictly increasing
  PTS, the same time base, and an inter-frame gap no greater than 0.350 seconds.
- Missing identity, repeated or regressed PTS, a time-base change, a generation
  change, or an excessive inter-frame gap resets the preflight count to zero.
- Advancing-camera preflight must pass before activation. A visible pear is not
  part of this gate because Woof turns toward the fruit-search area after the
  run starts.
- The perception service and model must be healthy before activation, but pear
  qualification starts during `TURN_TO_FRUIT`. The turn stops on five qualified
  detections and otherwise remains bounded by one measured revolution and 30
  seconds. `FIND_FRUIT` retains a bounded confirmation/reacquisition search.
  Search motion stops if the camera source becomes unhealthy.
- A replacement connection always receives a new generation and must pass the
  complete preflight. It cannot resume the prior Demo Run.

### Pear evidence

- A qualifying pear detection has confidence **greater than or equal to 0.65**.
- Pear acquisition requires **5 consecutive qualifying detections**, each
  bound to a distinct accepted frame in the current generation.
- A miss or a detection below 0.65 resets pear acquisition stability to zero.
- Warm-up must finish before preflight passes. After preflight, detector
  execution time must be **no greater than 0.200 seconds**.
- Detection age must be **no greater than 0.250 seconds**, measured with the
  local monotonic clock from source-frame receipt until the instant the
  detection is used to authorize or extend motion.
- A detection that violates execution-time or age limits is untrusted and
  cannot authorize or extend motion.

## Failure behavior

During an active Demo Run, any of the following terminates the run as
`CAMERA_FAILURE`, stops and disarms Woof, and records the failed phase:

- source progress exceeds 0.350 seconds;
- source identity is missing, repeats, or regresses;
- the source time base changes;
- the connection generation changes;
- detector execution exceeds 0.200 seconds while camera-guided motion is
  active;
- detection age exceeds 0.250 seconds while camera-guided motion is active; or
- camera-derived evidence otherwise cannot be proven fresh while it is needed
  for motion.

A replacement connection never resumes the failed run. It must pass fresh-frame
preflight before the operator can start a new Demo Run.

A fresh accepted frame without a qualifying pear is not itself a camera
failure. It means there is no trustworthy pear observation. Camera-guided
motion must stop, and the search/approach contract decides whether bounded
reacquisition is allowed or the run terminates as target loss.

## Evidence and remaining qualification

The thresholds above are based on the guarded clean-repository results in
[`validation-results.md`](validation-results.md):

- `CAMERA-SOURCE-001` measured 70 ms mean, 89 ms p95, and 146 ms maximum
  callback cadence with no missing, repeated, or regressed PTS.
- `PEAR-EVIDENCE-001` measured 100% pear matches, 0.812 minimum confidence,
  82 ms p95 detector execution, and 123 ms p95 detection age.
- `CAMERA-RECONNECT-001` proved distinct generation binding and full fresh
  replacement preflight 6.677 seconds after a planned client disconnect.

Packet loss, robot-side camera failure, and active-run `CAMERA_FAILURE` Run
Result behavior remain acceptance work. They do not authorize looser thresholds
or in-run recovery.
