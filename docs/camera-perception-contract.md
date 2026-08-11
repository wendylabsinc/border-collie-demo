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
  "supervision": {
    "state": "ready",
    "ready": true,
    "generation": "opaque-generation",
    "stable_frames": 10,
    "restart_budget": 5
  },
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
    "inference_passes": 2,
    "completed_monotonic_s": 100.05,
    "crop_confirmation": {
      "attempted": true,
      "promoted": true,
      "full_frame_confidence": 0.60,
      "crop_confidence": 0.81,
      "crop_xyxy": [497, 372, 753, 628],
      "agreement_iou": 0.72
    }
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

The sidecar process stays available while its SDK-neutral service supervisor
owns WebRTC connection attempts. A partial or stale session is disconnected
before each retry, failures use bounded exponential backoff, and exhausting the
configured restart budget produces an explicit `failed` state. Every retry
receives a new generation. The supervisor reports `degraded` until ten strictly
advancing frames from that one generation have arrived, then reports `ready`.
A frame stall revokes readiness before cleanup and reconnect. The mission-side
adapter requires the supervisor generation to match the camera generation and
still validates all source and detection evidence independently; the
supervisor's `ready` flag never authorizes motion by itself.

Connection setup and camera progress have separate deadlines. The 15-second
transport timeout bounds WebRTC and DataChannel setup. Only after the video
channel is enabled does the 3-second first-frame grace period begin. Once the
first frame arrives, the 0.75-second frame-stall timeout applies. Recovery uses
an 8-second initial robot-signaling cooldown with bounded exponential backoff;
this prevents rapid reconnects from overrunning the Go2 signaling endpoint.

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
  required by this gate. **The initial search is conditional:** if fresh
  qualified Target Fruit evidence is already present when a search stage
  begins, the stage records a zero-motion skip and sends no search rotation;
  otherwise Woof turns through the fruit-search area after the run starts.
- The perception service and model must be healthy before activation, but pear
  qualification starts no later than `TURN_TO_FRUIT`. Five fresh qualified
  detections already in view skip search before motion is armed. Otherwise the
  turn stops on five qualified detections and remains bounded by one measured
  revolution and 30 seconds. `FIND_FRUIT` retains the same zero-motion skip or
  a bounded confirmation/reacquisition search.
  Search motion stops if the camera source becomes unhealthy.
- A replacement connection always receives a new generation and must pass the
  complete preflight. It cannot resume the prior Demo Run.

### Pear evidence

- A qualifying pear detection has confidence **greater than or equal to 0.65**.
- Pear acquisition requires **5 consecutive qualifying detections**, each
  bound to a distinct accepted frame in the current generation.
- A miss or a detection below 0.65 resets pear acquisition stability to zero.
- A full-frame candidate at or above 0.35 receives at most one crop-confirm
  pass when its box occupies at most 0.5% of the frame or its confidence is
  below 0.65. The same engine and inference worker are reused. A crop result is
  promoted only when it reaches 0.55, improves the proposal, and overlaps it by
  at least 0.10 IoU. The combined time for both passes is the reported detector
  execution time; crop confirmation does not relax any freshness deadline.
- During bounded search, a crop-confirmed full-frame proposal at or above 0.50
  slows rotation with alternating reliable-rate yaw and zero-yaw heartbeats.
  The controller does not substitute a weaker yaw signal because smaller turn
  commands have not moved Woof reliably. If the enlarged result does not reach
  the unchanged 0.65 acquisition threshold for five frames, bounded rotation
  continues. Only qualified acquisition stops the search.
- After acquisition is complete, approach tracking uses hysteresis: **3
  consecutive pear detections at or above 0.55** may extend tracking. This
  lower threshold cannot acquire a pear, start a search result, or bypass any
  freshness, generation, geometry, or camera-health gate. A weaker or missing
  track commands zero motion until the approach contract either reacquires the
  pear or confirms a bounded sight-lost-close Arrival.
- Search qualification crosses into approach through one explicit, immutable
  handoff. It records the requested fruit, camera generation, source PTS and
  time base, qualification monotonic time, five-frame stability count,
  acquisition-grade confidence, and normalized geometry. Approach consumes
  that handoff at most once and only within 0.250 seconds when a current fresh
  same-fruit sample matches generation/time base, has non-regressed source
  identity, remains geometrically continuous and inside the broad centering
  corridor, and meets the fruit's existing tracking floor. The handoff seeds
  identity only: the current sample commands zero/centering, and three current
  fresh centered samples are still required before forward motion. A stale,
  mismatched, regressed, weak, invalid, discontinuous, or off-axis handoff is
  discarded and ordinary acquisition resumes at the unchanged per-fruit
  acquisition threshold.
- A previously acquired red-apple track may continue below the normal tracking
  floor, down to 0.10 confidence, only after its lower edge or the prior lower
  edge reaches 0.70 of frame height and its geometry remains continuous. Its
  horizontal center may move at most 0.20 of frame width between accepted
  samples; its vertical center and lower edge may retreat by at most 0.08.
  This rule cannot acquire an apple or accept a label change.
- Near-fruit geometry slows motion and establishes close-range track memory.
  In the deployed stage-camera profile, apple, banana, and pear share the same
  geometry thresholds. Three fresh centered near samples enter Final Approach;
  this transition is one-way for that approach. Questionable evidence revokes
  motion immediately without returning to ordinary tracking. Two fresh,
  advancing, camera-healthy weak or missing frames within the grace window
  confirm lower-edge loss and authorize exactly one 0.6 m/s, 1.0 second final
  push before the mandatory stop. Duplicate or stale frames cannot advance
  terminal confirmation, and stale evidence does not erase the latch. Wrong
  label or generation, off-axis fresh evidence, invalid geometry, camera
  failure, or latch expiry cancels push authority and remains fail-closed.
  A detection whose area shrinks by more than the unchanged 35% continuity
  limit counts as one loss-confirmation sample only when its source marker is
  fresh and advancing, its label and generation still match, the camera is
  healthy, both it and the preceding accepted box remain within 0.08 of image
  center and at the clipped lower edge, and neither center jump nor vertical
  retreat is present. This sample commands zero motion; it never extends the
  close approach. The second separately qualified loss sample is still
  required before final-push authorization. Area retreat elsewhere remains a
  discontinuity and cancels Final Approach.
- The optional metric profile instead treats image geometry only as entry to
  Final Approach and uses the following one-way handoff contract.
  Camera-to-LiDAR handoff requires the last three fresh filtered centers to
  remain within 0.08 of frame center. The final camera center is converted to a
  body-frame bearing using the calibrated horizontal field of view; post-loss
  LiDAR candidates must remain within six degrees of that bearing and satisfy
  temporal continuity. A nearer cluster on the wrong side is not the Target
  Fruit. The camera-to-LiDAR transition is one-way for the remainder of that
  approach: after LiDAR handoff is latched, weak, missing, contradictory, or
  newly qualified camera detections are diagnostic evidence only and cannot
  steer, stop, advance, or return control to camera tracking. Fresh LiDAR
  association and the metric Arrival gate exclusively authorize subsequent
  motion; unavailable LiDAR fails closed rather than falling back to vision.
  A camera transport or source-health failure may still abort the Demo Run,
  but cannot restore camera control.
  Off-axis disappearance before handoff sends zero motion, terminates as
  `TARGET_LOST_OFF_AXIS`, and never arms LiDAR or authorizes a blind push.
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

A fresh accepted frame without acquisition-grade pear evidence is not itself a
camera failure. Search motion must stop. During approach, only the bounded
tracking hysteresis above may extend an already-acquired track; otherwise
camera-guided motion stops and the approach contract decides whether bounded
reacquisition is allowed or the run terminates as target loss.

`TARGET_LOST_OFF_AXIS` and an `ARRIVAL_FAILURE` from the approach phase are
Recoverable Failures only after zero motion and disarm are confirmed. Their
automatic correction preserves the original Home, records terminal perception
evidence, lies down without barking, stands, waits for three fresh trusted
continuous Home-fusion samples, and performs one bounded position-only Home
recovery. Camera failure, Remote Takeover, stale or untrusted Home/pose/fusion,
missing or excessive outbound pulse evidence, hardware faults, active motion,
and unconfirmed posture or stop results cannot authorize translation. Recovery
never converts the failed fruit attempt into success.

## Evidence and remaining qualification

The provisioned engine also represents apple and banana. `RED-APPLE-001`
qualifies red apple at 0.70 confidence by five fresh detections; pear remains
qualified at 0.65 by five. Banana stays a camera-only **Supported Fruit**. The
green apple trial produced no apple proposal and was classified as pear when
class filtering was removed, so green apple is outside the qualified operating
envelope. The `/fruit-test` surface remains motion-free. Selecting a different
class clears detection stability, and starting a Demo Run explicitly restores
its chosen qualified class before preflight.

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

## Published detection confidence is raw by design

The sidecar publishes the model router's best candidate for the selected
fruit at whatever confidence the model produced, including very weak
detections. This is a deliberate contract, decided 2026-08-08 after a static
0.010-0.016 "pear" phantom held an arrival check open (the check lacked a
confidence floor; see the arrival-visibility fix on the approach-consistency
branch):

- **Every motion-relevant consumer MUST apply its own explicit confidence
  floor.** Current floors live in `fruits.py` (acquisition and close-range
  tracking per fruit) and in the arrival-visibility check. The lowest
  legitimate consumer floor is apple's 0.10 close-range tracking confidence,
  so a sidecar-side floor would have to sit below 0.10 to avoid starving the
  close-range continuation path - at which point it filters only absolute
  noise while adding a second policy location that must forever stay below
  every app floor.
- Raw publication is what makes failures diagnosable from telemetry: the
  phantom above was identified from recorded sub-floor samples, and the
  close-range confidence collapse of a real pear (0.27 at bottom 0.997) was
  distinguishable from track loss only because sub-floor frames were visible.
- The camera-only `/fruit-test` page intentionally displays weak detections
  for exploration and qualification evidence.

Consequently the sidecar stays runtime-neutral and policy-free: gating
belongs to consumers, and new consumers must not assume published detections
are qualified.
