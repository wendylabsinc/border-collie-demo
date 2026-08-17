# Border Collie Demo

This branch includes an on-device **Hey Wendy** voice service. Open
`http://woof.local:8092`, say “Hey Wendy,” then say an allowlisted request such
as “find the pear” or “follow the apple.” The custom wake model gates local
Parakeet ASR; the service calls the same idempotent `/api/run` safety boundary
as the audience UI and never owns a separate motion path.

A clean-room implementation of the Wendy Labs Border Collie routine for the
Unitree Go2.

This repository intentionally does not import from the original `collie-demo`
application. The old repository remains useful as test evidence and a hardware
reference, but it is not a runtime dependency.

## Intended routine

1. Verify the advancing camera source, perception service, pose, motion, media,
   and return-home readiness. A visible pear is not required yet.
2. Confirm that Woof is facing the person. This is initially an operator setup
   requirement rather than autonomous person detection.
3. Capture a stable Home position and heading.
4. Choose Pear or Red apple from the audience UI's qualified-fruit list, then
   use the single **Activate Demo** control.
5. **Conditional search:** if fresh qualified evidence already shows the
   requested fruit, do not perform the initial rotation. Record the
   turn/search stages as conditionally skipped and proceed directly to approach
   centering. Otherwise, turn through the fruit-search area until it is
   recognized, bounded by one measured revolution and a 30-second timeout. A 50%+
   full-frame fruit proposal triggers crop confirmation and a 50% duty-cycled
   turn using the same reliable yaw signal; an unqualified crop keeps rotating
   rather than becoming a false stop.
6. Confirm/reacquire and approach the requested fruit using fresh detections.
   No forward command is permitted before this stage. Once approach begins,
   forward and yaw inputs may be combined to steer toward the fruit.
7. Near-fruit geometry arms the lower-edge Arrival gate but does not stop the
   approach. While the requested fruit remains freshly visible, continue the
   bounded forward approach. Only after it disappears through the lower camera
   edge may Woof send the single bounded final movement.
8. First turn at a fixed 0.50 rad/s until the Target Fruit is within the middle
   16% of the camera, then hold zero yaw for three fresh centered samples. After
   qualified lower-edge disappearance, send the one-run bounded final push
   (default 0.60 m/s by 1.0 second), stop, lie down, attempt the best-effort
   bark, and remain down for 5 seconds.
9. Stand, align toward Home through regular Sports yaw, then replay the recorded
   number of outbound forward heartbeats through factory obstacle avoidance at
   1.0 m/s with bounded moving yaw. Fresh pose remains the authority for the
   10 cm position-only Home success gate and recorded Home Distance.
10. Stop all motion, record the result, and report completion.

Arbitrary typed commands remain on the separate debug surface. The supervised
voice adapter may activate a supported fruit request through the same narrow
mission API; it does not bypass any readiness or motion gate.

## Voice Demo V1 checkpoint

The first working voice-demo checkpoint supports a spoken request for a
Qualified Fruit and both initial camera conditions:

- when the requested fruit is already freshly qualified in view, the initial
  rotation/search is conditionally skipped; and
- when the requested fruit is not initially in view, Woof performs the bounded
  search before beginning the same approach sequence.

This is a working demo checkpoint, not a claim of precise motion. The next
motion-quality work is to make Return to Home finish more consistently at the
captured Home position and heading, and to make the final fruit distance more
consistent so Woof does not finish too close or too far away.

## Deferred interaction features

- **Unsupported voice command reaction (not implemented):** if speech is
  recognized but does not map to a supported fruit request, Woof should look
  back toward the person and tilt its head as expressive feedback. This future
  reaction must not start a Demo Run, infer a different command, or continue
  autonomous movement. Person reacquisition and the head-tilt behavior require
  their own safety and acceptance work before this can become runtime behavior.

### Test another fruit without motion

The provisioned TensorRT engine contains `apple`, `banana`, and `pear`. Pear,
red apple, and banana are **Qualified Fruits** available in the audience UI.
Banana's qualification is conditional on the resident banana specialist: the
media router publishes a banana detection only when the banana-only model
confirms the general proposal at 0.55 confidence with box agreement, so every
motion decision for banana already sits behind that specialist gate. The apple
qualification is explicitly limited to a red apple in the
tested placement and lighting: the green apple trial was misclassified as pear
and is not an interchangeable substitute.

Open `http://woof.local:8110/fruit-test`, choose Apple or Banana, and select
**Show this fruit**. The live overlay switches the engine to that class while
the page exposes no motion control. Start with an apple, then repeat with a
banana at the same placement and lighting used for the pear qualification.
Record the visible confidence, whether the box stays on the correct fruit, and
whether five consecutive detections are reached. Switching fruit clears prior
detection stability so evidence from one class cannot qualify another. The red
apple passed `RED-APPLE-001` at the unchanged 0.70 threshold and reached 15
consecutive qualifying detections.

The acquisition thresholds are 0.70 for apple, 0.20 for banana, and 0.65 for
pear. Banana's low app-side value rides on top of the specialist's 0.55 floor
rather than standing alone. Fruit-specific thresholds may change only after
fresh evidence; the qualified pear and red-apple thresholds remain unchanged.

Any physical remote-control input must eventually cause a latched
`REMOTE_TAKEOVER`. Autonomous control must stop and cannot resume until the
application process is restarted. Remote-input detection may be implemented
after the core routine, but verified takeover behavior is a required final
stage-qualification gate.

## Qualified operating envelope

The stage-ready milestone targets a controlled **Operating Envelope**, not
arbitrary rooms or fruit placements. Before the end-to-end routine can be
called qualified, its acceptance record must state the tested:

- Woof starting pose and heading;
- pear placement area and distance range;
- floor surface and clear travel corridor;
- lighting and camera visibility conditions;
- person and audience boundaries; and
- robot, compute, and network configuration.

The numeric and environmental limits are Wayfinder decisions and must not be
invented from unvalidated prototype behavior. If any qualified condition
changes, the affected acceptance tests must be repeated and their new results
recorded before relying on the demo. Outside the qualified envelope, Woof must
stop safely instead of improvising.

## Repository boundary

- `src/border_collie_demo/`: mission domain and narrow hardware contracts.
- `media/`: sole owner of Go2 WebRTC camera, fruit inference, and bark audio.
- `web/`: minimal audience and debug surfaces.
- `lab/`: isolated human-operated hardware tests, never imported by production.
- `tests/`: unit, contract, replay, and later hardware acceptance tests.
- `docs/`: behavior, known hardware facts, and validation results.

Camera and detector implementations must satisfy
[`docs/camera-perception-contract.md`](docs/camera-perception-contract.md)
before they can authorize motion.

Demo Run persistence and result APIs must satisfy
[`docs/run-result-contract.md`](docs/run-result-contract.md) before the
audience **Activate Demo** control can be enabled.

The recorder-backed activation, production stage orchestration, and diagnostic
stage-result slices are implemented.
**Activate Demo** durably creates a pear Demo Run before checking durable
storage, hardware connection, fresh pose, motion disarm, and advancing-camera
readiness. Missing readiness seals the run as `PREFLIGHT_FAILURE`; a fully ready
boundary captures one fresh robot-local pose as Home and advances to
`WAIT_FOR_COMMAND`. A configured stage executor then records turn, find,
approach, Arrival, sit/bark, stand, turn-home, return, and heading-restoration
evidence before sealing success. Loss of pose freshness during capture also fails closed.
**Stop Woof** seals an active run and permits another activation, while process
restart seals unfinished work as `PROCESS_INTERRUPTED`. The production executor
uses measured pose turns, bounded camera-guided search, geometry-gated approach,
one bounded per-run off-screen final push, Unitree posture actions, best-effort
bark, and closed-loop odometry return through factory obstacle avoidance.
The initial turn/search is explicitly **conditional**, not an unconditional
part of every run. Target Fruit qualification may already be present when
`TURN_TO_FRUIT` begins. Fresh qualified evidence for the selected Target Fruit
skips both broad-search stages without arming motion; otherwise Woof performs
the camera-guided turn and `FIND_FRUIT` confirms or reacquires it. The absence
of a fruit before activation is expected and does not block the button.
The audience UI also shows the media sidecar's latest annotated camera frame so
the operator can see the live image, pear box, confidence, and 5-frame model
qualification progress without granting the UI any motion authority.

After every run, `/debug` shows `COMPLETED`, `FAILED`, or `NOT_RUN` for each of
the eight executable stages and the exact recorded evidence. The terminal Run
Result also materializes the latest trustworthy Home Distance and heading error.
It links a bounded evidence archive containing the latest 40 raw, unannotated
camera frames sampled at 0.5-second intervals, an annotated terminal frame, and
the recognition summary used to explain a failed search. The rolling archive is
diagnostic evidence, not a complete video recording.

### Label captured frames in Fieldmark

Fieldmark remains a perception-only labeling tool and never receives motion
authority. There are two supported ways to give it images:

1. In Fieldmark, use **Capture from Go2** and set the frame endpoint to
   `http://woof.local:8111/api/camera/raw.jpg`.
2. After a Demo Run, open `http://woof.local:8110/debug`, download
   `evidence.zip`, extract `frames/*.jpg`, and upload those images to Fieldmark.
3. For a no-upload local pass, run the reusable
   [`lab/run-labeler`](lab/run-labeler/README.md) against `evidence.zip`. Its
   link opens the run directly, keeps human corrections in the browser, and
   exports normalized pear boxes as JSON.

The files under `frames/` are deliberately unannotated so the current detector
cannot bias manual boxes. `terminal/annotated.jpg`, `terminal.jpg`, and
`manifest.json` retain the model output and source metadata for comparison.
Choose the matching fruit class in Fieldmark, draw the boxes, then export the
dataset in the required YOLO or COCO format. Every fruit class uses this same
runtime-neutral evidence contract rather than adding motion controls to the
labeler.

Pear acquisition remains fixed at 0.65 confidence for five consecutive fresh
frames. Once that acquisition has succeeded, approach tracking may continue at
0.55 confidence after three consecutive fresh pear detections. These values are
separately configurable with
`BORDER_COLLIE_PEAR_TRACKING_MIN_CONFIDENCE` and
`BORDER_COLLIE_PEAR_TRACKING_CONFIRMATIONS`; changing them does not alter the
acquisition rule.

Red apple acquisition remains fixed at 0.70 confidence for five fresh frames.
After acquisition, a close red-apple track may continue down to 0.10 confidence
only when the box is already low in the image and remains spatially continuous:
its horizontal center cannot jump by more than 0.20 of the frame and its lower
edge or vertical center cannot retreat by more than 0.08. This close-range rule
cannot acquire an apple and must be requalified if the fruit, model, camera, or
stage setup changes.

For a complete zero-motion base Demo Run, explicitly start with
`BORDER_COLLIE_RUNTIME_MODE=simulation`. The audience UI shows a persistent
simulation banner, every stage records `motion_commands_sent: false`, and the
complete Run Result and diagnostic stage matrix can be exercised. The default
is `production`; simulation is never selected implicitly and never sends Go2
commands.

Audience and operator surfaces must satisfy
[`docs/ui-contract.md`](docs/ui-contract.md); diagnostic actions remain
separate from production mission control.

Return behavior is bounded by the draft
[`docs/return-home-contract.md`](docs/return-home-contract.md). Its state model is
defined, but marked Operating Envelope and motion values remain provisional
until WDY-2281 and WDY-2283 are qualified on hardware.

The live foundation includes the tested Unitree DDS connection,
`rt/sportmodestate` pose subscriber, factory `ObstaclesAvoidClient` motion
boundary, and production stage executor. Deployment gates remain disabled in
the checked-in manifest until the operator intentionally enables the real run.

## Live hardware gates

Two independent environment gates are required:

- `BORDER_COLLIE_RUNTIME_MODE=production` selects the fail-closed production
  runtime. Use `simulation` only for the explicit zero-motion base demo.
- `BORDER_COLLIE_HARDWARE_ENABLED=1` initializes DDS, pose, SportClient, and
  ObstaclesAvoidClient. This is sufficient for read-only connection checks.
- `BORDER_COLLIE_AUTONOMY_ENABLED=1` permits the production stage executor to
  issue its bounded motion and posture actions.
- `BORDER_COLLIE_LAB_MOTION_ENABLED=1` additionally permits the one guarded
  forward-pulse endpoint.
- `BORDER_COLLIE_PERCEPTION_ENABLED=1` enables the read-only production
  perception status adapter. It independently enforces the qualified source and
  pear thresholds against the configured sidecar evidence and fails closed when
  that evidence is missing, stale, malformed, or unreachable.
- `BORDER_COLLIE_BARK_ENABLED=1` enables the direct best-effort AudioHub bark
  request after Woof lies down. Bark readiness is observable but never blocks
  activation or motion, and bark failure never skips the five-second down hold.

The production media process is `media.perception_sidecar:app` on port `8111`.
It owns one Go2 WebRTC connection, advances PTS/time-base evidence, binds every
pear box to its exact generation and source marker, and exposes `/api/bark` on
the same connection. It also exposes the raw Fieldmark source at
`/api/camera/raw.jpg` and the bounded archive at `/api/evidence/clip.zip`. It
never imports or creates a motion client. The checkpointed TensorRT engine is
versioned at `media/model/model.engine` and deployed as `/media/model.engine`;
model binaries are part of each reproducible demo checkpoint. The planned
model adapter uses Modular MAX and the MAX/Mojo stack while preserving the same
runtime-neutral camera/perception contract; see
[`media/model/README.md`](media/model/README.md).

Banana recognition uses a resident specialist router: the general model must
first propose banana, then a banana-only model must confirm the same object.
Both models load at media startup, so the frame loop routes inference without a
cold model swap. This specialist gate is what qualifies banana for autonomous
motion; a banana run's supervised evidence is still its own validation work.

To make a real run eligible after the DLO comparison and deployment, set:

```text
BORDER_COLLIE_RUNTIME_MODE=production
BORDER_COLLIE_HARDWARE_ENABLED=1
BORDER_COLLIE_AUTONOMY_ENABLED=1
BORDER_COLLIE_PERCEPTION_ENABLED=1
BORDER_COLLIE_BARK_ENABLED=1
```

Keep `BORDER_COLLIE_LAB_MOTION_ENABLED=0` for the audience demo. Confirm
`GET /api/status` reports `activation.ready: true`, place Woof at Home facing
the person with the pear in the qualified search area, then use **Activate
Demo** once. Keep the physical remote and **Stop Woof** immediately available.

The pulse is fixed to the previously observed factory-path movement signal:
0.50 m/s for 0.40 seconds. It uses a private exclusive lease, 100 ms command
heartbeats, a 350 ms stale-command watchdog, factory avoidance verification,
and `StopMove` plus avoidance release in a `finally` boundary.

The endpoint is intentionally absent from the audience UI. A human-operated
acceptance run must use the exact confirmation documented in
[`lab/motion/README.md`](lab/motion/README.md). This repository has not yet
revalidated the physical pulse.

The clean app uses host port `8110` so it can be connection-tested beside the
legacy app on `8096` without replacing it.

Run Results default to `artifacts/runs/` for local use. On the device they are
durable: `wendy.json` gives the `app` service a `persist` volume named
`border-collie-run-store` mounted at `/run-store`, and points
`BORDER_COLLIE_RUNS_DIR` at `/run-store/runs` with `BORDER_COLLIE_COHORTS_DIR`
at `/run-store/cohorts`. Run Results, their black-box timelines, and their
evidence artifacts all live under the run directory, so they survive a
`wendy run` redeploy. The store is created on first use, so an empty volume is
a normal first boot.

## Configurable Demo Run cohorts

The audience UI and supervised soak harness both use the typed policy described
in [`docs/cohort-policy.md`](docs/cohort-policy.md). The default is five Demo
Runs in a seeded, randomized order with every failure stopping the cohort. A
fixed Target Fruit and explicitly tolerated terminal reasons or phases can be
selected. Tolerance never bypasses exact-zero disarm, fresh exact-run Home
clearance, or a camera, pose, motion, takeover, restart, or return safety stop.

The harness reads the deployed build's qualified fruits and balances randomized
schedules before shuffling them.
The result JSON is replaced atomically after every run and includes the exact
policy and sequence, per-run cohort decision, Home-clearance evidence,
per-stage telemetry, lighting frames, network observations, device temperatures,
dongle checks, terminal measurements, and the records-only scorecard.

Use an exact expected build label so an old or experimental deployment cannot
be activated accidentally:

```bash
python3 scripts/fruit_soak.py \
  --host 192.168.0.107 \
  --agent 192.168.0.107:50052 \
  --runs 10 \
  --seed 20260810 \
  --expected-build-label "base-soak-v1 (demo/base)" \
  --expected-fruits apple banana pear \
  --device-probes \
  --dongle-match "DJI MIC MINI" \
  --note "<fruit placements, lighting, and microphone setup>"
```

The build label and qualified-fruit set are checked before the first activation.
An activation request with an ambiguous response aborts the session without an
automatic retry. A restart-required application state also aborts immediately.
Individual terminal run failures are recorded and the supervised soak continues
unless the deployed application's safety state prevents another activation.
The root app, `media`, and `voice` services each have a committed
`build.stagefile.yaml` and digest-pinned lockfile. A Stagefile-capable Wendy CLI
selects all three automatically for a whole-project deployment; generated
Dockerfiles are build artifacts and are not committed. Deploy with
`wendy run --detach` and do not pass a Dockerfile override.

### Audience bark

Production uses the direct bark sidecar and does not own or change the Go2's
device-global VUI volume. Onboard prompts such as "I'm here" and obstacle-mode
announcements may therefore remain audible. Bark is audience polish, not a
motion or posture safety gate: a bark timeout or playback error is persisted as
`bark_played: false`, the five-second down hold still completes, and Stand and
Return Home continue. Stop, StandDown, hold, Stand, and Home failures remain
terminal.

### Per-run black box

Every Demo Run has an append-only, fsynced `black-box.ndjson` timeline beside
its materialized `result.json`. It records mission lifecycle events, stage
results, each camera-guidance decision, every resulting motion command, failure
epilogue evidence, and the final outcome/safety state. It contains numeric and
categorical evidence but no camera images. Download it at
`/api/results/{run_id}/black-box.ndjson`; terminal image/clip evidence remains
separate.

## Local validation

```bash
python -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

The suite currently covers every stage through the public production executor,
the physical adapter boundaries with deterministic pose/perception replays, the
complete zero-motion HTTP run, all default failure reasons, and diagnostic
stage classification.

## Historical Docker optimization proof

The repository retains an older Docker/DLO benchmark as historical evidence,
but Dockerfiles and DLO are not part of the current deployment path. The
committed Stagefiles declare stable dependency and model inputs before volatile
application source, and Wendy compiles them into ignored generated Dockerfiles.

The first controlled proof used three paired source-only trials plus no-op and
dependency-change controls. It reduced the source-edit median from 8.743 s to
0.598 s (93.16%), reduced median rebuilt steps from five to two, passed the
full then-current project suite and container import check, and had an estimated
verification break-even of 11.2 representative deployments. See
[`docs/docker-optimization-results.md`](docs/docker-optimization-results.md)
for the full result and limitations.
