# Border Collie Demo

A clean-room implementation of the Wendy Labs Border Collie routine for the
Unitree Go2.

This repository intentionally does not import from the original `collie-demo`
application. The old repository remains useful as test evidence and a hardware
reference, but it is not a runtime dependency.

## Project lineage

The demo has progressed through a physically validated base lane, a randomized
soak/recovery lane, an integrated durability lane, and the current stage-camera
candidate. Several other worktrees are parallel source, tooling, or research
lanes rather than later demo versions. See
[Demo and worktree progression](docs/worktree-progression.md) for the branch,
PR, integration, deployment, and physical-evidence map.

## Intended routine

1. Verify the advancing camera source, perception service, pose, motion, media,
   and return-home readiness. A visible pear is not required yet.
2. Confirm that Woof is facing the person. This is initially an operator setup
   requirement rather than autonomous person detection.
3. Capture a stable Home position and heading.
4. Choose apple, banana, or pear from the audience UI, then use the single
   **Activate Demo** control.
5. **Conditional search:** if fresh qualified evidence already shows the
   requested fruit, do not perform the initial rotation. Record the
   turn/search stages as conditionally skipped and proceed directly to approach
   centering. Otherwise, turn through the fruit-search area until it is
   recognized, bounded by one measured revolution and a 30-second timeout. A 50%+
   full-frame fruit proposal triggers crop confirmation and a 50% duty-cycled
   turn using the same reliable yaw signal; an unqualified crop keeps rotating
   rather than becoming a false stop.
6. Confirm/reacquire and approach the requested fruit using fresh detections.
   No forward command is permitted until the fruit remains within 5% of frame
   center for three fresh samples. Once approach begins, moderate corrections
   combine forward motion with proportional yaw. Track reacquisition does not
   repeat yaw-only centering unless the fruit is more than 25% off center.
7. Near-fruit geometry slows the approach. Three fresh centered near samples
   establish Final Approach for apple, banana, and pear. Woof continues at the
   close speed until the qualified track exits through the lower camera edge,
   then performs one bounded 0.6 m/s, 1.0 second final push and stops.
8. First turn at a fixed 1.00 rad/s until the Target Fruit is within the middle
   10% of the camera, then hold zero yaw for three fresh centered samples. After
   qualified sight-lost-close Arrival, complete the bounded final push, lie
   down, bark, and remain down for 5 seconds.
9. Stand, turn toward Home, follow sparse outbound pose breadcrumbs in reverse,
   and restore the original heading. The recorded heartbeat count remains a
   translation budget. Fresh fused pose remains the authority for the 10 cm
   Home success gate and recorded Home Distance.
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
detection stability so evidence from one class cannot qualify another. The
historical red-apple qualification passed `RED-APPLE-001` at 0.70 and reached
15 consecutive qualifying detections. The v23 candidate intentionally tests a
0.65 acquisition floor and requires 0.55 for continued tracking.

The acquisition thresholds are 0.65 for apple, 0.20 for banana, and 0.65 for
pear. Banana's low app-side value rides on top of the specialist's 0.55 floor
rather than standing alone. Fruit-specific thresholds require fresh physical
acceptance evidence before they are treated as qualified.

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
zero-motion sight-lost-close Arrival, Unitree posture actions, bark, and
closed-loop fused-odometry return through factory obstacle avoidance. The
planar estimator combines Go2 position/velocity, IMU yaw rate, loaded-foot
zero-velocity updates, and bounded scale-free visual direction/yaw while the
final 10 cm gate uses the farther of raw and filtered distance.
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
sidecar detections and three fresh application observations. Once that
acquisition has succeeded, approach tracking may continue at the fruit-policy
floor of 0.55 confidence. `BORDER_COLLIE_PEAR_TRACKING_MIN_CONFIDENCE` may make
that tracking floor stricter but cannot lower it, while
`BORDER_COLLIE_PEAR_TRACKING_CONFIRMATIONS` controls application-side temporal
acquisition. Neither setting alters the sidecar acquisition rule.

Red apple acquisition is 0.65 confidence for five fresh frames. After
acquisition, a close red-apple track may continue at 0.55 confidence
only while fresh observations remain spatially continuous: its horizontal
center cannot jump by more than 0.20 of the frame, its lower edge or vertical
center cannot retreat by more than 0.08, and its box area cannot collapse by
more than 35 percent between samples. This rule cannot acquire an apple. The
approach slows near the fruit and uses the shared bounded final-push contract;
stale frames, duplicate frames, identity changes, or discontinuous geometry
produce zero-motion recommendations. A same-fruit discontinuity preserves the
completed initial-centering gate: two fresh samples confirm reacquisition at
zero motion, then moderate horizontal error is corrected while moving. Only an
error greater than 0.25 of the frame authorizes another yaw-only recenter.

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
- `BORDER_COLLIE_BARK_ENABLED=1` requires the media sidecar to prove AudioHub
  bark readiness during preflight and permits the arrival bark request.

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

The pulse is fixed to the field-qualified factory-path movement signal:
0.55 m/s for 0.40 seconds. Any nonzero forward command below 0.55 m/s fails
closed before reaching the SDK. It uses a private exclusive lease, 100 ms command
heartbeats, a 350 ms stale-command watchdog, factory avoidance verification,
and `StopMove` plus avoidance release in a `finally` boundary.

The endpoint is intentionally absent from the audience UI. A human-operated
acceptance run must use the exact confirmation documented in
[`lab/motion/README.md`](lab/motion/README.md). This repository has not yet
revalidated the physical pulse.

The clean app uses host port `8110` so it can be connection-tested beside the
legacy app on `8096` without replacing it.

Run Results default to `artifacts/runs/`. A deployment must set
`BORDER_COLLIE_RUNS_DIR` to durable mounted storage before stage use.

## Ten-run randomized soak

The supervised soak runs ten complete missions in a seeded, randomized order.
It reads the deployed build's qualified fruits and balances the schedule before
shuffling it, so three qualified fruits receive three or four attempts each.
The same seed also assigns every run a relative orientation turn from 0 through
359 degrees. Woof captures Home, completes and records that measured turn, and
only then starts looking for the selected fruit.
The result JSON is replaced atomically after every run and includes the exact
sequence, per-stage telemetry, lighting frames, network observations, device
temperatures, dongle checks, terminal measurements, and the records-only
scorecard.

The short interface keeps the recovery contract and its regressions easy to
run:

```bash
scripts/fruit-soak test
scripts/fruit-soak run --host woof.local --runs 10 --seed 20260810 \
  --expected-build-label "<exact deployed build label>" \
  --expected-fruits apple banana pear
```

`run` enables failed-run recovery reconciliation and the `0.50 m` supervised
stage Home margin. Override the margin only with an explicit operator decision
using `FRUIT_SOAK_HOME_MARGIN_M` or `--stage-home-margin`.

Use an exact expected build label so an old or experimental deployment cannot
be activated accidentally:

```bash
python3 scripts/fruit_soak.py \
  --host 192.168.0.107 \
  --agent 192.168.0.107:50052 \
  --runs 10 \
  --seed 20260810 \
  --expected-build-label "base-soak-v3-recovery-retention (demo/base)" \
  --expected-fruits apple banana pear \
  --recover-failures \
  --device-probes \
  --dongle-match "DJI MIC MINI" \
  --note "<fruit placements, lighting, and microphone setup>"
```

The build label and qualified-fruit set are checked before the first activation.
An activation request with an ambiguous response aborts the session without an
automatic retry. A restart-required application state also aborts immediately.
Individual terminal run failures are recorded separately from their recovery
outcome. With `--recover-failures`, the harness first observes the app's
automatic recovery lifecycle. It waits for an active or recorded attempt and
never submits a duplicate manual request. Only when no attempt appears does it
submit one manual request; an ambiguous response is reconciled through read-only
status/result polling and is never retried. The next activation remains blocked
until recovery is terminal, `active_recovery` is clear, motion is disarmed, and
the attempt's authoritative Home distance is inside `--stage-home-margin`. A
rejected, failed, stopped, timed-out, or out-of-margin recovery aborts the soak.
Without the flag, failures retain the earlier records-only behavior and the
harness does not issue recovery motion.
Pass `--no-orientation-randomization` to run the original soak with a zero-degree
pre-search turn on every mission.

### Recover a failed run to its captured Home

Failed-run recovery is a separate, position-only operation. It reuses the
failed run's saved Home and recorded forward approach heartbeat count; it does
not capture a new Home or change the original failed outcome.

Eligible `ARRIVAL_FAILURE` and `TARGET_LOST_OFF_AXIS` approach failures start
one automatic no-bark recovery after confirmed disarm. That path requires a
fresh captured Home, fresh trusted pose and continuous fusion, released motion,
no fault, operation, or Remote Takeover, and a bounded recorded outbound pulse
count. An unsafe candidate is recorded as `AUTOMATIC_RECOVERY_SKIPPED` and does
not translate. The explicit endpoint below remains available for supervised
recovery of older failed records that did not receive an automatic attempt.

```bash
curl -X POST \
  http://woof.local:8110/api/results/RUN_ID/recover-home \
  -H 'content-type: application/json' \
  -d '{"confirmation":"RECOVER FAILED RUN TO CAPTURED HOME"}'
```

The endpoint returns `202 Accepted` with a recovery ID. Poll
`/api/results/RUN_ID` and read `recovery_attempts`; the active recovery also
appears in `/api/status`. One correction attempt is accepted only when the
first recovery itself failed with a confirmed disarm; completed, stopped, or
unconfirmed recoveries cannot be retried. `/api/stop` cancels and disarms an
active recovery.

Failed and stopped runs retain their full event, stage, failure, motion, and
frame evidence. Completed runs retain one compact `result.json` with the key
acceptance values and no frame archive or event journal.

## Deploy with Wendy Stagefiles

Run the multi-service deployment directly from the repository root:

```bash
cd /Users/olivertaylor/Documents/Wendy/border-collie-demo-stage-camera-v20-continuous-track
wendy run --detach
```

The exact underlying whole-project command is `wendy run --detach`; a device
flag is needed only when Woof is not already the selected Wendy target. The v19
candidate identity is `stage-camera-v20-continuous-track`, schema `4`, app
version `1.0.23-stage-camera`. Confirm the branch and commit against
[the progression map](docs/worktree-progression.md) before running it. Do not
use `--service`, `docker build`, or pass `--dockerfile`.

The root app and `media` service each have a committed `build.stagefile.yaml`
and digest-pinned lockfile. A Stagefile-capable Wendy CLI selects both
automatically for the multi-service deployment; generated Dockerfiles are build
artifacts, not deployment inputs selected by the operator. The root context
resolves `build.stagefile.yaml`, and the `media` context independently resolves
`media/build.stagefile.yaml`.

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

## Docker optimization proof

Docker build changes are measured with DLO using the repository contract in
`.dlo.yml`. The current Dockerfile keeps dependency installation in the stable
builder layer and copies application source directly into the final image, so a
source-only edit does not reinstall the project package.

The first controlled proof used three paired source-only trials plus no-op and
dependency-change controls. It reduced the source-edit median from 8.743 s to
0.598 s (93.16%), reduced median rebuilt steps from five to two, passed the
full then-current project suite and container import check, and had an estimated
verification break-even of 11.2 representative deployments. See
[`docs/docker-optimization-results.md`](docs/docker-optimization-results.md)
for the full result and limitations.
