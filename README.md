# Border Collie Demo

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
4. Start from the audience UI's single **Activate Demo** control with pear as
   the configured Target Fruit.
5. Turn through the fruit-search area until the requested fruit is recognized,
   bounded by one measured revolution and a 30-second timeout. A 50%+
   full-frame pear proposal triggers crop confirmation and a 50% duty-cycled
   turn using the same reliable yaw signal; an unqualified crop keeps rotating
   rather than becoming a false stop.
6. Confirm/reacquire and approach the requested fruit using fresh detections.
7. After confirmed near-fruit evidence, allow one bounded final approach when
   the fruit leaves the lower camera edge.
8. First turn at a fixed 0.50 rad/s until the pear is within the middle 16% of
   the camera, then hold zero yaw for three fresh centered samples. After
   qualified lower-edge disappearance, send one 0.3 m/s by
   1.0-second final push, stop, lie down, bark, and remain down for 5 seconds.
9. Stand, turn toward Home, replay the recorded number of outbound forward
   heartbeats at 1.0 m/s, and restore the original heading. Fresh pose remains
   the authority for the 10 cm Home success gate and recorded Home Distance.
10. Stop all motion, record the result, and report completion.

Arbitrary typed commands belong on the separate debug surface until more than
one fruit is qualified. Speech input remains outside this milestone until the
replacement microphone is available and independently validated.

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
- `media/`: sole owner of Go2 WebRTC camera, pear inference, and bark audio.
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
one 0.3 m/s by 1.0 s off-screen final push, Unitree posture actions, bark, and
closed-loop odometry return through factory obstacle avoidance.
Pear qualification begins inside `TURN_TO_FRUIT` and stops the camera-guided
turn; `FIND_FRUIT` confirms or reacquires it. The absence of a pear before
activation is expected and does not block the button.
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
Choose or create the `pear` class in Fieldmark, draw the boxes, then export the
dataset in the required YOLO or COCO format. Fruit selection is temporarily
fixed to pear; future fruit classes must use this same runtime-neutral evidence
contract rather than adding motion controls to the labeler.

Pear acquisition remains fixed at 0.65 confidence for five consecutive fresh
frames. Once that acquisition has succeeded, approach tracking may continue at
0.55 confidence after three consecutive fresh pear detections. These values are
separately configurable with
`BORDER_COLLIE_PEAR_TRACKING_MIN_CONFIDENCE` and
`BORDER_COLLIE_PEAR_TRACKING_CONFIRMATIONS`; changing them does not alter the
acquisition rule.

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
never imports or creates a motion client. The TensorRT
engine is a temporary provisioned deployment artifact at `/media/model.engine`;
it is not committed to Git. The planned model adapter uses Modular MAX and the
MAX/Mojo stack while preserving the same runtime-neutral camera/perception
contract; see [`media/model/README.md`](media/model/README.md).

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

Run Results default to `artifacts/runs/`. A deployment must set
`BORDER_COLLIE_RUNS_DIR` to durable mounted storage before stage use.

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
