# Validation results

No live motion behavior has been validated in this repository yet. The
read-only measurements below are the first live camera and pear-recognition
evidence captured by this repository.

Automated checks establish that the state model is deterministic, Remote
Takeover is process-latched, activation is recorder-backed, and production code
does not import the legacy project. They do not qualify physical motion or the
autonomous routine.

## PRODUCTION-OFFLINE-001 — passed

- Observed: 2026-08-03 in the clean repository
- Scope: 70 automated tests plus Ruff and Python bytecode compilation
- Complete-run boundary: production runtime wiring, all eight stage dispatches,
  durable Run Results, terminal Home measurements, and diagnostic stage status
- Motion replays: measured half-turn, bounded pear search, geometry-gated
  approach, one 1.0 m/s by 0.40 s lower-edge final push, posture actions,
  odometry return to 0.10 m, and heading restoration to 5 degrees
- Camera safety: generation/source-marker binding, 0.350 s source gap reset,
  frozen-source failure, and camera health rechecked during the final push
- Media boundary: one read-only WebRTC owner provides pear evidence and bark;
  it imports no motion client
- Result: passed the local implementation gate
- Limitation: deterministic adapters and recorded hardware facts do not qualify
  the combined physical sequence. The next step remains DLO comparison, then a
  supervised Woof deployment and one real end-to-end acceptance run.

## PREFLIGHT-FAIL-CLOSED-001 — passed

- Observed: 2026-08-03 at 11:15 PDT on the clean app at
  `http://127.0.0.1:8110/`
- Safety: hardware and lab motion disabled; no clients initialized and no
  physical motion command sent
- Activation: pear Demo Run `58c116e4-bb31-4a90-a21e-d25c4f6b3b31` was
  durably created before readiness evaluation
- Checks: durable storage and motion-disarmed passed; hardware connection,
  fresh pose, and production camera/perception readiness failed
- Terminal result: `FAILED / PREFLIGHT_FAILURE`, failed phase `preflight`, and
  final safety state `DISARMED_CONFIRMED`
- Audience UI: retained the failed Run ID and message and immediately made
  **Activate Demo** available for a new run
- Result: passed the fail-closed preflight rule; missing readiness cannot leave
  a run hanging or advance to Home capture
- Limitation: the all-ready path is integration-tested through qualified
  hardware and camera boundaries, but it has not been exercised on Woof and no
  production camera adapter is connected yet

Evidence: `lab/current-activation/results/PREFLIGHT-FAIL-CLOSED-001.json`.

## ACTIVATE-RECORDER-001 — passed

- Observed: 2026-08-03 at 11:04 PDT on the clean app at
  `http://127.0.0.1:8110/`
- Safety: hardware and lab motion disabled; no clients initialized and no
  physical motion command sent
- Audience surface: **Activate Demo** was enabled before the run, disabled while
  it was active, and enabled again after Stop
- Activation: created Demo Run
  `196ddbfc-57b2-49b0-9f98-3c6ac2a68935` for pear and visibly entered
  `preflight`
- Persistence: append-only `events.ndjson` and atomically materialized
  `result.json` were present; the result was readable by its full UUID
- Stop: the same run sealed as `STOPPED` with reason `OPERATOR_STOP`, final
  safety state `DISARMED_CONFIRMED`, and ordered `idle -> preflight -> stopped`
  events
- Result: passed the recorder-backed activation and operator-stop lifecycle
- Limitation: autonomous orchestration is not connected, so activation
  intentionally stops at `preflight`; this is not an end-to-end demo pass

Evidence: `lab/current-activation/results/ACTIVATE-RECORDER-001.json`.

## ACTIVATE-CURRENT-001 — blocked by design

- Observed: 2026-08-03 at 10:44 PDT on the clean app at
  `http://127.0.0.1:8110/`
- Safety: hardware and lab motion disabled; no clients initialized and no
  physical motion command sent
- Audience surface: **Activate Demo** was visible and disabled with the message
  that autonomous activation is not enabled
- Activation API: `POST /api/run` returned HTTP 404 and created no Demo Run
- Stop path: the visible **Stop** control moved the mission scaffold to
  `stopped` with zero stop errors
- Result: the current clean build cannot run the end-to-end demo yet. This is an
  expected fail-closed state, not an application crash and not a completed
  hardware test.
- Missing activation dependencies: durable Run Result recorder and APIs,
  production camera/perception adapter, autonomous phase orchestrator, qualified
  fruit approach/actions, and qualified return-to-Home control.

Evidence: `lab/current-activation/results/ACTIVATE-CURRENT-001.json`.

## Reused evidence boundary

The factory-avoidance motion implementation is adapted from the previous
prototype, where 0.50 m/s for a bounded 0.40-second pulse produced visible
movement and finished disarmed with zero velocity. That evidence justifies the
initial value and adapter choice; it does not count as validation of this new
build. The first deployment must repeat the guarded acceptance in
`lab/motion/README.md`.

The first combined clean-repository run later showed that operating continuously
at the exact 0.50 m/s deadband edge did not produce visible translation during
the approach. The production approach now requests the separately observed
1.0 m/s factory-avoidance signal while retaining continuous camera steering,
the 20-second timeout, lower-edge arrival evidence, and automatic disarm. This
change is an evidence-based correction, not a new physical qualification; the
next supervised run must confirm the approach actually advances.

## APPROACH-MOTION-001 — observed movement, intentionally stopped

- Observed: 2026-08-03 on Woof, Demo Run
  `a0258967-eb4d-4710-803d-858aa105b882`
- The operator confirmed that the revised 1.0 m/s factory-avoidance approach
  produced physical forward movement toward the pear.
- The operator stopped the run during `approach_fruit`; the terminal result is
  `STOPPED / OPERATOR_STOP`, motion is disarmed, the factory-avoidance remote
  owner is released, and the final command is zero.
- This qualifies the movement-input correction only as an observation. It does
  not complete Arrival, the off-screen final push, lie-down/bark, stand-up,
  pulse-count return, or the Home gates, so no completed-test snapshot is
  attached.

## ARRIVAL-ACTION-001 — completed stages, corrections required

- Observed: 2026-08-03 on Woof, Demo Run
  `f21054c5-598f-4008-a024-27f7b5e2d971`
- Arrival completed with 8 recorded forward heartbeats and the 0.4-second
  final push, but the operator reported that Woof remained too far from the
  pear.
- The pear was not required to be centered before the first forward command.
- `sit_and_bark` completed in 0.204 seconds and `stand` began approximately
  7 ms later. This confirms that the asynchronous StandDown action had no
  visible hold before StandUp interrupted it.
- The subsequent Home turn failed safely after the 0.30 rad/s, 15-second
  envelope timed out. The Run Result sealed as
  `FAILED / RETURN_HOME_FAILURE`; motion released and remained disarmed.
- Prepared corrections: three fresh center confirmations within the middle
  16% of the frame before translation, a 1.0 m/s by 1.0-second final push, a
  5-second down hold, and the already-working 0.50 rad/s by 30-second bounded
  turn input for the Home turn.
- This test is not marked passed and has no completed-test snapshot. The
  corrected sequence requires a new supervised run.

## APPROACH-CENTER-001 — failed, safely disarmed

- Observed: 2026-08-03 on Woof, Demo Run
  `2dd2ea43-8c5f-4a25-8698-54aa48fd31ad`
- The turn and find stages completed with fresh pear evidence, including 12
  stable detections in `find_fruit`, but `approach_fruit` timed out after its
  full 20-second envelope without recording Arrival.
- The operator observed that the initial centering correction was too small to
  produce a useful physical turn. The prior proportional controller requested
  only about 0.24–0.30 rad/s near the edge of the center gate and continued
  sending smaller yaw values while collecting centered confirmations.
- The Run Result sealed as `FAILED / ARRIVAL_FAILURE`; motion released, the
  factory-avoidance remote owner cleared, and the final command was zero.
- Prepared correction: outside the center band use a fixed 0.50 rad/s yaw—the
  signal already observed to turn Woof—and inside the band command exactly
  zero yaw while collecting three confirmations.
- This test is not marked passed and has no completed-test snapshot. The
  corrected centering behavior requires a new supervised run.

## FINAL-APPROACH-SPEED-001 — correction prepared

- Observed: 2026-08-04 during a supervised Woof Demo Run
- The operator confirmed that the overall sequence worked, but the 1.0 m/s by
  1.0-second movement after qualified lower-edge pear disappearance was too
  fast.
- Prepared correction: retain the 1.0 m/s camera-guided approach, reduce only
  the single bounded off-screen movement to 0.3 m/s, release motion, and keep
  the existing explicit stop before lie-down.
- This correction is covered by automated configuration tests but remains a
  physical qualification candidate until the next supervised run.

## CAMERA-SOURCE-001 — passed

- Observed: 2026-08-03 at 09:43 PDT on `woof.local`
- Safety: read-only WebRTC video; no motion client imported and no motion
  command sent
- Sample: 430 decoded 1280×720 frames over 32.411 seconds
- Source identity: PTS and time base present on every frame; zero repeated or
  regressed PTS values; stable `1/90000` time base
- Content continuity: zero identical consecutive luma frames
- Callback cadence: mean 0.070134 s, p95 0.089104 s, max 0.146180 s
- Result: decoded-frame PTS is suitable as a strictly advancing identity marker
  for this camera connection
- Limitation: the calculated PTS delta was 0.000667 s while callback cadence
  averaged 0.070134 s, so PTS must not be treated as a wall-clock age or cadence
  measurement. This run did not validate reconnect generations, stale-frame
  thresholds, pear confidence, or detection stability.

<img src="../lab/camera-source/results/CAMERA-SOURCE-001.jpg" alt="Woof camera source identity test scene" width="320">

Evidence: `lab/camera-source/results/CAMERA-SOURCE-001.json` and
`lab/camera-source/results/CAMERA-SOURCE-001.jpg`.

## PEAR-EVIDENCE-001 — passed

- Observed: 2026-08-03 at 16:52 PDT on `woof.local`
- Safety: read-only WebRTC video and GPU inference; no motion client imported
  and no motion command sent
- Sample: 858 decoded frames over 63.106 seconds; 727 inference samples were
  eligible for recognition scoring
- Recognition: 727/727 pear matches (100%); zero misses and zero consecutive
  misses
- Confidence: mean 0.870628, minimum 0.811677, p95 0.901923, maximum 0.922483
- Inference time: mean 0.071854 s, p95 0.081585 s; one 4.637798 s startup
  outlier
- Detection age: mean 0.091424 s, p95 0.122844 s; one 4.645201 s startup
  outlier
- Camera identity: zero missing, repeated, or regressed PTS values; zero
  identical consecutive luma frames
- Result: passed the provisional acceptance rule of at least 30 eligible
  samples and at least 80% pear matches. The saved annotated frame visually
  confirms that the detection is on the pear.
- Limitation: this run did not interrupt or reconnect the camera and therefore
  does not validate reconnect recovery or a production stale-frame envelope.

<img src="../lab/pear-evidence/results/PEAR-EVIDENCE-001.jpg" alt="Pear recognition test with an annotated detection on the pear" width="320">

Evidence: `lab/pear-evidence/results/PEAR-EVIDENCE-001.json` and
`lab/pear-evidence/results/PEAR-EVIDENCE-001.jpg`.

## CAMERA-RECONNECT-001 — passed

- Observed: 2026-08-03 at 17:04 PDT on `woof.local`
- Safety: read-only WebRTC video and GPU inference; no motion client imported
  and no motion command sent
- Method: qualify a baseline generation, deliberately close that client camera
  connection, hold it offline for 2.009 seconds, create a distinct replacement
  generation, and require the full pear preflight again
- Baseline: 117 decoded frames; 32/32 pear matches; strictly advancing PTS and
  stable `1/90000` time base
- Replacement: first decoded frame arrived 4.409 seconds after the planned
  disconnect; first pear match arrived approximately 4.499 seconds after the
  disconnect; full 30-sample preflight passed after 6.677 seconds
- Replacement evidence: 34 decoded frames; 32/32 pear matches; mean confidence
  0.889068; strictly advancing PTS and stable `1/90000` time base
- Generation boundary: the two opaque generations were distinct; the first
  replacement detection was bound to the replacement generation and its exact
  PTS/time base; the prior run was not resumed
- Startup behavior: the decoder logged transient missing-PPS/H.264 errors before
  baseline frames became available, then recovered without a recorded camera
  error. The replacement generation did not exhibit a measured evidence stall.
- Result: passed clean client disconnect/reconnect and fresh replacement
  preflight. A production Demo Run must still terminate the original run and
  log `CAMERA_FAILURE`; only a newly started run may use replacement evidence.
- Limitation: this did not simulate packet loss, a robot-side camera crash, or
  an application restart, and it does not yet prove the Demo Run's terminal
  `CAMERA_FAILURE` record implementation.

<img src="../lab/camera-reconnect/results/CAMERA-RECONNECT-001.jpg" alt="Replacement-generation pear detection after a controlled camera reconnect" width="320">

Evidence: `lab/camera-reconnect/results/CAMERA-RECONNECT-001.json` and
`lab/camera-reconnect/results/CAMERA-RECONNECT-001.jpg`.
