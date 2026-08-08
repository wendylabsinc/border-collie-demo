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

## DISTANT-PEAR-DIAGNOSTICS — deployed and reproduced on hardware

- A failed healthy-camera search is now classified separately as
  `TARGET_RECOGNITION_FAILURE`, not motion or camera failure.
- The Run Result retains search sample count, pear-candidate count, strongest
  confidence, largest box-area ratio, closest candidate, and sweep progress.
- The media process maintains a bounded approximately 20-second raw-frame
  sequence and an annotated terminal comparison. The app persists both before
  sealing an orchestrated terminal result and `/debug` links the files.
- Raw frames are compatible with manual upload to Fieldmark and the live raw
  endpoint can be configured as Fieldmark's Go2 source. The reusable local run
  labeler opens an archive directly and exports corrected normalized boxes.
- Automated contract tests pass for capture, rolling bounds, persistence,
  download authorization, and distinct failure classification.

Hardware run `f9de499a-014f-460c-9748-c2c875a85f27` reproduced the distant-pear
failure on 2026-08-04. The camera remained healthy and the model retained a
centered pear box, but `approach_fruit` stopped and timed out because the same
0.65 acquisition threshold was being reused for tracking. Its 40-frame archive
contains pear confidence from 0.518 to 0.686 (median 0.612): only 5 frames met
the acquisition threshold while 33 met 0.55. A focused regression test now
locks acquisition at 0.65 while permitting an already-acquired approach track
after three consecutive fresh detections at or above 0.55. The back-to-back
physical runs below exercised this tracking path successfully.

### Crop-and-confirm — deployed and exercised on hardware

- The existing TensorRT engine now performs at most one conditional crop pass
  for a small or uncertain full-frame pear proposal. Promotion requires higher
  confidence and at least 0.10 IoU with the original box.
- Replaying the trigger rules over all 40 frames from run
  `f9de499a-014f-460c-9748-c2c875a85f27` selects a 256×256 confirmation crop for
  every frame. Those boxes occupied only 0.12–0.14% of the full image.
- Recorded single-pass inference had a 0.060-second median and 0.076-second
  maximum. After deployment, 20 distinct live two-pass frames measured from
  0.115 to 0.137 seconds, below the unchanged 0.200-second deadline.
- Automated tests prove the second pass is bounded to one, maps crop boxes back
  to full-frame coordinates, rejects a high-confidence spatial disagreement,
  avoids extra work for large confident detections and extremely weak noise,
  and preserves crop diagnostics through the mission adapter.
- Search now responds to a 0.50+ full-frame crop candidate with a 50% duty
  cycle: it alternates the already validated yaw magnitude with zero instead of
  sending an unreliable weaker turn command. An unqualified enlarged result
  continues the bounded rotation; only the existing 0.65-by-five acquisition
  gate stops it. Automated motion tests lock the hold/turn sequence and its Run
  Result counters.

Three back-to-back supervised Demo Runs were recorded after deployment. Every
run completed `return_home` inside the 0.10-meter Home gate, at approximately
0.078, 0.065, and 0.063 meters. Runs
`6a613766-47d1-49f0-820b-bf1ba0a0c9cc` and
`da32e0c7-f642-464e-a166-9e6d6c76ef86` completed end to end in 50.47 and 38.61
seconds. Their terminal Home distances were approximately 0.085 and 0.048
meters. The remaining run reached Home at approximately 0.065 meters and then
failed safely in `restore_heading` when a yaw command produced insufficient
measured response; its final safety state was `DISARMED_CONFIRMED`.

The completed 50.47-second run recorded 18 crop-candidate search samples, 10
zero-yaw slowdown holds, and 7 reliable-rate slowdown turns before stable pear
acquisition. The completed 38.61-second run recorded 7 crop-candidate samples,
4 holds, and 2 slowdown turns. These results validate the new candidate
slowdown and the return-to-Home position behavior while preserving the final
heading-restoration no-response failure as a known follow-up.

## RED-APPLE-001 — passed read-only recognition

- Observed: 2026-08-04 on `woof.local`
- Safety: camera-only fruit-test surface; motion remained locked, disarmed, and
  inactive
- Recognition: the full-frame model placed the apple box on the red apple and
  reached 15 consecutive qualifying detections, exceeding the provisional
  0.70-confidence-by-five rule
- Confidence window: 20 samples, mean 0.699, minimum 0.667, maximum 0.740; 9 of
  20 sampled statuses were at or above 0.70
- Crop behavior: the optional crop pass remained unconfirmed and was not needed
  for qualification; the full-frame result supplied the qualifying evidence
- Result: passed the apple recognition gate and was promoted into the shared
  Qualified Fruit list for supervised Demo Runs
- Operating-envelope restriction: `apple` means the tested red apple; the
  failed green-apple trial is not qualified
- Limitation: the fruit-neutral motion path is enabled for supervised red-apple
  trials, but a red-apple end-to-end run has not yet been recorded

<img src="../lab/fruit-recognition/results/RED-APPLE-001.jpg" alt="Red apple recognition test with the detection box on the apple" width="320">

Evidence: `lab/fruit-recognition/results/RED-APPLE-001.json` and
`lab/fruit-recognition/results/RED-APPLE-001.jpg`.

## VOICE-DEMO-V1 — working checkpoint

- Recorded: 2026-08-05 from supervised operator validation on `woof.local`.
- Voice activation: a recognized supported fruit request reaches the same
  narrow Demo Run API as the audience control and does not bypass readiness or
  motion gates.
- Fruit already in view: fresh qualified evidence conditionally skips the
  initial turn/search stages and proceeds to approach centering.
- Fruit not initially in view: Woof performs the bounded camera-guided search,
  acquires the Target Fruit, and then enters the same approach path.
- Checkpoint meaning: this is the first working voice-triggered demo version
  with both initial-visibility paths. It is not a new general Operating
  Envelope or a claim of precise final pose.

Known motion-quality follow-ups:

- **Return to Home precision:** Woof returns approximately to Home, but may
  finish slightly over- or undershot and may retain a small heading error.
- **Fruit approach precision:** the final distance varies between runs and can
  place Woof too close to or too far from the fruit.
- These are calibration and closed-loop-control priorities for the next
  checkpoint. Existing camera, timeout, disarm, and bounded-motion failures
  remain fail-closed while that work is underway.

## BASE-BRANCH-CONFIRMATION-2026-08-08 — passed

- Observed: 2026-08-08, three consecutive supervised pear Demo Runs on Woof
- Build: `demo/base` at `648469f`, deployed with build label `base (demo/base)`
- Results: 3/3 `COMPLETED / SUCCESS`; terminal Home distances 0.0756, 0.0274,
  and 0.0578 m against the 0.10 m gate; heading errors 3.1, -4.3, and -1.7
  degrees against the 5-degree gate; durations 41.4, 40.4, and 40.2 s
- Coverage: run one skipped search (fruit already visible); runs two and three
  exercised `turn_to_fruit` before approach. All runs ended
  `DISARMED_CONFIRMED`
- Meaning: this configuration reproduces its 2026-08-05 record
  (0.063–0.078 m) on a different day, so `demo/base` is selected as the robust
  base branch. It also exonerates the environment for the 2026-08-07
  supervised failures (0.187–0.375 m): those ran the arrival-rework commits,
  which remain quarantined off this branch pending their own supervised pass
- Operator observation: the approach sometimes finishes too close to the
  fruit. This remains the "Fruit approach precision" follow-up above — a
  refinement item, not a gate failure. Any approach-distance tuning must
  revalidate against `benchmarks/results/supervised-three-run-2026-08-08.json`

Evidence: `benchmarks/results/supervised-three-run-2026-08-08.json`.

## APPROACH-CONSISTENCY-001 — implementation validated, hardware pending

- Recorded: 2026-08-08 on branch `demo/approach-consistency` (off the
  confirmed `demo/base`), addressing the operator-reported lateral drift at
  the fruit, the occasional overshoot, and the Home turn brushing the fruit
- Late-approach centering: once the track's lower edge crosses the 0.70
  close-range boundary, the horizontal center band tightens from 0.08 to
  0.04 of the frame while keeping the already-qualified fixed 0.30 rad/s
  correction and continuous forward translation. Forward-heartbeat counting
  is byte-for-byte unchanged: every visible-track iteration still records
  exactly one forward pulse
- Softer final push: the single blind off-screen movement keeps its 0.3 m/s
  signal and its unchanged trigger, but its window is shortened from 1.0 s to
  0.6 s so Woof stops slightly farther from the fruit with less variance
- New `STEP_BACK` stage between `stand` and `turn_toward_home`: one bounded
  1.0 m/s by 0.4 s reverse window through factory avoidance, sent through a
  dedicated reverse-only motion entry point (general velocity commands remain
  forward-only). The window timer starts only after the motion arm and its
  remote-API settle complete — the regression that silently sent zero reverse
  commands on 2026-08-07 run 2 — and fresh odometry must confirm at least
  0.05 m of backward movement or the run fails closed with `ACTION_FAILURE`
- Return-pulse accounting is untouched: `STEP_BACK` takes no credit against
  the outbound forward-heartbeat count, unlike the quarantined 2026-08-07
  attempt (0.187-0.375 m Home misses). `return_home` receives the same count
  it always did and keeps measuring the real pose from wherever Woof stands,
  so the step back only adds return-budget margin
- Scope: 152 automated tests pass (142 baseline plus 10 new covering the
  reverse motion boundary, the settle-then-window regression, the odometry
  fail-closed gate, the tighter late centering with unchanged pulse
  accounting, and the stage wiring); Ruff is clean on every touched file
- Limitation: no physical run has exercised these changes. The next
  supervised run must confirm arrival-distance consistency, a fruit-clear
  Home turn after the step back, and `return_home` still inside the 0.10 m
  gate against the BASE-BRANCH-CONFIRMATION-2026-08-08 baseline
  (0.0274-0.0756 m)

## ARRIVAL-VISIBILITY-001 — failed on hardware, corrected, hardware pending

- Observed: 2026-08-08 on Woof, supervised Demo Run
  `45a1e796-ba91-431c-8bae-d531ad63c3f8` on build
  `base+approach-fix (demo/approach-consistency)`; sealed
  `FAILED / ARRIVAL_FAILURE` ("qualified pear Arrival timed out") after
  31.5 s, final safety state `DISARMED_CONFIRMED`, no motion hazard
- Telemetry: the approach tracked the pear at 0.76/0.78/0.60 confidence with
  the box bottom rising 0.729 -> 0.858, then the operator confirmed the pear
  genuinely dropped below the camera at arrival distance. One tick later a
  static phantom appeared: label `pear`, confidence 0.010-0.016, box bottom
  frozen at exactly 0.3861 every sample for ~12 s until the 20 s approach
  deadline
- Root cause: the arrival visibility check accepted any matching-label
  detection with no confidence floor, so the phantom kept
  `target_still_visible` true and suppressed the one bounded final push.
  This defect is latent in `demo/base` as well; it was exposed here by an
  unlucky static false positive, not by the approach-consistency changes
- Fix: arrival visibility now requires the fruit's close-range tracking
  confidence (pear 0.55, banana 0.20, apple 0.10 from `fruits.py`) — the
  same floor that qualifies a close-range track. A sub-floor matching label
  counts as disappearance and is tallied as
  `subthreshold_visibility_samples` in the approach evidence
- Evidence sealing: the failed run recorded no `approach_fruit` evidence, so
  the suppression was invisible from the Run Result. Every terminal approach
  path (Arrival, identity change, timeout) now emits the same evidence dict;
  failures carry it in `failure_details.approach`, including
  `near_gate_confirmed`, which disambiguates the second possible suppression
  path (a never-confirmed near gate) if a failure recurs
- Late-band jitter hardening: the tighter 0.04 centering band now engages
  only after two consecutive off-band samples; a wide-band (0.08) offset
  still corrects immediately. `late_center_corrections` in the evidence will
  confirm or refute oscillation on hardware
- Scope: 156 automated tests pass (four new: the phantom-push regression,
  the sealed timeout evidence, the jitter hysteresis, and the
  `failure_details.approach` mapping); Ruff clean on every touched file
- Limitation: corrected in code and pinned by regression tests against the
  recorded telemetry; the supervised retest on Woof is pending
