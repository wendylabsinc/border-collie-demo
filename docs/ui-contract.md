# Audience and diagnostic UI contract

The audience surface and diagnostics are separate products over the same
mission state and durable Run Results. Neither surface owns autonomous state or
hardware directly.

## Audience surface

The audience route is intentionally minimal:

- one primary **Activate Demo** control;
- one always-available **Stop Woof** safety control;
- a narrow qualified-fruit selector containing Pear, Red apple, and Banana;
- bounded per-run search experiment controls, with effective values persisted
  before activation and no process-wide configuration mutation; selected-fruit
  focus and lock defaults/ranges update when the selector changes;
- a read-only annotated camera preview showing model state and Target Fruit
  evidence;
- readiness or the exact reason activation is unavailable;
- the active Run ID and current high-level phase while running; and
- the terminal outcome, reason, safety state, and Run ID when finished.

Activate Demo is enabled only when there is no active run or latched Remote
Takeover and the application reports advancing-camera, perception-service,
pose, motion, media, and result-storage readiness. A qualifying Target Fruit is
not required before activation. The initial search is explicitly conditional:
fresh qualified evidence already in view skips the two broad-search stages
without arming motion; otherwise Woof turns and acquires it during
`TURN_TO_FRUIT`, with `FIND_FRUIT` retaining the
confirmation/reacquisition boundary. One click creates the Run Result before hardware action and disables
repeated activation until that run is terminal.

The audience surface never provides an arbitrary fruit value, typed command,
arbitrary motion, phase skip, calibration value, retry-within-run, failure
clearing, Remote Takeover clearing, or Run Result editing. Its selector is
limited to the shared Qualified Fruit policy. It does not link to operator
diagnostics from the stage presentation.

Stop Woof immediately requests the shared safe-stop path. It does not pause or
create a resumable run. The resulting Run Result records `STOPPED`, reason
`OPERATOR_STOP`, the interrupted phase, and final safety evidence.

## Diagnostic surface

The operator-only diagnostic route displays:

- active Run ID, Target Fruit, phase, phase history, and elapsed time;
- every readiness gate and its explicit failure reason;
- camera generation, source marker/time base, source age, preflight count, and
  camera threshold;
- Target Fruit confidence, stability, bounding box, inference time, detection
  age, and acceptance/rejection reason;
- current pose freshness, captured Home, Home Distance, and heading error;
- motion owner, armed state, active primitive, watchdog, last command, and
  stop/release evidence;
- Remote Takeover latch and restart requirement;
- current and previous Run Results with their annotated snapshots, raw-frame
  evidence downloads, and recognition summaries; and
- application/build, robot, model, and threshold versions.

## Isolated diagnostic actions

Phase-level and mini-test actions live only on diagnostics. Each action is a
separate explicitly named test with its own safety gate, confirmation when
physical motion is possible, structured result, and small evidence snapshot.

Diagnostic actions:

- cannot run while an audience Demo Run is active;
- cannot mutate or advance the production mission state;
- cannot bypass camera, pose, ownership, watchdog, or Remote Takeover gates;
- cannot reuse an audience Run ID;
- record whether motion commands were possible and sent;
- stop their app or release their hardware owner after completion; and
- remain grouped under `lab/` so successful tests are reusable.

Debugging a production phase uses the same production adapter and thresholds
but a distinct diagnostic orchestration path. A stage tool does not silently
become part of the Demo Run merely because it works in isolation.

## Failure and terminal display

Failure display leads with:

- terminal outcome and stable reason code;
- failed phase;
- whether Woof is confirmed disarmed, stop is unconfirmed, authority is remote,
  or safety is unknown;
- active restart requirement;
- latest trustworthy Home Distance or its unavailable reason; and
- the relevant evidence snapshot labeled with its true age and source marker.

`CAMERA_FAILURE` never presents a replacement stream as continuation of the
failed run. `REMOTE_TAKEOVER` presents only the restart requirement; neither UI
offers resume or reset.

## Run Result access

Both surfaces read the contract in [`run-result-contract.md`](run-result-contract.md).
The audience surface shows a compact summary. Diagnostics exposes the complete
record, journal-derived phase timeline, metrics, and referenced evidence. A
failed-run archive is offered as a one-click download for extraction and manual
labeling in Fieldmark; the UI never sends those images to an external service
automatically.

The bounded yaw/lock comparison and its recorded evidence are defined in
[`search-experiment.md`](search-experiment.md).

Result routes are read-only. They accept only full Run IDs and snapshot
references already present in the Run Result. They cannot delete, relabel,
rewrite, resume, or clear evidence.

## API boundary

The UI contract requires narrow APIs with these semantics:

- activate once and return the new full Run ID;
- request the shared safe-stop path;
- read current readiness, active run, and mission status;
- list durable Run Result summaries newest first;
- read one complete Run Result by full Run ID; and
- read only snapshots or evidence archives referenced by that result.

Diagnostic execution APIs use a separate namespace and explicit feature gates.
They are absent or return unavailable in the audience deployment unless the
corresponding diagnostic capability is intentionally enabled.

## Acceptance requirements

Automated and browser acceptance must prove:

- repeated activation cannot create concurrent runs;
- the Run Result exists before the first hardware call;
- disabled activation explains the blocking readiness gate;
- Stop Woof remains reachable in every audience state;
- camera failure, target loss, operator stop, return failure, and Remote
  Takeover render distinct reasons and correct safety states;
- terminal or takeover runs offer no resume path;
- diagnostics cannot run tests during an audience run or bypass feature gates;
- both surfaces render from persisted Run Result data after restart; and
- result and snapshot routes reject path traversal, short IDs, and mutations.

The static comparison in `web/ui-contract-prototype.html` is a throwaway design
aid, not a production route or implementation.
