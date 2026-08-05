# Run Result contract

Every accepted **Activate Demo** request creates a durable Run Result before
preflight or any hardware action begins. A failed preflight, operator stop,
camera failure, process interruption, or Remote Takeover is still a Demo Run
and must remain inspectable.

## Identity and ownership

- `schema_version` begins at `1` and changes only for incompatible schemas.
- `run_id` is an immutable UUIDv4 generated when activation is accepted.
- The full UUID is used in storage, logs, snapshots, and APIs. A short prefix
  may be displayed but is never accepted as an authoritative identifier.
- Only one Demo Run may be active in the process. A second activation is
  rejected without creating another run.
- The initial record includes Target Fruit, activation source, application
  version, source revision, robot identity, and the qualified threshold set.

## Time

Every record includes UTC ISO-8601 timestamps for cross-process inspection and
local monotonic offsets for ordering and duration within the process:

- `started_at_utc` and `started_monotonic_s`;
- `ended_at_utc` and `duration_s` when terminal; and
- ordered event timestamps as UTC plus seconds since run start.

Monotonic values from different processes are never compared. Source-frame
age, detection age, pose age, and command duration use the local monotonic
clock that captured the corresponding evidence.

## Durability and storage

Production uses a configured persistent run directory. Activation fails safely
during preflight if the directory is not writable or durable storage is not
mounted.

Each run owns one directory:

```text
<runs-dir>/<run-id>/
├── events.ndjson
├── result.json
└── snapshots/
    ├── evidence.zip
    └── terminal.jpg
```

- `events.ndjson` is an append-only journal written and flushed after every
  accepted phase transition, safety event, failure, and terminal action.
- `result.json` is the current materialized Run Result. It is written to a
  sibling temporary file, flushed, and atomically renamed after every journal
  event.
- A terminal Run Result is sealed and never mutated. Corrections create a new
  explicitly linked administrative record; they do not rewrite evidence.
- On application startup, any unsealed prior run is sealed as `FAILED` with
  reason `PROCESS_INTERRUPTED` and safety state `UNKNOWN`. The new process does
  not infer that Woof stopped safely.
- Evidence is never silently overwritten or automatically deleted. If storage
  is full, a new Demo Run fails preflight. Retention may be added later only as
  an explicit operator policy.

## Outcome and terminal reason

`outcome` and `reason` are independent machine-readable fields.

Allowed outcomes are:

- `COMPLETED`
- `FAILED`
- `STOPPED`
- `REMOTE_TAKEOVER`

Initial terminal reason codes are:

- `SUCCESS`
- `PREFLIGHT_FAILURE`
- `CAMERA_FAILURE`
- `TARGET_RECOGNITION_FAILURE`
- `TARGET_LOST`
- `ARRIVAL_FAILURE`
- `ACTION_FAILURE`
- `MOTION_FAILURE`
- `RETURN_HOME_FAILURE`
- `OPERATOR_STOP`
- `REMOTE_TAKEOVER`
- `PROCESS_INTERRUPTED`
- `INTERNAL_ERROR`

The terminal record also includes `failed_phase`, a concise operator-facing
message, and structured failure details. Free text never replaces the stable
reason code.

## Phase history

Every phase event records:

- a strictly increasing event sequence number;
- the entered phase;
- UTC time and monotonic offset from run start;
- the prior phase;
- the triggering reason code and concise message; and
- relevant evidence references captured at that transition.

Terminalization is itself an event. A replacement camera generation or a new
application process cannot append autonomous work to a terminal run.

## Final safety state

Every terminal Run Result uses exactly one final safety state:

- `DISARMED_CONFIRMED`
- `STOP_REQUESTED_UNCONFIRMED`
- `REMOTE_OWNED`
- `UNKNOWN`

Safety evidence records the last non-zero application command, motion owner or
lease, watchdog state, stop/release methods attempted, request and completion
times, final armed state, final commanded velocity, and every stop error.

`COMPLETED` with reason `SUCCESS` is valid only when safety is
`DISARMED_CONFIRMED`. Remote Takeover uses `REMOTE_OWNED`: the application
records that it surrendered authority and does not claim that Woof is
disarmed. A failed stop attempt is never hidden by a successful mission phase.

## Home evidence

The Run Result records:

- the captured Home position and heading with source and freshness;
- the latest trustworthy terminal pose with source and freshness;
- Home Distance in meters;
- heading error in radians; and
- return-progress measurements required by the return-to-Home contract.

If terminal pose is not trustworthy, Home Distance and heading error are
`null` and `unavailable_reason` explains why. They are never estimated from
command duration or an earlier pose.

## Perception evidence

Perception evidence needed to explain a transition or failure retains:

- connection generation, PTS, time base, and local frame receipt time;
- source-progress age and progressing-frame preflight count;
- Target Fruit label, confidence, bounding box, and stability count;
- detector execution time, pass count, detection age, and any crop-confirm
  proposal confidence, crop confidence, crop bounds, spatial agreement, and
  promotion decision;
- the qualified threshold values used for the run; and
- whether the evidence was accepted or rejected and why.

The terminal record references the last trustworthy camera and Target Fruit
evidence. A `CAMERA_FAILURE` additionally records the violated threshold or
identity rule. A completed search sweep with fresh camera frames but no
qualified Target Fruit records `TARGET_RECOGNITION_FAILURE` plus sample count,
candidate count, maximum confidence, maximum box-area ratio, closest candidate,
and search progress. Track loss after a previously qualified acquisition may
still use `TARGET_LOST`; neither condition is mislabeled as a camera failure.

## Motion evidence

Each motion primitive records its phase, name, start and end offsets, requested
velocity or turn rate, requested bound, motion owner or lease, watchdog state,
pose before and after when trustworthy, completion status, and stop/release
evidence. Every accepted velocity heartbeat is retained in stage evidence with
its sequence, phase, monotonic timestamp, forward input, yaw input, and reason.
Failed-stage velocity heartbeats are retained in `failure_details`.

The approach summary records `forward_pulse_count`. The return summary records
`requested_forward_pulses` and `replayed_forward_pulses`, plus the latest fresh
measured `home_distance_m`. Matching pulse counts demonstrate playback only;
they do not demonstrate arrival without the measured Home gate.

## Snapshots

Small annotated JPEG snapshots are captured at these evidence boundaries:

- successful preflight;
- stable Target Fruit acquisition;
- Arrival; and
- terminal completion or failure.

Each snapshot entry records its kind, relative path, SHA-256 digest, dimensions,
generation, source marker, local receipt time, and evidence age. A camera
failure preserves the last trustworthy frame and labels its actual age; it is
never presented as current. If a required snapshot is unavailable, the Run
Result retains an entry with `unavailable_reason`.

Every orchestrated terminal run also attempts to persist a bounded rolling
archive of raw, unannotated JPEGs for manual labeling. The default is 40 frames
sampled at 0.5-second intervals, approximately the final 20 seconds. The archive
contains a source/detection manifest and one annotated terminal comparison
frame. Evidence-capture failure is recorded with an `unavailable_reason` and
must never mask the run's motion-safety result. Full continuous video and audio
recording remain outside this contract.

Every terminal record materializes `terminal_measurements.home_distance_m` and
`terminal_measurements.heading_error_rad` from the latest completed
return-stage evidence. A value is `null` when no trustworthy measurement was
completed; requested speed and elapsed command time never substitute for pose.

## Read interfaces

The audience and diagnostic surfaces read the same persisted record:

- current status exposes the active `run_id` or `null`;
- list results newest first with outcome, reason, final phase, and times;
- retrieve one complete Run Result by full `run_id`; and
- retrieve snapshots or evidence archives only through paths referenced by
  that result.

Result APIs are read-only. They cannot edit, delete, clear, resume, or relabel a
run. Arbitrary filesystem paths and short IDs are rejected.

## Acceptance requirements

Before the recorder can support a stage-ready run, automated tests must prove:

- activation persists before the first hardware call;
- every phase and terminal path is journaled in order;
- atomic materialization survives an interrupted write;
- startup seals an interrupted run without claiming a safe stop;
- camera, motion, return, operator-stop, and Remote Takeover failures retain
  the required reason and safety evidence;
- unavailable pose produces null Home measurements with a reason;
- terminal results cannot be resumed or mutated; and
- APIs cannot escape the configured run directory or alter evidence.
