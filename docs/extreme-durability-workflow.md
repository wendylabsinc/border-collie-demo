# Extreme durability workflow

This workflow hardens the base Demo Run without deploying intermediate motion
changes to Woof. Each layer must pass the complete local suite before the next
layer becomes an integration dependency.

## Worktrees

For the full current lineage, stacked PR structure, integrated commit mapping,
and parallel experiment lanes, see [Demo and worktree
progression](worktree-progression.md). This section describes the original
durability decomposition.

- Primary integration: `codex/extreme-durability-foundation` in
  `border-collie-demo-durability`.
- Mature Target Fruit tracking: `codex/durability-qualified-tracking` in
  `border-collie-demo-durability-tracking`.
- redundant Home localization: `codex/durability-home-localization` in
  `border-collie-demo-durability-home`.
- media supervision: `codex/durability-service-supervision` in
  `border-collie-demo-durability-supervision`.

The three original derived branches started at foundation commit `7acdc71`.
Their qualified tracking, Home localization, and supervision changes were
integrated deliberately into PR #5 under new commit hashes after their
interface tests and full suites passed. The source branches therefore do not
appear as direct ancestors of the current stage branch. Physical-run artifacts
in the original checkout stay untracked and are never copied into these
worktrees.

## Layer order

### 1. Durable Run Coordinator

The Run Coordinator is the sole module that orders Demo Run lifecycle changes.
Activation is idempotent, the durable transition is written before the
in-memory projection changes, and process restart seals interrupted work only
after a stop attempt establishes the final safety state.

Gate:

- repeated activation with the same key returns the same `run_id`;
- the same key cannot describe a different intent;
- journal transitions are hash-linked and fsynced;
- startup never claims a confirmed disarm without hardware evidence.

### 2. Motion Guardian

The Motion Guardian is SDK-neutral. Every Unitree command must hold one current
permit fenced to the Demo Run, run epoch, phase, and hardware operation. The
permit expires independently of the command watchdog and authorizes only the
configured velocity directions and limits.

Gate:

- stale generations cannot command motion;
- turn-only authority cannot command forward motion;
- expiry stops and releases the hardware adapter;
- emergency stop revokes the permit without exposing its token.

### 3. Qualified Target Fruit tracking

The primary branch carries the mature temporal tracker, so no approach decision
depends on one frame. Its interface returns explicit stop, align, approach,
slow, and Arrival recommendations.

Integration gate:

- raw weak detections remain available for `/fruit-test` and evidence;
- motion never acquires a 0.01-confidence phantom;
- close-range continuation requires prior acquisition and continuous geometry;
- wrong identity, stale frames, and discontinuous geometry stop or fail closed;
- initial centering is strict, while ordinary post-acquisition corrections
  combine forward motion with proportional yaw;
- same-fruit reacquisition preserves the completed centering gate, and yaw-only
  recentering is reserved for large horizontal error;
- pear close-range confidence collapse can reach Arrival without contact.

### 4. Redundant relative-motion localization

Fuse Go2 metric pose, body velocity, IMU yaw rate, loaded-foot stationary
updates, and bounded CPU-only sparse visual motion. Essential-matrix visual
translation is direction-only and cannot invent metres; Go2 displacement
supplies scale.

Integration gate:

- Go2-only behavior remains regression compatible;
- fresh agreeing visual direction/yaw reduces pose uncertainty;
- stationary foot contact drives velocity to zero and learns gyro bias;
- timestamp regressions, sample gaps, and persistent innovations fail closed;
- stale visual evidence falls back explicitly and persistent disagreement fails closed;
- visual processing stays within its CPU, latency, memory, and thermal budgets;
- the final measured Home Distance remains at most `0.10 m`.

### 5. Isolated media supervision

Supervision distinguishes process liveness, transport connection, advancing
frames, and stable generation readiness. Restart and stale-session cleanup use
bounded backoff, and motion remains unavailable until the new generation is
stable.

Integration gate:

- bounded startup failures recover without restarting the mission module;
- a frame stall removes readiness;
- generation replacement requires a new stability window;
- exhausted restart attempts become an explicit terminal supervisor state.

### 7. Durable flight recorder

The flight recorder persists a bounded, hash-linked rolling window outside the
container run directories. It records lifecycle transitions, motion authority,
accepted velocity commands, pose samples, and Target Fruit evidence. Failed and
stopped runs freeze the window as `flight-recorder.ndjson`; completed runs keep
only their compact Run Result.

Gate:

- the sequence and hash chain continue after restart;
- rotation has a fixed storage ceiling;
- failure evidence remains available when media capture fails;
- successful-run detail cleanup does not remove the rolling recorder.

## Deferred layer 6: recovery budgets

Recovery budgets follow the above modules because they consume their trusted
state. Implement one `RecoveryBudgetPolicy` module rather than scattering
timeouts across hardware methods. Its decision input will contain phase,
elapsed time, measured translation, measured yaw, perception-loss time,
reacquisition count, service restart count, and remaining motion authority.
Its output will be `continue`, `return_home`, or `safe_stop` with a stable reason
code.

Initial budgets to qualify later:

- maximum search yaw and search duration;
- maximum approach translation and time without a qualified track;
- maximum Home-search yaw and absolute-localization attempts;
- maximum recovery translation and total recovery duration;
- maximum media restarts per Demo Run;
- one total mission deadline that cannot be extended by state oscillation.

Budget values require replay and physical measurements. They are not selected
from intuition and are not part of the current implementation pass.

## Gated layers 8 and 9

Atomic release support and fault injection begin only after layers 1-5 and 7
are integrated and the complete suite is green.

For layer 8:

1. Run the Docker Layer Optimizer plan required by the repository workflow.
2. Record app and media image digests plus configuration schema in the release
   manifest.
3. Verify both containers before activating the group.
4. Preserve the prior known-good manifest for rollback.
5. Reject mixed app/media release identities.

For layer 9:

1. Add deterministic failure adapters at existing seams.
2. Inject frame loss, stale generations, weak phantoms, pose loss, API response
   loss, process restart, supervisor exhaustion, and partial evidence writes.
3. Assert the terminal Run Result, final safety state, flight-recorder evidence,
   and retry behavior for every fault.
4. Run replay and simulation volume before any supervised physical soak.

## Verification command

Run from the active worktree:

```bash
PYTHONPATH=src:. \
  /Users/olivertaylor/Documents/Wendy/border-collie-demo/.venv314/bin/pytest -q
```

Passing tests establish implemented behavior only. A later deploy must still
report built, deployed, readiness-verified, and physically executed states
separately.
