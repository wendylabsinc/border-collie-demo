# Demo Run cohort policy

The audience page can start a bounded cohort of independent Demo Runs. The
operator chooses the run count, a seeded randomized non-empty subset of
Qualified Fruits or one fixed Target Fruit, and terminal reasons or failed
phases that may be tolerated at the cohort boundary. Omitting the subset keeps
the compatible all-qualified randomized default. Empty or unqualified subsets
are rejected before a Demo Run is created.

Defaults are deliberately strict:

- five runs;
- seeded randomized qualified fruits; and
- every failure stops the cohort.

"Tolerated" never means continue a failed phase or resume a Demo Run. The run
must already be terminal. Before the next activation, the controller requires
all of the following from the existing `StageDemo` status boundary:

- the inter-run evidence names that exact prior `run_id`;
- fresh current pose is within `BORDER_COLLIE_STAGE_HOME_MARGIN_M` of that
  run's captured Home (default `0.50 m`);
- there is no active run or recovery;
- motion is disarmed with exact-zero forward and yaw commands;
- activation readiness passes; and
- no Remote Takeover or restart-required state is latched.

Camera, pose/preflight, motion, Remote Takeover, interrupted-process,
return-Home, unconfirmed-disarm, internal, and operator-stop failures are hard
stops even if a submitted tolerance selector matches them.

## HTTP interface

- `POST /api/cohorts` starts one cohort and returns its durable record.
- `GET /api/cohorts/active` observes the current or latest cohort.
- `POST /api/cohorts/active/stop` stops the cohort and the active Demo Run.
- `GET /api/cohorts/{cohort_id}` reads a durable cohort by full ID.

There is no shell, arbitrary command, or low-level motion endpoint. Every run
uses the same idempotent `StageDemo.activate()` path as a single run. A cohort
creates exactly one stable activation ID per scheduled run and never retries an
ambiguous activation.

Mission-level tuning is deliberately outside `CohortPolicy`. If the cohort API
accepts tuning, it should be one optional sibling request object and the
application must construct and persist a fresh immutable tuning snapshot for
each scheduled `FruitMission`; failure tolerance must never alter tuning or its
safety validation.

The durable cohort record contains the complete policy, exact selected fruit
set, seed, precomputed Target Fruit sequence, requested tuning template, one
effective immutable tuning snapshot per scheduled run, each Run Result
identity/outcome, the policy decision, and the exact inter-run Home-clearance
evidence.

By default cohort records use a `cohorts/` sibling of
`BORDER_COLLIE_RUNS_DIR`, so a deployment's durable Run Result mount also
covers cohort policy. `BORDER_COLLIE_COHORTS_DIR` can override that location.

## Harness interface

`scripts/fruit_soak.py` uses the same typed policy model. For example:

```bash
python3 scripts/fruit_soak.py --runs 5 --seed 20260813 \
  --fixed-fruit banana --tolerate-reason ACTION_FAILURE
```

Omit `--fixed-fruit` for a seeded randomized sequence. Repeat
`--tolerate-reason` or `--tolerate-phase` to add selectors. Safety hard stops
and Home clearance always override those selectors.
