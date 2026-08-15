# Border Collie deployment timing report

Snapshot: 2026-08-15, through ledger row 43 (`940c197`).

## Result

The controller-Start/autostart release deployed successfully as a four-service
whole project in **8.46 s** using
`scripts/deploy-stage-default --device woof.local -y`. The wrapper resolves the
normal native `wendy run --detach --restart-unless-stopped` path. Wendy emitted
build times for `home-recorder` (**0.741 s**), `media` (**0.205 s**), and
`voice` (**0.206 s**); the app build phase and readiness phase were not emitted
and remain `null`.

Post-deployment gates showed the exact v11 identity, controller input connected
and neutral-armed with zero accepted Start edges, activation ready, advancing
camera and passive pose sources, voice UI reachable, normal thermal state,
mission idle, and exact-zero disarm. This proves deployment and stationary
readiness. The controller was not pressed, and no Demo Run or motion was
started.

Sources: the central
[`deployment-timings.jsonl`](results/deployment-timings.jsonl), its
[`measurement contract`](deployment-timings.md), and the checked-in
[`deployment wrapper`](../scripts/deploy-stage-default).

## What the ledger contains

The ledger has **43 attempts**: **36 successful** and **7 non-successful**.
“Success” here means the ledger's final `outcome` is `success`; an accepted
command with a readiness timeout or unapplied configuration is not counted as
successful.

| Scope | Successful | Non-successful | Interpretation |
| --- | ---: | ---: | --- |
| Whole project | 6 | 4 | Includes scopes beginning `whole-project`; service sets and CLI versions vary. |
| Border Collie service-scoped | 29 | 3 | Includes app, recorder, media, paired app/media, and voice-with-dependencies. |
| External thermal monitor | 1 | 0 | Kept separate because it is not a Border Collie service build. |

Measurement completeness matters: complete command elapsed time exists for
**40/43** rows, at least one non-null service build phase for **33/43**, and a
separately measured readiness phase for only **15/43**. Missing phases are not
zero and are not reconstructed from other phases.

## Timing cohorts

| Cohort | n | Command median | Command range | Build median/range | Comparability |
| --- | ---: | ---: | ---: | ---: | --- |
| Successful app-only, command measured | 22 | 2.107 s | 1.110–64.240 s | — | Orientation only: CLI labels, targets, changes, and cache states differ. |
| Successful app-only with reported app build at most 0.6 s | 15 | 2.060 s | 1.110–2.634 s | 0.373 s / 0.088–0.479 s | A transparent low-build-time cohort, not proof that every row was warm. |
| Successful whole project, command measured | 6 | 7.678 s | 3.130–17.250 s | — | Orientation only: three- versus four-service layouts and emitted phases differ. |
| Current four-service v11 deployment | 1 | 8.460 s | — | app null; recorder 0.741 s; media 0.205 s; voice 0.206 s | Exact current result; readiness elapsed remains null. |

The 64.24-second app-only success (row 21) spent 62.29 seconds in the app build.
The ledger does not record its cache state or a root cause, so it must remain in
the all-success range and must not be silently reclassified or discarded.

## Cache state and build temperature

Cache state was not a schema-v1 field. Only four rows provide enough direct or
chronological evidence to classify, leaving **39/43 unreported**:

| Evidence | Classification | Observed result |
| --- | --- | --- |
| Row 8 says unchanged media reused its cached image. | Explicit warm cache reuse | Two sequential app/media commands took 5.563 s total; readiness took 4.0 s. Per-service build phases were not emitted. |
| Row 26 says Stagefile source layers rebuilt. | Explicit source-layer rebuild | App-only command 3.220 s; app build 1.153 s; readiness 0.042 s. |
| Row 42 says the changed recorder Stagefile key cold-rebuilt CycloneDDS. | Explicit cold dependency rebuild | Failed before replacement after 14.80 s; other reported service builds were short, but recorder build was null. |
| Row 1 is the first ledger entry after commit `1e8465a` made Stagefiles the only deployment builds. | Cache-migration candidate, inferred | App/media/voice builds were 108/210/116 s, command elapsed was not retained, and readiness later timed out. The ledger never labels this cache state, so it is not included in a warm/cold comparison. |

There is therefore no defensible warm-versus-cold speedup ratio in this ledger.
Future records need an explicit cache-state field before that comparison can be
made.

## CLI/source-version outcomes

| CLI/source group | Attempts | Success | Non-success | Evidence |
| --- | ---: | ---: | ---: | --- |
| PR 1698 development builds | 31 | 29 | 2 | One readiness timeout and one configuration-not-applied result. Several labels omit an exact binary hash, so these are one family, not necessarily one binary. |
| Unspecified WendyOS development binary | 2 | 2 | 0 | Exact source version was not retained. |
| Released `2026.08.13-010134` | 5 | 1 | 4 | All four Stagefile attempts failed before build because internal stage `native` was resolved as `docker.io/library/native`; the sole success was the external non-Stagefile thermal monitor. |
| Released `2026.08.15-001455` | 5 | 4 | 1 | Local-stage inheritance worked for successful Stagefile deployments. The one failure was a project dependency rebuild problem, not the earlier resolver regression. |

## Non-successful attempts and known causes

| Ledger row | Outcome | Cause and device boundary |
| ---: | --- | --- |
| 1 | `deployed_readiness_timeout` | Containers were replaced, then the app crash-looped because `VuiClient` was created before DDS initialization. The very long per-service builds are recorded; their cause is not. |
| 22 | `configuration_not_applied` | `wendy run` exited successfully, but live process and tuning evidence still showed the old value. No motion cohort was started. |
| 24, 30, 31, 37 | compile/failed-before-build | Released CLI `2026.08.13-010134` misresolved internal Stagefile stage `native` as a Docker Hub image and received `UNAUTHORIZED`. These were local failures before device mutation. |
| 42 | `failed-before-device-replacement` | Moving recorder runtime defaults into its Stagefile changed the cache key and cold-rebuilt CycloneDDS. CMake enabled OpenSSL but the image lacked `libssl-dev`, so `OpenSSL::SSL` was missing. The prior live release remained unchanged. |

## Why the recent loop became fast

The evidence supports four contributing practices, not one universal speedup:

1. **Narrow service replacement.** App-only work avoided rebuilding media/CUDA,
   voice, and the passive recorder. The 15-row low-build-time app cohort had a
   2.060-second command median. The 3.600-second all-fruit example is also
   documented in [`mostly-working-2026-08-13.md`](../docs/mostly-working-2026-08-13.md).
2. **Stable inputs precede volatile source.** The Stagefiles build native and
   Python dependencies before copying app source. The repository's current
   deployment guidance describes generated Dockerfiles as artifacts and keeps
   dependency/model inputs ahead of volatile code in
   [`README.md`](../README.md#historical-docker-optimization-proof).
3. **Independent build contexts.** `home-recorder` owns a separate Stagefile,
   lock, and context, so ordinary app iterations do not rebuild it. The initial
   shared-descriptor rollout remains whole-project by contract; later recorder
   or app-only changes can be scoped. See
   [`settled-home-recording.md`](../docs/settled-home-recording.md#one-recording-interface-two-build-contexts).
4. **The stable CLI repaired Stagefile inheritance.** Release
   `2026.08.15-001455` succeeded where the prior release failed locally. This
   removed a correctness blocker; it does not by itself explain every timing
   change.

Runtime tuning also avoids builds entirely where the contract permits it. The
run-tuning API changes one run without mutating process environment or requiring
a restart/deployment, as documented in
[`run-tuning.md`](../docs/run-tuning.md). The current controller release still
required a whole-project deployment because lifecycle policy and shared release
identity were part of the change.

## Practices to reuse

- Keep one append-only timing ledger and record failed or ambiguous attempts,
  CLI source/version, exact scope, command elapsed, per-service build,
  replacement/readiness evidence, and explicit nulls.
- Add `cache_state`, `build_lock_wait_s`, `stagefile_compile_s`, `upload_s`,
  `replacement_s`, and `functional_readiness_s` to the next schema. Without
  them, build, transfer, replacement, and readiness bottlenecks cannot be
  separated reliably.
- Partition services and layers by rate of change. Deploy only the affected
  service when schemas and dependencies are unchanged; use whole-project for a
  shared descriptor or cross-service contract.
- Keep runtime-only defaults in Wendy service environment or bounded runtime
  configuration. Do not change a stable native-dependency Stagefile merely to
  tune process values.
- Pin and record the released CLI version. Before a robot deployment, validate
  local Stagefile inheritance and generated output so resolver failures remain
  pre-device failures.
- Treat command exit as transport evidence, not functional success. Verify the
  exact release/configuration, fresh sensors, service health, idle mission,
  absence of competing motion owners, and exact-zero disarm.
- Continue using native `wendy run`; this report and its validator do not use
  DLO.

## Reproduce the calculations

Run, offline:

```sh
python3 scripts/deployment_timing_report.py
pytest -q tests/test_deployment_timing_report.py
```

The script validates every JSONL row before emitting the summary. The test pins
the 43-row outcome, scope, phase-coverage, cache-evidence, median, and range
calculations used above.
