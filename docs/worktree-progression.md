# Demo and worktree progression

Status snapshot verified from the local Git graph and GitHub PR metadata on
2026-08-11. Refresh the commands at the end before relying on this map later.

This document answers two different questions:

1. Which commits form the current stage candidate?
2. Which worktrees are parallel source, research, or tooling lanes rather than
   later versions of the demo?

A worktree is an isolated checkout, not a release. Its directory name is only
local scaffolding. The branch, commit, Wendy release label, and recorded
physical evidence are the authoritative identity.

## Current stage lineage

The current stage candidate is a stacked progression:

```text
main (clean scaffold)
  -> codex/live-go2-foundation
  -> codex/inference-worker-soft-final-push
  -> codex/voice-activation-source
  -> demo/base history
  -> codex/base-soak-recovery-harness        PR #4
  -> codex/durability-approach-cadence       PR #5
  -> codex/stage-camera-three-fruit          PR #6
  -> codex/stage-camera-final-approach-latch
  -> codex/stage-camera-bottom-clip-closeout
  -> codex/stage-camera-v18-closeout-recovery
  -> codex/stage-camera-v19-search-handoff
  -> codex/stage-camera-v20-continuous-track
  -> codex/stage-camera-v21-search-policy-abc
```

| Checkpoint | Purpose | Evidence status |
| --- | --- | --- |
| `main` | Clean-room application scaffold. | Code checkpoint only. |
| `codex/live-go2-foundation` | First guarded Go2 motion and physically exercised demo foundation. | Historical physical evidence; its PR is closed. |
| `codex/inference-worker-soft-final-push` | Multi-fruit recognition and resident banana specialist. | Incorporated into later base history; its standalone PR is closed. |
| `codex/voice-activation-source` | Voice-demo source checkpoint. | Open PR #3; voice activation is not part of the current button-driven acceptance gate. |
| `demo/base` | Golden camera-driven behavior used for the supervised baseline. | Historical 3/3 pear success at the recorded 2026-08-08 snapshot; do not project that result onto newer commits. |
| `codex/base-soak-recovery-harness` | Seeded randomized ten-run harness, compact success records, retained failure evidence, and bounded failed-run Home recovery. | Open PR #4. Local `demo/base` currently points at the same `c25a381` commit, while `origin/demo/base` is older; always identify the commit. |
| `codex/durability-approach-cadence` | Integrated durability foundation, supervision, qualified tracking, Home fusion, fault injection, smoother approach control, and the later metric/LiDAR experiments. | Open PR #5. Deployed and physically iterated, but the LiDAR Arrival path did not achieve repeatable stage success. |
| `codex/stage-camera-three-fruit` | Keeps PR #5's durability/recovery work while restoring a base-compatible camera closeout shared by apple, banana, and pear. | Open stacked PR #6. Commit `7a4686f` supplied the runtime change and `c6f70eb` added this lineage map; the latter was deployed with the same `stage-camera-v15-base-compatible-arrival` runtime identity. Two supervised pear runs have executed: run one completed the fruit/audience path but failed heading restoration after Home distance grew from 0.079 m to 0.124 m; run two reached centered close geometry but weak/stale/missing evidence prevented final-push authorization and timed out without automatic Home recovery. It is physically executed but not qualified. |
| `codex/stage-camera-final-approach-latch` | Repairs the v15 closeout state-transition hole with a bounded, one-way Final Approach latch while leaving the metric/LiDAR profile unchanged. | Deterministic replay of run `bce852b0-8645-46b4-9d26-dedcb6cb81a4` reaches one terminal camera final-push authorization instead of waiting for the overall timeout. Negative tests keep stale/frozen, wrong-label/generation, off-axis, invalid, unhealthy, and expired evidence fail-closed. This checkpoint is not physically qualified. |
| `codex/stage-camera-bottom-clip-closeout` | Specializes the latched camera closeout for a centered pear box shrinking as it clips out through the lower image edge. | Deterministic tracker and hardware replays of run `acfdb493-0afd-46e9-bff5-ee6caf8f152e` stop on the first qualifying area retreat and authorize one configured final push only after a second fresh advancing loss sample. The global 35% continuity limit remains unchanged, and non-bottom, off-axis, stale, duplicate, wrong-label/generation, invalid, unhealthy, and expired evidence remain fail-closed. This checkpoint is not physically qualified. |
| `codex/stage-camera-arrival-failure-home-recovery` | Extends the existing one-shot, no-bark failed-run recovery to eligible `ARRIVAL_FAILURE` outcomes with fresh Home, pose, fusion, disarm, motion-release, pulse-budget, posture, and takeover gates. | Parallel source lane derived from v16 at `5882aeb`; its implementation is integrated into the combined v18 checkpoint rather than merged independently into the progression. |
| `codex/stage-camera-v18-closeout-recovery` | Combines the bottom-clipped Final Approach closeout and automatic eligible Arrival-failure Home recovery under release `stage-camera-v18-closeout-recovery`, schema 4, app version `1.0.21-stage-camera`. | Integration checkpoint derived from the deployed/readiness-verified v17 branch. Software and deployment evidence are recorded separately; physical closeout and recovery behavior remain unqualified until supervised execution. |
| `codex/stage-camera-v19-search-handoff` | Preserves v18 closeout/recovery and adds an explicit bounded search-to-approach qualification handoff without lowering ordinary fruit acquisition thresholds. Release `stage-camera-v19-search-handoff`, schema 4, app version `1.0.22-stage-camera`. | Derived from v18 after physical apple run `5e2dc136` qualified search at 0.7106289 but approach saw 165 sub-0.70 same-label frames and never acquired. Deterministic replay proves the handoff seeds identity only; current tracking-floor geometry and three fresh centered approach samples remain mandatory. Physical repeatability is not yet qualified. |
| `codex/stage-camera-v20-continuous-track` | Preserves v19 identity handoff while separating its one-second cross-stage setup allowance from the unchanged 250 ms current-frame freshness gate. Close pear correction remains at the qualified 0.55 m/s approach speed with slew-limited yaw instead of stopping for an in-place recenter. Release `stage-camera-v20-continuous-track`, schema 4, app version `1.0.23-stage-camera`. | Derived from physical pear run `e3054a9a-5ecf-4526-ac5f-f3a980d6d22b`: v19 rejected the handoff as stale, issued 20 initial in-place centering commands, advanced for 17 pulses, then issued three close in-place turns and failed off-axis. Automatic recovery returned to 0.0529 m and disarmed. v20 software replays are green; physical repeatability is not yet qualified. |
| `codex/stage-camera-v21-search-policy-abc` | Integrates the three search experiments behind one immutable per-run selector: three-frame `fast-lock`, `0.50 rad/s` `slow-sweep`, and bounded `double-back`. Release `stage-camera-v23-apple-pear-confidence`, schema 7, app version `1.0.26-stage-camera`, retains v22 black-box traces and gives apple the same 0.65 acquisition and 0.55 tracking floors as pear. | Supervised randomized run `9a55f553-0ad8-43bf-b711-ea657962caf6` found Apple proposals up to 0.789 but held only two consecutive qualifying detections, so the five-frame slow-sweep lock failed closed without forward motion. |
| `codex/perception-throughput-v1` | Adds a reversible perception scheduler experiment: `baseline` preserves v22 serialization while `throughput-v1` publishes motion evidence before latest-only odometry/preview lanes and staggers missing apple/banana full-frame and search-crop passes across advancing frames. A crop candidate keeps that normalized route until loss so interleaved full-frame misses cannot reset acquisition. Standalone release `stage-camera-v23-perception-throughput`, schema 7, app version `1.0.26-stage-camera`. | Derived from v22 read-only evidence of 14.285 source FPS but 5.665 completed detector FPS. Software and synthetic scheduling checks passed on the source branch; physical throughput and reliability remained unqualified. |
| `codex/stage-camera-v21-search-policy-abc` + throughput integration | Combines the v23 Apple/Pear confidence policy with the reversible scheduler. Baseline cohort `stage-camera-v24-apple-pear-pipeline-baseline` used schema 8/app `1.0.27`; comparison cohort `stage-camera-v25-apple-pear-throughput-v1` used schema 9/app `1.0.28`. Candidate `stage-camera-v26-centered-second-scan` uses schema 10/app `1.0.29`. | The matched v24/v25 Apple/Pear/Banana cohorts each scored 0/3. Throughput-v1 improved processed cadence from 7.00 to 11.56 FPS and reduced the cumulative drop ratio from 50.5% to 10.9%. V26 keeps broad search at 0.50, centers the second scan at 0.20 with bounded hold/loss behavior, and reduces only Apple acquisition from five to three fully qualified frames. Its supervised randomized five-attempt cohort scored 0/5 complete missions: both Apple attempts reached Arrival and completed the audience action but failed the strict return gate; both Pear attempts failed off-axis and automatically recovered inside 0.10 m; Banana reached Arrival but was operator-stopped during the action after a stale terminal Home value was mistaken for current position. V26 is physically characterized, not qualified. |
| `codex/stage-camera-v21-search-policy-abc` v27 integration | Derives from deployed v26 and combines three reviewable changes: the Apple policy uses one 0.50 acquisition/tracking floor, approach translation is confined to the middle 40% of the frame with a bounded 0.20 rad/s recenter, and the soak harness waits for authoritative automatic recovery plus a 0.50 m inter-run stage margin. Release `stage-camera-v27-center-corridor-apple50`, schema 11, app version `1.0.30-stage-camera`. | The equal Apple floor avoids inverted acquisition/tracking hysteresis; Pear and Banana confidence policies remain unchanged. The corridor removes forward authority immediately outside center `0.30..0.70`, confirms recenter on a second fresh sample, and never relaxes stale/identity/camera stops. This candidate is software-qualified but physically unqualified until deployment and supervised runs. |

PR #6 is based on PR #5, and PR #5 is based on `demo/base`. This keeps each
review focused without pretending the latest stage candidate is already merged
into the public base branch.

## How the durability worktrees fed PR #5

The durability effort deliberately used parallel worktrees so safety modules
could mature independently. Three branches are source lanes, not direct Git
ancestors of the current stage branch, because their finished changes were
integrated under new commit hashes:

| Source worktree | Source branch and commit | Integrated commit in PR #5 | Contribution |
| --- | --- | --- | --- |
| `border-collie-demo-durability-home` | `codex/durability-home-localization` at `315c582` | `7b85145` | Fail-closed redundant Home localization. |
| `border-collie-demo-durability-supervision` | `codex/durability-service-supervision` at `9dd7f1b` | `33372a3` | Media restart supervision and generation fencing. |
| `border-collie-demo-durability-tracking` | `codex/durability-qualified-tracking` at `edb6273` | `0db0df9` | Mature temporal fruit tracking. |

These worktrees are direct ancestors of the current stage branch:

| Worktree | Branch and commit | Contribution |
| --- | --- | --- |
| `border-collie-demo-durability-visual-odometry` | `codex/durability-visual-odometry` at `9e213cd` | Bounded CPU visual odometry and breadcrumb return. |
| `border-collie-demo-durability-sensor-fusion` | `codex/durability-sensor-fusion` at `4524402` | Stateful planar fusion. |
| `border-collie-demo-durability` | `codex/extreme-durability-foundation` at `0d13e49` | Primary integration through direct SportClient turns and widened tracking corridor. |
| `border-collie-demo-durability-cadence` | `codex/durability-approach-cadence` at `f230493` | Cadence fixes followed by the metric/LiDAR Arrival experiments; head of PR #5. |
| `border-collie-demo-stage-camera` | `codex/stage-camera-three-fruit` at `c6f70eb` (`7a4686f` runtime change) | Current stage-oriented camera closeout and lineage documentation; head of PR #6 at the recorded run. |

Do not merge the source worktree branches again: their intended contributions
are already present in PR #5. Compare behavior or tests there when debugging,
but use the integrated branch for deployment.

## Parallel lanes that are not the current demo progression

| Lane | Role | Relationship to the stage candidate |
| --- | --- | --- |
| `demo/edge` | Newer risky runtime experiments. | Separate from the current stage lineage. |
| `demo/max` | MAX integration target. | Separate and not authorized for Woof deployment while its runtime artifacts remain incomplete. |
| `demo/approach-consistency` | Earlier approach-control experiment. | Separate reference lane. |
| `codex/woof-fruit-mcp` | Fail-closed MCP tools for starting/stopping fruit runs. | Tooling branch; not integrated into PR #6. |
| `codex/hey-woof-wakeword` | Wake-word exploration. | Voice/tooling lane; not the stage motion implementation. |
| `codex/go2-webcam-tracker-training` | Tracker-training checkpoint and v7 release stamp. | Its v7 checkpoint is in PR #5 history, but training work remains a separate concern. |
| `codex/prototype-camera-trust-contract` | Throwaway camera-source trust prototype. | Research input only. |
| `research/physical-remote-takeover-signal` | Remote takeover signal research. | Research input only. |
| `codex/prototype-max-native-skeleton` | MAX-native detector benchmark skeleton. | Performance prototype only. |
| `codex/stage-ready-integrated-demo` / `codex/robust-arrival-checkpoint` | Older supervised Arrival checkpoint. | Historical evidence branch, not an ancestor of PR #6. |

Temporary detached worktrees and `tmp/*` branches are evidence or integration
scratch space. They must never be selected for deployment based on their folder
name.

## Evidence labels

Use these terms precisely in issues and PR comments:

- **Implemented:** code exists at an identified commit.
- **Tested:** the stated local or simulated suite passed at that commit.
- **Built:** Wendy successfully built the complete root and media Stagefiles.
- **Deployed:** the identified app/media release is installed on Woof.
- **Readiness verified:** app and media identities match, frames advance, there
  are no activation blockers, and motion is disarmed.
- **Physically executed:** a supervised robot run actually occurred.
- **Qualified:** the required consecutive physical acceptance set passed and
  its result artifacts were retained.

Deployment is not qualification. In particular, v15 is deployed,
readiness-verified, and physically executed twice. Run
`7b5cfd93-7359-4f4c-908e-9b791e0bde93` proved the camera closeout and audience
action but ended `RETURN_HOME_FAILURE` during heading restoration, so v15 is
not qualified. Run `bce852b0-8645-46b4-9d26-dedcb6cb81a4` then ended
`ARRIVAL_FAILURE` after close-range evidence became weak/stale/missing and did
not trigger automatic Home recovery.

## Verifying the map

Before editing, merging, or deploying, refresh the live state:

```bash
git worktree list
git branch --show-current
git rev-parse HEAD
git status --short
git log --reverse --oneline demo/base..HEAD
gh pr list --state open
```

Before motion, also verify the installed build label, matching app/media release
IDs, active run/recovery state, readiness blockers, and confirmed disarm. A
worktree path or a healthy media port alone does not establish mission identity.
