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
| `codex/stage-camera-three-fruit` | Keeps PR #5's durability/recovery work while restoring a base-compatible camera closeout shared by apple, banana, and pear. | Open stacked PR #6. Commit `7a4686f` was built and deployed as `stage-camera-v15-base-compatible-arrival`; readiness and disarm were verified. No physical fruit run has yet qualified it. |

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
| `border-collie-demo-stage-camera` | `codex/stage-camera-three-fruit` at `7a4686f` | Current stage-oriented camera closeout; head of PR #6. |

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

Deployment is not qualification. In particular, v15 is deployed and readiness
verified, but it is not yet physically executed or qualified.

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
