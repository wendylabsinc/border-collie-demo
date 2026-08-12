# V28 lie-down evidence deployment runbook

Release: `stage-camera-v28-lie-down-evidence`  
Schema: `12`  
App version: `1.0.31-stage-camera`

V28 changes evidence and operator tooling only. It does not change search,
confidence, approach, Arrival, action, or Home motion policy.

## Offline software gate

```sh
scripts/fruit-soak test
```

This must pass before device access. The full repository suite is:

```sh
PYTHONPATH=src:. /Users/olivertaylor/Documents/Wendy/border-collie-demo/.venv314/bin/pytest -q
```

## Read-only pre-deployment device gate

Run only after Woof is back online:

```sh
scripts/fruit-soak check \
  --host woof.local \
  --expected-search-policy slow-sweep \
  --expected-fruits apple banana pear
```

This command cannot activate motion. It requires idle/no takeover/no recovery,
fresh pose, exact disarm and zero command, matching app/media identity, stable
camera generation, advancing PTS across two samples, fresh frames, bark
readiness, and a valid JPEG.

## Whole-project Stagefile deployment

```sh
wendy run --detach
```

Do not use DLO. Do not retry an ambiguous deployment. Resolve the original
command to a definite result first.

## Exact post-deployment gate

```sh
scripts/fruit-soak check \
  --host woof.local \
  --expected-build-label "stage-camera-v28-lie-down-evidence (codex/stage-camera-v21-search-policy-abc)" \
  --expected-search-policy slow-sweep \
  --expected-fruits apple banana pear
```

## Quick physical cohort

This schedules one balanced attempt per fruit with reproducible randomized
orientations, follows automatic recovery, and blocks the next activation unless
the client is disarmed and inside the 0.50 m stage Home margin:

```sh
scripts/fruit-soak run \
  --host woof.local \
  --runs 3 \
  --seed 2026081201 \
  --expected-build-label "stage-camera-v28-lie-down-evidence (codex/stage-camera-v21-search-policy-abc)" \
  --expected-search-policy slow-sweep \
  --expected-fruits apple banana pear \
  --search-policy slow-sweep \
  --output benchmarks/results/2026-08-12-v28-quick-three.json
```

After each audience action or eligible failed-run recovery, open `/debug` and
inspect the inline **Fruit position when Woof lay down** image. A missing image
is recorded as unavailable and never changes the mission or recovery result.
