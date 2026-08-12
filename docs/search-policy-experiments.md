# Search policy A/B/C experiment

Release `stage-camera-v28-lie-down-evidence` compares three bounded Target Fruit
search policies without changing approach, Arrival, action, or Home behavior.
The application default is configured by `BORDER_COLLIE_SEARCH_POLICY`; a run
may select one canonical policy in `POST /api/run`. The selected policy is
persisted before preflight, carried immutably through the run, and retained in
both success summaries and full failure evidence. Reusing an idempotency key
with a different policy is rejected.

| Policy | Experimental change | Unchanged safety gates |
| --- | --- | --- |
| `fast-lock` | Use three consecutive fully qualified fresh frames for acquisition. | Per-fruit acquisition confidence, camera health, source/detection freshness, generation, timebase, advancing PTS, inference budget, fruit identity, and normalized geometry. |
| `slow-sweep` | Use `0.50 rad/s` for the broad sweep. A same-fruit proposal at `0.50` confidence enters a yaw-only second scan: hold for `0.75 s`, then center at `0.20 rad/s` after two samples outside `+/-0.12` frame width. Brief loss holds zero for at most `1.0 s`; at most two alignment episodes may start. | Apple requires three fully qualified acquisition frames; pear and banana require five. Candidate alignment cannot authorize translation or relax any perception gate. |
| `double-back` | Dwell up to `0.75 s` on a qualified proposal, then reverse toward it after `0.50 s` loss. A reverse is capped at `0.35 rad`, `0.75 s`, two episodes, and a two-second absolute episode budget. | Apple requires three fully qualified acquisition frames; pear and banana require five. Full freshness/identity checks, absolute search deadline, yaw-only search, and fail-closed generation changes remain. |

The matched physical trial uses one deployment, one seed, the same fruit and
orientation sequence, and ten runs per policy:

```bash
python3 scripts/fruit_soak.py --host woof.local --runs 10 --seed 2026081104 \
  --no-orientation-randomization --recover-failures \
  --expected-build-label "stage-camera-v28-lie-down-evidence (codex/stage-camera-v21-search-policy-abc)" \
  --expected-search-policy slow-sweep --expected-fruits apple banana pear \
  --search-policy fast-lock --output benchmarks/results/search-fast-lock.json
```

Repeat with `slow-sweep` and `double-back`, preserving every other argument.
Then generate the comparison:

```bash
python3 scripts/compare_search_policies.py \
  benchmarks/results/search-fast-lock.json \
  benchmarks/results/search-slow-sweep.json \
  benchmarks/results/search-double-back.json \
  --output benchmarks/results/search-policy-abc-comparison.json
```

The comparison refuses unmatched seeds, fruit sequences, or orientation
sequences. It reports completion/error rates, failures by reason/phase/fruit,
median acquisition time, Home distance, recovery success, and network poll
errors. A software replay or one successful run is not policy qualification.
