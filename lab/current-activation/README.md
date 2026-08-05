# Current Activate Demo readiness check

These checks answer one narrow question: what happens when an operator opens the
clean application's current audience surface and uses **Activate Demo**?

They run the clean app locally on port `8110` with hardware disabled. They must
not fall back to the legacy application on port `8096` or send a physical
motion command.

The results are readiness snapshots, not hardware acceptance tests. The first
captured the intentionally blocked scaffold. The second verifies the
recorder-backed activation lifecycle while documenting that autonomous
orchestration is still absent.

Evidence is stored in:

- `results/ACTIVATE-CURRENT-001.json`
- `results/ACTIVATE-RECORDER-001.json`
- `results/PREFLIGHT-FAIL-CLOSED-001.json`
