# Deployment timing ledger

All Border Collie deployment measurements belong in
[`results/deployment-timings.jsonl`](results/deployment-timings.jsonl).

Record one JSON object per deployment, including failed or ambiguous attempts.
Keep these measurements separate:

- `command_elapsed_s`: wall time for the complete `wendy run` command;
- `service_build_s`: Wendy's reported build time for each service;
- `readiness_s`: readiness time when the CLI reports it;
- `scope`: `whole-project` or the exact selectively deployed service.

Never infer end-to-end time from build time. Use `null` when a phase was not
measured. The deployment command must be the native Wendy workflow; DLO is not
used in this project.

