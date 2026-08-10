# Fault injection workflow

Fault injection is deterministic and simulation-only. Constructing the
injector in production mode is rejected before a fault plan can run.

Each `FaultSpec` names a module seam, exact occurrence, action, and parameters.
The injector records every applied fault so a failed replay can be reproduced.
Supported actions cover exceptions, response loss after commit, simulated
process restart, dropped values, stale perception, weak phantoms, camera
generation replacement, and partial writes.

Initial durability scenarios:

| Fault | Required result |
| --- | --- |
| Activate response lost after durable commit | Retry the same idempotency key; no second Demo Run |
| Weak `0.01` Target Fruit phantom | Never acquire or authorize forward motion |
| Stale source/detection timestamps | Camera readiness fails closed |
| Connection generation replacement | Old tracking evidence cannot continue motion |
| Media restart-budget exhaustion | Supervisor becomes explicit `failed` |
| Go2 pose loss or persistent visual disagreement | Home estimate becomes unavailable; no inferred arrival |
| Partial flight-recorder write | Restart truncates the torn tail and continues the hash chain |
| Process restart with an active run | Attempt stop, seal interrupted run, never resume motion |

Run these scenarios in local tests and high-volume simulation before a
supervised physical soak. The injector is not packaged as a production toggle,
and fault plans never authorize robot deployment.
