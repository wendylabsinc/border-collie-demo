# Banana reliability replay

Run the minimized physical-failure traces and all-fruit bearing-map contracts
without a robot, network, or motion client:

```bash
python scripts/replay_banana_failures.py
```

The command exits nonzero if any contract regresses. It replays:

- `af45a566-d0c8-4c97-9640-28abe5ee1512`: 267.8 ms Banana evidence removes
  motion authority but may recover before the 500 ms terminal deadline;
- `689e9005-5ae9-4579-ad4c-3e25022c1779`: an off-center Banana retains the
  fine-alignment direction through a bounded missing-evidence interval; and
- `1d5129e4-9f18-4cb3-8f74-59e81b95dbc1`: a Home heading-gate escape is
  reconciled only by a fresh post-disarm position inside 0.10 m.
- one general-model full-frame pass publishes raw Apple, Banana, and Pear
  observations while selected-target guidance remains isolated;
- centered observations map confidence-weighted robot-local bearings, route by
  the shortest signed turn, and reject stale or contradictory evidence; and
- reuse requires the same camera generation and odometry epoch; the shortest
  turn is calculated from the new run's fresh captured Home heading. Home
  position offset is recorded but does not invalidate yaw-only routing.

Routing is an A/B control, not a deployment-time fork. The library default is
`BORDER_COLLIE_BEARING_ROUTING_ENABLED=0`; the current stage descriptor sets it
to `1`. Each immutable Demo Run exposes
the matching UI field `search.bearing_routing_enabled`. Recording remains on in
both variants. A baseline run with `false` performs the established broad scan
and can populate the process-local map without a map-induced turn. A subsequent
run with `true` may reuse a qualified bearing. The run, black box, API status,
and cohort record expose whether routing was enabled, whether a route was used,
its fallback reason, and route-planning time. Changing this boolean never lets
the map authorize translation or Arrival.

These are deterministic software replays. Passing them proves the controller
contracts, not physical motion, sensor quality, or deployment readiness.
