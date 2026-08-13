# Banana reliability replay

Run the three minimized physical-failure traces without a robot, network, or
motion client:

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

These are deterministic software replays. Passing them proves the controller
contracts, not physical motion, sensor quality, or deployment readiness.
