# Final Demo: First Verified 5/5

## Canonical version

- Working branch: `demo/final`
- Immutable tag: `demo-final-5of5-2026-08-13`
- Deployed behavior commit: `1f783ecb51188135fca314ab101b7021b7b89e10`
- Cohort evidence commit: `01d286e`
- Live build label: `stage-default-v3-runtime-tuning-cohorts`
- Device: Woof

Future Border Collie demo work starts from `demo/final`. The historical
`demo/base` branch remains intact as the earlier 3/3 supervised Pear proof; it
is not the current three-fruit baseline.

## Qualification result

On 2026-08-13, seeded randomized cohort `2026081315` completed five consecutive
Demo Runs with five successes:

| Run | Target Fruit | Run ID | Forward pulses | Terminal Home Distance |
| --- | --- | --- | ---: | ---: |
| 1 | Banana | `09479c45-627b-4e17-8468-0c7948257aef` | 20 | 0.0507 m |
| 2 | Pear | `9f601a90-b8d8-48b1-aef5-d37bc42fc739` | 25 | 0.0266 m |
| 3 | Banana | `4b60464c-e1eb-4e1c-bffa-091cf84b6984` | 20 | 0.0568 m |
| 4 | Apple | `f4342d5b-c342-4ecc-9159-0b4795cae5f4` | 19 | 0.0375 m |
| 5 | Pear | `96ca95f1-66ae-4efa-a3b2-bebf0d910258` | 30 | 0.0218 m |

Every run reached direct lower-edge Arrival after positive forward motion,
completed the audience action, returned inside the application `0.10 m` Home
gate, and ended `DISARMED_CONFIRMED`. One operator-side HTTP poll reported
`Network is unreachable` during Apple; the robot-side Demo Run continued and
completed safely.

The durable cohort record and five start frames are:

- `benchmarks/results/2026-08-13-random-five-stationary-jump.json`
- `benchmarks/results/2026-08-13-random-five-stationary-jump-frames/`

## Qualified behavior snapshot

- Apple focus/lock confidence: `0.40` / `0.40`
- Banana lock/tracking confidence: `0.20` / `0.20`
- Pear lock/tracking confidence: `0.65` / `0.55`
- Approach: factory avoidance at `1.0 m/s`
- Direct Arrival: fresh valid fruit box bottom at or beyond `0.90`
- Disappearance Arrival: immediately after a qualified `0.80` lower-edge frame
- Stationary lower-edge jumps: rejected until two ordinary, agreeing Target
  Fruit observations restore the existing lock
- Moving Home correction: exact zero inside the configured deadband and at
  least the physically verified `0.50 rad/s` outside it
- Home completion gate: fresh measured position within `0.10 m`
- Inter-run operator margin: fresh measured position within `0.50 m`

## Evidence boundary

This is the first verified five-of-five cohort in the recorded Border Collie
history, and the first such cohort to include Apple, Banana, and Pear. It is a
strong stage baseline, not a statistical reliability guarantee: Apple occurred
once, Banana and Pear occurred twice, voice activation was not part of this
cohort, and the stationary-jump rejection branch was replay-tested but did not
trigger during these five physical runs.

## Starting new work

```sh
git fetch origin --tags
git switch --track origin/demo/final
```

Experimental changes should branch from `demo/final` and preserve this tag and
cohort evidence unchanged.
