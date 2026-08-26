# Mostly Working Checkpoint — 2026-08-13

Git tag: `mostly-working-2026-08-13`

This checkpoint is the current stage-default implementation after the
all-fruit lower-edge closeout change. It is **mostly working**, not qualified
for an unattended stage demo.

## What worked physically

- Pear run `19d47cc5-04a4-4856-bf5d-c496242f2ff5` completed camera search,
  approach, stopped Arrival, lie-down/bark, stand, and began Home return.
- The Pear lower-edge disappearance no longer timed out in `APPROACH_FRUIT`.
- All tested terminal paths released motion and reported
  `DISARMED_CONFIRMED`.
- App-only Stagefile deployments remained fast: the latest all-fruit change
  deployed in 3.600 seconds without rebuilding media/CUDA or voice.

## Known blockers

### Tomorrow: Apple candidate-focus can hold forever

Run `7085ee1d-fe63-47ef-a887-b0faf11de3af` detected Apple strongly but failed
in `TURN_TO_FRUIT` after the 30-second guidance timeout.

- Maximum confidence: `0.7676874`.
- Twelve observations met the `0.50` focus threshold and eighteen met the
  `0.40` lock-confidence threshold.
- Those strong detections were far left; the best frames were around
  `center_x=0.106`, outside the centered `0.42..0.58` corridor.
- While Woof aligned, confidence fell below `0.40` before Apple reached the
  centered corridor. Centered observations later peaked at only `0.08449`.
- `apple_candidate_focus_below_acquisition` and
  `apple_candidate_focus_missing` hold yaw at zero indefinitely. The
  controller never resumed alignment or broad search, so it timed out.

Tomorrow's smallest experiment should make candidate focus bounded: preserve
the Apple identity through a short alignment grace, then resume broad search
if qualified evidence does not recover. It must never remain in a permanent
zero-yaw hold. Replay this exact trace before another physical Apple run.

### Home return remains separate

The Pear fruit sequence worked, but the same run finished
`RETURN_HOME_FAILURE` because the heading escaped the forward steering gate.
Live post-run Home distance was about `0.109 m`, inside the operator's
`0.50 m` inter-run margin but outside the strict `0.10 m` application target.

### Application outcomes still require operator review

The all-fruit closeout now assumes stopped Arrival after a qualified
lower-edge target disappears or loses identity. Stale/frozen frames, camera
failure, and generation changes still fail closed. Physical fruit clearance
and posture must still be checked by the operator until repeatability is
qualified.

## Next test order

1. Fix and replay the bounded Apple candidate-focus state.
2. Run one supervised Apple attempt.
3. Only after a full Apple success and Home/disarm gate, run Banana.
4. Only after a full Banana success and Home/disarm gate, run Pear.

