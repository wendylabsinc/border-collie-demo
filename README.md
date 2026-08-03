# Border Collie Demo

A clean-room implementation of the Wendy Labs Border Collie routine for the
Unitree Go2.

This repository intentionally does not import from the original `collie-demo`
application. The old repository remains useful as test evidence and a hardware
reference, but it is not a runtime dependency.

## Intended routine

1. Verify camera, detector, pose, motion, media, and return-home readiness.
2. Confirm that Woof is facing the person. This is initially an operator setup
   requirement rather than autonomous person detection.
3. Capture a stable Home position and heading.
4. Accept a typed `go to pear` command. Speech support comes later.
5. Turn toward the fruit-search area.
6. Find and approach the requested fruit using fresh detections.
7. After confirmed near-fruit evidence, allow one bounded final approach when
   the fruit leaves the lower camera edge.
8. Stop, sit, and bark.
9. Stand, turn toward Home, return to the captured position, and restore the
   original heading.
10. Stop all motion, record the result, and report completion.

Any physical remote-control input will eventually cause a latched
`REMOTE_TAKEOVER`. Autonomous control must stop and cannot resume until the
application process is restarted. Remote-input detection is a planned adapter,
not part of the initial implementation.

## Repository boundary

- `src/border_collie_demo/`: mission domain and narrow hardware contracts.
- `media/`: future sole owner of Go2 WebRTC camera, microphone, and bark audio.
- `web/`: minimal audience and debug surfaces.
- `lab/`: isolated human-operated hardware tests, never imported by production.
- `tests/`: unit, contract, replay, and later hardware acceptance tests.
- `docs/`: behavior, known hardware facts, and validation results.

The initial scaffold contains no live Go2 adapter and cannot move the robot.

## Local validation

```bash
python -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```
