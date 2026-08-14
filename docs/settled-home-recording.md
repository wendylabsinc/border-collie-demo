# Settled Home verification and recorder

Home completion is now a stopped-position claim, not a single-pose event. An
inside-gate pose commands exact zero and disarms, waits briefly, then requires
an advancing same-epoch window to remain inside the unchanged 0.10 m gate with
bounded spread. One bounded position-only retry is allowed; no Home-heading
restoration movement is issued.

The deployable identity for this contract is application version
`1.1.5-stage-default`, build label
`stage-default-v6-settled-home-recorder`.

## Runtime settings

All settings are read when the app service starts and are also reported in the
`run_tuning` contract returned by `GET /api/status`.

| Environment variable | Units | Default | Valid range | Safety meaning |
| --- | --- | ---: | ---: | --- |
| `BORDER_COLLIE_HOME_SETTLE_INTERVAL_S` | seconds | `0.30` | `0..2` | Exact-zero quiet time before verification. |
| `BORDER_COLLIE_HOME_SETTLED_SAMPLE_COUNT` | samples | `4` | `3..5` | Consecutive advancing fresh poses required. |
| `BORDER_COLLIE_HOME_SETTLED_MAX_SPREAD_M` | meters | `0.03` | `0.005..0.05` | Maximum pairwise planar spread in the window. |
| `BORDER_COLLIE_HOME_SETTLED_SAMPLE_TIMEOUT_S` | seconds | `1.0` | `0.25..3` | Bounded time to collect the advancing window. |
| `BORDER_COLLIE_HOME_SETTLED_RETRY_COUNT` | retries | `1` | `0..1` | Bounded position-only correction attempts. |

These settings do not widen the 0.10 m Home gate, permit stale samples, accept
repeated or regressed timestamps, cross odometry epochs, authorize heading
restoration, or bypass the existing course/progress/timeout gates.

## One recording interface, two build contexts

The app retains its ordinary per-run black box and mirrors Home events through
one `HomeRecordingHub` into a bounded, redacted, version-1 journal. Recorder
errors are best effort and can never change a motion decision or mask a safety
failure.

The separate `home-recorder` service is passive and GET-only. It reads the
shared app journal and `rt/sportmodestate`; it imports no Unitree command
client, writer, or RPC client. It correlates every active-run pose with Home,
stage, odometry epoch, armed/mode state, and the latest app motion command, then
writes `/state/home-recorder/runs/<run-id>/home-deep.ndjson`. Pose retention
starts at `turn_toward_home` (or failure recovery), so high-rate DDS traffic
during fruit search cannot exhaust the bounded Home window. The app exposes
that file read-only at `GET /api/results/<run-id>/home-deep.ndjson`, and the
recorder exposes health only at port 8112 `GET /status`.

The first deployment changes the shared descriptor and must therefore use a
whole-project `wendy run --detach`. After that, the recorder owns a separate
Stagefile, lock, and build context, so an app source iteration with
`wendy run --service app --detach` does not rebuild the passive recorder.

Each deep row is bounded by `HOME_RECORDER_MAX_EVENTS_PER_RUN` (default 10,000)
and contains the applicable raw x/y/yaw, Home delta/distance, source timestamp
and local sequence, odometry epoch, stage, forward/yaw command and path,
armed/mode state, gate/window evidence from app events, and terminal reason.
Frames and credentials are not recorded.

## Deterministic replay

Run the two saved inside-then-outside physical sequences plus a stable control:

```bash
.venv/bin/python scripts/replay_home_settling.py
```

The command exits nonzero if either physical failure is prematurely accepted
or the stable four-sample control is rejected. It performs no robot access.
