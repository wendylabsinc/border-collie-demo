# Go2 Start button and app lifecycle

This release adds one read-only physical-controller activation path. After the
Border Collie app is ready, release the Go2 controller's **Start** button once,
then press it. One rising edge requests one Pear Demo Run through the same
`StageDemo.activate(FruitMission(...))` interface used by HTTP, voice, MCP, and
soak callers.

## Controller evidence contract

The app subscribes read-only to Unitree `rt/lf/lowstate`. The pinned Unitree SDK
defines `LowState_.wireless_remote` as 40 bytes and its joystick decoder maps
Start to bit `2` of the little-endian button word in bytes `2:4`. `LowState.tick`
is retained as the source sequence. Every accepted or rejected Start edge that
can be associated with a Demo Run is written as `kind=controller_start` with:

- source topic and source sequence;
- local monotonic receive time;
- exact button word, Start mask, and pressed state;
- deterministic activation ID and fixed Target Fruit (`pear`);
- acceptance, idempotent replay, not-ready, or active-run disposition.

The adapter requires a released sample before arming. Holding Start across app
or Go2 boot is inert. Repeated DDS delivery and button hold cannot submit more
than one activation. A release followed by a new press is a new request, but an
active Demo Run is still rejected by the existing exclusive-run contract.

The controller path does not own a Unitree writer or motion client. It cannot
bypass camera, fresh-pose, Remote Takeover, exclusive-motion, watchdog, Home,
or exact-disarm checks. If preflight is not ready, the ordinary failed Run
Result is produced and search/motion does not begin.

`BORDER_COLLIE_CONTROLLER_START_ENABLED=1` enables the production subscription.
The status endpoint reports both DDS-source health and whether a neutral sample
has armed the Start edge.

## Boot and process restart

Wendy already owns this lifecycle. The released CLI and current WendyOS agent
support `unless-stopped`: the agent persists the policy, monitors exits, and
reconciles eligible multi-service containers after agent/device boot. Deploy
this app group with:

```sh
scripts/deploy-stage-default --device woof.local
```

The wrapper uses the Stagefiles through normal `wendy run` resolution and makes
`--restart-unless-stopped` explicit. It never activates a Demo Run. After boot,
the controller adapter still waits for a released sample and a new Start edge.

A deliberate `wendy stop` remains authoritative and persists across reboot.
Redeploy or explicitly start the app to clear that deliberate stopped state;
do not add a competing system daemon that overrides operator intent.

## Qualification boundary

The mapping and lifecycle behavior are locally regression-tested against the
pinned SDK and Wendy source. The physical Go2 Start button has not been pressed
and this build has not been deployed to Woof. Before stage use, perform one
supervised qualification from a clear operating envelope and confirm exactly
one `controller_start` event, one Pear Run ID, normal terminal recovery, and
exact-zero disarm.
