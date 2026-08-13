# RobotKit

RobotKit is a runnable, multi-container perception–planning–action app for a Unitree Go2 on Wendy. It consumes ROS2 camera, LIDAR, audio, temperature, and battery topics, persists their interpretations, runs a durable mission state machine, and emits bounded robot actions.

```text
ROS2 / simulator / command website
       │
       ▼
B: replaceable interpreters ── immutable observations ──┐
                                                       ▼
                                      A: world-state service
                                      ├─ current projection
                                      └─ append-only black box
                                                │
                         ┌──────────────────────┴──────────────────────┐
                         ▼                                             ▼
              C: A → high-level goal G                  D: (G + A) → effect E
                         │                                             │
                         └──────── durable G/E records ────────────────┘
                                                                       │
                                                                       ▼
                                                safety gate → ROS2 executor
```

## What is implemented

- **A — `world-state`:** FastAPI plus SQLite in WAL mode with full synchronous writes. A Wendy named volume persists the database. Every observation, goal, effect, claim, rejection, and acknowledgement is revisioned in the event log. A transactional projection exposes the current interpretation for each `(producer_id, stream)`.
- **B — `yolo-fruits`:** Ultralytics YOLOE-11m on the Go2 DDS video service or a standard ROS2 `sensor_msgs/Image`, open-vocabulary prompted for apple, banana, grapes, orange, and pear. It publishes normalized boxes on `vision.fruits`, including empty frames.
- **B — `lidar-voxel`:** ROS2 `PointCloud2` plus odometry into a bounded sparse room map, planar voxel scan-matched `localization.pose`, and fixed angular `lidar.proximity` sectors for safe target ranging.
- **B — `transcription`:** Local `faster-whisper` inference over ROS2 PCM audio, WAV/raw PCM, or microphone input. It publishes the transcript and an auditable command intent separately.
- **B — `website-command`:** A stateless command page and JSON API. Typed instructions publish immutable `website.intent` observations, using the same deterministic intent parser as voice.
- **B — `health-high-low`:** Standard ROS2 `Temperature` and `BatteryState` into validated critical-low/low/normal/high/critical-high observations.
- **C — `planner`:** A pure state-machine function over snapshot, active durable goal, and latest durable effect. It has no process-local mission state.
- **D — `controller`:** A pure function that turns the current mission stage and fresh perception into one bounded, short-lived effect.
- **Executor:** claims exactly one durable effect, revalidates it against A, calls the Go2 high-level Unitree Sport API or plays the configured bark WAV, then durably acknowledges the outcome.

The shared wire models are in [`src/robotkit/contracts.py`](src/robotkit/contracts.py). Unknown fields are rejected and each contract carries `schema_version: "1"`.

## Run it

Run tests locally:

```sh
PYTHONPATH=src pytest
```

On a configured Wendy device:

```sh
wendy run
```

Open `http://<wendy-device>:8090` and submit commands such as `find pear` or
`go to the orange`.
The page sends `POST /v1/command`; the service parses the text and publishes it
to A. This endpoint is intentionally unauthenticated, so restrict port `8090`
to a trusted robot network before physical operation.

Open `http://<wendy-device>:8090/debug` to diagnose the robot without attaching
a shell. It refreshes from A once per second and shows command → perception →
planner → controller → executor as one pipeline, with observation freshness,
the current goal/effect, and a bounded black-box event timeline. YOLO also
publishes `diagnostics.yolo`, including model name, inference latency, frame age,
detection count, supported classes, and runtime failures. Fruit missions support
apple, banana, grapes, orange, and pear. Commands for other targets are rejected
explicitly and stop an active fruit search.

The equivalent API call is:

```sh
curl -X POST http://<wendy-device>:8090/v1/command \
  -H 'Content-Type: application/json' \
  -d '{"command":"find apple","request_id":"93070866-7198-4c5e-8a6f-38cdfbf3295d"}'
```

Wendy builds each service from its own Stagefile. `build.stagefile.yaml` is A's
world-state image; planner, controller, YOLO, LIDAR, transcription, website,
high/low, and executor each use a named variant. Their `copy` declarations list
only the service package and the shared modules it imports. That separation is
intentional: changing website code invalidates only the website image instead
of rebuilding and pushing every service that happens to live in the same source
tree. Shared modules such as `contracts.py` still invalidate every consumer, as
they should. This uses Wendy's Stagefile-family support because the services
share one source context. Stagefiles are the only container build definitions
in this repository.

YOLO's CUDA stage resolves the target-specific PyTorch wheel index and runtime
from the selected Wendy device; no Jetson URL is baked into this project. The
ABI-critical CUDA runtime, cuBLAS, and cuDNN packages are pinned to the CUDA
12.6 / cuDNN 9.3 versions used to build the JetPack 6 PyTorch wheel in the
checked-in Stagefile lock, preventing an unpinned PyPI runtime refresh from
mixing CUDA 12.9 libraries into the image.
The Go2 does not expose its front camera as a standard ROS Image. The deployed YOLO
producer obtains JPEG frames from Unitree's ROS 2-compatible DDS video service,
so it does not compete for the robot's single WebRTC camera slot. It retains
standard ROS Image, HTTP, direct WebRTC, and file modes for other hardware and
tests.
Neither YOLO nor Whisper weights are downloaded by tests or image builds. They
are fetched by their model libraries on first use into the shared persistent
`/models` cache, then survive blue/green image deployments.

The companion [`wendy.json`](wendy.json) grants ROS2-compatible host networking, GPU access to YOLO, audio access to transcription/execution, and a persistent `/data` volume only to A. Adjust the ROS topic environment variables in [`docker-compose.yml`](docker-compose.yml) to match the installed Go2 driver.

The deployed mission is:

```text
unsafe temperature or critical-low battery
  └─ go_home → lie_down

"find <fruit>" or "go to <fruit>" from voice or website
  └─ search_<fruit> → approach_<fruit> (≤ 0.30 m) → bark → go_home → lie_down

otherwise
  └─ idle_at_home
```

Search rotates until a fresh detection of the requested fruit exists. The deployed detector runs on
CUDA device 0 at 1280-pixel inference size with a `0.15` confidence threshold
for the Go2's wide-angle camera. It prompts both canonical fruit names and
their toy-fruit variants, while publishing only canonical labels. Approach
steers from that fruit's normalized image bearing and uses the
corresponding fresh LIDAR angular sector for range; it sends zero velocity
rather than moving without a valid range. Home return recomputes a command from
fresh `localization.pose` and the configured `HOME_X_M`, `HOME_Y_M`, and
`HOME_YAW_RAD` on every cycle. The stateless controller renews effects at 10 Hz
on A-derived time buckets, searches at `0.8 rad/s` (about eight seconds per full
sweep), and the executor polls at 20 Hz. Pending velocity
renewals use latest-wins semantics, so device load cannot turn them into a FIFO
backlog of expired commands; the executor keeps Unitree's 0.8-second velocity
dead-man fed without process-local mission state. `MAX_STATE_DRIFT` defaults to
32 revisions because the camera, LIDAR, pose, and diagnostic streams can
legitimately advance A many times while an effect crosses the HTTP boundary;
the one-second effect TTL and required-stream freshness checks provide the
time-based safety bounds.
An explicit `stop` command immediately emits zero velocity and cancels any
active motion mission, including a health return. A fresh failed YOLO
diagnostic also cancels the mission, while the controller independently emits
zero velocity so perception failure cannot leave search rotation running.
`FRUIT_STOP_DISTANCE_M` configures the approach threshold for every fruit;
`APPLE_STOP_DISTANCE_M` remains the backward-compatible fallback.

Run without containers:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest
```

### Replay a ROS2 bag through perception containers

The bag test runner starts an isolated world-state service plus only the
perception containers named on the command line, replays a bag on an isolated
ROS domain, and checks the observations that reached world-state. It never
starts the planner, controller, or physical executor.

```sh
make test-bag BAG=/absolute/path/to/recording \
  ARGS='--service lidar-voxel --expect lidar.proximity --expect lidar.room_map'
```

Repeat `--service` to exercise several producers from one recording. Available
services are `yolo-fruits`, `lidar-voxel`, `transcription`, and
`health-high-low`. The first run builds the selected service images and the
small ROS2 bag-player image. YOLO and transcription may also fetch their model
weights on their first inference.

For payload assertions, pass `--expect-file`. Each item is a recursive JSON
subset of an observation; arrays are treated as unordered subsets, so a test
can name one expected detection without copying the entire frame:

```json
{
  "observations": [
    {
      "stream": "vision.fruits",
      "frame_id": "front_camera",
      "payload": {"detections": [{"class_name": "apple"}]}
    }
  ]
}
```

```sh
make test-bag BAG=/absolute/path/to/camera-bag \
  ARGS='--service yolo-fruits --expect-file tests/bags/apple.expected.json'
```

Use `--rate`, `--startup-seconds`, `--discovery-seconds`, and `--timeout` for
slow inference, DDS discovery, or large recordings. `--keep` preserves the
generated Compose project for inspecting logs; the runner otherwise removes
its containers and ephemeral database.

## Swapping and scaling B/C/D

Workers retain no correctness-critical local state. A restart may repeat a request, but `idempotency_key` makes retries harmless. To scale a component, run replicas with:

- the same logical ID (`ROBOTKIT_PRODUCER_ID`, `ROBOTKIT_PLANNER_ID`, or `ROBOTKIT_CONTROLLER_ID`);
- the same `ROBOTKIT_DEPLOYMENT_GENERATION`; and
- distinct `ROBOTKIT_INSTANCE_ID` values (the hostname is the default).

For a blue/green replacement, start green with the same logical ID and a higher integer `ROBOTKIT_DEPLOYMENT_GENERATION`. A immediately fences outputs from lower generations while still retaining them in the black-box log. After health/audit checks, remove blue. Generations are monotonic; a rollback is another deployment with a still-higher generation.

A B component needs only this boundary:

```python
from robotkit.contracts import Observation

observation = Observation(
    producer_id="another-fruit-model",
    instance_id="yolo-green-7f8d",
    deployment_generation=12,
    idempotency_key="camera-front:frame:92834",
    stream="vision.fruits",
    observation_type="vision.coco_fruits.v1",
    observed_at=source_timestamp,
    frame_id="camera_front",
    confidence=0.94,
    ttl_seconds=0.5,
    payload={"detections": detections},
)
client.publish_observation(observation)
```

Keep model loading and ROS2 subscription inside the worker adapter; keep interpretation in a pure function that accepts recorded raw input. That makes the same fixture usable for unit tests, audit replay, and simulation.

## State and audit semantics

- `GET /v1/state` returns a consistent current projection, a global `revision`, and `state_revision` for the newest projected observation.
- An older source timestamp cannot replace a newer current observation.
- A lower deployment generation cannot replace a higher one, even if its event arrives later.
- Stale data remains visible with its source timestamp and TTL; planners decide whether it is usable.
- `GET /v1/events` is the ordered black box. Pagination uses `after=<revision>`.
- Goals and effects have validity deadlines. Effects also name required fresh streams and maximum allowed world-state drift.
- Effect claims and completion are transactional. Competing executors cannot apply the same claimed effect.

For high-bandwidth artifacts such as point clouds, voxel grids, video, or audio, publish a content hash, metadata, and durable object/volume reference in `payload`; do not copy large raw blobs into SQLite.

SQLite is appropriate for one Go2 edge node with a single A process and many HTTP writers. `synchronous=FULL`, WAL, the Wendy persistent volume, and process-reopen tests provide crash recovery. For fleet-wide replication or storage beyond one node, keep the HTTP contracts and replace `WorldStateStore` with a replicated log/database implementation.

## Physical ROS2 execution

Each hardware-facing B component has a dedicated ROS2 stage. The executor is
built from [`executor.stagefile.yaml`](executor.stagefile.yaml):

```yaml
executor:
  build:
    context: .
    dockerfile: executor.stagefile.yaml
  environment:
    ROBOTKIT_EXECUTOR_MODE: unitree_sport
    GO2_NETWORK_INTERFACE: enP8p1s0
    BARK_WAV_PATH: /opt/robotkit/assets/bark.wav
```

Velocity effects call the high-level `SportClient.Move(vx, 0, yaw)` API; lie-down calls `SportClient.StandDown()`. The executor pins the same Unitree SDK and CycloneDDS versions as Wendy's proven Go2 motion template. A local 0.8-second dead-man issues `StopMove` if A-derived controller renewals cease; this ephemeral safety timer is not mission/progress state. Bark uses a deployment-configured WAV; the image includes a short synthesized default. Confirm the robot-facing interface and motion in simulation before enabling motors. This layer is not a substitute for the Go2 hardware emergency stop, collision/drop-off protection, or Unitree low-level safety controls.

## Test boundaries

The suite independently covers:

- YOLO filtering, ROS/DDS image decoding, DDS/WebRTC input plumbing, and producer metadata without model downloads;
- sparse voxel mapping, Go2 mount transformation, scan matching, and angular proximity using synthetic rooms;
- Whisper adaptation, PCM framing, transcript normalization, and intent extraction without model downloads;
- website authentication, deterministic intent publication, and cross-channel command ordering;
- temperature and battery band boundaries;
- deterministic simulator output for offline development;
- stateless mission transitions and durable trigger correlation;
- search, fused 30 cm approach, bark, and pose-based home policies;
- pure planner decisions;
- pure controller decisions;
- actuator safety rejection paths;
- API serialization and idempotency;
- out-of-order observation handling;
- persistent reopen/recovery;
- transactional effect claim/ack; and
- blue/green generation fencing.
