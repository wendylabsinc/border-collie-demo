# RobotKit

RobotKit is a runnable, multi-container perception–planning–action app for a Unitree Go2 on Wendy. It consumes ROS2 camera, LIDAR, audio, temperature, and battery topics, persists their interpretations, runs a durable mission state machine, and emits bounded robot actions.

```text
ROS2 / simulator
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
- **B — `yolo-fruits`:** Ultralytics YOLO on ROS2 `sensor_msgs/Image`, filtered to the COCO apple, banana, and orange classes. It publishes normalized boxes on `vision.fruits`, including empty frames.
- **B — `lidar-voxel`:** ROS2 `PointCloud2` plus odometry into a bounded sparse room map, planar voxel scan-matched `localization.pose`, and fixed angular `lidar.proximity` sectors for safe target ranging.
- **B — `transcription`:** Local `faster-whisper` inference over ROS2 PCM audio, WAV/raw PCM, or microphone input. It publishes the transcript and an auditable command intent separately.
- **B — `health-high-low`:** Standard ROS2 `Temperature` and `BatteryState` into validated critical-low/low/normal/high/critical-high observations.
- **C — `planner`:** A pure state-machine function over snapshot, active durable goal, and latest durable effect. It has no process-local mission state.
- **D — `controller`:** A pure function that turns the current mission stage and fresh perception into one bounded, short-lived effect.
- **Executor:** claims exactly one durable effect, revalidates it against A, emits ROS2 velocity/posture or plays the configured bark WAV, then durably acknowledges the outcome.

The shared wire models are in [`src/robotkit/contracts.py`](src/robotkit/contracts.py). Unknown fields are rejected and each contract carries `schema_version: "1"`.

## Run it

Local Docker:

```sh
docker compose up --build
curl http://localhost:8080/v1/state
curl 'http://localhost:8080/v1/events?after=0&limit=100'
```

On a configured Wendy device:

```sh
wendy run
```

The companion [`wendy.json`](wendy.json) grants ROS2-compatible host networking, GPU access to YOLO, audio access to transcription/execution, and a persistent `/data` volume only to A. Adjust the ROS topic environment variables in [`docker-compose.yml`](docker-compose.yml) to match the installed Go2 driver.

The deployed mission is:

```text
unsafe temperature or critical-low battery
  └─ go_home → lie_down

"find apple" or "go to apple"
  └─ search_apple → approach_apple (≤ 0.30 m) → bark → go_home → lie_down

otherwise
  └─ idle_at_home
```

Search rotates until a fresh YOLO apple exists. Approach steers from the apple's normalized image bearing and uses the corresponding fresh LIDAR angular sector for range; it sends zero velocity rather than moving without a valid range. Home return recomputes a command from fresh `localization.pose` and the configured `HOME_X_M`, `HOME_Y_M`, and `HOME_YAW_RAD` on every cycle.

Run without containers:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest
```

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

Each hardware-facing B component has a dedicated ROS2 image. [`Dockerfile.ros2`](Dockerfile.ros2) provides the executor selected by Compose:

```yaml
executor:
  build:
    context: .
    dockerfile: Dockerfile.ros2
  environment:
    ROBOTKIT_EXECUTOR_MODE: ros2
    ROS2_CMD_VEL_TOPIC: /cmd_vel
    ROS2_POSTURE_COMMAND_TOPIC: /robotkit/posture_command
    BARK_WAV_PATH: /opt/robotkit/assets/bark.wav
```

Velocity effects emit `geometry_msgs/Twist`. Bark uses a deployment-configured WAV; the image includes a short synthesized default. Lie-down publishes the narrow `std_msgs/String` command `lie_down` on the posture topic. Bridge that topic to the audited Unitree Sport API version installed on the robot. Confirm all topics and the bridge in simulation before enabling motors. This layer is not a substitute for the Go2 hardware emergency stop, command watchdog, collision layer, or Unitree low-level safety controls.

## Test boundaries

The suite independently covers:

- YOLO filtering, ROS image decoding, and producer metadata without model downloads;
- sparse voxel mapping, scan matching, and angular proximity using synthetic rooms;
- Whisper adaptation, PCM framing, transcript normalization, and intent extraction without model downloads;
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
