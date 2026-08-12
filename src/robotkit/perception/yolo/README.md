# YOLO COCO fruit producer

This replaceable B component runs an Ultralytics COCO checkpoint and publishes
the latest apple, banana, and orange detections to `vision.fruits`. Every frame
is an immutable world-state event; A retains the history and projects the latest
event. Empty frames are published so stale fruit does not remain present.

Run against ROS2:

```sh
WORLD_STATE_URL=http://world-state:8080 \
YOLO_IMAGE_TOPIC=/camera/image_raw \
python -m robotkit.perception.yolo
```

Run against the real Go2 front camera through its ROS 2-compatible DDS video
service (the deployed default):

```sh
WORLD_STATE_URL=http://127.0.0.1:8080 \
YOLO_INPUT_MODE=go2_dds \
GO2_NETWORK_INTERFACE=enP8p1s0 \
python -m robotkit.perception.yolo
```

This mode calls the Go2 video service on DDS domain `ROS_DOMAIN_ID` and decodes
its JPEG response. It does not consume the robot's single WebRTC camera slot,
so it can coexist with the Unitree app or another WebRTC camera owner. HTTP and
direct WebRTC modes remain available for other deployments.

Run against a changing image file for simulation or smoke tests:

```sh
WORLD_STATE_URL=http://localhost:8080 \
YOLO_INPUT_MODE=file \
YOLO_IMAGE_PATH=/data/frame.jpg \
python -m robotkit.perception.yolo
```

Configuration variables include `YOLO_MODEL` (default `yolo11n.pt`),
`YOLO_CONFIDENCE` (default `0.25`), `YOLO_DEVICE`, `YOLO_CLASSES`,
`YOLO_TTL_SECONDS`, `YOLO_CAMERA_URL`, `YOLO_CAMERA_TIMEOUT_SECONDS`,
`YOLO_INTERVAL_SECONDS`, `GO2_NETWORK_INTERFACE`, `GO2_VIDEO_TIMEOUT_SECONDS`,
`GO2_IP`, `GO2_AES_KEY`, `YOLO_FRAME_ID`, and normal
RobotKit instance/deployment variables. `GO2_RETRY_SECONDS` controls the delay
between failed WebRTC connection attempts (default `5`). Model weights are resolved by
Ultralytics and should be pre-cached in production.

Build the dedicated ROS2 image from the repository root:

```sh
docker build -f src/robotkit/perception/yolo/Dockerfile -t robotkit-yolo .
```
