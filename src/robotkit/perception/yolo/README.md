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

Run against a changing image file for simulation or smoke tests:

```sh
WORLD_STATE_URL=http://localhost:8080 \
YOLO_INPUT_MODE=file \
YOLO_IMAGE_PATH=/data/frame.jpg \
python -m robotkit.perception.yolo
```

Configuration variables include `YOLO_MODEL` (default `yolo11n.pt`),
`YOLO_CONFIDENCE` (default `0.25`), `YOLO_DEVICE`, `YOLO_CLASSES`,
`YOLO_TTL_SECONDS`, and normal RobotKit instance/deployment variables. Model
weights are resolved by Ultralytics and should be pre-cached in production.

Build the dedicated ROS2 image from the repository root:

```sh
docker build -f src/robotkit/perception/yolo/Dockerfile -t robotkit-yolo .
```
