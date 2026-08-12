"""Runnable ROS2 or file-input entrypoint for the YOLO B container."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robotkit.client import WorldStateClient
from robotkit.perception.yolo.adapters import UltralyticsDetector, decode_ros_image
from robotkit.perception.yolo.core import COCO_FRUIT_CLASSES
from robotkit.perception.yolo.producer import YoloProducer
from robotkit.runtime import (
    configure_logging,
    deployment_generation,
    instance_id,
    run_loop,
    world_state_url,
)


LOGGER = logging.getLogger(__name__)


def _configured_classes() -> frozenset[str]:
    configured = os.getenv("YOLO_CLASSES")
    if not configured:
        return COCO_FRUIT_CLASSES
    classes = frozenset(
        value.strip().casefold()
        for value in configured.split(",")
        if value.strip()
    )
    if not classes:
        raise ValueError("YOLO_CLASSES must contain at least one class")
    return classes


def _make_producer(client: WorldStateClient) -> YoloProducer:
    confidence = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
    detector = UltralyticsDetector(
        os.getenv("YOLO_MODEL", "yolo11n.pt"),
        confidence=confidence,
        device=os.getenv("YOLO_DEVICE") or None,
    )
    return YoloProducer(
        detector,
        client,
        producer_id=os.getenv("ROBOTKIT_PRODUCER_ID", "yolo-coco-fruits"),
        instance_id=instance_id(),
        deployment_generation=deployment_generation(),
        allowed_classes=_configured_classes(),
        min_confidence=confidence,
        ttl_seconds=float(os.getenv("YOLO_TTL_SECONDS", "2.0")),
    )


def _stamp(message: Any) -> tuple[datetime, str]:
    stamp = message.header.stamp
    seconds = int(stamp.sec)
    nanoseconds = int(stamp.nanosec)
    if seconds == 0 and nanoseconds == 0:
        observed_at = datetime.now(timezone.utc)
        key = f"received:{observed_at.timestamp():.9f}"
    else:
        observed_at = datetime.fromtimestamp(
            seconds + nanoseconds / 1_000_000_000, tz=timezone.utc
        )
        key = f"stamp:{seconds}:{nanoseconds}"
    return observed_at, key


def run_ros2(producer: YoloProducer) -> None:
    """Subscribe to a ROS2 Image topic until the process is stopped."""

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
    except ImportError as exc:  # pragma: no cover - runtime integration
        raise RuntimeError("ROS2 rclpy and sensor_msgs are required in ros2 mode") from exc

    topic = os.getenv("YOLO_IMAGE_TOPIC", "/camera/image_raw")

    class YoloNode(Node):
        def __init__(self) -> None:
            super().__init__("robotkit_yolo_coco_fruits")
            self.create_subscription(
                Image, topic, self._on_image, qos_profile_sensor_data
            )

        def _on_image(self, message: Any) -> None:
            try:
                observed_at, source_key = _stamp(message)
                producer.process_image(
                    decode_ros_image(message),
                    source_key=f"{topic}:{source_key}",
                    observed_at=observed_at,
                    frame_id=message.header.frame_id or "camera",
                )
            except Exception as exc:
                self.get_logger().error(f"YOLO frame processing failed: {exc!r}")

    rclpy.init()
    node = YoloNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run_file(producer: YoloProducer) -> None:
    """Poll a local image; useful for simulation and deployment smoke tests."""

    image_path = Path(os.environ["YOLO_IMAGE_PATH"])

    def step() -> None:
        stat = image_path.stat()
        producer.process_image(
            str(image_path),
            source_key=f"file:{image_path.resolve()}:{stat.st_mtime_ns}",
            observed_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            frame_id=os.getenv("YOLO_FRAME_ID", "camera"),
        )

    run_loop(step, float(os.getenv("YOLO_INTERVAL_SECONDS", "1.0")))


def main() -> None:
    configure_logging()
    client = WorldStateClient(world_state_url())
    try:
        producer = _make_producer(client)
        mode = os.getenv("YOLO_INPUT_MODE", "ros2").casefold()
        LOGGER.info("starting YOLO producer in %s mode", mode)
        if mode == "ros2":
            run_ros2(producer)
        elif mode == "file":
            run_file(producer)
        else:
            raise ValueError("YOLO_INPUT_MODE must be 'ros2' or 'file'")
    finally:
        client.close()
