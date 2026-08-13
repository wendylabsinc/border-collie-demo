"""Runnable ROS2 or file-input entrypoint for the YOLO B container."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robotkit.client import WorldStateClient
import httpx

from robotkit.perception.yolo.adapters import (
    UltralyticsDetector,
    decode_image_bytes,
    decode_ros_image,
)
from robotkit.perception.yolo.core import FRUIT_CLASSES
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
        return FRUIT_CLASSES
    classes = frozenset(
        value.strip().casefold()
        for value in configured.split(",")
        if value.strip()
    )
    if not classes:
        raise ValueError("YOLO_CLASSES must contain at least one class")
    return classes


def _configured_prompt_templates() -> tuple[str, ...]:
    configured = os.getenv("YOLO_PROMPT_TEMPLATES", "{name}")
    templates = tuple(
        value.strip() for value in configured.split(",") if value.strip()
    )
    if not templates or any("{name}" not in template for template in templates):
        raise ValueError("YOLO_PROMPT_TEMPLATES must contain {name} templates")
    return templates


def _make_producer(client: WorldStateClient) -> YoloProducer:
    confidence = float(os.getenv("YOLO_CONFIDENCE", "0.25"))
    classes = _configured_classes()
    model_name = os.getenv("YOLO_MODEL", "yoloe-11m-seg.pt")
    detector = UltralyticsDetector(
        model_name,
        confidence=confidence,
        device=os.getenv("YOLO_DEVICE") or None,
        classes=sorted(classes),
        prompt_templates=_configured_prompt_templates(),
        image_size=int(os.getenv("YOLO_IMAGE_SIZE", "640")),
    )
    return YoloProducer(
        detector,
        client,
        producer_id=os.getenv("ROBOTKIT_PRODUCER_ID", "yoloe-fruits"),
        instance_id=instance_id(),
        deployment_generation=deployment_generation(),
        allowed_classes=classes,
        min_confidence=confidence,
        ttl_seconds=float(os.getenv("YOLO_TTL_SECONDS", "2.0")),
        model_name=model_name,
        diagnostics_ttl_seconds=float(os.getenv("YOLO_DIAGNOSTICS_TTL_SECONDS", "10")),
    )


def _stamp(message: Any) -> tuple[datetime, str]:
    """Use this computer's receipt clock for TTL and the ROS stamp for identity.

    The Go2 and its companion computer can have different wall clocks. A ROS
    header remains the correct idempotency identity, but using it as A's
    freshness clock can make every live frame stale on arrival.
    """
    stamp = message.header.stamp
    seconds = int(stamp.sec)
    nanoseconds = int(stamp.nanosec)
    observed_at = datetime.now(timezone.utc)
    if seconds == 0 and nanoseconds == 0:
        key = f"received:{observed_at.timestamp():.9f}"
    else:
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


def run_http(producer: YoloProducer) -> None:
    """Poll the existing camera owner's read-only JPEG endpoint.

    A Go2 only permits one WebRTC camera client. Reusing the local demo camera
    endpoint lets YOLO coexist with that owner instead of competing for its
    slot. Identical payloads retain one source key so a frozen feed goes stale.
    """

    url = os.getenv(
        "YOLO_CAMERA_URL", "http://127.0.0.1:8111/api/camera/frame.jpg"
    )
    client = httpx.Client(
        timeout=float(os.getenv("YOLO_CAMERA_TIMEOUT_SECONDS", "2.0")),
        follow_redirects=False,
    )

    def step() -> None:
        try:
            response = client.get(
                url,
                headers={"Accept": "image/jpeg,image/*", "Cache-Control": "no-cache"},
            )
            response.raise_for_status()
            payload = response.content
            producer.process_image(
                decode_image_bytes(payload),
                source_key=f"http:{url}:{hashlib.sha256(payload).hexdigest()}",
                observed_at=datetime.now(timezone.utc),
                frame_id=os.getenv("YOLO_FRAME_ID", "camera_link"),
            )
        except Exception as exc:
            producer.publish_failure(exc, source_key=f"http:{url}:{time.time_ns()}")
            raise

    try:
        run_loop(step, float(os.getenv("YOLO_INTERVAL_SECONDS", "0.2")))
    finally:
        client.close()


def _make_go2_video_client() -> Any:
    """Connect to Unitree's video service on the ROS 2-compatible DDS graph."""

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.video.video_client import VideoClient
    except ImportError as exc:  # pragma: no cover - runtime integration
        raise RuntimeError(
            "go2_dds mode requires the pinned Unitree SDK2 Python package"
        ) from exc

    domain_id = int(os.getenv("ROS_DOMAIN_ID", "0"))
    network_interface = os.getenv("GO2_NETWORK_INTERFACE") or None
    if network_interface:
        ChannelFactoryInitialize(domain_id, network_interface)
    else:
        ChannelFactoryInitialize(domain_id)

    client = VideoClient()
    client.SetTimeout(float(os.getenv("GO2_VIDEO_TIMEOUT_SECONDS", "3")))
    client.Init()
    return client


def run_go2_dds(producer: YoloProducer) -> None:
    """Poll the Go2 video service over its ROS 2-compatible DDS transport."""

    client = _make_go2_video_client()

    def step() -> None:
        try:
            code, data = client.GetImageSample()
            if code != 0:
                raise RuntimeError(f"Go2 video service returned error code {code}")
            payload = bytes(data)
            if not payload:
                raise ValueError("Go2 video service returned an empty frame")
            producer.process_image(
                decode_image_bytes(payload),
                source_key=f"go2-dds:{hashlib.sha256(payload).hexdigest()}",
                observed_at=datetime.now(timezone.utc),
                frame_id=os.getenv("YOLO_FRAME_ID", "camera_link"),
            )
        except Exception as exc:
            producer.publish_failure(exc, source_key=f"go2-dds:{time.time_ns()}")
            raise

    run_loop(step, float(os.getenv("YOLO_INTERVAL_SECONDS", "0.2")))


async def _consume_go2_frames(
    producer: YoloProducer,
    frames: asyncio.Queue[tuple[Any, str, datetime]],
    fatal: asyncio.Future[Exception],
) -> None:
    """Infer newest-first without blocking aiortc's packet receive loop."""
    while True:
        image, source_key, observed_at = await frames.get()
        try:
            await asyncio.to_thread(
                producer.process_image,
                image,
                source_key=source_key,
                observed_at=observed_at,
                frame_id=os.getenv("YOLO_FRAME_ID", "camera_link"),
            )
        except Exception as exc:
            if not fatal.done():
                fatal.set_result(exc)
            return
        finally:
            frames.task_done()


async def _read_go2_track(
    track: Any,
    frames: asyncio.Queue[tuple[Any, str, datetime]],
    first_frame: asyncio.Event,
    fatal: asyncio.Future[Exception],
) -> None:
    errors = 0
    while not fatal.done():
        try:
            frame = await track.recv()
            errors = 0
        except Exception as exc:
            errors += 1
            if errors >= 5:
                fatal.set_result(
                    RuntimeError(
                        f"Go2 video track failed {errors} consecutive times: {exc}"
                    )
                )
                return
            await asyncio.sleep(0.2)
            continue

        observed_at = datetime.now(timezone.utc)
        item = (
            frame.to_ndarray(format="bgr24"),
            f"go2-webrtc:{time.time_ns()}",
            observed_at,
        )
        if frames.full():
            frames.get_nowait()
            frames.task_done()
        frames.put_nowait(item)
        first_frame.set()


async def _request_go2_keyframes(connection: Any, first_frame: asyncio.Event) -> None:
    """Request an H.264 I-frame while a mid-GOP join has decoded nothing."""
    while not first_frame.is_set():
        try:
            await asyncio.wait_for(first_frame.wait(), timeout=3.0)
            return
        except TimeoutError:
            pass
        for transceiver in connection.pc.getTransceivers():
            receiver = transceiver.receiver
            track = getattr(receiver, "track", None)
            send_pli = getattr(receiver, "_send_rtcp_pli", None)
            ssrc = getattr(receiver, "_ssrc", None)
            if track is not None and track.kind == "video" and send_pli and ssrc:
                asyncio.ensure_future(send_pli(ssrc))


async def _run_go2_webrtc_async(producer: YoloProducer) -> None:
    try:
        from unitree_webrtc_connect import (
            RobotBusyError,
            UnitreeWebRTCConnection,
            WebRTCConnectionMethod,
        )
    except ImportError as exc:
        raise RuntimeError(
            "go2_webrtc mode requires unitree-webrtc-connect and aiortc"
        ) from exc

    connection = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip=os.getenv("GO2_IP", "192.168.123.161"),
        aes_128_key=os.getenv("GO2_AES_KEY") or None,
    )
    consumer: asyncio.Task[None] | None = None
    keyframes: asyncio.Task[None] | None = None
    try:
        try:
            await connection.connect()
        except RobotBusyError as exc:
            raise RuntimeError(
                "the Go2 camera has one WebRTC slot and another client holds it; "
                "close the Unitree phone app or other camera service"
            ) from exc

        loop = asyncio.get_running_loop()
        fatal: asyncio.Future[Exception] = loop.create_future()
        frames: asyncio.Queue[tuple[Any, str, datetime]] = asyncio.Queue(maxsize=1)
        first_frame = asyncio.Event()
        consumer = asyncio.create_task(_consume_go2_frames(producer, frames, fatal))
        keyframes = asyncio.create_task(_request_go2_keyframes(connection, first_frame))
        connection.video.switchVideoChannel(True)
        connection.video.add_track_callback(
            lambda track: _read_go2_track(track, frames, first_frame, fatal)
        )
        LOGGER.info("Go2 WebRTC connected; waiting for front-camera frames")
        failure = await fatal
        raise RuntimeError("Go2 camera perception stopped") from failure
    finally:
        tasks = [task for task in (consumer, keyframes) if task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        close = getattr(connection.pc, "close", None)
        if close is not None:
            await close()


async def _run_go2_webrtc_forever(producer: YoloProducer) -> None:
    retry_seconds = float(os.getenv("GO2_RETRY_SECONDS", "5"))
    while True:
        try:
            await _run_go2_webrtc_async(producer)
        except Exception as exc:
            LOGGER.exception(
                "Go2 WebRTC session failed; retrying in %.1fs", retry_seconds
            )
            try:
                producer.publish_failure(
                    exc, source_key=f"go2-webrtc:{time.time_ns()}"
                )
            except Exception:
                LOGGER.exception("failed to publish Go2 WebRTC diagnostic")
            await asyncio.sleep(retry_seconds)


def run_go2_webrtc(producer: YoloProducer) -> None:
    """Consume the Go2's real front camera (the robot exposes no ROS image)."""
    asyncio.run(_run_go2_webrtc_forever(producer))


def main() -> None:
    configure_logging()
    client = WorldStateClient(world_state_url())
    producer: YoloProducer | None = None
    try:
        producer = _make_producer(client)
        mode = os.getenv("YOLO_INPUT_MODE", "ros2").casefold()
        LOGGER.info("starting YOLO producer in %s mode", mode)
        if mode == "ros2":
            run_ros2(producer)
        elif mode == "file":
            run_file(producer)
        elif mode == "http":
            run_http(producer)
        elif mode == "go2_dds":
            run_go2_dds(producer)
        elif mode == "go2_webrtc":
            run_go2_webrtc(producer)
        else:
            raise ValueError(
                "YOLO_INPUT_MODE must be 'go2_dds', 'http', 'go2_webrtc', "
                "'ros2', or 'file'"
            )
    except Exception as exc:
        if producer is not None:
            producer.publish_failure(exc, source_key=f"runtime:{time.time_ns()}")
        raise
    finally:
        client.close()
