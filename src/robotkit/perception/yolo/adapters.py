"""Lazy runtime adapters for Ultralytics and ROS2 images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robotkit.perception.yolo.core import Detection


@dataclass(frozen=True)
class DetectionFrame:
    detections: tuple[Detection, ...]
    width: int
    height: int


def _tolist(value: Any) -> list[Any]:
    """Convert Torch-like values without importing Torch."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


class UltralyticsDetector:
    """Thin adapter over an Ultralytics YOLO COCO checkpoint.

    ``model`` is injectable so adapter parsing is testable without importing
    Ultralytics or downloading weights.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        *,
        confidence: float = 0.25,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.model_name = model_name
        self.confidence = confidence
        self.device = device
        self._model = model

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - depends on runtime image
                raise RuntimeError(
                    "Ultralytics is required at runtime; install yolo/requirements.txt"
                ) from exc
            self._model = YOLO(self.model_name)
        return self._model

    def detect(self, image: Any) -> DetectionFrame:
        kwargs: dict[str, Any] = {"source": image, "conf": self.confidence, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = self._load_model().predict(**kwargs)
        if not results:
            raise RuntimeError("Ultralytics returned no result for the image")

        result = results[0]
        height, width = (int(value) for value in result.orig_shape[:2])
        boxes = result.boxes
        if boxes is None:
            return DetectionFrame((), width, height)

        class_ids = _tolist(boxes.cls)
        confidences = _tolist(boxes.conf)
        coordinates = _tolist(boxes.xyxy)
        names = getattr(result, "names", None) or getattr(self._model, "names", {})
        detections = tuple(
            Detection(
                class_id=int(class_id),
                class_name=str(names[int(class_id)]),
                confidence=float(confidence),
                xyxy=tuple(float(value) for value in xyxy),  # type: ignore[arg-type]
            )
            for class_id, confidence, xyxy in zip(class_ids, confidences, coordinates)
        )
        return DetectionFrame(detections, width, height)


def decode_ros_image(message: Any) -> Any:
    """Convert a ``sensor_msgs/Image`` into an Ultralytics-compatible array.

    NumPy is imported only when a ROS image is actually processed. Common
    8-bit encodings are supported without requiring cv_bridge.
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError("NumPy is required to decode ROS2 images") from exc

    encoding = str(message.encoding).casefold()
    channels_by_encoding = {
        "mono8": 1,
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
    }
    if encoding not in channels_by_encoding:
        raise ValueError(f"unsupported ROS image encoding: {message.encoding}")
    channels = channels_by_encoding[encoding]
    row_bytes = int(message.width) * channels
    step = int(message.step)
    if step < row_bytes:
        raise ValueError("ROS image step is smaller than the encoded row")

    raw = np.frombuffer(message.data, dtype=np.uint8)
    required = int(message.height) * step
    if raw.size < required:
        raise ValueError("ROS image data is truncated")
    rows = raw[:required].reshape(int(message.height), step)[:, :row_bytes]
    if channels == 1:
        return rows.reshape(int(message.height), int(message.width)).copy()

    image = rows.reshape(int(message.height), int(message.width), channels)
    if encoding == "rgb8":
        image = image[:, :, ::-1]
    elif encoding == "rgba8":
        image = image[:, :, [2, 1, 0]]
    elif encoding == "bgra8":
        image = image[:, :, :3]
    return np.ascontiguousarray(image)


def decode_image_bytes(payload: bytes) -> Any:
    """Decode a JPEG/PNG payload into an Ultralytics-compatible BGR array."""

    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError("OpenCV and NumPy are required to decode HTTP images") from exc

    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("camera endpoint returned an invalid image")
    return image
