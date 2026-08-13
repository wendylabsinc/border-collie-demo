"""Lazy runtime adapters for Ultralytics and ROS2 images."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    """Thin adapter over an Ultralytics YOLO/YOLOE checkpoint.

    ``model`` is injectable so adapter parsing is testable without importing
    Ultralytics or downloading weights.
    """

    def __init__(
        self,
        model_name: str = "yoloe-11m-seg.pt",
        *,
        confidence: float = 0.25,
        device: str | None = None,
        classes: Sequence[str] | None = None,
        prompt_templates: Sequence[str] = ("{name}",),
        image_size: int | None = None,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if image_size is not None and image_size <= 0:
            raise ValueError("image_size must be positive")
        if not prompt_templates or any("{name}" not in item for item in prompt_templates):
            raise ValueError("prompt_templates must contain {name}")
        self.model_name = model_name
        self.confidence = confidence
        self.device = device
        self.classes = tuple(classes or ())
        self.prompt_templates = tuple(prompt_templates)
        self.image_size = image_size
        self._model = model
        self._classes_configured = False
        self._prompt_to_class: Mapping[str, str] = {}

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - depends on runtime image
                raise RuntimeError(
                    f"Ultralytics failed to import: {exc}"
                ) from exc
            self._model = YOLO(self.model_name)
        if self.classes and not self._classes_configured:
            # YOLOE-11 is open-vocabulary. Encode the configured fruit names
            # once at startup, then reuse those embeddings for every frame.
            # Ultralytics 8.3.x requires embeddings explicitly here.
            names: list[str] = []
            prompt_to_class: dict[str, str] = {}
            for canonical_name in self.classes:
                for template in self.prompt_templates:
                    prompt = template.format(name=canonical_name)
                    previous = prompt_to_class.setdefault(prompt, canonical_name)
                    if previous != canonical_name:
                        raise ValueError(
                            f"YOLO prompt {prompt!r} maps to multiple classes"
                        )
                    if prompt not in names:
                        names.append(prompt)
            embeddings = self._model.get_text_pe(names)
            self._model.set_classes(names, embeddings)
            self._prompt_to_class = prompt_to_class
            self._classes_configured = True
        return self._model

    def detect(self, image: Any) -> DetectionFrame:
        kwargs: dict[str, Any] = {"source": image, "conf": self.confidence, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        if self.image_size is not None:
            kwargs["imgsz"] = self.image_size
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
                class_name=self._prompt_to_class.get(
                    str(names[int(class_id)]), str(names[int(class_id)])
                ),
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
