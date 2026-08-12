"""Stateless YOLO inference-to-world-state pipeline."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Protocol

from robotkit.contracts import Observation, PublishResult
from robotkit.perception.yolo.adapters import DetectionFrame
from robotkit.perception.yolo.core import COCO_FRUIT_CLASSES, interpret_detections


class Detector(Protocol):
    def detect(self, image: Any) -> DetectionFrame: ...


class ObservationPublisher(Protocol):
    def publish_observation(self, observation: Observation) -> PublishResult: ...


class YoloProducer:
    """A stateless producer whose dependencies are explicit and injectable."""

    def __init__(
        self,
        detector: Detector,
        publisher: ObservationPublisher,
        *,
        producer_id: str = "yolo-coco-fruits",
        instance_id: str,
        deployment_generation: int = 0,
        allowed_classes: frozenset[str] = COCO_FRUIT_CLASSES,
        min_confidence: float = 0.25,
        ttl_seconds: float = 2.0,
    ) -> None:
        self.detector = detector
        self.publisher = publisher
        self.producer_id = producer_id
        self.instance_id = instance_id
        self.deployment_generation = deployment_generation
        self.allowed_classes = allowed_classes
        self.min_confidence = min_confidence
        self.ttl_seconds = ttl_seconds

    def process_image(
        self,
        image: Any,
        *,
        source_key: str,
        observed_at: datetime | None = None,
        frame_id: str | None = "camera",
    ) -> PublishResult:
        observed_at = observed_at or datetime.now(timezone.utc)
        frame = self.detector.detect(image)
        interpreted = interpret_detections(
            frame.detections,
            image_width=frame.width,
            image_height=frame.height,
            allowed_classes=self.allowed_classes,
            min_confidence=self.min_confidence,
        )
        digest = hashlib.sha256(
            f"{self.instance_id}\0{source_key}".encode("utf-8")
        ).hexdigest()
        observation = Observation(
            idempotency_key=f"yolo-frame-v1:{digest}",
            producer_id=self.producer_id,
            instance_id=self.instance_id,
            deployment_generation=self.deployment_generation,
            stream=interpreted.stream,
            observation_type=interpreted.observation_type,
            observed_at=observed_at,
            frame_id=frame_id,
            confidence=interpreted.confidence,
            ttl_seconds=self.ttl_seconds,
            payload=interpreted.payload,
        )
        return self.publisher.publish_observation(observation)
