"""Stateless YOLO inference-to-world-state pipeline."""

from __future__ import annotations

import hashlib
import logging
import time
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
        model_name: str = "unknown",
        diagnostics_ttl_seconds: float = 10.0,
    ) -> None:
        self.detector = detector
        self.publisher = publisher
        self.producer_id = producer_id
        self.instance_id = instance_id
        self.deployment_generation = deployment_generation
        self.allowed_classes = allowed_classes
        self.min_confidence = min_confidence
        self.ttl_seconds = ttl_seconds
        self.model_name = model_name
        self.diagnostics_ttl_seconds = diagnostics_ttl_seconds

    def _publish_diagnostic(
        self,
        *,
        source_key: str,
        observed_at: datetime,
        status: str,
        inference_ms: float,
        detection_count: int | None = None,
        error: str | None = None,
    ) -> None:
        digest = hashlib.sha256(
            f"{self.instance_id}\0{source_key}".encode("utf-8")
        ).hexdigest()
        payload: dict[str, Any] = {
            "status": status,
            "model": self.model_name,
            "supported_classes": sorted(self.allowed_classes),
            "inference_ms": round(inference_ms, 3),
            "frame_age_ms": round(
                max(
                    0.0,
                    (datetime.now(timezone.utc) - observed_at).total_seconds() * 1000,
                ),
                3,
            ),
            "detection_count": detection_count,
        }
        if error:
            payload["error"] = error[:500]
        diagnostic = Observation(
            idempotency_key=f"yolo-diagnostic-v1:{digest}",
            producer_id=self.producer_id,
            instance_id=self.instance_id,
            deployment_generation=self.deployment_generation,
            stream="diagnostics.yolo",
            observation_type="diagnostic.perception.yolo.v1",
            observed_at=observed_at,
            frame_id="diagnostic",
            confidence=1.0 if status == "ok" else 0.0,
            ttl_seconds=self.diagnostics_ttl_seconds,
            payload=payload,
        )
        try:
            self.publisher.publish_observation(diagnostic)
        except Exception:
            # Observability must never turn successful perception into a
            # failed perception cycle. A's ordinary request logs retain the
            # diagnostic publication failure.
            logging.getLogger(__name__).exception("failed to publish YOLO diagnostic")

    def publish_failure(self, error: Exception, *, source_key: str) -> None:
        """Best-effort durable failure report for camera/runtime boundaries."""
        self._publish_diagnostic(
            source_key=source_key,
            observed_at=datetime.now(timezone.utc),
            status="failed",
            inference_ms=0.0,
            error=f"{type(error).__name__}: {error}",
        )

    def process_image(
        self,
        image: Any,
        *,
        source_key: str,
        observed_at: datetime | None = None,
        frame_id: str | None = "camera",
    ) -> PublishResult:
        observed_at = observed_at or datetime.now(timezone.utc)
        started = time.perf_counter()
        try:
            frame = self.detector.detect(image)
        except Exception as exc:
            self._publish_diagnostic(
                source_key=source_key,
                observed_at=observed_at,
                status="failed",
                inference_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        inference_ms = (time.perf_counter() - started) * 1000
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
        result = self.publisher.publish_observation(observation)
        self._publish_diagnostic(
            source_key=source_key,
            observed_at=observed_at,
            status="ok",
            inference_ms=inference_ms,
            detection_count=int(interpreted.payload["count"]),
        )
        return result
