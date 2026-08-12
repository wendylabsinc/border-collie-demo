"""Pure conversion of a typed website command into a world-state observation."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from robotkit.contracts import Observation
from robotkit.perception.transcription.interpretation import (
    extract_intent,
    normalize_transcript,
)


class WebsiteCommandProducer:
    """Build website observations without retaining process-local state."""

    def __init__(
        self,
        *,
        producer_id: str = "website-command",
        instance_id: str = "local",
        deployment_generation: int = 0,
        intent_ttl_seconds: float = 30.0,
    ) -> None:
        self.producer_id = producer_id
        self.instance_id = instance_id
        self.deployment_generation = deployment_generation
        self.intent_ttl_seconds = intent_ttl_seconds

    def observation_from_command(
        self,
        command: str,
        *,
        request_id: UUID | str | None = None,
        observed_at: datetime | None = None,
    ) -> Observation:
        """Return the exact immutable observation to publish to container A."""
        text = normalize_transcript(command)
        intent = extract_intent(text)
        correlation_id = str(request_id or uuid4())
        return Observation(
            producer_id=self.producer_id,
            instance_id=self.instance_id,
            deployment_generation=self.deployment_generation,
            idempotency_key=f"{self.producer_id}:{correlation_id}:intent",
            stream="website.intent",
            observation_type="website.intent.command",
            observed_at=observed_at or datetime.now(timezone.utc),
            frame_id="website",
            confidence=intent.confidence,
            ttl_seconds=self.intent_ttl_seconds,
            payload={
                "intent": intent.name,
                "slots": intent.slots,
                "source_text": text,
                "request_id": correlation_id,
            },
        )
