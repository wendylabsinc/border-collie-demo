"""World-state publishing boundary for voice interpretations."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Protocol

from robotkit.contracts import Observation
from robotkit.perception.transcription.audio import AudioFormat
from robotkit.perception.transcription.backend import (
    TranscriptionBackend,
    TranscriptionResult,
)
from robotkit.perception.transcription.interpretation import (
    extract_intent,
    normalize_transcript,
)


class ObservationPublisher(Protocol):
    def publish_observation(self, observation: Observation) -> object: ...


class VoicePerceptionProducer:
    def __init__(
        self,
        backend: TranscriptionBackend,
        publisher: ObservationPublisher,
        *,
        producer_id: str = "voice-transcription",
        instance_id: str = "local",
        deployment_generation: int = 0,
        transcript_ttl_seconds: float = 300.0,
        intent_ttl_seconds: float = 30.0,
    ) -> None:
        self.backend = backend
        self.publisher = publisher
        self.producer_id = producer_id
        self.instance_id = instance_id
        self.deployment_generation = deployment_generation
        self.transcript_ttl_seconds = transcript_ttl_seconds
        self.intent_ttl_seconds = intent_ttl_seconds

    def transcribe_pcm(
        self,
        pcm: bytes,
        audio_format: AudioFormat = AudioFormat(),
        *,
        observed_at: datetime | None = None,
        chunk_id: str | None = None,
    ) -> list[Observation]:
        """Interpret and publish one utterance, returning exactly what was sent."""
        observed_at = observed_at or datetime.now(timezone.utc)
        result = self.backend.transcribe(pcm, audio_format)
        observations = self.observations_from_result(
            result,
            observed_at=observed_at,
            chunk_id=chunk_id or _chunk_id(pcm, observed_at),
        )
        for observation in observations:
            self.publisher.publish_observation(observation)
        return observations

    def observations_from_result(
        self,
        result: TranscriptionResult,
        *,
        observed_at: datetime,
        chunk_id: str,
    ) -> list[Observation]:
        text = normalize_transcript(result.text)
        intent = extract_intent(text)
        segment_confidences = [
            segment.confidence
            for segment in result.segments
            if segment.confidence is not None
        ]
        transcript_confidence = (
            sum(segment_confidences) / len(segment_confidences)
            if segment_confidences
            else result.language_probability
        )
        common = {
            "producer_id": self.producer_id,
            "instance_id": self.instance_id,
            "deployment_generation": self.deployment_generation,
            "observed_at": observed_at,
            "frame_id": "microphone",
        }
        transcript = Observation(
            **common,
            idempotency_key=f"{self.producer_id}:{chunk_id}:transcript",
            stream="voice.transcript",
            observation_type="voice.transcript.final",
            confidence=transcript_confidence,
            ttl_seconds=self.transcript_ttl_seconds,
            payload={
                "text": text,
                "language": result.language,
                "language_probability": result.language_probability,
                "segments": [
                    {
                        "start_seconds": segment.start_seconds,
                        "end_seconds": segment.end_seconds,
                        "text": normalize_transcript(segment.text),
                        "confidence": segment.confidence,
                    }
                    for segment in result.segments
                ],
            },
        )
        command = Observation(
            **common,
            idempotency_key=f"{self.producer_id}:{chunk_id}:intent",
            stream="voice.intent",
            observation_type="voice.intent.command",
            confidence=intent.confidence,
            ttl_seconds=self.intent_ttl_seconds,
            payload={
                "intent": intent.name,
                "slots": intent.slots,
                "source_transcript": text,
            },
        )
        return [transcript, command]


def _chunk_id(pcm: bytes, observed_at: datetime) -> str:
    digest = hashlib.sha256(pcm).hexdigest()[:16]
    return f"{observed_at.isoformat()}:{digest}"
