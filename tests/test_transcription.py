from __future__ import annotations

import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from robotkit.perception.transcription.audio import (
    AudioFormat,
    PCMChunker,
    pcm_to_wav,
    read_wav,
)
from robotkit.perception.transcription.backend import (
    FasterWhisperBackend,
    TranscriptionResult,
    TranscriptionSegment,
)
from robotkit.perception.transcription.interpretation import (
    extract_intent,
    normalize_transcript,
)
from robotkit.perception.transcription.producer import VoicePerceptionProducer


NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


class FakeBackend:
    def __init__(self, result: TranscriptionResult) -> None:
        self.result = result
        self.calls = []

    def transcribe(self, pcm: bytes, audio_format: AudioFormat) -> TranscriptionResult:
        self.calls.append((pcm, audio_format))
        return self.result


class FakePublisher:
    def __init__(self) -> None:
        self.observations = []

    def publish_observation(self, observation):
        self.observations.append(observation)


def test_normalization_and_auditable_intent_extraction():
    text = normalize_transcript("  Hey,   Go2, move forward  3 feet! \n")
    intent = extract_intent(text)

    assert text == "Hey, Go2, move forward 3 feet!"
    assert intent.name == "move"
    assert intent.slots == {"direction": "forward", "distance_m": 0.914}
    assert extract_intent("stop").name == "stop"
    assert extract_intent("Robot, stop!").name == "stop"
    assert extract_intent("emergency stop now").name == "stop"
    assert extract_intent("find the red apple").slots == {"target": "red apple"}
    assert extract_intent("go to apple").name == "find"
    assert extract_intent("go to apple").slots == {"target": "apple"}
    assert extract_intent("Today is pleasant").name == "unknown"


def test_producer_publishes_transcript_and_intent_from_synthetic_pcm():
    result = TranscriptionResult(
        text=" Robot, turn left 90 degrees. ",
        language="en",
        language_probability=0.99,
        segments=(TranscriptionSegment(0.0, 1.2, "turn left", 0.8),),
    )
    backend = FakeBackend(result)
    publisher = FakePublisher()
    producer = VoicePerceptionProducer(
        backend,
        publisher,
        producer_id="voice-test",
        instance_id="green-1",
        deployment_generation=7,
    )
    pcm = b"\x00\x00\x01\x00" * 100

    observations = producer.transcribe_pcm(
        pcm, AudioFormat(), observed_at=NOW, chunk_id="audio-42"
    )

    assert backend.calls == [(pcm, AudioFormat())]
    assert publisher.observations == observations
    assert [item.stream for item in observations] == [
        "voice.transcript",
        "voice.intent",
    ]
    assert observations[0].payload["text"] == "Robot, turn left 90 degrees."
    assert observations[0].confidence == pytest.approx(0.8)
    assert observations[1].payload["intent"] == "turn"
    assert observations[1].payload["slots"] == {
        "direction": "left",
        "angle_degrees": 90.0,
    }
    assert observations[1].idempotency_key == "voice-test:audio-42:intent"
    assert all(item.deployment_generation == 7 for item in observations)


def test_pcm_wav_round_trip_and_chunking_need_no_audio_dependency():
    audio_format = AudioFormat(sample_rate_hz=10, channels=1, sample_width_bytes=2)
    pcm = bytes(range(40))
    restored, restored_format = read_wav(pcm_to_wav(pcm, audio_format))
    chunker = PCMChunker(audio_format, duration_seconds=0.5)

    assert restored == pcm
    assert restored_format == audio_format
    assert chunker.feed(pcm[:7]) == []
    assert chunker.feed(pcm[7:23]) == [pcm[:10], pcm[10:20]]
    assert chunker.flush() == pcm[20:23]


def test_faster_whisper_adapter_accepts_injected_model_without_optional_import():
    class FakeModel:
        def transcribe(self, path, **kwargs):
            assert path.endswith(".wav")
            assert kwargs["vad_filter"] is True
            return iter(
                [SimpleNamespace(start=0, end=1, text=" hello ", avg_logprob=-0.1)]
            ), SimpleNamespace(language="en", language_probability=0.97)

    before = sys.modules.get("faster_whisper")
    backend = FasterWhisperBackend(model=FakeModel())
    result = backend.transcribe(b"\x00\x00" * 100, AudioFormat())

    assert result.text == "hello"
    assert result.language == "en"
    assert result.language_probability == 0.97
    assert result.segments[0].confidence == pytest.approx(0.904837, rel=1e-5)
    assert sys.modules.get("faster_whisper") is before
