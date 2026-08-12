"""Whisper adapter with lazy optional model loading."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Protocol

from robotkit.perception.transcription.audio import AudioFormat, pcm_to_wav


@dataclass(frozen=True)
class TranscriptionSegment:
    start_seconds: float
    end_seconds: float
    text: str
    confidence: float | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None = None
    language_probability: float | None = None
    segments: tuple[TranscriptionSegment, ...] = field(default_factory=tuple)


class TranscriptionBackend(Protocol):
    def transcribe(self, pcm: bytes, audio_format: AudioFormat) -> TranscriptionResult:
        """Transcribe one complete PCM utterance."""


class FasterWhisperBackend:
    """Production adapter for ``faster-whisper``.

    Importing this module and constructing the adapter are cheap.  The optional
    dependency and model weights are touched only by the first ``transcribe`` call.
    """

    def __init__(
        self,
        model_name: str = "small.en",
        *,
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model = model

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is required for transcription; install "
                    "src/robotkit/perception/transcription/requirements.txt"
                ) from exc
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, pcm: bytes, audio_format: AudioFormat) -> TranscriptionResult:
        if not pcm:
            return TranscriptionResult(text="")

        wav = pcm_to_wav(pcm, audio_format)
        path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                handle.write(wav)
                path = handle.name
            raw_segments, info = self._get_model().transcribe(
                path,
                beam_size=self.beam_size,
                language=self.language,
                vad_filter=self.vad_filter,
            )
            segments = tuple(self._convert_segment(segment) for segment in raw_segments)
        finally:
            if path:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

        return TranscriptionResult(
            text=" ".join(item.text.strip() for item in segments if item.text.strip()),
            language=getattr(info, "language", self.language),
            language_probability=_probability(
                getattr(info, "language_probability", None)
            ),
            segments=segments,
        )

    @staticmethod
    def _convert_segment(segment: Any) -> TranscriptionSegment:
        probability = None
        avg_logprob = getattr(segment, "avg_logprob", None)
        if avg_logprob is not None:
            probability = _probability(math.exp(float(avg_logprob)))
        return TranscriptionSegment(
            start_seconds=float(segment.start),
            end_seconds=float(segment.end),
            text=str(segment.text),
            confidence=probability,
        )


def _probability(value: Any) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))
