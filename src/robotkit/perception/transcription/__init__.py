"""Speech transcription perception producer.

The core interpretation functions and backend protocol intentionally have no ROS,
audio-device, or model dependency.  Deployments opt into those integrations through
the command-line entrypoint.
"""

from robotkit.perception.transcription.backend import (
    FasterWhisperBackend,
    TranscriptionBackend,
    TranscriptionResult,
    TranscriptionSegment,
)
from robotkit.perception.transcription.interpretation import (
    VoiceIntent,
    extract_intent,
    normalize_transcript,
)
from robotkit.perception.transcription.producer import VoicePerceptionProducer

__all__ = [
    "FasterWhisperBackend",
    "TranscriptionBackend",
    "TranscriptionResult",
    "TranscriptionSegment",
    "VoiceIntent",
    "VoicePerceptionProducer",
    "extract_intent",
    "normalize_transcript",
]
