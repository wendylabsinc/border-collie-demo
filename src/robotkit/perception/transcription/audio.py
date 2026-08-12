"""PCM utilities shared by live and test audio sources."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioFormat:
    sample_rate_hz: int = 16_000
    channels: int = 1
    sample_width_bytes: int = 2

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.sample_width_bytes not in (1, 2, 3, 4):
            raise ValueError("sample_width_bytes must be between 1 and 4")

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate_hz * self.channels * self.sample_width_bytes


def pcm_to_wav(pcm: bytes, audio_format: AudioFormat) -> bytes:
    """Wrap little-endian integer PCM in a valid WAV container."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(audio_format.channels)
        wav.setsampwidth(audio_format.sample_width_bytes)
        wav.setframerate(audio_format.sample_rate_hz)
        wav.writeframes(pcm)
    return output.getvalue()


def read_wav(data: bytes) -> tuple[bytes, AudioFormat]:
    """Extract PCM and its format without requiring a media library."""
    with wave.open(io.BytesIO(data), "rb") as wav:
        if wav.getcomptype() != "NONE":
            raise ValueError("only uncompressed PCM WAV input is supported")
        audio_format = AudioFormat(
            sample_rate_hz=wav.getframerate(),
            channels=wav.getnchannels(),
            sample_width_bytes=wav.getsampwidth(),
        )
        return wav.readframes(wav.getnframes()), audio_format


class PCMChunker:
    """Stateful transport buffer; interpretation and the producer stay stateless."""

    def __init__(self, audio_format: AudioFormat, duration_seconds: float) -> None:
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        self._chunk_size = max(1, round(audio_format.bytes_per_second * duration_seconds))
        frame_size = audio_format.channels * audio_format.sample_width_bytes
        self._chunk_size -= self._chunk_size % frame_size
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        self._buffer.extend(data)
        chunks: list[bytes] = []
        while len(self._buffer) >= self._chunk_size:
            chunks.append(bytes(self._buffer[: self._chunk_size]))
            del self._buffer[: self._chunk_size]
        return chunks

    def flush(self) -> bytes:
        final = bytes(self._buffer)
        self._buffer.clear()
        return final
