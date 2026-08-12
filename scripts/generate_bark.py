"""Generate a tiny deterministic bark-like WAV for the safe default image."""

from __future__ import annotations

import math
import random
import struct
import sys
import wave


def main(path: str) -> None:
    sample_rate = 16_000
    duration = 0.65
    random_source = random.Random(42)
    samples: list[bytes] = []
    previous = 0.0
    for index in range(int(sample_rate * duration)):
        time = index / sample_rate
        burst_time = time if time < 0.26 else time - 0.34
        active = 0 <= burst_time < (0.26 if time < 0.3 else 0.22)
        if not active:
            sample = 0.0
        else:
            envelope = math.sin(math.pi * burst_time / (0.26 if time < 0.3 else 0.22)) ** 2
            noise = random_source.uniform(-1.0, 1.0)
            previous = 0.82 * previous + 0.18 * noise
            growl = math.sin(2 * math.pi * (115 - 45 * burst_time) * burst_time)
            sample = envelope * (0.65 * previous + 0.35 * growl)
        samples.append(struct.pack("<h", int(max(-1, min(1, sample)) * 20_000)))
    with wave.open(path, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"".join(samples))


if __name__ == "__main__":
    main(sys.argv[1])
