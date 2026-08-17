"""Microphone capture: opens the selected input device, downmixes to mono, and
resamples to 16 kHz float32 frames.

Uses sounddevice/PortAudio. The device is opened at its native sample rate and
resampled with stdlib `audioop.ratecv` (no scipy dependency). Frames are pushed
from the PortAudio callback thread onto a queue and yielded by `frames()`.
"""

from __future__ import annotations

import audioop
import queue
import time
from collections.abc import Callable

import numpy as np
from devices import InputDevice


class MicrophoneSilent(RuntimeError):
    """A live stream delivered nothing but digital silence.

    Distinct from a stalled stream (`TimeoutError`): frames keep arriving on
    schedule, they just carry no signal. A USB receiver whose transmitter is
    off, muted, or out of range stays enumerated and keeps clocking out zeroed
    buffers, which is otherwise indistinguishable from a healthy microphone.

    The test is bit-exact zero, not a level threshold. A live analog input
    always carries a noise floor -- on this stage speech measured around
    -55 dBFS and even a silent room sits well above zero -- so requiring every
    sample in every frame across the whole window to be exactly zero cannot fire
    on a quiet room. It only fires when no signal path exists at all.
    """


DEFAULT_SILENCE_TIMEOUT_S = 20.0


class Capture:
    def __init__(
        self,
        device: InputDevice,
        target_rate: int = 16000,
        frame_ms: int = 20,
        max_queue: int = 256,
    ) -> None:
        self.device = device
        self.target_rate = target_rate
        self.frame_ms = frame_ms
        self.native_rate = round(device.default_samplerate) or 48000
        self.channels = device.channels
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue)
        self._ratecv_state = None
        self._stream = None
        # Observable health, read by the status endpoint so an operator can tell
        # "no mic" from "mic present but sending nothing" without reading logs.
        self.signal_seen = False
        self.silent_seconds = 0.0
        # Native frames per callback block, sized so the resampled output is
        # ~frame_ms of audio.
        self._blocksize = int(self.native_rate * frame_ms / 1000)

    def _callback(self, indata, frames, time_info, status):
        # indata: int16 numpy array, shape (frames, channels)
        raw = bytes(indata)
        if self.channels == 2:
            raw = audioop.tomono(raw, 2, 0.5, 0.5)
        elif self.channels > 2:
            mono = np.asarray(indata, dtype=np.int16).mean(axis=1).astype(np.int16)
            raw = mono.tobytes()
        if self.native_rate != self.target_rate:
            raw, self._ratecv_state = audioop.ratecv(
                raw, 2, 1, self.native_rate, self.target_rate, self._ratecv_state
            )
        frame = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            pass  # drop under backpressure rather than block the audio thread

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            device=self.device.index,
            channels=self.channels,
            samplerate=self.native_rate,
            dtype="int16",
            blocksize=self._blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def frames(
        self,
        *,
        timeout_s: float | None = None,
        silence_timeout_s: float | None = DEFAULT_SILENCE_TIMEOUT_S,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        """Yield frames, failing when a live stream stops producing or goes dead.

        Two distinct failures, because they need different operator responses:
        `TimeoutError` means frames stopped arriving (unplugged, driver gone),
        and `MicrophoneSilent` means frames keep arriving with no signal in them
        (receiver on, transmitter off/muted/out of range).
        """
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero")
        if silence_timeout_s is not None and silence_timeout_s <= 0:
            raise ValueError("silence_timeout_s must be greater than zero")

        silent_since: float | None = None
        while True:
            try:
                frame = self._q.get(timeout=timeout_s)
            except queue.Empty as exc:
                raise TimeoutError(
                    f"no microphone frame received for {timeout_s:.3f} seconds"
                ) from exc

            if frame.any():
                silent_since = None
                self.signal_seen = True
            elif silence_timeout_s is not None:
                now = monotonic()
                if silent_since is None:
                    silent_since = now
                elif now - silent_since >= silence_timeout_s:
                    raise MicrophoneSilent(
                        f"{self.device.name} delivered only digital silence for "
                        f"{now - silent_since:.1f} seconds; the microphone is "
                        "connected but sending no audio (check it is powered on, "
                        "unmuted and paired to its receiver)"
                    )
            self.silent_seconds = 0.0 if silent_since is None else monotonic() - silent_since
            yield frame
