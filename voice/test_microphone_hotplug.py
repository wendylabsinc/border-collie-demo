"""Microphone hot-plug discovery and dead-microphone detection.

Two failures used to be invisible on stage. A microphone plugged in after boot
was never discovered, because PortAudio caches its device list at initialization
and every retry re-read the same stale snapshot. And a receiver that was plugged
in but sending nothing looked healthy, because frames kept arriving on schedule.

Neither may ever fail a Demo Run: the voice service runs in its own container
and only ever asks the demo to start a run, so waiting for a microphone is a
normal idle state, not a fault.
"""

from __future__ import annotations

import sys
import types
import unittest

# Python 3.14 removed audioop and these seams do not exercise resampling.
sys.modules.setdefault("audioop", types.ModuleType("audioop"))
_numpy = types.ModuleType("numpy")
_numpy.ndarray = object
sys.modules.setdefault("numpy", _numpy)

import devices as devices_module
from capture import Capture, MicrophoneSilent
from devices import InputDevice, list_input_devices


class _Frame:
    """Stand-in for a numpy frame exposing only what the silence gate reads."""

    def __init__(self, *, has_signal: bool) -> None:
        self._has_signal = has_signal

    def any(self) -> bool:
        return self._has_signal


SILENCE = _Frame(has_signal=False)
SIGNAL = _Frame(has_signal=True)


def _mic(name: str = "DJI MIC MINI") -> InputDevice:
    return InputDevice(index=7, name=name, channels=1, default_samplerate=48_000.0)


class _FakeSounddevice(types.ModuleType):
    """A sounddevice whose device list only changes when the backend is cycled.

    This is the real PortAudio behaviour that caused the bug: the snapshot is
    taken at `_initialize` and `query_devices` never revisits the hardware.
    """

    def __init__(self, snapshots: list[list[dict]]) -> None:
        super().__init__("sounddevice")
        self._snapshots = snapshots
        self._current = snapshots[0]
        self.initializations = 0
        self.terminations = 0

    def _terminate(self) -> None:
        self.terminations += 1

    def _initialize(self) -> None:
        self.initializations += 1
        index = min(self.initializations - 1, len(self._snapshots) - 1)
        self._current = self._snapshots[index]

    def query_devices(self) -> list[dict]:
        return self._current


class HotPlugDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_sd = sys.modules.get("sounddevice")

    def tearDown(self) -> None:
        if self._real_sd is None:
            sys.modules.pop("sounddevice", None)
        else:
            sys.modules["sounddevice"] = self._real_sd

    def _install(self, snapshots: list[list[dict]]) -> _FakeSounddevice:
        fake = _FakeSounddevice(snapshots)
        sys.modules["sounddevice"] = fake
        return fake

    def test_microphone_plugged_in_after_boot_becomes_discoverable(self) -> None:
        """The regression that made the mic work on boot only."""
        empty: list[dict] = []
        plugged = [{"name": "DJI MIC MINI", "max_input_channels": 1,
                    "default_samplerate": 48_000.0}]
        self._install([empty, plugged])

        # Boot: nothing plugged in. This is a waiting state, not an error.
        self.assertEqual(list_input_devices(), [])

        # Operator plugs the mic in; the next retry must see it.
        found = list_input_devices()
        self.assertEqual([d.name for d in found], ["DJI MIC MINI"])

    def test_enumeration_cycles_the_backend_so_the_snapshot_is_never_stale(self) -> None:
        fake = self._install([[]])
        list_input_devices()
        self.assertEqual(fake.terminations, 1)
        self.assertEqual(fake.initializations, 1)

    def test_refresh_can_be_skipped_while_a_stream_is_open(self) -> None:
        fake = self._install([[]])
        list_input_devices(refresh=False)
        self.assertEqual(fake.terminations, 0)
        self.assertEqual(fake.initializations, 0)

    def test_a_backend_that_refuses_to_cycle_still_enumerates(self) -> None:
        """A refresh failure must not become a discovery failure."""
        fake = self._install([[{"name": "DJI MIC MINI", "max_input_channels": 1,
                                "default_samplerate": 48_000.0}]])

        def explode() -> None:
            raise OSError("PortAudio refused to reinitialize")

        fake._initialize = explode  # type: ignore[method-assign]
        self.assertEqual([d.name for d in list_input_devices()], ["DJI MIC MINI"])


class DeadMicrophoneTests(unittest.TestCase):
    def test_sustained_digital_silence_is_reported_as_a_dead_microphone(self) -> None:
        capture = Capture(_mic())
        for _ in range(4):
            capture._q.put_nowait(SILENCE)
        ticks = iter([0.0, 0.0, 1.0, 1.0, 30.0, 30.0])

        stream = capture.frames(silence_timeout_s=20.0, monotonic=lambda: next(ticks))
        next(stream)
        next(stream)
        with self.assertRaisesRegex(MicrophoneSilent, "connected but sending no audio"):
            next(stream)

    def test_any_signal_resets_the_silence_window(self) -> None:
        """A quiet room must never be mistaken for a dead microphone."""
        capture = Capture(_mic())
        for frame in (SILENCE, SILENCE, SIGNAL, SILENCE):
            capture._q.put_nowait(frame)
        ticks = iter([0.0, 0.0, 10.0, 10.0, 21.0, 21.0, 22.0, 22.0])

        stream = capture.frames(silence_timeout_s=20.0, monotonic=lambda: next(ticks))
        for _ in range(4):
            next(stream)  # must not raise: the signal at t=21 cleared the window
        self.assertTrue(capture.signal_seen)

    def test_silence_detection_is_optional(self) -> None:
        capture = Capture(_mic())
        for _ in range(3):
            capture._q.put_nowait(SILENCE)
        stream = capture.frames(silence_timeout_s=None)
        for _ in range(3):
            next(stream)

    def test_silence_timeout_must_be_positive(self) -> None:
        capture = Capture(_mic())
        with self.assertRaisesRegex(ValueError, "silence_timeout_s"):
            next(capture.frames(silence_timeout_s=0.0))


if __name__ == "__main__":
    unittest.main()
