from __future__ import annotations

import sys
import types
import unittest

# Python 3.14 removed audioop and this test does not exercise resampling. Keep
# the queue-timeout seam independent of host audio packages.
audioop = types.ModuleType("audioop")
numpy = types.ModuleType("numpy")
numpy.ndarray = object
sys.modules.setdefault("audioop", audioop)
sys.modules.setdefault("numpy", numpy)

from capture import Capture
from devices import InputDevice


class CaptureRecoveryTests(unittest.TestCase):
    def test_frame_starvation_raises_so_the_session_can_restart(self) -> None:
        capture = Capture(
            InputDevice(
                index=7,
                name="DJI MIC MINI",
                channels=1,
                default_samplerate=48_000.0,
            )
        )

        with self.assertRaisesRegex(TimeoutError, "no microphone frame"):
            next(capture.frames(timeout_s=0.001))


if __name__ == "__main__":
    unittest.main()
