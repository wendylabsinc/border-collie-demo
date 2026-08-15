from __future__ import annotations

import unittest

from microphone import run_microphone_session_loop


class MicrophoneReconnectTests(unittest.TestCase):
    def test_missing_open_failure_and_disconnect_all_retry_until_capture_recovers(
        self,
    ) -> None:
        outcomes = iter(
            [
                ("receiver missing", False),
                ("microphone capture failed to start: device busy", False),
                ("microphone capture stopped unexpectedly: receiver unplugged", False),
                (None, True),
            ]
        )
        stop = False
        sleeps: list[float] = []
        retries: list[str] = []
        state: dict[str, object] = {
            "ready": False,
            "device": None,
            "error": "not started",
        }

        def session() -> None:
            nonlocal stop
            error, ready = next(outcomes)
            state.update(
                ready=ready,
                device="DJI MIC MINI" if ready or "capture" in str(error) else None,
                error=error,
            )
            if ready:
                stop = True

        run_microphone_session_loop(
            session=session,
            state=state,
            retry_interval_s=0.25,
            sleep=sleeps.append,
            should_stop=lambda: stop,
            on_retry=retries.append,
        )

        self.assertEqual(sleeps, [0.25, 0.25, 0.25])
        self.assertEqual(
            retries,
            [
                "receiver missing",
                "microphone capture failed to start: device busy",
                "microphone capture stopped unexpectedly: receiver unplugged",
            ],
        )
        self.assertEqual(
            state,
            {
                "ready": True,
                "device": "DJI MIC MINI",
                "error": None,
                "attempts": 4,
                "retry_count": 3,
            },
        )

    def test_retry_interval_is_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "retry_interval_s"):
            run_microphone_session_loop(
                session=lambda: None,
                state={},
                retry_interval_s=0.0,
                sleep=lambda _seconds: None,
                should_stop=lambda: True,
            )


if __name__ == "__main__":
    unittest.main()
