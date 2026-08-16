from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

from collie_adapter import (
    FULL_DEMO_SEED,
    BorderCollieAdapter,
    VoiceIntent,
    display_command,
    interpret_command,
)
from page import INDEX_HTML


class _Handler(BaseHTTPRequestHandler):
    requests: ClassVar[list[tuple[str, dict[str, object]]]] = []
    status_payload: ClassVar[dict[str, object]] = {}
    #: ``{path: (status_code, detail)}`` forces a refusal for one endpoint.
    refusals: ClassVar[dict[str, tuple[int, str]]] = {}

    def do_GET(self) -> None:
        if self.path != "/api/status":
            self.send_error(404)
            return
        body = json.dumps(self.__class__.status_payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.__class__.requests.append((self.path, payload))
        refusal = self.__class__.refusals.get(self.path)
        if refusal is not None:
            code, detail = refusal
            body = json.dumps({"detail": detail}).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/cohorts":
            response = {
                "cohort": {
                    "cohort_id": "c0f0a1b2-0000-4000-8000-abcdefabcdef",
                    "status": "RUNNING",
                    "fruit_sequence": ["pear", "apple", "mango"],
                }
            }
            status = 201
        elif self.path == "/api/run":
            response = {
                "run": {
                    "run_id": "32fd8de4-2d08-4f29-8304-f07413ad9a8f",
                    "target_fruit": payload["target_fruit"],
                    "activation_source": payload["activation_source"],
                    "activation_id": payload.get("activation_id"),
                    "current_phase": "wait_for_command",
                    "outcome": None,
                }
            }
            status = 201
        else:
            response = {"status": "stopped"}
            status = 200
        body = json.dumps(response).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        pass


class InterpretCommandTests(unittest.TestCase):
    def test_accepts_singular_and_plural_supported_fruit_phrases(self) -> None:
        self.assertEqual(
            interpret_command("Go to the pear"),
            VoiceIntent(action="activate_demo", target_fruit="pear"),
        )
        self.assertEqual(
            interpret_command("go to apple"),
            VoiceIntent(action="activate_demo", target_fruit="apple"),
        )
        self.assertEqual(
            interpret_command("go to apples"),
            VoiceIntent(action="activate_demo", target_fruit="apple"),
        )
        self.assertEqual(
            interpret_command("find the mango"),
            VoiceIntent(action="activate_demo", target_fruit="mango"),
        )
        self.assertEqual(
            interpret_command("find mangoes"),
            VoiceIntent(action="activate_demo", target_fruit="mango"),
        )
        self.assertIsNone(interpret_command("find the banana"))
        self.assertIsNone(interpret_command("go forward"))

    def test_accepts_natural_find_requests_and_the_pear_homophone(self) -> None:
        self.assertEqual(
            interpret_command("can you find a pair for me uh"),
            VoiceIntent(action="activate_demo", target_fruit="pear"),
        )
        self.assertEqual(
            interpret_command("please locate the red apple"),
            VoiceIntent(action="activate_demo", target_fruit="apple"),
        )

    def test_requires_an_allowlisted_movement_intent(self) -> None:
        self.assertIsNone(interpret_command("I like pears"))
        self.assertIsNone(interpret_command("banana"))
        self.assertIsNone(interpret_command("the red one is an apple please"))
        self.assertEqual(
            interpret_command("Hey Wendy, follow the apple"),
            VoiceIntent(action="activate_demo", target_fruit="apple"),
        )

    def test_rejects_no_fruit_or_multiple_different_fruits(self) -> None:
        self.assertIsNone(interpret_command("go forward"))
        self.assertIsNone(interpret_command("find an apple and a pear"))
        self.assertEqual(
            interpret_command("find bananas and apples"),
            VoiceIntent(action="activate_demo", target_fruit="apple"),
        )

    def test_accepts_a_narrow_stop_vocabulary(self) -> None:
        self.assertEqual(interpret_command("stop the demo"), VoiceIntent("stop_demo"))


class BareMangoPhraseTests(unittest.TestCase):
    MANGO = VoiceIntent(action="activate_demo", target_fruit="mango")

    def test_accepts_the_bare_word_and_its_asr_near_misses(self) -> None:
        for text in (
            "mango",
            "Mango.",
            "mangos",
            "mangoes",
            "man go",
            "Man Go",
            "mango run",
            "mango demo",
            "run mango",
            "start the mango",
            "hey wendy mango please",
        ):
            with self.subTest(text=text):
                self.assertEqual(interpret_command(text), self.MANGO)

    def test_mango_inside_ordinary_speech_does_not_activate(self) -> None:
        for text in (
            "I like mango",
            "the mango is on the left",
            "is that a mango or a pear",
            "so the mango",
            "put the mango down over there",
            "mango is my favourite fruit",
        ):
            with self.subTest(text=text):
                self.assertIsNone(interpret_command(text))

    def test_an_explicit_verb_still_reaches_mango(self) -> None:
        self.assertEqual(interpret_command("find the mango"), self.MANGO)


class FullDemoPhraseTests(unittest.TestCase):
    FULL_DEMO = VoiceIntent(action="full_demo")

    def test_accepts_the_phrase_and_its_asr_near_misses(self) -> None:
        for text in (
            "full demo",
            "Full demo.",
            "fulldemo",
            "full demos",
            "full demo run",
            "run the full demo",
            "start full demo",
            "hey wendy full demo please",
        ):
            with self.subTest(text=text):
                self.assertEqual(interpret_command(text), self.FULL_DEMO)

    def test_full_demo_wins_over_a_fruit_named_in_the_same_breath(self) -> None:
        # Ambiguous: refuse rather than start the wrong thing on a real robot.
        for text in (
            "run the full demo and find the mango",
            "full demo starting with mango",
            "after the full demo go to the pear",
        ):
            with self.subTest(text=text):
                self.assertIsNone(interpret_command(text))

    def test_conversation_about_a_full_demo_does_not_activate(self) -> None:
        for text in (
            "we should do the full demo later",
            "that was a full demo of the system",
            "demo",
            "full",
        ):
            with self.subTest(text=text):
                self.assertIsNone(interpret_command(text))

    def test_display_labels_do_not_mislabel_the_cohort_as_stop(self) -> None:
        self.assertEqual(display_command(self.FULL_DEMO), "full demo")
        self.assertEqual(display_command(BareMangoPhraseTests.MANGO), "mango")
        self.assertEqual(display_command(VoiceIntent("stop_demo")), "stop")


class VoicePageTests(unittest.TestCase):
    def test_missing_microphone_is_rendered_as_a_loud_blocker(self) -> None:
        self.assertIn("MIC ERROR", INDEX_HTML)
        self.assertIn("MIC NOT CONNECTED", INDEX_HTML)
        self.assertIn("listening will start automatically", INDEX_HTML)
        self.assertIn("Microphone unavailable", INDEX_HTML)
        self.assertIn("armButton.disabled = true", INDEX_HTML)

    def test_feed_shows_only_the_canonical_command_and_validity(self) -> None:
        self.assertIn("msg.display_command.toUpperCase()", INDEX_HTML)
        self.assertIn("NO VALID COMMAND", INDEX_HTML)
        self.assertIn("Valid dog command", INDEX_HTML)
        self.assertNotIn("msg.audio_ms", INDEX_HTML)
        self.assertNotIn("msg.input_dbfs", INDEX_HTML)

    def test_wake_resets_to_waiting_after_a_short_visible_window(self) -> None:
        self.assertIn('id="wakePanel"', INDEX_HTML)
        self.assertIn('id="commandPanel"', INDEX_HTML)
        self.assertIn("HEARD at", INDEX_HTML)
        self.assertIn("VALID DOG COMMAND", INDEX_HTML)
        self.assertIn("msg.kind === 'snapshot'", INDEX_HTML)
        self.assertIn("const WAKE_RESET_MS = 5000", INDEX_HTML)
        self.assertIn("function resetWake()", INDEX_HTML)
        self.assertIn("window.clearTimeout(wakeResetTimer)", INDEX_HTML)
        self.assertIn("window.setTimeout(resetWake, WAKE_RESET_MS)", INDEX_HTML)
        self.assertIn("Listening for &ldquo;{{WAKE}}&rdquo;", INDEX_HTML)

    def test_demo_auto_arm_state_is_visible(self) -> None:
        self.assertIn("ARMED · ALWAYS ACTIVE", INDEX_HTML)
        self.assertIn("{{WAKE_TITLE}} · Border Collie Voice", INDEX_HTML)


class DispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        _Handler.requests = []
        _Handler.refusals = {}
        _Handler.status_payload = {
            "build_label": "stage camera test build",
            "release": {
                "release_id": "stage-camera-v28-lie-down-evidence",
                "config_schema": 12,
            },
            "runtime_mode": "production",
            "mission": {"restart_required": False},
            "active_run_id": None,
            "activation": {"ready": True, "blockers": []},
        }
        self.server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.adapter = BorderCollieAdapter(
            f"http://{host}:{port}",
            expected_release_id="stage-camera-v28-lie-down-evidence",
            expected_config_schema=12,
        )

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_starts_disarmed(self) -> None:
        result = self.adapter.dispatch("go to pear")
        self.assertIn("disarmed", result["error"])
        self.assertEqual(_Handler.requests, [])

    def test_armed_activation_uses_the_existing_demo_api(self) -> None:
        self.adapter.arm()
        result = self.adapter.dispatch(
            "go to pear", activation_id="voice-wake-pear-1"
        )
        self.assertEqual(
            _Handler.requests,
            [
                (
                    "/api/run",
                    {
                        "target_fruit": "pear",
                        "activation_source": "voice",
                        "activation_id": "voice-wake-pear-1",
                    },
                )
            ],
        )
        self.assertEqual(result["calls"][0]["tool"], "activate_demo")
        self.assertEqual(
            result["calls"][0]["run_id"],
            "32fd8de4-2d08-4f29-8304-f07413ad9a8f",
        )

    def test_disabled_banana_does_not_dispatch(self) -> None:
        self.adapter.arm()
        result = self.adapter.dispatch(
            "find bananas", activation_id="voice-wake-banana-1"
        )
        self.assertEqual(_Handler.requests, [])
        self.assertEqual(result["calls"], [])

    def test_activation_requires_a_durable_voice_idempotency_key(self) -> None:
        self.adapter.arm()

        with self.assertRaisesRegex(ValueError, "activation_id"):
            self.adapter.dispatch("find the pear")

        self.assertEqual(_Handler.requests, [])

    def test_cannot_arm_when_the_dog_reports_a_preflight_blocker(self) -> None:
        _Handler.status_payload["activation"] = {
            "ready": False,
            "blockers": [{"name": "fresh_pose", "detail": "fresh pose unavailable"}],
        }

        with self.assertRaisesRegex(RuntimeError, "fresh pose unavailable"):
            self.adapter.arm()
        self.assertFalse(self.adapter.armed)

    def test_cannot_arm_the_wrong_dog_build(self) -> None:
        adapter = BorderCollieAdapter(
            self.adapter.base_url,
            expected_build_label="approved build",
        )

        with self.assertRaisesRegex(RuntimeError, "expected dog build"):
            adapter.arm()
        self.assertFalse(adapter.armed)

    def test_cannot_arm_the_wrong_dog_release(self) -> None:
        _Handler.status_payload["release"]["release_id"] = "stage-camera-v29"

        with self.assertRaisesRegex(RuntimeError, "expected dog release"):
            self.adapter.arm()
        self.assertFalse(self.adapter.armed)

    def test_cannot_arm_the_wrong_config_schema(self) -> None:
        _Handler.status_payload["release"]["config_schema"] = 13

        with self.assertRaisesRegex(RuntimeError, "expected dog config schema"):
            self.adapter.arm()
        self.assertFalse(self.adapter.armed)

    def test_cannot_arm_a_simulation_runtime(self) -> None:
        _Handler.status_payload["runtime_mode"] = "simulation"

        with self.assertRaisesRegex(RuntimeError, "runtime must be 'production'"):
            self.adapter.arm()
        self.assertFalse(self.adapter.armed)

    def test_bare_mango_starts_a_single_voice_demo_run(self) -> None:
        self.adapter.arm()
        result = self.adapter.dispatch("mango", activation_id="voice-wake-mango-1")
        self.assertEqual(
            _Handler.requests,
            [
                (
                    "/api/run",
                    {
                        "target_fruit": "mango",
                        "activation_source": "voice",
                        "activation_id": "voice-wake-mango-1",
                    },
                )
            ],
        )
        self.assertEqual(result["calls"][0]["tool"], "activate_demo")

    def test_full_demo_posts_the_three_fruit_cohort_the_ui_button_sends(self) -> None:
        self.adapter.arm()
        result = self.adapter.dispatch("full demo", activation_id="voice-wake-full-1")

        self.assertEqual(len(_Handler.requests), 1)
        path, payload = _Handler.requests[0]
        self.assertEqual(path, "/api/cohorts")
        self.assertEqual(payload["runs"], 3)
        self.assertIs(payload["randomized"], True)
        self.assertIsNone(payload["target_fruit"])
        self.assertEqual(payload["fruit_subset"], ["apple", "mango", "pear"])
        self.assertEqual(payload["seed"], FULL_DEMO_SEED)
        self.assertEqual(
            payload["tuning"],
            {"search": {"yaw_rps": 0.8}, "home": {"align_yaw_rps": 0.8}},
        )
        self.assertIn({"reason": "TARGET_LOST"}, payload["tolerated_failures"])
        self.assertIn({"failed_phase": "sit_and_bark"}, payload["tolerated_failures"])
        self.assertEqual(len(payload["tolerated_failures"]), 10)

        call = result["calls"][0]
        self.assertEqual(call["tool"], "start_full_demo")
        self.assertEqual(call["cohort_id"], "c0f0a1b2-0000-4000-8000-abcdefabcdef")
        self.assertIn("RUNNING", call["result"])

    def test_full_demo_never_falls_back_to_a_single_run(self) -> None:
        self.adapter.arm()
        result = self.adapter.dispatch(
            "run the full demo and find the mango",
            activation_id="voice-wake-ambiguous-1",
        )
        self.assertEqual(_Handler.requests, [])
        self.assertEqual(result["calls"], [])

    def test_full_demo_is_blocked_before_dispatch_when_a_run_is_active(self) -> None:
        self.adapter.arm()
        _Handler.status_payload["active_run_id"] = "run-in-flight"
        result = self.adapter.dispatch("full demo")
        self.assertEqual(_Handler.requests, [])
        self.assertIn("already active", result["error"])

    def test_cohort_conflict_409_reaches_the_operator_with_the_api_detail(self) -> None:
        self.adapter.arm()
        _Handler.refusals = {
            "/api/cohorts": (409, "a Demo Run is already active"),
        }

        with self.assertRaises(RuntimeError) as caught:
            self.adapter.dispatch("full demo")

        message = str(caught.exception)
        self.assertIn("409", message)
        self.assertIn("a Demo Run is already active", message)
        self.assertNotIn("{", message)

    def test_takeover_latch_423_reaches_the_operator_with_the_api_detail(self) -> None:
        self.adapter.arm()
        _Handler.refusals = {
            "/api/run": (423, "physical remote takeover latched"),
        }

        with self.assertRaises(RuntimeError) as caught:
            self.adapter.dispatch("mango", activation_id="voice-wake-mango-2")

        message = str(caught.exception)
        self.assertIn("423", message)
        self.assertIn("physical remote takeover latched", message)
        self.assertIn("restart the demo", message)

    def test_full_demo_requires_arming_like_every_other_motion_command(self) -> None:
        result = self.adapter.dispatch("full demo")
        self.assertIn("disarmed", result["error"])
        self.assertEqual(_Handler.requests, [])

    def test_cannot_arm_an_unready_dog_without_named_blockers(self) -> None:
        _Handler.status_payload["activation"] = {"ready": False, "blockers": []}

        with self.assertRaisesRegex(RuntimeError, "activation is not ready"):
            self.adapter.arm()
        self.assertFalse(self.adapter.armed)


if __name__ == "__main__":
    unittest.main()
