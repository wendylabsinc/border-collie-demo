from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from collie_adapter import BorderCollieAdapter, VoiceIntent, interpret_command
from page import INDEX_HTML


class _Handler(BaseHTTPRequestHandler):
    requests = []
    status_payload = {}

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
        if self.path == "/api/run":
            response = {
                "run": {
                    "run_id": "32fd8de4-2d08-4f29-8304-f07413ad9a8f",
                    "target_fruit": payload["target_fruit"],
                    "activation_source": payload["activation_source"],
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
            interpret_command("find the banana"),
            VoiceIntent(action="activate_demo", target_fruit="banana"),
        )
        self.assertEqual(
            interpret_command("find bananas"),
            VoiceIntent(action="activate_demo", target_fruit="banana"),
        )
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

    def test_any_single_fruit_mention_triggers_without_an_action_verb(self) -> None:
        self.assertEqual(
            interpret_command("I like pears"),
            VoiceIntent(action="activate_demo", target_fruit="pear"),
        )
        self.assertEqual(
            interpret_command("banana"),
            VoiceIntent(action="activate_demo", target_fruit="banana"),
        )
        self.assertEqual(
            interpret_command("the red one is an apple please"),
            VoiceIntent(action="activate_demo", target_fruit="apple"),
        )

    def test_rejects_no_fruit_or_multiple_different_fruits(self) -> None:
        self.assertIsNone(interpret_command("go forward"))
        self.assertIsNone(interpret_command("find an apple and a pear"))
        self.assertIsNone(interpret_command("find bananas and apples"))

    def test_accepts_a_narrow_stop_vocabulary(self) -> None:
        self.assertEqual(interpret_command("stop the demo"), VoiceIntent("stop_demo"))


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

    def test_wake_and_command_remain_visible_as_separate_steps(self) -> None:
        self.assertIn('id="wakePanel"', INDEX_HTML)
        self.assertIn('id="commandPanel"', INDEX_HTML)
        self.assertIn("HEARD at", INDEX_HTML)
        self.assertIn("VALID DOG COMMAND", INDEX_HTML)
        self.assertIn("msg.kind === 'snapshot'", INDEX_HTML)
        self.assertNotIn("wake.classList.remove", INDEX_HTML)

    def test_demo_auto_arm_state_is_visible(self) -> None:
        self.assertIn("ARMED · ALWAYS ACTIVE", INDEX_HTML)
        self.assertIn("{{WAKE_TITLE}} · Border Collie Voice", INDEX_HTML)


class DispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        _Handler.requests = []
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
        result = self.adapter.dispatch("go to pear")
        self.assertEqual(
            _Handler.requests,
            [("/api/run", {"target_fruit": "pear", "activation_source": "voice"})],
        )
        self.assertEqual(result["calls"][0]["tool"], "activate_demo")
        self.assertEqual(
            result["calls"][0]["run_id"],
            "32fd8de4-2d08-4f29-8304-f07413ad9a8f",
        )

    def test_plural_banana_dispatches_the_canonical_target(self) -> None:
        self.adapter.arm()
        result = self.adapter.dispatch("find bananas")
        self.assertEqual(
            _Handler.requests,
            [("/api/run", {"target_fruit": "banana", "activation_source": "voice"})],
        )
        self.assertEqual(result["calls"][0]["tool"], "activate_demo")

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

    def test_cannot_arm_an_unready_dog_without_named_blockers(self) -> None:
        _Handler.status_payload["activation"] = {"ready": False, "blockers": []}

        with self.assertRaisesRegex(RuntimeError, "activation is not ready"):
            self.adapter.arm()
        self.assertFalse(self.adapter.armed)


if __name__ == "__main__":
    unittest.main()
