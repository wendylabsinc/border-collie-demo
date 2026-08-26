from __future__ import annotations

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

from collie_adapter import (
    FULL_DEMO_SEED,
    BorderCollieAdapter,
    VoiceIntent,
    _edit_distance,
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


class FuzzyMangoAcceptanceTests(unittest.TestCase):
    """Near-misses a -54 dBFS channel produces, which must still start a run."""

    MANGO = VoiceIntent(action="activate_demo", target_fruit="mango")

    def test_accepts_vowel_substitutions_as_the_whole_utterance(self) -> None:
        # The final /oh/ is unstressed and the first vowel is short; both are the
        # first things to go when the input is 25 dB down.
        for text in ("mengo", "mingo", "mangoe", "mangu", "mango."):
            with self.subTest(text=text):
                self.assertEqual(interpret_command(text), self.MANGO)

    def test_accepts_truncation_as_the_whole_utterance(self) -> None:
        self.assertEqual(interpret_command("mang"), self.MANGO)
        self.assertEqual(interpret_command("start the mang"), self.MANGO)

    def test_accepts_word_splits_only_as_the_whole_utterance(self) -> None:
        for text in ("man go", "men go", "mang o", "m ango"):
            with self.subTest(text=text):
                self.assertEqual(interpret_command(text), self.MANGO)

    def test_accepts_mango_shaped_english_words_only_when_nothing_else_is_said(
        self,
    ) -> None:
        # As a bare utterance during an armed demo these are botched "mango"s.
        for text in ("manga", "mangy", "mange", "mongo"):
            with self.subTest(text=text):
                self.assertEqual(interpret_command(text), self.MANGO)

    def test_near_misses_survive_the_filler_and_qualifier_wrappers(self) -> None:
        for text in (
            "mengo run",
            "mengo demo",
            "one mengo",
            "single mangoe",
            "hey wendy mengo please",
            "ok run the mengo now",
        ):
            with self.subTest(text=text):
                self.assertEqual(interpret_command(text), self.MANGO)

    def test_near_misses_work_inside_the_operators_real_sentences(self) -> None:
        # These are the captured live phrasings with the word degraded.
        for text in (
            "go to the mengo",
            "can you go find the mengo?",
            "Can you go to the mangoe?",
            "find the mang",
            "please locate the mingo",
        ):
            with self.subTest(text=text):
                self.assertEqual(interpret_command(text), self.MANGO)

    def test_the_captured_live_utterances_still_match(self) -> None:
        for text in (
            "Go to the mango",
            "can you go find the mango?",
            "Can you go to the mango?",
        ):
            with self.subTest(text=text):
                self.assertEqual(interpret_command(text), self.MANGO)


class FuzzyMangoRejectionTests(unittest.TestCase):
    """The important half: ordinary speech must never walk the robot."""

    def test_rejects_onset_substituted_forms(self) -> None:
        # /m/ is the identity-bearing onset.  Accepting b-/n-/t-/d- onsets would
        # open the whole bingo/banjo/bongo/tango neighbourhood, so a dropped or
        # mangled onset is a deliberate false negative.
        for text in ("bango", "nango", "ango", "tango", "bingo", "dingo", "lingo"):
            with self.subTest(text=text):
                self.assertIsNone(interpret_command(text))

    def test_rejects_real_words_two_edits_away(self) -> None:
        for text in ("man", "many", "mangle", "mangled", "manage", "mandarin"):
            with self.subTest(text=text):
                self.assertIsNone(interpret_command(text))

    def test_rejects_words_that_only_share_a_phonetic_key(self) -> None:
        # Why Soundex/Metaphone alone was rejected: all of these key the same as
        # "mango" once the vowels are discarded.
        for text in ("monkey", "mink", "manic", "mongoose", "meaning"):
            with self.subTest(text=text):
                self.assertIsNone(interpret_command(text))

    def test_mango_shaped_english_words_do_not_fire_inside_a_sentence(self) -> None:
        for text in (
            "go and read the manga",
            "go look at that mangy dog",
            "can you go and find Margo",
            "go to the mongo database",
            "the dog has mange, go check",
        ):
            with self.subTest(text=text):
                self.assertIsNone(interpret_command(text))

    def test_similar_sounding_ordinary_speech_with_a_motion_verb_is_refused(
        self,
    ) -> None:
        for text in (
            "how many are there, go ahead",
            "can you go and manage the queue",
            "lets go play bingo",
            "go dance the tango",
            "go find the man",
            "go to the mangle in the corner",
            "search for the banjo",
            "seek out a dingo",
        ):
            with self.subTest(text=text):
                self.assertIsNone(interpret_command(text))

    def test_splits_are_not_rejoined_inside_a_sentence(self) -> None:
        # "man goes" concatenates to "mangoes"; re-joining is anchored-only so
        # ordinary sentences about a man going somewhere stay inert.
        for text in (
            "the man goes over there",
            "can you go and see where the man goes",
            "watch which way the man goes next",
        ):
            with self.subTest(text=text):
                self.assertIsNone(interpret_command(text))

    def test_a_near_miss_without_a_motion_verb_is_still_refused(self) -> None:
        for text in (
            "I like mengo",
            "the mengo is on the left",
            "put the mangoe down over there",
            "that mang over there is ours",
        ):
            with self.subTest(text=text):
                self.assertIsNone(interpret_command(text))

    def test_a_near_miss_never_disambiguates_a_two_fruit_utterance(self) -> None:
        for text in (
            "find an apple and a mengo",
            "is that a mangoe or a pear",
        ):
            with self.subTest(text=text):
                self.assertIsNone(interpret_command(text))

    def test_apple_and_pear_get_no_fuzzy_budget(self) -> None:
        # Deliberate: their edit-distance-1 neighbourhoods are full of ordinary
        # words ("ample", "apply", "pea", "peak", "peer", "par"), and no field
        # failure has been reported for either.
        for text in (
            "go to the ample space",
            "can you apply the brake and go",
            "go find the peer review",
            "go to the pea",
            "find the peak",
            "go over to par",
            "go and check the peas",
        ):
            with self.subTest(text=text):
                self.assertIsNone(interpret_command(text))
        # ...while the exact spellings and the existing "pair" homophone stay.
        self.assertEqual(
            interpret_command("locate the apple"),
            VoiceIntent(action="activate_demo", target_fruit="apple"),
        )
        self.assertEqual(
            interpret_command("go find a pair"),
            VoiceIntent(action="activate_demo", target_fruit="pear"),
        )


class FuzzyMangoDictionarySweepTests(unittest.TestCase):
    """Sweep a real dictionary so the false-positive surface stays measurable.

    Asserts the *invariant* rather than a fixed word list, so it survives a
    different dictionary but still fails loudly if the threshold is ever
    loosened (edit distance 2, or dropping the /m/ onset gate).
    """

    WORDS = "/usr/share/dict/words"

    def _dictionary(self) -> list[str]:
        if not os.path.exists(self.WORDS):
            self.skipTest(f"no dictionary at {self.WORDS}")
        with open(self.WORDS, encoding="utf-8", errors="ignore") as handle:
            return sorted({
                word.strip().lower() for word in handle if word.strip().isalpha()
            })

    def test_only_mango_shaped_words_can_trigger_a_run_inside_a_sentence(self) -> None:
        allowed_fruits = {"mango", "mangos", "mangoes", "apple", "apples",
                          "pear", "pears", "pair", "pairs"}
        offenders = []
        for word in self._dictionary():
            if interpret_command(f"go to the {word}") is None:
                continue
            if word in allowed_fruits:
                continue
            # Anything else that fires must be an /m/-onset, one-edit near miss.
            if word.startswith("m") and _edit_distance(word, "mango", budget=1) <= 1:
                continue
            offenders.append(word)
        self.assertEqual(offenders, [])

    def test_the_in_sentence_false_positive_surface_stays_tiny(self) -> None:
        dictionary = self._dictionary()
        firing = [
            word for word in dictionary
            if interpret_command(f"go to the {word}") is not None
        ]
        # 3 fruits + the "pair" homophone + a handful of obscure m-words.  If a
        # future change makes this balloon, the robot's exposure ballooned too.
        self.assertLess(len(firing), 20, firing)
        self.assertGreater(len(dictionary), 50_000, "dictionary looks truncated")

    def test_no_everyday_word_triggers_a_run(self) -> None:
        for word in (
            "man", "many", "mangle", "manage", "manager", "monkey", "money",
            "mink", "manic", "mandarin", "mongoose", "meaning", "morning",
            "bingo", "tango", "banjo", "dingo", "lingo", "mangled", "mangold",
            "ample", "apply", "pea", "peas", "peak", "peer", "par", "person",
        ):
            with self.subTest(word=word):
                self.assertIsNone(interpret_command(word))
                self.assertIsNone(interpret_command(f"go to the {word}"))
                self.assertIsNone(interpret_command(f"can you go and find the {word}"))


class FuzzyMangoDoesNotWeakenOtherCommandsTests(unittest.TestCase):
    def test_full_demo_still_wins_over_a_fuzzy_mango_in_the_same_utterance(
        self,
    ) -> None:
        for text in (
            "run the full demo and find the mengo",
            "full demo starting with mangoe",
            "after the full demo go to the mang",
        ):
            with self.subTest(text=text):
                self.assertIsNone(interpret_command(text))

    def test_full_demo_is_matched_before_any_mango_canonicalisation(self) -> None:
        self.assertEqual(interpret_command("full demo"), VoiceIntent("full_demo"))
        self.assertEqual(interpret_command("fulldemo"), VoiceIntent("full_demo"))
        self.assertEqual(
            interpret_command("hey wendy full demo please"), VoiceIntent("full_demo")
        )

    def test_stop_stays_exact_and_gets_no_fuzzy_budget(self) -> None:
        for text in ("stop", "stop demo", "stop the demo"):
            with self.subTest(text=text):
                self.assertEqual(interpret_command(text), VoiceIntent("stop_demo"))
        for text in ("stopp", "stap", "stop the mengo", "stop it"):
            with self.subTest(text=text):
                self.assertNotEqual(interpret_command(text), VoiceIntent("stop_demo"))

    def test_a_fuzzy_mango_never_hijacks_the_banana_refusal(self) -> None:
        self.assertIsNone(interpret_command("find the banana"))
        self.assertIsNone(interpret_command("bananas"))


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

    def test_a_fuzzy_near_miss_dispatches_a_real_mango_run(self) -> None:
        self.adapter.arm()
        result = self.adapter.dispatch(
            "go to the mengo", activation_id="voice-wake-mango-fuzzy-1"
        )
        self.assertEqual(
            _Handler.requests,
            [
                (
                    "/api/run",
                    {
                        "target_fruit": "mango",
                        "activation_source": "voice",
                        "activation_id": "voice-wake-mango-fuzzy-1",
                    },
                )
            ],
        )
        self.assertEqual(result["calls"][0]["tool"], "activate_demo")

    def test_a_mango_shaped_word_in_a_sentence_never_reaches_the_dog(self) -> None:
        self.adapter.arm()
        result = self.adapter.dispatch(
            "go and read the manga", activation_id="voice-wake-manga-1"
        )
        self.assertEqual(_Handler.requests, [])
        self.assertEqual(result["calls"], [])

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
