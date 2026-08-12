from __future__ import annotations

import unittest

from wake_phrase import extract_wake_command


class WakePhraseTests(unittest.TestCase):
    def test_extracts_same_utterance_command(self) -> None:
        self.assertEqual(
            extract_wake_command("Hey Walker, go find the banana", "hey walker"),
            "go find the banana",
        )

    def test_accepts_punctuation_between_wake_words(self) -> None:
        self.assertEqual(
            extract_wake_command("hey, walker... banana", "hey walker"),
            "banana",
        )

    def test_wake_phrase_alone_opens_a_followup_window(self) -> None:
        self.assertEqual(extract_wake_command("hey walker", "hey walker"), "")

    def test_rejects_nonleading_or_different_phrase(self) -> None:
        self.assertIsNone(
            extract_wake_command("go find banana hey walker", "hey walker")
        )
        self.assertIsNone(extract_wake_command("hey wendy banana", "hey walker"))


if __name__ == "__main__":
    unittest.main()
