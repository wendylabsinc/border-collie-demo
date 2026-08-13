from __future__ import annotations

import unittest

from wake_phrase import extract_wake_command


class WakePhraseTests(unittest.TestCase):
    def test_extracts_same_utterance_command(self) -> None:
        self.assertEqual(
            extract_wake_command("Hey Wendy, go find the banana", "hey wendy"),
            "go find the banana",
        )

    def test_accepts_punctuation_between_wake_words(self) -> None:
        self.assertEqual(
            extract_wake_command("hey, wendy... banana", "hey wendy"),
            "banana",
        )

    def test_wake_phrase_alone_opens_a_followup_window(self) -> None:
        self.assertEqual(extract_wake_command("hey wendy", "hey wendy"), "")

    def test_rejects_nonleading_or_different_phrase(self) -> None:
        self.assertIsNone(
            extract_wake_command("go find banana hey wendy", "hey wendy")
        )
        self.assertIsNone(extract_wake_command("hey walter banana", "hey wendy"))


if __name__ == "__main__":
    unittest.main()
