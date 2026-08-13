"""Text-level wake phrase matching for always-listening local ASR."""

from __future__ import annotations

import re
from typing import Optional


def extract_wake_command(text: str, phrase: str) -> Optional[str]:
    """Return text after a leading wake phrase, or ``None`` when absent."""
    words = [
        word
        for word in re.split(r"[^a-z0-9]+", phrase.lower())
        if word and not word.isdigit()
    ]
    if not words:
        return None
    pattern = r"^\W*" + r"\W+".join(re.escape(word) for word in words) + r"\W*"
    match = re.match(pattern, text.strip(), flags=re.IGNORECASE)
    if match is None:
        return None
    return text.strip()[match.end():].strip()
