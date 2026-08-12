"""Pure transcript cleanup and deterministic command interpretation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VoiceIntent:
    name: str
    slots: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


_WAKE_PREFIX = re.compile(r"^(?:hey[ ,]+)?(?:unitree|go[ -]?2|robot)[, :!-]*", re.I)
_DISTANCE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(meters?|metres?|m|feet|foot|ft)\b", re.I)
_ANGLE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(degrees?|deg)\b", re.I)


def normalize_transcript(text: str) -> str:
    """Canonicalize Unicode/spacing while retaining human-readable punctuation."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(character for character in text if character.isprintable())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_intent(transcript: str) -> VoiceIntent:
    """Extract a small, auditable Go2 command vocabulary.

    Unknown speech is deliberately retained as ``unknown``; downstream planning
    can decide whether an LLM should interpret it without hiding this producer's
    uncertainty.
    """
    normalized = normalize_transcript(transcript)
    command = _WAKE_PREFIX.sub("", normalized).strip(" .!?").casefold()
    if not command:
        return VoiceIntent("unknown", {"utterance": normalized}, 0.0)

    if re.search(r"\b(emergency stop|stop|halt|freeze)\b", command):
        return VoiceIntent("stop", {"emergency": "emergency" in command}, 1.0)
    if re.search(r"\b(sit|sit down)\b", command):
        return VoiceIntent("posture", {"posture": "sit"}, 0.98)
    if re.search(r"\b(stand|stand up)\b", command):
        return VoiceIntent("posture", {"posture": "stand"}, 0.98)
    if re.search(r"\b(lie down|lay down)\b", command):
        return VoiceIntent("posture", {"posture": "lie"}, 0.96)

    turn = re.search(r"\bturn\s+(left|right)\b", command)
    if turn:
        slots: dict[str, Any] = {"direction": turn.group(1)}
        angle = _ANGLE.search(command)
        if angle:
            slots["angle_degrees"] = float(angle.group(1))
        return VoiceIntent("turn", slots, 0.96)

    move = re.search(r"\b(?:go|move|walk)\s+(forward|backward|back|left|right)\b", command)
    if move:
        direction = "backward" if move.group(1) == "back" else move.group(1)
        slots = {"direction": direction}
        distance = _DISTANCE.search(command)
        if distance:
            value = float(distance.group(1))
            if distance.group(2).casefold() in {"feet", "foot", "ft"}:
                value *= 0.3048
            slots["distance_m"] = round(value, 3)
        return VoiceIntent("move", slots, 0.94)

    target = re.search(r"\b(?:find|locate|look for|search for)\s+(?:the\s+)?(.+)$", command)
    if target:
        return VoiceIntent("find", {"target": target.group(1).strip(" .!?")}, 0.9)
    target = re.search(r"\b(?:go|move|walk)\s+to\s+(?:the\s+)?(.+)$", command)
    if target:
        return VoiceIntent("find", {"target": target.group(1).strip(" .!?")}, 0.9)
    if re.search(r"\b(?:come here|come to me)\b", command):
        return VoiceIntent("come", {}, 0.9)
    if re.search(r"\bfollow\s+(?:me|the speaker)\b", command):
        return VoiceIntent("follow", {"target": "speaker"}, 0.9)

    return VoiceIntent("unknown", {"utterance": normalized}, 0.0)
