"""Shared fruit-target vocabulary used across perception and missions."""

from __future__ import annotations


SUPPORTED_FRUITS = frozenset({"apple", "banana", "grapes", "orange", "pear"})
DEFAULT_FRUIT = "apple"


def normalize_fruit(value: object) -> str | None:
    """Return a supported, normalized fruit name or ``None``."""

    normalized = str(value).strip().casefold()
    return normalized if normalized in SUPPORTED_FRUITS else None

