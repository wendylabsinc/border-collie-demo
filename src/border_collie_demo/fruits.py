from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FruitPolicy:
    acquisition_confidence: float
    close_range_tracking_confidence: float
    search_slowdown_confidence: float
    motion_qualified: bool


FRUIT_POLICIES: dict[str, FruitPolicy] = {
    "apple": FruitPolicy(
        acquisition_confidence=0.70,
        close_range_tracking_confidence=0.10,
        search_slowdown_confidence=0.50,
        motion_qualified=True,
    ),
    "banana": FruitPolicy(
        acquisition_confidence=0.20,
        close_range_tracking_confidence=0.20,
        search_slowdown_confidence=0.55,
        motion_qualified=True,
    ),
    "pear": FruitPolicy(
        acquisition_confidence=0.65,
        close_range_tracking_confidence=0.55,
        search_slowdown_confidence=0.50,
        motion_qualified=True,
    ),
}

SUPPORTED_FRUITS = tuple(FRUIT_POLICIES)
QUALIFIED_FRUITS = tuple(
    fruit for fruit, policy in FRUIT_POLICIES.items() if policy.motion_qualified
)


def fruit_policy(target_fruit: str) -> FruitPolicy:
    normalized = target_fruit.casefold().strip()
    try:
        return FRUIT_POLICIES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported Target Fruit: {target_fruit}") from exc
