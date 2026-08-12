from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FruitPolicy:
    acquisition_confidence: float
    close_range_tracking_confidence: float
    motion_qualified: bool


FRUIT_POLICIES: dict[str, FruitPolicy] = {
    "apple": FruitPolicy(
        acquisition_confidence=0.65,
        close_range_tracking_confidence=0.55,
        motion_qualified=True,
    ),
    # Banana is motion-qualified only because every published banana detection
    # is already gated by the resident banana specialist (0.55 confidence with
    # IoU agreement against the general proposal) in the media router. The low
    # acquisition threshold here rides on top of that specialist floor.
    "banana": FruitPolicy(
        acquisition_confidence=0.20,
        close_range_tracking_confidence=0.20,
        motion_qualified=True,
    ),
    "pear": FruitPolicy(
        acquisition_confidence=0.65,
        close_range_tracking_confidence=0.55,
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
