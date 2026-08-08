from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FruitPolicy:
    acquisition_confidence: float
    close_range_tracking_confidence: float
    motion_qualified: bool


FRUIT_POLICIES: dict[str, FruitPolicy] = {
    "apple": FruitPolicy(
        acquisition_confidence=0.70,
        close_range_tracking_confidence=0.10,
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
    # Pear confidence collapses when the fruit fills the frame at arrival
    # distance. Supervised run d740a5f2 (2026-08-08) qualified a pear at
    # 0.57-0.89 confidence, then measured 0.2658 at bbox bottom 0.9972 —
    # a real pear clipping the frame — while the old 0.55 close-range value
    # (identical to the normal tracking floor, so zero relief) rejected all
    # 18 bridgeable sub-floor frames; the near gate starved at zero
    # confirmations and Arrival timed out. 0.20 accepts the observed
    # collapse with margin while staying an order of magnitude above every
    # recorded phantom detection (0.010-0.024). The spatial-continuity
    # guards, not this floor, remain the real close-range gate.
    "pear": FruitPolicy(
        acquisition_confidence=0.65,
        close_range_tracking_confidence=0.20,
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
