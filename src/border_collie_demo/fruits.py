from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class FruitPolicy:
    acquisition_confidence: float
    close_range_tracking_confidence: float
    motion_qualified: bool
    focus_confidence: float | None = None


FRUIT_POLICIES: dict[str, FruitPolicy] = {
    "apple": FruitPolicy(
        acquisition_confidence=0.40,
        close_range_tracking_confidence=0.10,
        motion_qualified=True,
        focus_confidence=0.50,
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
        policy = FRUIT_POLICIES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported Target Fruit: {target_fruit}") from exc
    if normalized != "apple":
        return policy
    acquisition = _bounded_confidence_env(
        "BORDER_COLLIE_APPLE_ACQUISITION_CONFIDENCE",
        policy.acquisition_confidence,
        minimum=0.40,
        maximum=0.70,
    )
    assert policy.focus_confidence is not None
    focus = _bounded_confidence_env(
        "BORDER_COLLIE_APPLE_FOCUS_CONFIDENCE",
        policy.focus_confidence,
        minimum=0.50,
        maximum=0.70,
    )
    if focus < acquisition:
        raise ValueError(
            "BORDER_COLLIE_APPLE_FOCUS_CONFIDENCE must be greater than or equal "
            "to BORDER_COLLIE_APPLE_ACQUISITION_CONFIDENCE"
        )
    return FruitPolicy(
        acquisition_confidence=acquisition,
        close_range_tracking_confidence=policy.close_range_tracking_confidence,
        motion_qualified=policy.motion_qualified,
        focus_confidence=focus,
    )


def _bounded_confidence_env(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must stay within {minimum:.2f}..{maximum:.2f}")
    return value
