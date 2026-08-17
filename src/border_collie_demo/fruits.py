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
    # 0.40 -> 0.30 on 2026-08-17. Apple was the clearest case for this: run
    # a22eaba1 held a real apple in view for 234 detections peaking at 0.815 and
    # still failed, because the sustained value sat at 0.23-0.37 just under the
    # 0.40 bar; 4da2a662 and 55eefa3d peaked at 0.358 and 0.379.
    "apple": FruitPolicy(
        acquisition_confidence=0.30,
        close_range_tracking_confidence=0.10,
        motion_qualified=True,
        focus_confidence=0.30,
    ),
    # Banana is motion-qualified only because every published banana detection
    # is already gated by the resident banana specialist (0.55 confidence with
    # IoU agreement against the general proposal) in the media router. The low
    # acquisition threshold here rides on top of that specialist floor.
    "banana": FruitPolicy(
        acquisition_confidence=0.20,
        close_range_tracking_confidence=0.20,
        motion_qualified=False,
    ),
    # Mango is a first-class class in the general model, so it carries its own
    # measured thresholds rather than the old derived-route floors.
    #
    # 0.65 was copied from pear on the strength of mango reading 0.88-0.92 when
    # well presented. Three physical runs then failed TARGET_RECOGNITION_FAILURE
    # with the mango plainly in frame: 45482f40, 049481c6 and a397090b each swept
    # a full 2pi, produced 23-37 detections, and scored a median of 0.26-0.50
    # with a peak of 0.69. So the weak-presentation regime sits at 0.25-0.60 and
    # 0.65 cut through the middle of the real distribution instead of below it.
    #
    # 0.45 clears all three. The frozen 40-frame MANGO-EVAL-001 clip puts the
    # worst false positive at 0.209, so this still keeps better than 2x headroom
    # over measured noise. Tracking stays 0.10 below acquisition, matching pear's
    # spacing, so a lock is easier to keep than to win.
    # 0.45 -> 0.30 on 2026-08-17: mango failed a full sweep at 0.349 (512d8790)
    # on the same stage, and only one sample in three failed mango sweeps ever
    # reached 0.30, so the noise headroom holds.
    "mango": FruitPolicy(
        acquisition_confidence=0.30,
        close_range_tracking_confidence=0.35,
        motion_qualified=True,
        focus_confidence=0.30,
    ),
    # Pear kept 0.65 from when it measured 0.957 on the reference frame. On the
    # 2026-08-17 stage that bar is unreachable. With the robot STATIONARY and the
    # pear plainly in frame, the live preview read 0.36, 0.36, 0.36, 0.46, 0.46
    # and 0.59 — so the pear could not lock even with no motion, no blur and the
    # fruit centred. During a sweep it only reaches 0.086-0.248.
    #
    # False-positive exposure at 0.30 is measured, not assumed: across eight
    # failed pear sweeps (95 samples each) not one sample ever reached 0.30, and
    # the highest pear confidence recorded in any failing sweep was 0.248. With
    # required_frames=3 on top, 0.30 has real headroom over observed noise.
    #
    # This makes the pear lockable once the dog is settled. It does NOT rescue
    # the sweep, where the pear tops out at 0.248 — that needs better frames,
    # not a lower bar.
    "pear": FruitPolicy(
        acquisition_confidence=0.30,
        close_range_tracking_confidence=0.55,
        motion_qualified=True,
        focus_confidence=0.30,
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
        minimum=0.30,
        maximum=0.70,
    )
    assert policy.focus_confidence is not None
    focus = _bounded_confidence_env(
        "BORDER_COLLIE_APPLE_FOCUS_CONFIDENCE",
        policy.focus_confidence,
        minimum=0.30,
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


def mango_color_confidence() -> float:
    """Return the defense-in-depth floor for derived Mango identity evidence."""
    return _bounded_confidence_env(
        "BORDER_COLLIE_MANGO_COLOR_CONFIDENCE",
        0.80,
        minimum=0.65,
        maximum=1.0,
    )


def mango_raw_confidence() -> float:
    """Return the strict raw COCO bowl floor for every Mango observation."""
    return _bounded_confidence_env(
        "BORDER_COLLIE_MANGO_RAW_CONFIDENCE",
        0.08,
        minimum=0.01,
        maximum=0.20,
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
