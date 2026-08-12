from __future__ import annotations

import pytest

from border_collie_demo.fruits import FruitPolicy, fruit_policy


def test_fruit_policy_rejects_inverted_acquisition_and_tracking_floors() -> None:
    with pytest.raises(
        ValueError,
        match="acquisition confidence must not be below tracking confidence",
    ):
        FruitPolicy(
            acquisition_confidence=0.50,
            close_range_tracking_confidence=0.55,
            motion_qualified=True,
        )


def test_apple_experiment_does_not_change_pear_or_banana_policy() -> None:
    assert fruit_policy("apple") == FruitPolicy(0.50, 0.50, True)
    assert fruit_policy("pear") == FruitPolicy(0.65, 0.55, True)
    assert fruit_policy("banana") == FruitPolicy(0.20, 0.20, True)
