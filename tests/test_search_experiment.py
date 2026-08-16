from __future__ import annotations

import pytest

from border_collie_demo.search_experiment import (
    SearchExperimentTuning,
    search_experiment_scorecard,
)


@pytest.mark.parametrize(
    ("fruit", "focus", "lock"),
    [
        ("apple", 0.40, 0.40),
        ("pear", None, 0.65),
        ("banana", None, 0.20),
        ("mango", None, 0.65),
    ],
)
def test_search_experiment_tuning_preserves_each_fruits_baseline_policy(
    fruit: str,
    focus: float | None,
    lock: float,
) -> None:
    tuning = SearchExperimentTuning.defaults(fruit)

    assert tuning.to_dict() == {
        "target_fruit": fruit,
        "search_yaw_rps": 0.40,
        "focus_confidence": focus,
        "lock_confidence": lock,
        "center_confirmations": 3,
        "center_tolerance_ratio": 0.08,
    }


def test_search_experiment_tuning_rejects_unsafe_or_inverted_values() -> None:
    with pytest.raises(ValueError, match="search_yaw_rps"):
        SearchExperimentTuning(target_fruit="apple", search_yaw_rps=0.30)
    with pytest.raises(ValueError, match="focus confidence"):
        SearchExperimentTuning(
            target_fruit="apple",
            focus_confidence=0.50,
            lock_confidence=0.55,
        )
    with pytest.raises(ValueError, match="Pear lock confidence"):
        SearchExperimentTuning(
            target_fruit="pear",
            focus_confidence=0.65,
            lock_confidence=0.64,
        )
    with pytest.raises(ValueError, match="finite"):
        SearchExperimentTuning(
            target_fruit="banana",
            focus_confidence=float("nan"),
            lock_confidence=0.20,
        )


def test_search_experiment_rejects_a_persisted_target_mismatch() -> None:
    with pytest.raises(ValueError, match="Target Fruit"):
        SearchExperimentTuning.from_mapping(
            "pear",
            {
                "target_fruit": "apple",
                "focus_confidence": 0.65,
                "lock_confidence": 0.65,
            },
        )


def test_search_experiment_scorecard_groups_lock_and_success_by_fruit_and_yaw() -> None:
    tuning = SearchExperimentTuning(
        target_fruit="apple",
        search_yaw_rps=0.45,
    ).to_dict()
    scorecard = search_experiment_scorecard(
        [
            {
                "target_fruit": "apple",
                "search_experiment": tuning,
                "stage_results": {"turn_to_fruit": {"acquisition_epoch": 1}},
                "outcome": "COMPLETED",
                "reason": "SUCCESS",
            },
            {
                "target_fruit": "apple",
                "search_experiment": tuning,
                "stage_results": {},
                "outcome": "FAILED",
                "reason": "TARGET_RECOGNITION_FAILURE",
            },
        ]
    )

    assert scorecard == {
        "cohorts": [
            {
                "target_fruit": "apple",
                "search_yaw_rps": 0.45,
                "runs": 2,
                "locks": 1,
                "successes": 1,
                "lock_rate_percent": 50.0,
                "success_rate_percent": 50.0,
            }
        ]
    }
