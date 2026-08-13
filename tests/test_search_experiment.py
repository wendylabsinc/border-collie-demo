from __future__ import annotations

import pytest

from border_collie_demo.search_experiment import (
    SearchExperimentTuning,
    search_experiment_scorecard,
)


def test_search_experiment_tuning_has_bounded_stage_defaults() -> None:
    tuning = SearchExperimentTuning.defaults()

    assert tuning.to_dict() == {
        "search_yaw_rps": 0.40,
        "apple_focus_confidence": 0.50,
        "apple_acquisition_confidence": 0.40,
        "center_confirmations": 3,
        "center_tolerance_ratio": 0.08,
    }


def test_search_experiment_tuning_rejects_unsafe_or_inverted_values() -> None:
    with pytest.raises(ValueError, match="search_yaw_rps"):
        SearchExperimentTuning(search_yaw_rps=0.30)
    with pytest.raises(ValueError, match="focus confidence"):
        SearchExperimentTuning(
            apple_focus_confidence=0.50,
            apple_acquisition_confidence=0.55,
        )


def test_search_experiment_scorecard_groups_lock_and_success_by_fruit_and_yaw() -> None:
    tuning = SearchExperimentTuning(search_yaw_rps=0.45).to_dict()
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
