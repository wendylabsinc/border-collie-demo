from __future__ import annotations

import json
from pathlib import Path

import pytest

from border_collie_demo.search_experiment import (
    SearchExperimentTuning,
    search_experiment_evidence,
    search_experiment_scorecard,
)


@pytest.mark.parametrize(
    ("fruit", "focus", "lock"),
    [
        ("apple", 0.40, 0.40),
        ("pear", None, 0.65),
        ("banana", None, 0.20),
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
        "center_tolerance_ratio": 0.05,
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
                "run_tuning": {"search": {"bearing_routing_enabled": False}},
                "stage_results": {"turn_to_fruit": {"acquisition_epoch": 1}},
                "outcome": "COMPLETED",
                "reason": "SUCCESS",
            },
            {
                "target_fruit": "apple",
                "search_experiment": tuning,
                "run_tuning": {"search": {"bearing_routing_enabled": False}},
                "stage_results": {},
                "outcome": "FAILED",
                "reason": "TARGET_RECOGNITION_FAILURE",
            },
        ]
    )

    assert scorecard == {
        "schema_version": 2,
        "experiments": search_experiment_evidence()["experiments"],
        "note": (
            "Only terminal runs are counted. Routing is NOT_RECORDED for runs "
            "created before the per-run routing flag was persisted."
        ),
        "non_terminal_runs_excluded": 0,
        "cohorts": [
            {
                "target_fruit": "apple",
                "search_yaw_rps": 0.45,
                "bearing_routing": "OFF",
                "runs": 2,
                "locks": 1,
                "successes": 1,
                "lock_rate_percent": 50.0,
                "success_rate_percent": 50.0,
            }
        ]
    }


def test_search_experiment_scorecard_separates_routing_and_excludes_unfinished() -> None:
    tuning = SearchExperimentTuning(target_fruit="banana").to_dict()
    runs = []
    for enabled in (False, True):
        runs.append(
            {
                "target_fruit": "banana",
                "search_experiment": tuning,
                "run_tuning": {"search": {"bearing_routing_enabled": enabled}},
                "stage_results": {"turn_to_fruit": {}},
                "outcome": "COMPLETED",
                "reason": "SUCCESS",
            }
        )
    runs.append(
        {
            "target_fruit": "banana",
            "search_experiment": tuning,
            "run_tuning": {"search": {"bearing_routing_enabled": True}},
            "outcome": None,
        }
    )

    scorecard = search_experiment_scorecard(runs)

    assert [item["bearing_routing"] for item in scorecard["cohorts"]] == [
        "OFF",
        "ON",
    ]
    assert scorecard["non_terminal_runs_excluded"] == 1


def test_checked_in_experiment_index_matches_physical_result_artifacts() -> None:
    repository = Path(__file__).resolve().parents[1]
    evidence = search_experiment_evidence()["experiments"]
    by_id = {item["experiment_id"]: item for item in evidence}

    routing = json.loads(
        (repository / by_id["bearing-routing-flag-ab"]["evidence_paths"][0])
        .read_text(encoding="utf-8")
    )
    assert routing["experiment"]["name"] == "bearing-routing flag A/B"
    assert routing["experiment"]["target_fruit"] == "banana"
    assert by_id["bearing-routing-flag-ab"]["successful_runs"] == sum(
        result["outcome"] == "COMPLETED" and result["reason"] == "SUCCESS"
        for result in (routing["baseline"], routing["treatment"])
    )

    for experiment_id in (
        "stationary-jump-random-five",
        "final-demo-random-ten",
        "final-demo-second-random-ten",
    ):
        experiment = by_id[experiment_id]
        runs = []
        for relative_path in experiment["evidence_paths"]:
            artifact = json.loads(
                (repository / relative_path).read_text(encoding="utf-8")
            )
            assert artifact["policy"]["randomized"] is True
            runs.extend(artifact["runs"])
        assert experiment["attempted_runs"] == len(runs)
        assert experiment["successful_runs"] == sum(
            run["outcome"] == "COMPLETED" and run["reason"] == "SUCCESS"
            for run in runs
        )

    planned = by_id["bearing-routing-on-random-ten"]
    assert planned["status"] == "NOT_RUN"
    assert planned["attempted_runs"] == 0
    assert planned["evidence_paths"] == []
