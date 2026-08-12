from __future__ import annotations

import pytest

from scripts.compare_search_policies import compare_sessions


def session(
    policy: str,
    outcomes: list[tuple[str, str, str | None, float | None]],
) -> dict[str, object]:
    runs = []
    for number, (fruit, outcome, reason, home) in enumerate(outcomes, 1):
        runs.append(
            {
                "number": number,
                "target_fruit": fruit,
                "outcome": outcome,
                "reason": reason or "SUCCESS",
                "failed_phase": None if outcome == "COMPLETED" else "turn_to_fruit",
                "home_distance_m": home,
                "network": {"poll_errors": 1 if number == 2 else 0},
                "stage_durations": {"turn_to_fruit": float(number)},
                "recovery": (
                    {"outcome": "COMPLETED"} if outcome != "COMPLETED" else None
                ),
            }
        )
    return {
        "search_policy": policy,
        "seed": 19,
        "fruit_sequence": [item[0] for item in outcomes],
        "orientation_sequence_degrees": [0] * len(outcomes),
        "runs": runs,
    }


def test_compare_sessions_reports_rates_and_failure_breakdowns() -> None:
    fruit_runs = [
        ("apple", "COMPLETED", None, 0.04),
        ("pear", "FAILED", "TARGET_RECOGNITION_FAILURE", 0.08),
    ]
    comparison = compare_sessions(
        [
            session("fast-lock", fruit_runs),
            session(
                "slow-sweep",
                [
                    ("apple", "FAILED", "TARGET_RECOGNITION_FAILURE", 0.06),
                    ("pear", "FAILED", "RETURN_HOME_FAILURE", 0.09),
                ],
            ),
            session(
                "double-back",
                [
                    ("apple", "COMPLETED", None, 0.03),
                    ("pear", "COMPLETED", None, 0.05),
                ],
            ),
        ]
    )

    assert comparison["trial"] == {
        "seed": 19,
        "runs_per_policy": 2,
        "fruit_sequence": ["apple", "pear"],
        "orientation_sequence_degrees": [0, 0],
    }
    assert comparison["profiles"]["fast-lock"]["completion_rate"] == 0.5
    assert comparison["profiles"]["slow-sweep"]["failures_by_reason"] == {
        "RETURN_HOME_FAILURE": 1,
        "TARGET_RECOGNITION_FAILURE": 1,
    }
    assert comparison["profiles"]["double-back"]["completion_rate"] == 1.0
    assert comparison["profiles"]["fast-lock"]["recovery_success_rate"] == 1.0
    assert comparison["profiles"]["double-back"]["network_poll_errors"] == 1
    assert comparison["ranking"][0]["search_policy"] == "double-back"


def test_compare_sessions_rejects_nonidentical_trial_sequences() -> None:
    first = session(
        "fast-lock",
        [("apple", "COMPLETED", None, 0.04)],
    )
    second = session(
        "slow-sweep",
        [("pear", "COMPLETED", None, 0.04)],
    )
    third = session(
        "double-back",
        [("apple", "COMPLETED", None, 0.04)],
    )

    with pytest.raises(ValueError, match="identical fruit and orientation sequences"):
        compare_sessions([first, second, third])
