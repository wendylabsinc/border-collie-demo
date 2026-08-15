from __future__ import annotations

import pytest

from border_collie_demo.cohort_policy import (
    CohortPolicy,
    FailureSelector,
    choose_fruit_sequence,
    decide_terminal_run,
    evaluate_home_clearance,
)


def failed_run(
    *,
    reason: str = "ACTION_FAILURE",
    failed_phase: str = "sit_and_bark",
    safety: str = "DISARMED_CONFIRMED",
) -> dict[str, object]:
    return {
        "outcome": "FAILED",
        "reason": reason,
        "failed_phase": failed_phase,
        "final_safety_state": safety,
    }


def test_policy_defaults_to_five_randomized_runs_and_stops_on_failure() -> None:
    policy = CohortPolicy(seed=20260813)

    assert policy.to_dict() == {
        "runs": 5,
        "randomized": True,
        "target_fruit": None,
        "fruit_subset": None,
        "seed": 20260813,
        "tolerated_failures": [],
    }
    assert decide_terminal_run(failed_run(), policy) == {
        "action": "STOP_COHORT",
        "code": "FAILURE_POLICY_STOP",
        "detail": "ACTION_FAILURE in sit_and_bark is configured to stop the cohort",
        "matched_selector": None,
    }


def test_failure_can_be_tolerated_by_reason_or_failed_phase() -> None:
    by_reason = CohortPolicy(
        seed=7,
        tolerated_failures=(FailureSelector(reason="ACTION_FAILURE"),),
    )
    by_phase = CohortPolicy(
        seed=7,
        tolerated_failures=(FailureSelector(failed_phase="sit_and_bark"),),
    )

    expected = {
        "action": "AWAIT_HOME_CLEARANCE",
        "code": "FAILURE_TOLERATED",
        "detail": "ACTION_FAILURE in sit_and_bark is tolerated only for the cohort boundary",
        "matched_selector": {"failed_phase": None, "reason": "ACTION_FAILURE"},
    }
    assert decide_terminal_run(failed_run(), by_reason) == expected
    assert decide_terminal_run(failed_run(), by_phase)["action"] == "AWAIT_HOME_CLEARANCE"
    assert decide_terminal_run(failed_run(), by_phase)["matched_selector"] == {
        "failed_phase": "sit_and_bark",
        "reason": None,
    }


@pytest.mark.parametrize(
    ("reason", "phase"),
    [
        ("CAMERA_FAILURE", "find_fruit"),
        ("MOTION_FAILURE", "approach_fruit"),
        ("REMOTE_TAKEOVER", "remote_takeover"),
        ("PROCESS_INTERRUPTED", "failed"),
        ("RETURN_HOME_FAILURE", "return_home"),
        ("PREFLIGHT_FAILURE", "capture_home"),
    ],
)
def test_safety_failures_override_tolerance(reason: str, phase: str) -> None:
    policy = CohortPolicy(
        seed=7,
        tolerated_failures=(FailureSelector(reason=reason),),
    )

    decision = decide_terminal_run(
        failed_run(reason=reason, failed_phase=phase), policy
    )

    assert decision["action"] == "STOP_COHORT"
    assert decision["code"] == "HARD_SAFETY_STOP"
    assert decision["matched_selector"] is None


def test_unconfirmed_disarm_overrides_tolerance() -> None:
    policy = CohortPolicy(
        seed=7,
        tolerated_failures=(FailureSelector(reason="ACTION_FAILURE"),),
    )

    decision = decide_terminal_run(
        failed_run(safety="STOP_REQUESTED_UNCONFIRMED"), policy
    )

    assert decision["code"] == "HARD_SAFETY_STOP"


def test_completed_run_with_unconfirmed_disarm_is_still_a_hard_stop() -> None:
    policy = CohortPolicy(seed=7)

    decision = decide_terminal_run(
        {
            "outcome": "COMPLETED",
            "reason": "SUCCESS",
            "failed_phase": None,
            "final_safety_state": "STOP_REQUESTED_UNCONFIRMED",
        },
        policy,
    )

    assert decision["code"] == "HARD_SAFETY_STOP"


def test_structured_motion_blocker_overrides_a_tolerated_arrival_failure() -> None:
    policy = CohortPolicy(
        seed=7,
        tolerated_failures=(FailureSelector(reason="ARRIVAL_FAILURE"),),
    )
    run = failed_run(reason="ARRIVAL_FAILURE", failed_phase="approach_fruit")
    run["failure_details"] = {"safety_class": "motion"}

    assert decide_terminal_run(run, policy)["code"] == "HARD_SAFETY_STOP"


def test_seeded_random_sequence_and_fixed_fruit_have_exact_run_count() -> None:
    random_policy = CohortPolicy(runs=5, randomized=True, seed=19)
    fixed_policy = CohortPolicy(
        runs=4, randomized=False, target_fruit="banana", seed=19
    )

    first = choose_fruit_sequence(random_policy, ["pear", "apple", "banana"])
    second = choose_fruit_sequence(random_policy, ["banana", "pear", "apple"])

    assert first == second
    assert len(first) == 5
    assert set(first) == {"apple", "banana", "pear"}
    assert choose_fruit_sequence(fixed_policy, ["pear", "banana"]) == [
        "banana",
        "banana",
        "banana",
        "banana",
    ]


def test_seeded_random_sequence_is_restricted_to_the_persisted_fruit_subset() -> None:
    policy = CohortPolicy(
        runs=5,
        randomized=True,
        seed=919,
        fruit_subset=("pear", "apple"),
    )

    sequence = choose_fruit_sequence(policy, ["apple", "banana", "pear"])

    assert policy.to_dict()["fruit_subset"] == ["apple", "pear"]
    assert sequence == ["pear", "apple", "pear", "pear", "apple"]
    assert set(sequence) == {"apple", "pear"}


def test_randomized_fruit_subset_rejects_empty_and_unqualified_values() -> None:
    with pytest.raises(ValueError, match="fruit_subset must not be empty"):
        CohortPolicy(runs=2, randomized=True, seed=1, fruit_subset=())

    policy = CohortPolicy(
        runs=2,
        randomized=True,
        seed=1,
        fruit_subset=("apple", "mango"),
    )
    with pytest.raises(ValueError, match="not qualified"):
        choose_fruit_sequence(policy, ["apple", "banana", "pear"])


def test_fixed_policy_requires_a_qualified_target() -> None:
    with pytest.raises(ValueError, match="target_fruit is required"):
        CohortPolicy(runs=2, randomized=False, seed=1)

    policy = CohortPolicy(
        runs=2, randomized=False, target_fruit="apple", seed=1
    )
    with pytest.raises(ValueError, match="not qualified"):
        choose_fruit_sequence(policy, ["pear"])


def test_home_clearance_requires_exact_run_zero_disarm_and_no_restart_latch() -> None:
    status = {
        "mission": {"restart_required": False},
        "active_run_id": None,
        "hardware": {
            "motion": {
                "armed": False,
                "last_command": {"forward_mps": 0.0, "yaw_rps": 0.0},
            }
        },
        "activation": {
            "ready": True,
            "inter_run": {"prior_run_id": "run-1", "returned_home": True},
        },
    }

    assert evaluate_home_clearance(status, "run-1")["safe_to_continue"] is True
    assert evaluate_home_clearance(status, "run-2")["safe_to_continue"] is False
    status["mission"]["restart_required"] = True
    assert evaluate_home_clearance(status, "run-1")["safe_to_continue"] is False
    status["mission"]["restart_required"] = False
    status["hardware"]["motion"]["last_command"]["yaw_rps"] = 0.01
    assert evaluate_home_clearance(status, "run-1")["safe_to_continue"] is False
