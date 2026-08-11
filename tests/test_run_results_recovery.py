import pytest

from border_collie_demo.run_results import ActiveRunError, RunResultStore


def test_one_disarmed_failed_recovery_can_be_corrected_once(tmp_path) -> None:
    results = RunResultStore(tmp_path)
    run = results.start_run(target_fruit="pear", activation_source="audience_ui")
    run_id = run["run_id"]
    results.seal(
        run_id,
        phase="failed",
        outcome="FAILED",
        reason="ARRIVAL_FAILURE",
        message="arrival failed",
        final_safety_state="DISARMED_CONFIRMED",
        failed_phase="approach_fruit",
    )

    first = results.start_recovery(run_id, confirmation="confirmed")
    results.seal_recovery(
        run_id,
        first["recovery_id"],
        outcome="FAILED",
        reason="RECOVERY_FAILURE",
        message="pulse reserve exhausted",
        final_safety_state="DISARMED_CONFIRMED",
        final_evidence={},
    )
    second = results.start_recovery(run_id, confirmation="confirmed")

    assert first["attempt_number"] == 1
    assert second["attempt_number"] == 2

    results.seal_recovery(
        run_id,
        second["recovery_id"],
        outcome="FAILED",
        reason="RECOVERY_FAILURE",
        message="correction failed",
        final_safety_state="DISARMED_CONFIRMED",
        final_evidence={},
    )
    with pytest.raises(ActiveRunError, match="exhausted"):
        results.start_recovery(run_id, confirmation="confirmed")


def test_unconfirmed_failed_recovery_cannot_be_retried(tmp_path) -> None:
    results = RunResultStore(tmp_path)
    run = results.start_run(target_fruit="pear", activation_source="audience_ui")
    run_id = run["run_id"]
    results.seal(
        run_id,
        phase="failed",
        outcome="FAILED",
        reason="ARRIVAL_FAILURE",
        message="arrival failed",
        final_safety_state="DISARMED_CONFIRMED",
        failed_phase="approach_fruit",
    )
    attempt = results.start_recovery(run_id, confirmation="confirmed")
    results.seal_recovery(
        run_id,
        attempt["recovery_id"],
        outcome="FAILED",
        reason="RECOVERY_FAILURE",
        message="stop could not be confirmed",
        final_safety_state="STOP_REQUESTED_UNCONFIRMED",
        final_evidence={},
    )

    with pytest.raises(ActiveRunError, match="non-retryable"):
        results.start_recovery(run_id, confirmation="confirmed")
