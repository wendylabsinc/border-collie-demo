from __future__ import annotations

import json

import pytest

from border_collie_demo.mission import MissionMachine
from border_collie_demo.models import MissionPhase
from border_collie_demo.run_coordinator import RunActivation, RunCoordinator
from border_collie_demo.run_results import ActiveRunError, RunResultStore


def test_activation_key_reuses_the_same_durable_run(tmp_path) -> None:
    machine = MissionMachine()
    results = RunResultStore(tmp_path)
    coordinator = RunCoordinator(machine, results)
    activation = RunActivation(
        target_fruit="pear",
        activation_source="audience_ui",
        idempotency_key="soak-42-run-3",
    )

    first = coordinator.activate(activation)
    second = coordinator.activate(activation)

    assert first.created is True
    assert second.created is False
    assert second.run["run_id"] == first.run["run_id"]
    assert len(results.list_results()) == 1
    assert machine.phase is MissionPhase.PREFLIGHT


def test_activation_key_cannot_be_reused_for_different_intent(tmp_path) -> None:
    coordinator = RunCoordinator(MissionMachine(), RunResultStore(tmp_path))
    coordinator.activate(
        RunActivation(
            target_fruit="pear",
            activation_source="audience_ui",
            idempotency_key="fixed-key",
        )
    )

    with pytest.raises(ActiveRunError, match="different Demo Run"):
        coordinator.activate(
            RunActivation(
                target_fruit="apple",
                activation_source="audience_ui",
                idempotency_key="fixed-key",
            )
        )


def test_transition_is_durable_before_memory_projection_changes(tmp_path) -> None:
    class FailingProjection(MissionMachine):
        def advance(self, reason: str) -> MissionPhase:
            raise RuntimeError("projection failed")

    machine = FailingProjection()
    results = RunResultStore(tmp_path)
    coordinator = RunCoordinator(machine, results)
    started = coordinator.activate(
        RunActivation(target_fruit="pear", activation_source="audience_ui")
    ).run

    with pytest.raises(RuntimeError, match="projection failed"):
        coordinator.advance(
            started["run_id"],
            reason="preflight passed",
            event_reason="CAPTURE_HOME_STARTED",
            message="capture Home",
        )

    persisted = results.get(started["run_id"])
    assert persisted["current_phase"] == "capture_home"
    assert machine.phase is MissionPhase.PREFLIGHT


def test_startup_recovery_seals_an_interrupted_run_with_verified_safety(tmp_path) -> None:
    results = RunResultStore(tmp_path)
    coordinator = RunCoordinator(MissionMachine(), results)
    run = coordinator.activate(
        RunActivation(target_fruit="pear", activation_source="audience_ui")
    ).run

    restarted_machine = MissionMachine()
    restarted = RunCoordinator(restarted_machine, RunResultStore(tmp_path))
    sealed = restarted.recover_interrupted(
        final_safety_state="DISARMED_CONFIRMED"
    )

    assert [item["run_id"] for item in sealed] == [run["run_id"]]
    assert sealed[0]["reason"] == "PROCESS_INTERRUPTED"
    assert sealed[0]["final_safety_state"] == "DISARMED_CONFIRMED"
    assert restarted_machine.phase is MissionPhase.FAILED


def test_journal_events_form_a_hash_chain(tmp_path) -> None:
    coordinator = RunCoordinator(MissionMachine(), RunResultStore(tmp_path))
    run = coordinator.activate(
        RunActivation(target_fruit="pear", activation_source="audience_ui")
    ).run
    events = [
        json.loads(line)
        for line in (tmp_path / run["run_id"] / "events.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert events[0]["previous_event_sha256"] is None
    assert events[1]["previous_event_sha256"] == events[0]["event_sha256"]
    assert len({event["event_id"] for event in events}) == len(events)
