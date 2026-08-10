import json

import pytest

from border_collie_demo.fault_injection import (
    AmbiguousResponseInjected,
    DeterministicFaultInjector,
    FaultAction,
    FaultInjectingCallable,
    FaultSpec,
)
from border_collie_demo.flight_recorder import FlightRecorder
from border_collie_demo.mission import MissionMachine
from border_collie_demo.run_coordinator import RunActivation, RunCoordinator
from border_collie_demo.run_results import RunResultStore


def test_fault_injection_cannot_be_enabled_in_production() -> None:
    with pytest.raises(ValueError, match="only in simulation"):
        DeterministicFaultInjector([], runtime_mode="production")


def test_ambiguous_activation_response_retries_the_same_durable_run(tmp_path) -> None:
    coordinator = RunCoordinator(MissionMachine(), RunResultStore(tmp_path))
    activation = RunActivation(
        target_fruit="pear",
        activation_source="audience_ui",
        idempotency_key="fault-run-1",
    )
    injector = DeterministicFaultInjector(
        [FaultSpec("activation.after", 1, FaultAction.RESPONSE_LOST)],
        runtime_mode="simulation",
    )
    activate = FaultInjectingCallable(coordinator.activate, injector, "activation")

    with pytest.raises(AmbiguousResponseInjected):
        activate(activation)
    retried = activate(activation)

    assert retried.created is False
    assert len(RunResultStore(tmp_path).list_results()) == 1
    assert injector.events[0]["point"] == "activation.after"


def test_perception_faults_are_deterministic_and_do_not_mutate_input() -> None:
    status = {
        "ready": True,
        "target_ready": True,
        "generation": "generation-1",
        "source": {"received_monotonic_s": 100.0},
        "detection": {
            "label": "pear",
            "confidence": 0.9,
            "consecutive_detections": 5,
            "completed_monotonic_s": 100.0,
        },
    }
    injector = DeterministicFaultInjector(
        [
            FaultSpec("camera", 2, FaultAction.WEAK_PHANTOM),
            FaultSpec(
                "camera",
                3,
                FaultAction.GENERATION_CHANGE,
                {"generation": "generation-2"},
            ),
        ],
        runtime_mode="simulation",
    )

    assert injector.hit("camera", status)["detection"]["confidence"] == 0.9
    assert injector.hit("camera", status)["detection"]["confidence"] == 0.01
    assert injector.hit("camera", status)["generation"] == "generation-2"
    assert status["generation"] == "generation-1"
    assert status["detection"]["confidence"] == 0.9


def test_flight_recorder_recovers_after_a_partial_tail_write(tmp_path) -> None:
    recorder = FlightRecorder(tmp_path, segment_max_bytes=4096)
    first = recorder.record("before_fault", {"ok": True})
    injector = DeterministicFaultInjector(
        [FaultSpec("evidence.write", 1, FaultAction.PARTIAL_WRITE, {"bytes": 13})],
        runtime_mode="simulation",
    )
    partial = injector.hit("evidence.write", b'{"sequence":999,"kind":"torn"}\n')
    with (tmp_path / "active.ndjson").open("ab") as handle:
        handle.write(partial)

    restarted = FlightRecorder(tmp_path, segment_max_bytes=4096)
    second = restarted.record("after_restart", {"ok": True})
    lines = (tmp_path / "active.ndjson").read_text().splitlines()

    assert second["sequence"] == first["sequence"] + 1
    assert second["previous_event_sha256"] == first["event_sha256"]
    assert [json.loads(line)["kind"] for line in lines] == [
        "before_fault",
        "after_restart",
    ]
