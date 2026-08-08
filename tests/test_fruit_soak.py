import json
from pathlib import Path

import pytest

from scripts.fruit_soak import (
    HarnessAbort,
    draw_fruit_sequence,
    run_session,
    summarize_run,
    wait_for_ready,
    wait_for_terminal,
)


class FakeClient:
    """Scripted API double: statuses, run results, and activations."""

    def __init__(self, statuses, results_by_id=None, qualified=("apple", "pear")):
        self._statuses = list(statuses)
        self._results = dict(results_by_id or {})
        self._qualified = list(qualified)
        self.activated: list[str] = []
        self.stop_calls = 0

    def status(self):
        return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]

    def fruits(self):
        return {"qualified_fruits": self._qualified}

    def activate(self, fruit):
        self.activated.append(fruit)
        run_id = f"run-{len(self.activated)}"
        return {"run": {"run_id": run_id}}

    def result(self, run_id):
        payloads = self._results[run_id]
        return payloads.pop(0) if len(payloads) > 1 else payloads[0]

    def stop(self):
        self.stop_calls += 1
        return {}


READY = {
    "build_label": "base (demo/base)",
    "mission": {"restart_required": False},
    "activation": {"ready": True, "blockers": []},
    "active_run_id": None,
}
LATCHED = {
    "build_label": "base (demo/base)",
    "mission": {"restart_required": True, "reason": "REMOTE_TAKEOVER"},
    "activation": {"ready": False, "blockers": []},
    "active_run_id": None,
}


def terminal(run_id, outcome="COMPLETED", home=0.05):
    return {
        "run": {
            "run_id": run_id,
            "outcome": outcome,
            "reason": "SUCCESS" if outcome == "COMPLETED" else outcome,
            "message": "done",
            "terminal_measurements": {"home_distance_m": home, "heading_error_rad": 0.01},
        }
    }


def test_sequence_is_seeded_and_only_qualified():
    first = draw_fruit_sequence(["pear", "apple", "banana"], 10, seed=7)
    second = draw_fruit_sequence(["banana", "apple", "pear"], 10, seed=7)
    assert first == second
    assert set(first) <= {"pear", "apple", "banana"}
    assert draw_fruit_sequence(["pear"], 3, seed=1) == ["pear", "pear", "pear"]


def test_sequence_requires_qualified_fruits():
    with pytest.raises(HarnessAbort):
        draw_fruit_sequence([], 10, seed=7)


def test_wait_for_ready_aborts_on_restart_required():
    with pytest.raises(HarnessAbort, match="restart-required"):
        wait_for_ready(FakeClient([LATCHED]), sleep=lambda _: None)


def test_wait_for_ready_times_out_with_blockers():
    blocked = {
        "mission": {"restart_required": False},
        "activation": {
            "ready": False,
            "blockers": [{"name": "camera_perception_ready", "detail": "down"}],
        },
        "active_run_id": None,
    }
    clock_values = iter([0.0, 0.0, 100.0, 100.0])
    with pytest.raises(HarnessAbort, match="camera_perception_ready"):
        wait_for_ready(
            FakeClient([blocked]),
            sleep=lambda _: None,
            clock=lambda: next(clock_values),
        )


def test_wait_for_terminal_stops_robot_on_overrun():
    pending = {"run": {"run_id": "run-1", "outcome": None}}
    client = FakeClient([READY], results_by_id={"run-1": [pending]})
    clock_values = iter([0.0, 500.0])
    run, note = wait_for_terminal(
        client, "run-1", sleep=lambda _: None, clock=lambda: next(clock_values)
    )
    assert client.stop_calls == 1
    assert "harness stop" in note


def test_session_records_every_run_and_build_label(tmp_path: Path):
    client = FakeClient(
        [READY],
        results_by_id={
            "run-1": [terminal("run-1")],
            "run-2": [terminal("run-2", outcome="FAILED", home=0.3)],
        },
    )
    output = tmp_path / "soak.json"
    session = run_session(
        client, runs=2, seed=7, output_path=output, sleep=lambda _: None, log=lambda *_: None
    )
    saved = json.loads(output.read_text())
    assert saved["build_label"] == "base (demo/base)"
    assert saved["seed"] == 7
    assert saved["fruit_sequence"] == client.activated
    assert [r["outcome"] for r in saved["runs"]] == ["COMPLETED", "FAILED"]
    assert saved["runs"][1]["home_distance_m"] == 0.3
    assert session["aborted"] is None


def test_session_aborts_and_persists_partial_on_latch(tmp_path: Path):
    client = FakeClient(
        [READY, READY, LATCHED],
        results_by_id={"run-1": [terminal("run-1")]},
    )
    output = tmp_path / "soak.json"
    session = run_session(
        client, runs=3, seed=7, output_path=output, sleep=lambda _: None, log=lambda *_: None
    )
    saved = json.loads(output.read_text())
    assert len(saved["runs"]) == 1
    assert "restart-required" in saved["aborted"]
    assert session["aborted"] == saved["aborted"]


def test_summarize_run_flattens_terminal_measurements():
    record = summarize_run(terminal("run-9")["run"], "pear", 9)
    assert record["number"] == 9
    assert record["home_distance_m"] == 0.05
    assert record["run_id"] == "run-9"
