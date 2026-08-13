from __future__ import annotations

from border_collie_demo.black_box import RunBlackBox
from border_collie_demo.run_results import RunResultStore


def test_black_box_is_an_append_only_per_run_timeline(tmp_path) -> None:
    recorder = RunBlackBox(tmp_path)
    results = RunResultStore(tmp_path, black_box=recorder)
    run = results.start_run(
        target_fruit="pear",
        activation_source="audience_ui",
        activation_id="activation-1",
    )

    recorder.record(
        run["run_id"],
        "guidance_decision",
        phase="approach_fruit",
        payload={
            "source_pts": 120,
            "confidence": 0.71,
            "guidance_action": "drive",
            "resulting_command": {"forward_mps": 1.0, "yaw_rps": 0.1},
        },
    )
    results.seal(
        run["run_id"],
        phase="failed",
        outcome="FAILED",
        reason="ARRIVAL_FAILURE",
        message="camera guidance timed out",
        final_safety_state="DISARMED_CONFIRMED",
        failed_phase="approach_fruit",
    )

    trace = recorder.read(run["run_id"])

    assert [event["kind"] for event in trace] == [
        "run_started",
        "mission_event",
        "guidance_decision",
        "mission_event",
        "run_sealed",
    ]
    assert [event["sequence"] for event in trace] == [1, 2, 3, 4, 5]
    assert trace[2]["phase"] == "approach_fruit"
    assert trace[2]["payload"]["source_pts"] == 120
    assert trace[-1]["payload"] == {
        "outcome": "FAILED",
        "reason": "ARRIVAL_FAILURE",
        "failed_phase": "approach_fruit",
        "final_safety_state": "DISARMED_CONFIRMED",
    }
