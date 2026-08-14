from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "home-recorder" / "home_recorder.py"


def load_module():
    spec = importlib.util.spec_from_file_location("home_recorder_service", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def app_event(run_id: str, sequence: int, kind: str, phase: str, payload: dict):
    return {
        "schema_version": 1,
        "run_id": run_id,
        "sequence": sequence,
        "recorded_at_utc": "2026-08-14T00:00:00Z",
        "recorded_monotonic_s": float(sequence),
        "kind": kind,
        "phase": phase,
        "payload": payload,
    }


def test_passive_service_correlates_high_rate_pose_with_home_and_commands(tmp_path) -> None:
    module = load_module()
    run_id = str(uuid4())
    recorder = module.HomeTimelineRecorder(tmp_path, maximum_events_per_run=20)

    recorder.process_app_event(
        app_event(
            run_id,
            1,
            "home_captured",
            "capture_home",
            {"x_m": 1.0, "y_m": 2.0, "yaw_rad": 0.5, "odometry_epoch": "odom-a"},
        )
    )
    # A high-rate pose during fruit work is deliberately not retained. The
    # service keeps the Home reference but spends its bounded budget only once
    # the Home-return timeline begins.
    recorder.record_pose(9.0, 9.0, 0.0, captured_monotonic_s=11.0)
    recorder.process_app_event(
        app_event(
            run_id,
            2,
            "motion_command",
            "return_home",
            {"forward_mps": 1.0, "yaw_rps": 0.5, "motion_path": "factory_avoidance"},
        )
    )
    recorder.record_pose(1.08, 2.06, 0.45, captured_monotonic_s=12.0)
    recorder.process_app_event(
        app_event(
            run_id,
            3,
            "run_sealed",
            "failed",
            {"reason": "RETURN_HOME_FAILURE", "final_safety_state": "DISARMED"},
        )
    )

    rows = [
        json.loads(line)
        for line in recorder.path(run_id).read_text(encoding="utf-8").splitlines()
    ]
    pose = next(row for row in rows if row["kind"] == "pose")
    assert pose["schema_version"] == 1
    assert pose["pose_sequence"] == 2
    assert pose["raw_pose"] == {"x_m": 1.08, "y_m": 2.06, "yaw_rad": 0.45}
    assert pose["home_delta"] == {"x_m": 0.08, "y_m": 0.06}
    assert pose["home_distance_m"] == 0.10
    assert pose["odometry_epoch"] == "odom-a"
    assert pose["command"] == {
        "motion_path": "factory_avoidance",
        "forward_mps": 1.0,
        "yaw_rps": 0.5,
    }
    assert pose["stage"] == "return_home"
    assert rows[-1]["terminal_reason"] == "RETURN_HOME_FAILURE"
    assert recorder.status()["active_run_ids"] == []


def test_passive_service_is_bounded_and_contains_no_robot_writer_or_rpc_imports(
    tmp_path,
) -> None:
    module = load_module()
    run_id = str(uuid4())
    recorder = module.HomeTimelineRecorder(tmp_path, maximum_events_per_run=3)
    recorder.process_app_event(
        app_event(
            run_id,
            1,
            "home_captured",
            "capture_home",
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0, "odometry_epoch": "odom-a"},
        )
    )
    recorder.process_app_event(
        app_event(
            run_id,
            2,
            "home_motion_state",
            "return_home",
            {"armed": False, "forward_mps": 0.0, "yaw_rps": 0.0},
        )
    )
    for index in range(10):
        recorder.record_pose(0.01, 0.0, 0.0, captured_monotonic_s=float(index))

    assert len(recorder.path(run_id).read_text(encoding="utf-8").splitlines()) == 3
    assert recorder.status()["dropped_events"] == 9
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "ChannelPublisher" not in source
    assert "SportClient" not in source
    assert "ObstaclesAvoidClient" not in source
    assert "VuiClient" not in source
    assert "@app.post" not in source
