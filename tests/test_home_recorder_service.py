from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
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


def test_passive_dds_reader_recovers_after_one_nonfinite_pose_sample(
    tmp_path, monkeypatch
) -> None:
    module = load_module()
    callbacks = []

    class FakeSubscriber:
        def __init__(self, topic, message_type) -> None:
            assert topic == "rt/sportmodestate"
            assert message_type is FakeSportModeState

        def Init(self, callback, queue_depth) -> None:
            assert queue_depth == 1
            callbacks.append(callback)

    class FakeSportModeState:
        pass

    channel = types.ModuleType("unitree_sdk2py.core.channel")
    channel.ChannelFactoryInitialize = lambda domain, interface: (domain, interface)
    channel.ChannelSubscriber = FakeSubscriber
    sport_state = types.ModuleType("unitree_sdk2py.idl.unitree_go.msg.dds_")
    sport_state.SportModeState_ = FakeSportModeState
    monkeypatch.setitem(sys.modules, "unitree_sdk2py", types.ModuleType("unitree_sdk2py"))
    monkeypatch.setitem(
        sys.modules, "unitree_sdk2py.core", types.ModuleType("unitree_sdk2py.core")
    )
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", channel)
    monkeypatch.setitem(
        sys.modules, "unitree_sdk2py.idl", types.ModuleType("unitree_sdk2py.idl")
    )
    monkeypatch.setitem(
        sys.modules,
        "unitree_sdk2py.idl.unitree_go",
        types.ModuleType("unitree_sdk2py.idl.unitree_go"),
    )
    monkeypatch.setitem(
        sys.modules,
        "unitree_sdk2py.idl.unitree_go.msg",
        types.ModuleType("unitree_sdk2py.idl.unitree_go.msg"),
    )
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.idl.unitree_go.msg.dds_", sport_state)
    captured_times = iter((1.0, 2.0, 3.0, 3.0, 4.0, 5.0))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(captured_times))

    recorder = module.HomeTimelineRecorder(tmp_path)
    module.start_pose_subscriber(recorder, "enP8p1s0")
    assert len(callbacks) == 1

    def publish(x_m: float) -> None:
        callbacks[0](
            SimpleNamespace(
                position=[x_m, 2.0],
                imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.25]),
            )
        )

    publish(1.0)
    assert recorder.status()["ready"] is True

    publish(float("nan"))
    fault = recorder.status()
    assert fault["ready"] is False
    assert fault["last_error"] == "non-finite pose sample"
    assert fault["pose_sequence"] == 1
    assert fault["rejected_pose_samples"] == 1

    publish(1.1)
    # A duplicate timestamp is finite but not fresh and cannot qualify
    # recovery by itself.
    publish(1.15)
    publish(1.2)
    assert recorder.status()["ready"] is False

    publish(1.3)
    recovered = recorder.status()
    assert recovered["ready"] is True
    assert recovered["last_error"] is None
    assert recovered["pose_sequence"] == 5
    assert recovered["rejected_pose_samples"] == 1
    assert recovered["consecutive_finite_pose_samples"] == 3


def test_finite_pose_recovery_does_not_clear_independent_recorder_errors(
    tmp_path,
) -> None:
    module = load_module()
    recorder = module.HomeTimelineRecorder(tmp_path)
    recorder.set_error("event journal unavailable")

    recorder.record_pose(float("nan"), 0.0, 0.0, captured_monotonic_s=1.0)
    for sequence in range(2, 5):
        recorder.record_pose(
            float(sequence),
            0.0,
            0.0,
            captured_monotonic_s=float(sequence),
        )

    status = recorder.status()
    assert status["ready"] is False
    assert status["last_error"] == "event journal unavailable"
    assert status["rejected_pose_samples"] == 1


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
    # Every fresh pose is retained from Home capture through the terminal
    # result, including fruit work before the Home-return phases.
    recorder.record_pose(9.0, 9.0, 0.0, captured_monotonic_s=11.0)
    recorder.process_app_event(
        app_event(
            run_id,
            2,
            "motion_command",
            "return_home",
            {
                "forward_mps": 1.0,
                "yaw_rps": 0.5,
                "motion_path": "factory_avoidance",
                "sender_function": "HardwareManager.return_home_position",
                "active_operation": "return_home",
                "recorded_monotonic_s": 11.9,
            },
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
    poses = [row for row in rows if row["kind"] == "pose"]
    assert len(poses) == 2
    assert poses[0]["stage"] == "capture_home"
    pose = poses[1]
    assert pose["schema_version"] == 1
    assert pose["pose_sequence"] == 2
    assert pose["raw_pose"] == {"x_m": 1.08, "y_m": 2.06, "yaw_rad": 0.45}
    assert pose["home_delta"] == {"x_m": 0.08, "y_m": 0.06}
    assert pose["home_distance_m"] == 0.10
    assert pose["odometry_epoch"] == "odom-a"
    assert pose["command"] == {
        "motion_path": "factory_avoidance",
        "sender_function": "HardwareManager.return_home_position",
        "active_operation": "return_home",
        "forward_mps": 1.0,
        "yaw_rps": 0.5,
    }
    assert pose["stage"] == "return_home"
    assert rows[-1]["terminal_reason"] == "RETURN_HOME_FAILURE"
    assert recorder.status()["active_run_ids"] == []


def test_yaw_only_drift_is_recorded_and_warned_once_without_motion_authority(
    tmp_path,
) -> None:
    module = load_module()
    run_id = str(uuid4())

    class CaptureLogger:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def warning(self, message: str, *args: object) -> None:
            self.messages.append(message % args)

    logger = CaptureLogger()
    recorder = module.HomeTimelineRecorder(
        tmp_path,
        yaw_drift_threshold_m=0.03,
        command_active_s=0.5,
        logger=logger,
    )
    recorder.process_app_event(
        app_event(
            run_id,
            1,
            "home_captured",
            "capture_home",
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0, "odometry_epoch": "odom-a"},
        )
    )
    recorder.record_pose(0.0, 0.0, 0.0, captured_monotonic_s=0.9)
    recorder.process_app_event(
        app_event(
            run_id,
            2,
            "motion_command",
            "turn_to_fruit",
            {
                "forward_mps": 0.0,
                "yaw_rps": 0.5,
                "motion_path": "sport_yaw",
                "sender_function": "HardwareManager.turn_relative",
                "active_operation": "measured_turn",
                "recorded_monotonic_s": 1.0,
            },
        )
    )

    recorder.record_pose(0.01, 0.0, 0.1, captured_monotonic_s=1.1)
    recorder.record_pose(0.031, 0.004, 0.2, captured_monotonic_s=1.2)
    recorder.record_pose(0.050, 0.010, 0.3, captured_monotonic_s=1.3)

    rows = [
        json.loads(line)
        for line in recorder.path(run_id).read_text(encoding="utf-8").splitlines()
    ]
    drift_rows = [row for row in rows if row["kind"] == "drift_detected"]
    assert len(drift_rows) == 1
    assert drift_rows[0]["stage"] == "turn_to_fruit"
    assert drift_rows[0]["motion_path"] == "sport_yaw"
    assert drift_rows[0]["command"] == {
        "motion_path": "sport_yaw",
        "sender_function": "HardwareManager.turn_relative",
        "active_operation": "measured_turn",
        "forward_mps": 0.0,
        "yaw_rps": 0.5,
    }
    assert drift_rows[0]["drift_m"] > 0.03
    assert drift_rows[0]["threshold_m"] == 0.03
    assert logger.messages == [
        "DRIFT DETECTED "
        + json.dumps(
            {
                "run_id": run_id,
                "stage": "turn_to_fruit",
                "motion_path": "sport_yaw",
                "sender_function": "HardwareManager.turn_relative",
                "active_operation": "measured_turn",
                "drift_m": drift_rows[0]["drift_m"],
                "threshold_m": 0.03,
                "forward_mps": 0.0,
                "yaw_rps": 0.5,
            },
            separators=(",", ":"),
        )
    ]
    assert recorder.status()["drift_detection_count"] == 1

    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "ChannelPublisher" not in source
    assert "SportClient" not in source
    assert "ObstaclesAvoidClient" not in source


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
