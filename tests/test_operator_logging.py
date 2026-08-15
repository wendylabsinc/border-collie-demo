from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import yaml

from border_collie_demo.black_box import RunBlackBox
from border_collie_demo.operator_logging import OperatorEventLogger

ROOT = Path(__file__).resolve().parents[1]


class _CaptureLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args: object) -> None:
        self.messages.append(message % args)


class _CaptureSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, run_id: str, event: dict[str, object]) -> None:
        self.events.append((run_id, event))


def test_operator_logger_projects_state_failure_and_motion_without_trace_noise() -> None:
    capture = _CaptureLogger()
    logger = OperatorEventLogger(logger=capture)
    run_id = str(uuid4())

    logger.emit(
        run_id,
        {
            "sequence": 4,
            "kind": "mission_event",
            "phase": "approach_fruit",
            "payload": {
                "reason": "APPROACH_FRUIT_STARTED",
                "message": "approach_fruit started",
                "irrelevant_large_value": [1, 2, 3],
            },
        },
    )
    logger.emit(
        run_id,
        {
            "sequence": 5,
            "kind": "motion_command",
            "phase": "approach_fruit",
            "payload": {
                "sequence": 19,
                "motion_path": "factory_avoidance",
                "sender_function": "HardwareManager.return_home_position",
                "active_operation": "return_home",
                "forward_mps": 1.0,
                "yaw_rps": -0.5,
                "reason": "return_home_position",
            },
        },
    )
    logger.emit(
        run_id,
        {
            "sequence": 6,
            "kind": "guidance_decision",
            "phase": "approach_fruit",
            "payload": {"confidence": 0.71},
        },
    )
    logger.emit(
        run_id,
        {
            "sequence": 7,
            "kind": "run_sealed",
            "phase": "failed",
            "payload": {
                "outcome": "FAILED",
                "reason": "RETURN_HOME_FAILURE",
                "failed_phase": "return_home",
                "final_safety_state": "DISARMED_CONFIRMED",
            },
        },
    )

    entries = [json.loads(message.removeprefix("demo_event ")) for message in capture.messages]
    assert entries == [
        {
            "run_id": run_id,
            "sequence": 4,
            "event": "state",
            "phase": "approach_fruit",
            "reason": "APPROACH_FRUIT_STARTED",
            "message": "approach_fruit started",
        },
        {
            "run_id": run_id,
            "sequence": 5,
            "event": "motion",
            "phase": "approach_fruit",
            "command_sequence": 19,
            "motion_path": "factory_avoidance",
            "sender_function": "HardwareManager.return_home_position",
            "active_operation": "return_home",
            "forward_mps": 1.0,
            "yaw_rps": -0.5,
            "reason": "return_home_position",
        },
        {
            "run_id": run_id,
            "sequence": 7,
            "event": "terminal",
            "phase": "failed",
            "outcome": "FAILED",
            "reason": "RETURN_HOME_FAILURE",
            "failed_phase": "return_home",
            "final_safety_state": "DISARMED_CONFIRMED",
        },
    ]


def test_black_box_fans_the_same_event_to_operator_sink(tmp_path) -> None:
    sink = _CaptureSink()
    recorder = RunBlackBox(tmp_path, operator_sink=sink)
    run_id = str(uuid4())

    recorded = recorder.record(
        run_id,
        "motion_command",
        phase="return_home",
        payload={"forward_mps": 1.0, "yaw_rps": 0.0},
    )
    recorder.close()

    assert sink.events == [(run_id, recorded)]


def test_production_servers_suppress_polling_access_noise() -> None:
    main_source = (ROOT / "src/border_collie_demo/main.py").read_text()
    media = yaml.safe_load((ROOT / "media/build.stagefile.yaml").read_text())
    descriptor = yaml.safe_load((ROOT / "wendy.json").read_text())

    assert "access_log=False" in main_source
    assert "--no-access-log" in media["stages"][-1]["cmd"]
    assert (
        descriptor["services"]["app"]["env"]["BORDER_COLLIE_OPERATOR_LOG_ENABLED"]
        == "1"
    )
